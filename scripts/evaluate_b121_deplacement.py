"""Evaluate B121 deplacement traces with the exported inference model.

This script keeps `data/B121/deplacement` fully out of training. It simply
loads each CSV as an online inference batch, aggregates predictions per file,
and writes machine-readable summaries for challenging OOD/motion cases.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization import VertiLocInferenceModel

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
DEFAULT_INPUT_DIR = ROOT / "data" / "B121" / "deplacement"
DEFAULT_MODEL = ROOT / "reports" / "localizer.joblib"
DEFAULT_METRICS = ROOT / "reports" / "latest_metrics.json"
DEFAULT_CSV = ROOT / "reports" / "benchmarks" / "b121_deplacement_eval.csv"
DEFAULT_JSON = ROOT / "reports" / "benchmarks" / "b121_deplacement_eval.json"

TRACE_DESCRIPTIONS = {
    "rond_411": "Cercle en mouvement demarrant autour de 4_11.",
    "diag_44": "Diagonale demarrant a 4_4.",
    "diag_4-4": "Diagonale demarrant a 4_-4.",
    "hauteur_47": "Trajet vertical demarrant a 4_7.",
    "b_c": "ESP32 au fond de la piece, centre.",
    "b_g": "ESP32 au fond de la piece, gauche.",
    "b_d": "ESP32 au fond de la piece, droite.",
    "c_c": "ESP32 au centre de la piece, centre.",
    "c_d": "ESP32 au centre de la piece, droite.",
    "c_g": "ESP32 au centre/cote gauche.",
}


@dataclass(frozen=True)
class ExpectedTarget:
    name: str
    kind: str
    target_cell: str | None = None
    description: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate B121 deplacement traces with the inference model.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--room", choices=["auto", "B121"], default="B121")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def infer_expected_target(name: str) -> ExpectedTarget:
    match = re.fullmatch(r"(-?\d+_-?\d+)", name)
    if match:
        return ExpectedTarget(name=name, kind="static_cell", target_cell=match.group(1), description="Static hold-out point.")
    return ExpectedTarget(
        name=name,
        kind="trajectory_or_semantic",
        description=TRACE_DESCRIPTIONS.get(name, "Non-standard inference-only trace."),
    )


def parse_cell(cell: str) -> tuple[int, int]:
    left, right = str(cell).split("_", 1)
    return int(left), int(right)


def euclidean_error_m(cell_true: str, cell_pred: str) -> float:
    tx, ty = parse_cell(cell_true)
    px, py = parse_cell(cell_pred)
    dx = (ty - py) * 0.25
    dy = (tx - px) * 0.30
    return float((dx * dx + dy * dy) ** 0.5)


def compress_prediction_path(cells: list[str]) -> list[str]:
    if not cells:
        return []
    compressed = [cells[0]]
    for cell in cells[1:]:
        if cell != compressed[-1]:
            compressed.append(cell)
    return compressed


def summarize_prediction_file(csv_path: Path, model: VertiLocInferenceModel, room: str, top_k: int) -> tuple[dict[str, object], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    pred_df = model.predict_dataframe(
        df[FEATURE_COLUMNS].copy(),
        top_k=top_k,
        room=room,
        include_neighbors=False,
    )
    expected = infer_expected_target(csv_path.stem)
    dominant_cell = str(pred_df["pred_cell"].mode().iloc[0])
    compressed_path = compress_prediction_path(pred_df["pred_cell"].astype(str).tolist())
    summary: dict[str, object] = {
        "trace_name": csv_path.stem,
        "trace_kind": expected.kind,
        "description": expected.description,
        "n_samples": int(len(pred_df)),
        "target_cell": expected.target_cell,
        "dominant_pred_cell": dominant_cell,
        "dominant_pred_share": float((pred_df["pred_cell"] == dominant_cell).mean()),
        "n_unique_pred_cells": int(pred_df["pred_cell"].nunique()),
        "compressed_path": " -> ".join(compressed_path[:20]),
        "compressed_path_len": int(len(compressed_path)),
        "mean_confidence": float(pred_df["confidence"].mean()),
        "median_confidence": float(pred_df["confidence"].median()),
        "ood_reject_rate": float(pred_df["ood_is_unknown"].mean()),
        "mean_ood_energy": float(pred_df["ood_energy"].mean()),
        "mean_ood_embedding_distance": float(pred_df["ood_embedding_distance"].mean()),
        "pred_router_distance_mode_m": float(pred_df["pred_router_distance_m"].mode(dropna=True).iloc[0])
        if pred_df["pred_router_distance_m"].notna().any()
        else None,
        "within_board_rate": float(pred_df["within_board"].mean()) if "within_board" in pred_df else None,
    }
    if expected.target_cell is not None:
        train_labels = getattr(model.localizer, "train_labels_", None)
        known_labels = set(map(str, np.unique(train_labels))) if train_labels is not None else set()
        summary["target_known_in_training_labels"] = bool(expected.target_cell in known_labels)
        summary["exact_match_rate"] = float((pred_df["pred_cell"] == expected.target_cell).mean())
        summary["dominant_cell_error_m"] = euclidean_error_m(expected.target_cell, dominant_cell)
    else:
        summary["target_known_in_training_labels"] = None
        summary["exact_match_rate"] = None
        summary["dominant_cell_error_m"] = None

    pred_out = pred_df.copy()
    pred_out.insert(0, "trace_name", csv_path.stem)
    pred_out.insert(1, "trace_kind", expected.kind)
    if expected.target_cell is not None:
        pred_out.insert(2, "target_cell", expected.target_cell)
    return summary, pred_out


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = VertiLocInferenceModel.load(args.model, metrics_path=args.metrics)

    summaries: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for csv_path in sorted(args.input_dir.glob("*.csv")):
        summary, pred_df = summarize_prediction_file(csv_path, model, room=str(args.room), top_k=int(args.top_k))
        summaries.append(summary)
        prediction_frames.append(pred_df)

    summary_df = pd.DataFrame(summaries).sort_values(["trace_kind", "trace_name"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)

    predictions_json_path = args.output_json.with_name(args.output_json.stem + "_predictions.csv")
    predictions_df.to_csv(predictions_json_path, index=False)

    payload = {
        "input_dir": str(args.input_dir),
        "model": str(args.model),
        "metrics": str(args.metrics),
        "room": str(args.room),
        "n_traces": int(len(summary_df)),
        "summary": summaries,
        "prediction_csv": str(predictions_json_path),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary_df.to_string(index=False))
        print(f"\nSaved summary CSV to {args.output_csv}")
        print(f"Saved summary JSON to {args.output_json}")
        print(f"Saved per-sample predictions to {predictions_json_path}")


if __name__ == "__main__":
    main()
