#!/usr/bin/env python3
"""Compare fine cell prediction vs macro-zone prediction on E101."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.board_geometry import BoardGeometry, add_board_geometry, add_board_zones  # noqa: E402
from localization.data import CampaignSpec, load_measurements  # noqa: E402

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
ROUTER_HEIGHT_M = 0.75
CAMPAIGNS: dict[str, tuple[CampaignSpec, str]] = {
    "E101/back": (CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "back", router_distance_m=3.0), "back"),
    "E101/front": (CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "front", router_distance_m=3.0), "front"),
    "E101/left": (CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "left", router_distance_m=3.0), "left"),
    "E101/right": (CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "right", router_distance_m=3.0), "right"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone-cols", type=int, default=3, help="Number of horizontal macro-zones.")
    parser.add_argument("--zone-rows", type=int, default=3, help="Number of vertical macro-zones.")
    parser.add_argument("--seeds", default="7,17,27,37,47", help="Comma-separated random seeds.")
    parser.add_argument("--output-prefix", default="e101_zone_benchmark", help="Artifact prefix.")
    return parser.parse_args()


def load_e101_dataframe(zone_cols: int, zone_rows: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    geometry = BoardGeometry()
    for campaign_name, (spec, orientation_label) in CAMPAIGNS.items():
        df = load_measurements([spec]).copy()
        df["campaign"] = campaign_name
        df["orientation_label"] = orientation_label
        df = add_board_geometry(
            df,
            geometry=geometry,
            router_height_m=ROUTER_HEIGHT_M,
            grid_x_top_is_zero=True,
            coordinate_mode="fit_to_data",
        )
        df = add_board_zones(df, geometry=geometry, n_cols=zone_cols, n_rows=zone_rows, use_clamped_coordinates=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fit_predict_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def zone_neighbor_accuracy(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    row_gap = np.abs(true_df["zone_row"].to_numpy(dtype=int) - pred_df["zone_row"].to_numpy(dtype=int))
    col_gap = np.abs(true_df["zone_col"].to_numpy(dtype=int) - pred_df["zone_col"].to_numpy(dtype=int))
    return float(((row_gap <= 1) & (col_gap <= 1)).mean())


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_e101_dataframe(zone_cols=int(args.zone_cols), zone_rows=int(args.zone_rows))
    seeds = [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]

    print(
        f"Loaded E101 zoning benchmark: samples={len(df)} | cells={df['grid_cell'].nunique()} | "
        f"zones={df['zone_id'].nunique()} | zone_grid={args.zone_rows}x{args.zone_cols}"
    )

    results: list[dict[str, float]] = []
    pred_frames: list[pd.DataFrame] = []
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for seed in seeds:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        stratify_labels = df["grid_cell"] + "|" + df["orientation_label"]
        train_idx, test_idx = next(split.split(df, stratify_labels))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        pred_cell = fit_predict_rf(X_train, train_df["grid_cell"].to_numpy(), X_test, seed=seed)
        pred_zone = fit_predict_rf(X_train, train_df["zone_id"].to_numpy(), X_test, seed=seed)

        cell_lookup = (
            train_df[["grid_cell", "coord_x_m", "coord_y_m"]]
            .drop_duplicates("grid_cell")
            .set_index("grid_cell")
        )
        pred_coords = cell_lookup.loc[pred_cell, ["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
        true_coords = test_df[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
        cell_errors_m = np.linalg.norm(true_coords - pred_coords, axis=1)

        zone_lookup = (
            train_df[["zone_id", "zone_row", "zone_col"]]
            .drop_duplicates("zone_id")
            .set_index("zone_id")
        )
        pred_zone_meta = zone_lookup.loc[pred_zone].reset_index(drop=True)
        true_zone_meta = test_df[["zone_row", "zone_col"]].reset_index(drop=True)

        split_pred = test_df[
            ["campaign", "grid_cell", "orientation_label", "zone_id", "zone_row", "zone_col", "coord_x_m", "coord_y_m"]
        ].copy()
        split_pred["seed"] = seed
        split_pred["pred_cell"] = pred_cell
        split_pred["pred_zone"] = pred_zone
        split_pred["cell_error_m"] = cell_errors_m
        pred_frames.append(split_pred)

        results.append(
            {
                "seed": seed,
                "cell_accuracy": float(accuracy_score(test_df["grid_cell"], pred_cell)),
                "cell_f1_macro": float(f1_score(test_df["grid_cell"], pred_cell, average="macro")),
                "cell_mean_distance_m": float(cell_errors_m.mean()),
                "zone_accuracy": float(accuracy_score(test_df["zone_id"], pred_zone)),
                "zone_f1_macro": float(f1_score(test_df["zone_id"], pred_zone, average="macro")),
                "zone_neighbor_accuracy": zone_neighbor_accuracy(true_zone_meta, pred_zone_meta),
            }
        )
        print(
            f"seed={seed} "
            f"cell_acc={results[-1]['cell_accuracy']:.3f} "
            f"cell_err={results[-1]['cell_mean_distance_m']:.3f}m "
            f"zone_acc={results[-1]['zone_accuracy']:.3f} "
            f"zone_neighbor_acc={results[-1]['zone_neighbor_accuracy']:.3f}"
        )

    pred_df = pd.concat(pred_frames, ignore_index=True)
    results_df = pd.DataFrame(results)
    zone_labels = sorted(df["zone_id"].unique().tolist())
    zone_cm = confusion_matrix(pred_df["zone_id"], pred_df["pred_zone"], labels=zone_labels)

    summary = {
        "dataset": {
            "room": "E101",
            "samples": int(len(df)),
            "cells": int(df["grid_cell"].nunique()),
            "zones": int(df["zone_id"].nunique()),
            "zone_rows": int(args.zone_rows),
            "zone_cols": int(args.zone_cols),
            "router_height_m": ROUTER_HEIGHT_M,
            "zone_labels": zone_labels,
        },
        "cell_task": {
            "accuracy_mean": float(results_df["cell_accuracy"].mean()),
            "accuracy_std": float(results_df["cell_accuracy"].std(ddof=0)),
            "f1_macro_mean": float(results_df["cell_f1_macro"].mean()),
            "mean_distance_m": float(results_df["cell_mean_distance_m"].mean()),
        },
        "zone_task": {
            "accuracy_mean": float(results_df["zone_accuracy"].mean()),
            "accuracy_std": float(results_df["zone_accuracy"].std(ddof=0)),
            "f1_macro_mean": float(results_df["zone_f1_macro"].mean()),
            "neighbor_accuracy_mean": float(results_df["zone_neighbor_accuracy"].mean()),
            "confusion_matrix": zone_cm.tolist(),
        },
        "per_zone": [
            {
                "zone_id": zone_id,
                "samples": int(len(group)),
                "zone_accuracy": float(accuracy_score(group["zone_id"], group["pred_zone"])),
                "cell_accuracy": float(accuracy_score(group["grid_cell"], group["pred_cell"])),
                "cell_mean_distance_m": float(group["cell_error_m"].mean()),
            }
            for zone_id, group in pred_df.groupby("zone_id", sort=True)
        ],
    }

    prefix = str(args.output_prefix).strip() or "e101_zone_benchmark"
    summary_path = REPORT_DIR / f"{prefix}_summary.json"
    pred_path = REPORT_DIR / f"{prefix}_predictions.csv"
    zone_cm_path = REPORT_DIR / f"{prefix}_zone_confusion_matrix.csv"
    per_seed_path = REPORT_DIR / f"{prefix}_per_seed.csv"
    snippet_path = REPORT_DIR / f"{prefix}_snippet.md"

    summary_path.write_text(json.dumps(summary, indent=2))
    pred_df.to_csv(pred_path, index=False)
    results_df.to_csv(per_seed_path, index=False)
    pd.DataFrame(zone_cm, index=zone_labels, columns=zone_labels).to_csv(zone_cm_path)
    snippet = (
        "### Benchmark zones E101\n\n"
        f"- Cellule exacte: accuracy={summary['cell_task']['accuracy_mean']:.3f}, "
        f"F1 macro={summary['cell_task']['f1_macro_mean']:.3f}, "
        f"erreur moyenne={summary['cell_task']['mean_distance_m']:.3f} m.\n"
        f"- Zone {args.zone_rows}x{args.zone_cols}: accuracy={summary['zone_task']['accuracy_mean']:.3f}, "
        f"F1 macro={summary['zone_task']['f1_macro_mean']:.3f}, "
        f"accuracy voisinage={summary['zone_task']['neighbor_accuracy_mean']:.3f}.\n"
    )
    snippet_path.write_text(snippet)

    print("\nSummary")
    print(json.dumps(summary["cell_task"], indent=2))
    print(json.dumps(summary["zone_task"], indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
