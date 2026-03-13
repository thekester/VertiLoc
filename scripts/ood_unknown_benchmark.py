"""OOD benchmark for unknown room/campaign detection with NN+L-KNN.

This script evaluates lightweight OOD signals already exposed by
EmbeddingKnnLocalizer:
- softmax energy score,
- nearest-neighbor distance in embedding space,
- calibrated OR rule (energy OR distance threshold exceedance).

Protocols:
- unknown_room: hold out one room as OOD, train on the remaining rooms.
- unknown_campaign_e102: hold out one E102 campaign as OOD, train on the others.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer
from scripts.benchmark_models import FEATURE_COLUMNS, load_cross_room

OUT_DIR = ROOT / "reports" / "benchmarks"
DEFAULT_CSV = OUT_DIR / "ood_unknown_benchmark.csv"
DEFAULT_JSON = OUT_DIR / "ood_unknown_benchmark.json"


def _group_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_keys = df[["room", "campaign", "grid_cell"]].astype(str).agg("::".join, axis=1).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=group_keys))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _fpr_at_tpr95(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    mask = tpr >= 0.95
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


def _score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, score)),
        "aupr_ood": float(average_precision_score(y_true, score)),
        "fpr95": _fpr_at_tpr95(y_true, score),
    }


def _evaluate_one_split(
    known_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    *,
    random_state: int,
    id_test_size: float,
    config: EmbeddingKnnConfig,
    ood_energy_percentile: float,
    ood_distance_percentile: float,
    ood_temperature: float,
) -> dict[str, float | int | str]:
    train_df, id_df = _group_split(known_df, test_size=id_test_size, random_state=random_state)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["grid_cell"].to_numpy()
    X_id = id_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    X_ood = ood_df[FEATURE_COLUMNS].to_numpy(dtype=float)

    localizer = EmbeddingKnnLocalizer(config=config).fit(X_train, y_train)
    localizer.calibrate_ood(
        energy_percentile=ood_energy_percentile,
        distance_percentile=ood_distance_percentile,
        temperature=ood_temperature,
    )

    id_scores = localizer.ood_scores(X_id)
    ood_scores = localizer.ood_scores(X_ood)
    id_flags = localizer.is_ood(X_id, scores=id_scores)
    ood_flags = localizer.is_ood(X_ood, scores=ood_scores)

    y_bin = np.concatenate(
        [
            np.zeros(len(X_id), dtype=int),  # 0 = in-distribution
            np.ones(len(X_ood), dtype=int),  # 1 = OOD
        ]
    )
    energy_all = np.concatenate([id_scores["ood_energy"], ood_scores["ood_energy"]])
    dist_all = np.concatenate([id_scores["ood_embedding_distance"], ood_scores["ood_embedding_distance"]])

    # Combined continuous score from relative exceedance to calibrated thresholds.
    e_thr = float(localizer.ood_energy_threshold_)
    d_thr = float(localizer.ood_distance_threshold_)
    eps = 1e-9
    energy_rel = (energy_all - e_thr) / (abs(e_thr) + eps)
    dist_rel = (dist_all - d_thr) / (abs(d_thr) + eps)
    combo_score = np.maximum(energy_rel, dist_rel)

    row: dict[str, float | int | str] = {
        "n_train": int(len(train_df)),
        "n_id": int(len(X_id)),
        "n_ood": int(len(X_ood)),
        "ood_energy_threshold": e_thr,
        "ood_distance_threshold": d_thr,
        "ood_energy_percentile": float(localizer.ood_energy_percentile_),
        "ood_distance_percentile": float(localizer.ood_distance_percentile_),
        "ood_temperature": float(localizer.ood_temperature_),
        "rule_tpr": float(np.mean(ood_flags)) if len(ood_flags) else float("nan"),
        "rule_fpr": float(np.mean(id_flags)) if len(id_flags) else float("nan"),
        "rule_id_accept_rate": float(1.0 - np.mean(id_flags)) if len(id_flags) else float("nan"),
        "rule_ood_reject_rate": float(np.mean(ood_flags)) if len(ood_flags) else float("nan"),
    }

    for prefix, score in (
        ("energy", energy_all),
        ("distance", dist_all),
        ("combo", combo_score),
    ):
        for name, value in _score_metrics(y_bin, score).items():
            row[f"{prefix}_{name}"] = float(value)
    return row


def _iter_splits(protocol: str) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    if protocol == "unknown_room":
        df = load_cross_room()
        splits = []
        for holdout in sorted(df["room"].unique()):
            known = df[df["room"] != holdout].copy()
            ood = df[df["room"] == holdout].copy()
            splits.append((str(holdout), known, ood))
        return splits

    if protocol == "unknown_campaign_e102":
        df = load_cross_room(room_filter=["E102"])
        splits = []
        for holdout in sorted(df["campaign"].unique()):
            known = df[df["campaign"] != holdout].copy()
            ood = df[df["campaign"] == holdout].copy()
            splits.append((str(holdout), known, ood))
        return splits

    raise ValueError(f"Unknown protocol: {protocol}")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    protocols = [args.protocol] if args.protocol != "all" else ["unknown_room", "unknown_campaign_e102"]
    rows: list[dict] = []

    config = EmbeddingKnnConfig(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation=args.activation,
        max_iter=args.max_iter,
        learning_rate_init=args.learning_rate,
        alpha=args.alpha,
        k_neighbors=args.k_neighbors,
        random_state=args.random_state,
    )

    for protocol in protocols:
        for holdout, known_df, ood_df in _iter_splits(protocol):
            row = _evaluate_one_split(
                known_df,
                ood_df,
                random_state=args.random_state,
                id_test_size=args.id_test_size,
                config=config,
                ood_energy_percentile=args.ood_energy_percentile,
                ood_distance_percentile=args.ood_distance_percentile,
                ood_temperature=args.ood_temperature,
            )
            row["protocol"] = protocol
            row["holdout"] = holdout
            rows.append(row)
            print(
                f"[{protocol}] holdout={holdout} | "
                f"rule_tpr={row['rule_tpr']:.3f} rule_fpr={row['rule_fpr']:.3f} | "
                f"combo_auroc={row['combo_auroc']:.3f} combo_fpr95={row['combo_fpr95']:.3f}"
            )

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c not in {"protocol", "holdout"}]
    summary = (
        df.groupby("protocol")[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    payload = {
        "rows": int(len(df)),
        "protocols": protocols,
        "params": {
            "id_test_size": float(args.id_test_size),
            "random_state": int(args.random_state),
            "hidden_layers": [int(v) for v in args.hidden_layers],
            "activation": args.activation,
            "max_iter": int(args.max_iter),
            "learning_rate": float(args.learning_rate),
            "alpha": float(args.alpha),
            "k_neighbors": int(args.k_neighbors),
            "ood_energy_percentile": float(args.ood_energy_percentile),
            "ood_distance_percentile": float(args.ood_distance_percentile),
            "ood_temperature": float(args.ood_temperature),
        },
        "summary_by_protocol_mean": summary,
    }
    return df, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark OOD detection for unknown room/campaign.")
    parser.add_argument(
        "--protocol",
        type=str,
        default="all",
        choices=["all", "unknown_room", "unknown_campaign_e102"],
        help="Evaluation protocol.",
    )
    parser.add_argument("--id-test-size", type=float, default=0.2, help="Fraction of known-data used as ID test.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "logistic", "identity"])
    parser.add_argument("--max-iter", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--ood-energy-percentile", type=float, default=95.0)
    parser.add_argument("--ood-distance-percentile", type=float, default=95.0)
    parser.add_argument("--ood-temperature", type=float, default=1.0)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df, payload = run(args)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(payload, indent=2))

    print(f"Saved per-split metrics to {args.output_csv}")
    print(f"Saved summary to {args.output_json}")


if __name__ == "__main__":
    main()
