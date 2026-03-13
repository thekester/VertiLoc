#!/usr/bin/env python3
"""LOCO E102 benchmark for post-prediction spatial smoothing.

Compares NN+L-KNN raw predictions vs smoothed predictions (neighbor / crf-lite)
on the 6 leave-one-campaign-out folds of E102.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer  # noqa: E402
from localization.pipeline import apply_spatial_smoothing  # noqa: E402


REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
_E102_ROOT = PROJECT_ROOT / "data" / "E102"

CAMPAIGNS: dict[str, CampaignSpec] = {
    "exp1_back_right": CampaignSpec(_E102_ROOT / "exp1", router_distance_m=4.0),
    "exp2_front_right": CampaignSpec(_E102_ROOT / "exp2", router_distance_m=4.0),
    "exp3_front_left": CampaignSpec(_E102_ROOT / "exp3", router_distance_m=4.0),
    "exp4_back_left": CampaignSpec(_E102_ROOT / "exp4", router_distance_m=4.0),
    "exp5_ground": CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0),
    "exp6_1m50": CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0),
}


def localization_error(y_true_cells: pd.Series, y_pred: np.ndarray, cell_lookup: pd.DataFrame) -> dict:
    pred_coords = cell_lookup.loc[y_pred][["coord_x_m", "coord_y_m"]].to_numpy()
    true_coords = cell_lookup.loc[y_true_cells.to_numpy()][["coord_x_m", "coord_y_m"]].to_numpy()
    errors = np.linalg.norm(true_coords - pred_coords, axis=1)
    acc = float((y_pred == y_true_cells.to_numpy()).mean())
    return {
        "cell_accuracy": acc,
        "mean_error_m": float(errors.mean()),
        "p90_error_m": float(np.percentile(errors, 90)),
    }


def fit_nn_lknn(X_train: np.ndarray, y_train: pd.Series, *, max_iter: int) -> EmbeddingKnnLocalizer:
    cfg = EmbeddingKnnConfig(
        hidden_layer_sizes=(48, 24),
        k_neighbors=5,
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=max_iter,
        tol=1e-3,
        random_state=7,
    )
    return EmbeddingKnnLocalizer(config=cfg).fit(X_train, y_train)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOCO E102 benchmark for spatial smoothing.")
    parser.add_argument("--mode", type=str, default="neighbor", choices=["neighbor", "crf-lite"])
    parser.add_argument("--metric", type=str, default="hex", choices=["hex", "chebyshev", "manhattan"])
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--max-confidence", type=float, default=0.95)
    parser.add_argument("--max-iter", type=int, default=300, help="NN encoder max_iter.")
    parser.add_argument("--output-prefix", type=str, default="spatial_smoothing_loco_e102")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading E102 campaigns...")
    frames = {}
    for name, spec in CAMPAIGNS.items():
        df = load_measurements([spec])
        df["campaign"] = name
        frames[name] = df

    all_df = pd.concat(frames.values(), ignore_index=True)
    cell_lookup = (
        all_df[["grid_cell", "grid_x", "grid_y", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )
    campaign_names = list(CAMPAIGNS.keys())
    print(f"Total samples={len(all_df)} | campaigns={len(campaign_names)}")

    rows: list[dict] = []
    for held_out in campaign_names:
        train_df = pd.concat([df for name, df in frames.items() if name != held_out], ignore_index=True)
        test_df = frames[held_out]

        X_train = train_df[FEATURE_COLUMNS].to_numpy()
        X_test = test_df[FEATURE_COLUMNS].to_numpy()
        y_train = train_df["grid_cell"]
        y_test = test_df["grid_cell"]

        model = fit_nn_lknn(X_train, y_train, max_iter=args.max_iter)
        class_order = list(model.knn_.classes_)

        y_pred_raw = np.asarray(model.predict(X_test))
        y_proba_raw = np.asarray(model.predict_proba(X_test))

        y_proba_s = apply_spatial_smoothing(
            y_proba_raw,
            class_order,
            cell_lookup,
            mode=args.mode,
            alpha=float(args.alpha),
            beta=float(args.beta),
            n_iters=int(args.iters),
            metric=args.metric,
            radius=int(args.radius),
            max_confidence=float(args.max_confidence),
        )
        idx_s = np.argmax(y_proba_s, axis=1)
        y_pred_s = np.array([class_order[i] for i in idx_s], dtype=object)

        raw_m = localization_error(y_test, y_pred_raw, cell_lookup)
        sm_m = localization_error(y_test, y_pred_s, cell_lookup)
        y_true = y_test.to_numpy()
        fixed = int(np.sum((y_pred_raw != y_true) & (y_pred_s == y_true)))
        broken = int(np.sum((y_pred_raw == y_true) & (y_pred_s != y_true)))

        row = {
            "held_out": held_out,
            "n_test": int(len(test_df)),
            "raw_cell_accuracy": raw_m["cell_accuracy"],
            "smooth_cell_accuracy": sm_m["cell_accuracy"],
            "delta_cell_accuracy": sm_m["cell_accuracy"] - raw_m["cell_accuracy"],
            "raw_mean_error_m": raw_m["mean_error_m"],
            "smooth_mean_error_m": sm_m["mean_error_m"],
            "delta_mean_error_m": sm_m["mean_error_m"] - raw_m["mean_error_m"],
            "raw_p90_error_m": raw_m["p90_error_m"],
            "smooth_p90_error_m": sm_m["p90_error_m"],
            "delta_p90_error_m": sm_m["p90_error_m"] - raw_m["p90_error_m"],
            "n_fixed": fixed,
            "n_broken": broken,
            "net_fixed": fixed - broken,
        }
        rows.append(row)
        print(
            f"{held_out}: acc {row['raw_cell_accuracy']:.4f}->{row['smooth_cell_accuracy']:.4f} "
            f"(d={row['delta_cell_accuracy']:+.4f}) | err {row['raw_mean_error_m']:.3f}->{row['smooth_mean_error_m']:.3f}m "
            f"(d={row['delta_mean_error_m']:+.3f}m) | fix={fixed} break={broken}"
        )

    df_out = pd.DataFrame(rows).sort_values("held_out").reset_index(drop=True)

    summary = {
        "mode": args.mode,
        "metric": args.metric,
        "radius": int(args.radius),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "iters": int(args.iters),
        "max_confidence": float(args.max_confidence),
        "nn_max_iter": int(args.max_iter),
        "mean_raw_cell_accuracy": float(df_out["raw_cell_accuracy"].mean()),
        "mean_smooth_cell_accuracy": float(df_out["smooth_cell_accuracy"].mean()),
        "mean_delta_cell_accuracy": float(df_out["delta_cell_accuracy"].mean()),
        "mean_raw_mean_error_m": float(df_out["raw_mean_error_m"].mean()),
        "mean_smooth_mean_error_m": float(df_out["smooth_mean_error_m"].mean()),
        "mean_delta_mean_error_m": float(df_out["delta_mean_error_m"].mean()),
        "total_fixed": int(df_out["n_fixed"].sum()),
        "total_broken": int(df_out["n_broken"].sum()),
        "total_net_fixed": int(df_out["net_fixed"].sum()),
    }

    stem = f"{args.output_prefix}_{args.mode}_{args.metric}_r{args.radius}"
    csv_path = REPORT_DIR / f"{stem}.csv"
    json_path = REPORT_DIR / f"{stem}.json"
    df_out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))

    print("\nAverage across LOCO folds:")
    print(
        f"acc {summary['mean_raw_cell_accuracy']:.4f}->{summary['mean_smooth_cell_accuracy']:.4f} "
        f"(delta={summary['mean_delta_cell_accuracy']:+.4f})"
    )
    print(
        f"err {summary['mean_raw_mean_error_m']:.3f}->{summary['mean_smooth_mean_error_m']:.3f}m "
        f"(delta={summary['mean_delta_mean_error_m']:+.3f}m)"
    )
    print(
        f"fix={summary['total_fixed']} break={summary['total_broken']} "
        f"net={summary['total_net_fixed']:+d}"
    )
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
