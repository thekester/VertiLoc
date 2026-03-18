#!/usr/bin/env python3
"""Evaluate raw vs clamped board errors from a predictions CSV."""

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

from localization.board_geometry import BoardGeometry, add_board_geometry  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "predictions.csv",
        help="Predictions CSV containing true/pred grid indices.",
    )
    parser.add_argument(
        "--router-height-m",
        type=float,
        default=0.75,
        help="Router height used for the board projection.",
    )
    parser.add_argument(
        "--output-prefix",
        default="board_clamping_eval",
        help="Prefix used for output artifacts inside reports/benchmarks.",
    )
    return parser.parse_args()


def _project_grid_columns(
    df: pd.DataFrame,
    *,
    grid_x_col: str,
    grid_y_col: str,
    router_distance_col: str,
    geometry: BoardGeometry,
    router_height_m: float,
    prefix: str,
) -> pd.DataFrame:
    temp = pd.DataFrame(
        {
            "grid_x": df[grid_x_col].to_numpy(dtype=float),
            "grid_y": df[grid_y_col].to_numpy(dtype=float),
            "router_distance_m": df[router_distance_col].to_numpy(dtype=float),
        }
    )
    projected = add_board_geometry(
        temp,
        geometry=geometry,
        router_height_m=router_height_m,
        grid_x_top_is_zero=True,
    )
    out = pd.DataFrame(index=df.index)
    for col in [
        "board_x_m",
        "board_z_m",
        "board_x_clamped_m",
        "board_z_clamped_m",
        "is_within_board",
        "board_projection_distance_m",
        "router_esp_3d_m",
        "router_esp_3d_clamped_m",
    ]:
        out[f"{prefix}_{col}"] = projected[col].to_numpy()
    return out


def main() -> None:
    args = parse_args()
    if not args.predictions_csv.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {args.predictions_csv}")

    geometry = BoardGeometry()
    pred_df = pd.read_csv(args.predictions_csv)
    required = {"true_grid_x", "true_grid_y", "pred_grid_x", "pred_grid_y"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {sorted(missing)}")

    if "true_router_distance_m" in pred_df.columns:
        router_distance_col = "true_router_distance_m"
    elif "router_distance_m" in pred_df.columns:
        router_distance_col = "router_distance_m"
    else:
        raise ValueError("Predictions CSV must contain `true_router_distance_m` or `router_distance_m`.")

    true_proj = _project_grid_columns(
        pred_df,
        grid_x_col="true_grid_x",
        grid_y_col="true_grid_y",
        router_distance_col=router_distance_col,
        geometry=geometry,
        router_height_m=float(args.router_height_m),
        prefix="true",
    )
    pred_proj = _project_grid_columns(
        pred_df,
        grid_x_col="pred_grid_x",
        grid_y_col="pred_grid_y",
        router_distance_col=router_distance_col,
        geometry=geometry,
        router_height_m=float(args.router_height_m),
        prefix="pred",
    )

    out_df = pd.concat([pred_df.copy(), true_proj, pred_proj], axis=1)
    out_df["board_error_raw_m"] = np.sqrt(
        np.square(out_df["true_board_x_m"] - out_df["pred_board_x_m"])
        + np.square(out_df["true_board_z_m"] - out_df["pred_board_z_m"])
    )
    out_df["board_error_clamped_m"] = np.sqrt(
        np.square(out_df["true_board_x_clamped_m"] - out_df["pred_board_x_clamped_m"])
        + np.square(out_df["true_board_z_clamped_m"] - out_df["pred_board_z_clamped_m"])
    )
    out_df["pred_outside_board"] = ~out_df["pred_is_within_board"].astype(bool)
    out_df["true_outside_board"] = ~out_df["true_is_within_board"].astype(bool)
    out_df["board_error_gain_m"] = out_df["board_error_raw_m"] - out_df["board_error_clamped_m"]

    summary = {
        "predictions_csv": str(args.predictions_csv),
        "router_height_m": float(args.router_height_m),
        "n_samples": int(len(out_df)),
        "raw_board_error_mean_m": float(out_df["board_error_raw_m"].mean()),
        "clamped_board_error_mean_m": float(out_df["board_error_clamped_m"].mean()),
        "raw_board_error_p90_m": float(np.percentile(out_df["board_error_raw_m"], 90)),
        "clamped_board_error_p90_m": float(np.percentile(out_df["board_error_clamped_m"], 90)),
        "mean_error_gain_m": float(out_df["board_error_gain_m"].mean()),
        "pred_outside_board_ratio": float(out_df["pred_outside_board"].mean()),
        "true_outside_board_ratio": float(out_df["true_outside_board"].mean()),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix).strip() or "board_clamping_eval"
    csv_path = REPORT_DIR / f"{prefix}.csv"
    json_path = REPORT_DIR / f"{prefix}.json"
    out_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved detailed CSV to {csv_path}")


if __name__ == "__main__":
    main()
