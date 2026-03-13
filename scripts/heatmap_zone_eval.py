"""Zone-aware evaluation for raw RSSI vs heatmap KNN under generalization splits.

Focuses on neighborhood quality instead of exact-cell accuracy:
- exact cell accuracy,
- within 1-ring neighborhood (hex distance <= 1),
- within 2-ring neighborhood (hex distance <= 2),
- within all neighborhood radii (Zone±k for every k up to the grid hex diameter),
- metric-distance thresholds (<= 0.5 m, <= 1.0 m),
- average grid and metric errors.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_models import FEATURE_COLUMNS, _rssi_to_heatmaps, load_cross_room

OUT_DIR = Path("reports/benchmarks")


def _odd_r_to_cube(row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert odd-r offset coordinates (row, col) to cube coordinates.

    Assumption: grid_x is the row index and grid_y is the column index on an
    odd-r horizontal hex layout.
    """
    cube_x = col - ((row - (row & 1)) // 2)
    cube_z = row
    cube_y = -cube_x - cube_z
    return cube_x, cube_y, cube_z


def _hex_distance(
    true_row: np.ndarray,
    true_col: np.ndarray,
    pred_row: np.ndarray,
    pred_col: np.ndarray,
) -> np.ndarray:
    tx, ty, tz = _odd_r_to_cube(true_row, true_col)
    px, py, pz = _odd_r_to_cube(pred_row, pred_col)
    return np.maximum.reduce([np.abs(px - tx), np.abs(py - ty), np.abs(pz - tz)])


def _max_hex_diameter(cell_lookup: pd.DataFrame) -> int:
    rows = cell_lookup["grid_x"].to_numpy(dtype=int)
    cols = cell_lookup["grid_y"].to_numpy(dtype=int)
    rx, ry, rz = _odd_r_to_cube(rows, cols)
    max_d = 0
    for i in range(len(cell_lookup)):
        d = np.maximum.reduce([np.abs(rx[i] - rx), np.abs(ry[i] - ry), np.abs(rz[i] - rz)])
        max_d = max(max_d, int(np.max(d)))
    return max_d


def _zone_curve(hex_dist: np.ndarray, max_zone: int) -> dict[str, float]:
    curve: dict[str, float] = {}
    for k in range(max_zone + 1):
        curve[f"zone{k}_acc"] = float(np.mean(hex_dist <= k))
    return curve


def _zone_metrics(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    cell_lookup: pd.DataFrame,
    *,
    max_zone: int,
) -> dict[str, float]:
    pred_meta = cell_lookup.loc[y_pred]

    true_x = test_df["grid_x"].to_numpy(dtype=int)
    true_y = test_df["grid_y"].to_numpy(dtype=int)
    pred_x = pred_meta["grid_x"].to_numpy(dtype=int)
    pred_y = pred_meta["grid_y"].to_numpy(dtype=int)

    dx = np.abs(pred_x - true_x)
    dy = np.abs(pred_y - true_y)
    cheb = np.maximum(dx, dy)
    manhattan = dx + dy
    hex_dist = _hex_distance(true_x, true_y, pred_x, pred_y)

    true_coords = test_df[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    pred_coords = pred_meta[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    dist_m = np.linalg.norm(true_coords - pred_coords, axis=1)

    metrics = {
        "exact_acc": float(np.mean(hex_dist == 0)),
        "zone1_acc": float(np.mean(hex_dist <= 1)),
        "zone2_acc": float(np.mean(hex_dist <= 2)),
        "within_0_5m_acc": float(np.mean(dist_m <= 0.5)),
        "within_1_0m_acc": float(np.mean(dist_m <= 1.0)),
        "mean_error_m": float(np.mean(dist_m)),
        "p90_error_m": float(np.percentile(dist_m, 90)),
        "mean_hex_cells": float(np.mean(hex_dist)),
        "mean_chebyshev_cells": float(np.mean(cheb)),
        "mean_manhattan_cells": float(np.mean(manhattan)),
    }
    metrics.update(_zone_curve(hex_dist, max_zone))
    return metrics


def _run_split(df: pd.DataFrame, group_col: str, out_csv: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    cell_lookup = (
        df[["grid_cell", "grid_x", "grid_y", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )

    max_zone = _max_hex_diameter(cell_lookup)

    for held_out in sorted(df[group_col].unique()):
        train_df = df[df[group_col] != held_out].reset_index(drop=True)
        test_df = df[df[group_col] == held_out].reset_index(drop=True)

        X_train_raw = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        X_test_raw = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)

        raw_scaler = StandardScaler()
        X_train_raw_z = raw_scaler.fit_transform(X_train_raw)
        X_test_raw_z = raw_scaler.transform(X_test_raw)
        raw_knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
        raw_knn.fit(X_train_raw_z, train_df["grid_cell"])
        raw_pred = raw_knn.predict(X_test_raw_z)
        raw_metrics = _zone_metrics(test_df, raw_pred, cell_lookup, max_zone=max_zone)

        X_train_hm = _rssi_to_heatmaps(X_train_raw).reshape(len(train_df), -1)
        X_test_hm = _rssi_to_heatmaps(X_test_raw).reshape(len(test_df), -1)
        hm_scaler = StandardScaler()
        X_train_hm_z = hm_scaler.fit_transform(X_train_hm)
        X_test_hm_z = hm_scaler.transform(X_test_hm)
        hm_knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
        hm_knn.fit(X_train_hm_z, train_df["grid_cell"])
        hm_pred = hm_knn.predict(X_test_hm_z)
        hm_metrics = _zone_metrics(test_df, hm_pred, cell_lookup, max_zone=max_zone)

        row = {group_col: held_out, "n_test": int(len(test_df))}
        for k, v in raw_metrics.items():
            row[f"raw_{k}"] = v
        for k, v in hm_metrics.items():
            row[f"heatmap_{k}"] = v
        rows.append(row)

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_csv, index=False)
    avg = res_df.mean(numeric_only=True).to_dict()
    avg["max_zone"] = max_zone
    return res_df, avg


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cross_df = load_cross_room()
    loro_df, loro_avg = _run_split(cross_df, "room", OUT_DIR / "heatmap_zone_loro_room.csv")

    e102_df = load_cross_room(room_filter=["E102"])
    loco_df, loco_avg = _run_split(e102_df, "campaign", OUT_DIR / "heatmap_zone_loco_e102.csv")

    summary = {
        "date": "2026-03-12",
        "loro_room_rows": int(len(loro_df)),
        "loco_e102_rows": int(len(loco_df)),
        "loro_room_avg": loro_avg,
        "loco_e102_avg": loco_avg,
    }
    (OUT_DIR / "heatmap_zone_summary.json").write_text(json.dumps(summary, indent=2))

    print("Saved:", OUT_DIR / "heatmap_zone_loro_room.csv")
    print("Saved:", OUT_DIR / "heatmap_zone_loco_e102.csv")
    print("Saved:", OUT_DIR / "heatmap_zone_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
