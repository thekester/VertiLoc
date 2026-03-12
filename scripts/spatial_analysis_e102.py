#!/usr/bin/env python3
"""Spatial analysis: which cells/zones are hardest to predict in E102?

Uses 5-fold CV on all 6 campaigns combined. Reports per-cell and per-zone metrics.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from localization.data import CampaignSpec, load_measurements  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
FEATURES = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
_E102_ROOT = PROJECT_ROOT / "data" / "E102"

CAMPAIGNS = {
    "exp1": CampaignSpec(_E102_ROOT / "exp1",               router_distance_m=4.0),
    "exp2": CampaignSpec(_E102_ROOT / "exp2",               router_distance_m=4.0),
    "exp3": CampaignSpec(_E102_ROOT / "exp3",               router_distance_m=4.0),
    "exp4": CampaignSpec(_E102_ROOT / "exp4",               router_distance_m=4.0),
    "exp5": CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0),
    "exp6": CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0),
}

# Cell width=0.25m (y axis), height=0.30m (x axis)
CELL_W = 0.25
CELL_H = 0.30


def main():
    print("Loading E102...")
    frames = []
    for name, spec in CAMPAIGNS.items():
        df = load_measurements([spec])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    cell_lookup = (all_df[["grid_cell","coord_x_m","coord_y_m","grid_x","grid_y"]]
                   .drop_duplicates("grid_cell").set_index("grid_cell"))

    X = all_df[FEATURES].to_numpy()
    y = all_df["grid_cell"].to_numpy()

    # 5-fold CV with RF — collect per-sample predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=7, n_jobs=-1)

    all_true, all_pred = [], []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        all_true.extend(y[te])
        all_pred.extend(preds)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    pred_coords = cell_lookup.loc[all_pred][["coord_x_m","coord_y_m"]].to_numpy()
    true_coords = cell_lookup.loc[all_true][["coord_x_m","coord_y_m"]].to_numpy()
    errors = np.linalg.norm(true_coords - pred_coords, axis=1)

    # Per-cell stats
    cell_stats = []
    for cell in sorted(cell_lookup.index):
        mask = all_true == cell
        if not mask.any():
            continue
        cell_errors = errors[mask]
        cell_preds  = all_pred[mask]
        acc = float((cell_preds == cell).mean())
        meta = cell_lookup.loc[cell]
        cell_stats.append({
            "cell": cell,
            "grid_x": int(meta["grid_x"]),
            "grid_y": int(meta["grid_y"]),
            "coord_x_m": float(meta["coord_x_m"]),
            "coord_y_m": float(meta["coord_y_m"]),
            "n_samples": int(mask.sum()),
            "cell_accuracy": acc,
            "mean_error_m": float(cell_errors.mean()),
            "p90_error_m": float(np.percentile(cell_errors, 90)),
        })

    df_cells = pd.DataFrame(cell_stats).sort_values("mean_error_m", ascending=False)

    print("\n=== TOP 10 hardest cells (highest mean error) ===")
    print(df_cells[["cell","grid_x","grid_y","cell_accuracy","mean_error_m","p90_error_m"]].head(10).to_string(index=False))

    print("\n=== TOP 10 easiest cells (lowest mean error) ===")
    print(df_cells[["cell","grid_x","grid_y","cell_accuracy","mean_error_m","p90_error_m"]].tail(10).to_string(index=False))

    # Per-row analysis (grid_x = ligne)
    print("\n=== Per-row (ligne) accuracy ===")
    row_stats = df_cells.groupby("grid_x").agg(
        n_cells=("cell","count"),
        mean_acc=("cell_accuracy","mean"),
        mean_err=("mean_error_m","mean"),
    ).reset_index()
    print(row_stats.to_string(index=False))

    # Per-column zone: negative (gauche), center (0-5), positive far (6-10)
    def zone(gy):
        if gy < 0:   return "left_neg"
        elif gy <= 5: return "center"
        else:        return "right_far"

    df_cells["zone"] = df_cells["grid_y"].apply(zone)
    print("\n=== Per-zone accuracy ===")
    zone_stats = df_cells.groupby("zone").agg(
        n_cells=("cell","count"),
        mean_acc=("cell_accuracy","mean"),
        mean_err=("mean_error_m","mean"),
    ).reset_index()
    print(zone_stats.to_string(index=False))

    # Save
    df_cells.to_csv(REPORT_DIR / "spatial_analysis_e102_cells.csv", index=False)
    row_stats.to_csv(REPORT_DIR / "spatial_analysis_e102_rows.csv", index=False)
    zone_stats.to_csv(REPORT_DIR / "spatial_analysis_e102_zones.csv", index=False)
    print(f"\nSaved -> {REPORT_DIR}/spatial_analysis_e102_*.csv")

    # Overall stats
    overall_acc = float((all_true == all_pred).mean())
    overall_err = float(errors.mean())
    print(f"\nOverall (5-fold CV, RF): acc={overall_acc:.3f}  mean_err={overall_err:.3f}m")


if __name__ == "__main__":
    main()
