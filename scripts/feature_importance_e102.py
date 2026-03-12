#!/usr/bin/env python3
"""Feature importance analysis on E102.

Tests all subsets of features: Signal alone, Signal+Noise,
all 5 features, with and without each antenna signal.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from localization.data import CampaignSpec, load_measurements  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
ALL_FEATURES = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
_E102_ROOT = PROJECT_ROOT / "data" / "E102"

CAMPAIGNS = {
    "exp1": CampaignSpec(_E102_ROOT / "exp1",               router_distance_m=4.0),
    "exp2": CampaignSpec(_E102_ROOT / "exp2",               router_distance_m=4.0),
    "exp3": CampaignSpec(_E102_ROOT / "exp3",               router_distance_m=4.0),
    "exp4": CampaignSpec(_E102_ROOT / "exp4",               router_distance_m=4.0),
    "exp5": CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0),
    "exp6": CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0),
}

FEATURE_SUBSETS = {
    "Signal only":              ["Signal"],
    "Signal+Noise":             ["Signal", "Noise"],
    "Signal+A1":                ["Signal", "signal_A1"],
    "Signal+A2":                ["Signal", "signal_A2"],
    "Signal+A3":                ["Signal", "signal_A3"],
    "Signal+A1+A2":             ["Signal", "signal_A1", "signal_A2"],
    "Signal+A1+A2+A3":          ["Signal", "signal_A1", "signal_A2", "signal_A3"],
    "All 5 features":           ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"],
    "No Noise":                 ["Signal", "signal_A1", "signal_A2", "signal_A3"],
    "No Signal (antennes only)":["Noise", "signal_A1", "signal_A2", "signal_A3"],
    "Antennes only":            ["signal_A1", "signal_A2", "signal_A3"],
}


def cv_metrics(X, y, clf, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, errs = [], []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        accs.append((preds == y[te]).mean())
    return float(np.mean(accs))


def main():
    print("Loading E102...")
    frames = []
    for name, spec in CAMPAIGNS.items():
        df = load_measurements([spec])
        df["campaign"] = name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    results = []
    models = {
        "RF":     RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1),
        "KNN":    KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "HistGB": HistGradientBoostingClassifier(learning_rate=0.1, max_depth=8, max_iter=150, random_state=7),
    }

    print(f"{'Feature subset':<30} {'RF acc':>8} {'KNN acc':>8} {'HistGB acc':>10}")
    print("-" * 60)
    for subset_name, cols in FEATURE_SUBSETS.items():
        X = all_df[cols].to_numpy()
        y = all_df["grid_cell"].to_numpy()
        row = {"feature_subset": subset_name, "n_features": len(cols), "features": ", ".join(cols)}
        for mname, clf in models.items():
            acc = cv_metrics(X, y, clf)
            row[f"{mname}_acc"] = acc
        results.append(row)
        print(f"  {subset_name:<28} {row['RF_acc']:>8.3f} {row['KNN_acc']:>8.3f} {row['HistGB_acc']:>10.3f}")

    # RF feature importances
    print("\n--- RF feature importances (trained on all data) ---")
    rf_full = RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=-1)
    X_full = all_df[ALL_FEATURES].to_numpy()
    y_full = all_df["grid_cell"].to_numpy()
    rf_full.fit(X_full, y_full)
    for feat, imp in sorted(zip(ALL_FEATURES, rf_full.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<14} {imp:.4f}")

    df_res = pd.DataFrame(results)
    out = REPORT_DIR / "feature_importance_e102.csv"
    df_res.to_csv(out, index=False)

    importances = dict(zip(ALL_FEATURES, [float(x) for x in rf_full.feature_importances_]))
    json_out = REPORT_DIR / "feature_importance_e102.json"
    with open(json_out, "w") as f:
        json.dump({"rf_importances": importances, "subset_results": results}, f, indent=2)

    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
