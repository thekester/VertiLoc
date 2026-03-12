#!/usr/bin/env python3
"""Learning curve: how many E102 campaigns are needed for good accuracy?

For each subset size k=1..5, tries all C(6,k) combinations as training,
tests on the held-out campaign(s), reports mean accuracy.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from localization.data import CampaignSpec, load_measurements  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
_E102_ROOT = PROJECT_ROOT / "data" / "E102"

CAMPAIGNS = {
    "exp1_back_right":  CampaignSpec(_E102_ROOT / "exp1",               router_distance_m=4.0),
    "exp2_front_right": CampaignSpec(_E102_ROOT / "exp2",               router_distance_m=4.0),
    "exp3_front_left":  CampaignSpec(_E102_ROOT / "exp3",               router_distance_m=4.0),
    "exp4_back_left":   CampaignSpec(_E102_ROOT / "exp4",               router_distance_m=4.0),
    "exp5_ground":      CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0),
    "exp6_1m50":        CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0),
}


def loc_metrics(y_true, y_pred, cell_lookup):
    errors = np.linalg.norm(
        cell_lookup.loc[y_true][["coord_x_m","coord_y_m"]].values -
        cell_lookup.loc[y_pred][["coord_x_m","coord_y_m"]].values, axis=1)
    return float((y_pred == y_true).mean()), float(errors.mean()), float(np.percentile(errors, 90))


def main():
    print("Loading campaigns...")
    frames = {}
    for name, spec in CAMPAIGNS.items():
        df = load_measurements([spec])
        df["campaign"] = name
        frames[name] = df

    all_df = pd.concat(frames.values(), ignore_index=True)
    cell_lookup = (all_df[["grid_cell","coord_x_m","coord_y_m"]]
                   .drop_duplicates("grid_cell").set_index("grid_cell"))

    names = list(CAMPAIGNS.keys())
    results = []  # {n_train, train_set, test_set, model, acc, err, p90}

    for n_train in range(1, 6):
        print(f"\n--- n_train={n_train} ({len(list(itertools.combinations(names, n_train)))} combos) ---")
        for train_names in itertools.combinations(names, n_train):
            test_names = [n for n in names if n not in train_names]
            train_df = pd.concat([frames[n] for n in train_names], ignore_index=True)
            test_df  = pd.concat([frames[n] for n in test_names],  ignore_index=True)
            X_tr = train_df[FEATURE_COLUMNS].to_numpy()
            X_te = test_df [FEATURE_COLUMNS].to_numpy()
            y_tr = train_df["grid_cell"]
            y_te = test_df ["grid_cell"].to_numpy()

            for model_name, clf in [
                ("KNN",    KNeighborsClassifier(n_neighbors=7, weights="distance")),
                ("RF",     RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)),
                ("HistGB", HistGradientBoostingClassifier(learning_rate=0.1, max_depth=8, max_iter=150, random_state=7)),
            ]:
                clf.fit(X_tr, y_tr)
                acc, err, p90 = loc_metrics(y_te, clf.predict(X_te), cell_lookup)
                results.append({
                    "n_train": n_train,
                    "train_campaigns": "+".join(train_names),
                    "n_test_samples": len(test_df),
                    "model": model_name,
                    "cell_accuracy": acc,
                    "mean_error_m": err,
                    "p90_error_m": p90,
                })

        # Summary for this n_train
        subset = [r for r in results if r["n_train"] == n_train]
        for model_name in ["KNN", "RF", "HistGB"]:
            vals = [r for r in subset if r["model"] == model_name]
            mean_acc = np.mean([v["cell_accuracy"] for v in vals])
            mean_err = np.mean([v["mean_error_m"]  for v in vals])
            print(f"  {model_name:<8} mean_acc={mean_acc:.3f}  mean_err={mean_err:.3f}m  (over {len(vals)} combos)")

    # n_train=5 (train on 5, test on 1) — equivalent to LOCO
    print("\n--- n_train=5 (= LOCO, 6 combos) ---")
    for train_names in itertools.combinations(names, 5):
        test_names = [n for n in names if n not in train_names]
        train_df = pd.concat([frames[n] for n in train_names], ignore_index=True)
        test_df  = frames[test_names[0]]
        X_tr = train_df[FEATURE_COLUMNS].to_numpy()
        X_te = test_df [FEATURE_COLUMNS].to_numpy()
        y_tr = train_df["grid_cell"]
        y_te = test_df ["grid_cell"].to_numpy()
        for model_name, clf in [
            ("KNN",    KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ("RF",     RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)),
            ("HistGB", HistGradientBoostingClassifier(learning_rate=0.1, max_depth=8, max_iter=150, random_state=7)),
        ]:
            clf.fit(X_tr, y_tr)
            acc, err, p90 = loc_metrics(y_te, clf.predict(X_te), cell_lookup)
            results.append({
                "n_train": 5,
                "train_campaigns": "+".join(train_names),
                "n_test_samples": len(test_df),
                "model": model_name,
                "cell_accuracy": acc,
                "mean_error_m": err,
                "p90_error_m": p90,
            })

    df_res = pd.DataFrame(results)
    out = REPORT_DIR / "learning_curve_e102.csv"
    df_res.to_csv(out, index=False)

    print("\n" + "="*60)
    print("LEARNING CURVE SUMMARY — mean over all combos")
    print("="*60)
    summary_rows = []
    for n in range(1, 6):
        for model in ["KNN", "RF", "HistGB"]:
            sub = df_res[(df_res["n_train"]==n) & (df_res["model"]==model)]
            row = {"n_train": n, "model": model,
                   "mean_acc": sub["cell_accuracy"].mean(),
                   "mean_err": sub["mean_error_m"].mean(),
                   "mean_p90": sub["p90_error_m"].mean()}
            summary_rows.append(row)
            print(f"  n={n} {model:<8} acc={row['mean_acc']:.3f}  err={row['mean_err']:.3f}m")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORT_DIR / "learning_curve_e102_summary.csv", index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
