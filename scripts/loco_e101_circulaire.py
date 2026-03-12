"""
LOCO E101 Circulaire — Leave-One-Orientation-Out
Entraîne sur 3 orientations (+ dtroismetres + dcinqmetres), teste sur la 4e.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.benchmark_models import (
    ROOM_CAMPAIGNS, load_cross_room, build_features, localization_summary, save_confusion
)

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CIRCULAIRE_ORIENTATIONS = ["back", "front", "left", "right"]
FIXED_CAMPAIGNS = ["dtroismetres", "dcinqmetres"]

print("Loading E101 (all campaigns)...")
df_all = load_cross_room(["E101"])
print(f"Total samples: {len(df_all)} | campaigns: {sorted(df_all.campaign.unique())}\n")

# Cell lookup for distance computation
cell_lookup = df_all[["grid_cell","grid_x","grid_y","coord_x_m","coord_y_m"]].drop_duplicates().set_index("grid_cell")

MODELS = {
    "KNN":         KNeighborsClassifier(n_neighbors=5),
    "Bagging KNN": BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_estimators=15, random_state=7),
    "LDA+KNN":     None,  # special
    "RF":          RandomForestClassifier(n_estimators=200, random_state=7),
    "HistGB":      HistGradientBoostingClassifier(max_iter=200, random_state=7),
}

results = []

for held_out in CIRCULAIRE_ORIENTATIONS:
    held_campaign = f"E101/{held_out}"
    test_mask = df_all.campaign == held_campaign
    train_mask = ~test_mask

    df_train = df_all[train_mask]
    df_test  = df_all[test_mask]

    X_train = build_features(df_train, include_room=False)
    X_test  = build_features(df_test,  include_room=False)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train["grid_cell"])
    y_test  = le.transform(df_test["grid_cell"])

    print(f"--- Held-out: {held_out} ({len(df_test)} samples) ---")

    row = {"campaign": held_out, "n_test": len(df_test)}

    for name, clf in MODELS.items():
        if name == "LDA+KNN":
            lda = LinearDiscriminantAnalysis()
            Xtr_lda = lda.fit_transform(X_train, y_train)
            Xte_lda = lda.transform(X_test)
            clf_knn = KNeighborsClassifier(n_neighbors=5)
            clf_knn.fit(Xtr_lda, y_train)
            y_pred_idx = clf_knn.predict(Xte_lda)
        else:
            clf.fit(X_train, y_train)
            y_pred_idx = clf.predict(X_test)

        y_pred_cells = le.inverse_transform(y_pred_idx)
        y_true_cells = df_test["grid_cell"].values
        res = localization_summary(df_test, y_pred_cells, cell_lookup)

        print(f"  {name:<12}: acc={res.cell_accuracy:.3f}  err={res.mean_error_m:.3f}m  p90={res.p90_error_m:.3f}m")
        row[f"{name}_acc"]  = res.cell_accuracy
        row[f"{name}_err"]  = res.mean_error_m
        row[f"{name}_p90"]  = res.p90_error_m

    results.append(row)

df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("LOCO E101 Circulaire — Cell accuracy per held-out orientation")
print("="*80)
acc_cols = [c for c in df_results.columns if c.endswith("_acc")]
print(df_results[["campaign","n_test"] + acc_cols].to_string(index=False))

print("\n" + "="*80)
print("LOCO E101 Circulaire — Mean error (m) per held-out orientation")
print("="*80)
err_cols = [c for c in df_results.columns if c.endswith("_err")]
print(df_results[["campaign","n_test"] + err_cols].to_string(index=False))

print("\n" + "="*80)
print("LOCO E101 Circulaire — MEAN across all held-out orientations")
print("="*80)
for name in MODELS:
    mean_acc = df_results[f"{name}_acc"].mean()
    mean_err = df_results[f"{name}_err"].mean()
    mean_p90 = df_results[f"{name}_p90"].mean()
    print(f"  {name:<12}  acc={mean_acc:.3f}  mean_err={mean_err:.3f}m  p90={mean_p90:.3f}m")

out_path = REPORT_DIR / "loco_e101_circulaire.csv"
df_results.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")
