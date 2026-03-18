#!/usr/bin/env python3
"""Leave-One-Campaign-Out (LOCO) benchmark within B121.

For each B121 campaign, trains on the 2 others and tests on the held-out one.
This measures robustness to router lateral placement changes within the same room.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]

_B121_ROOT = PROJECT_ROOT / "data" / "B121" / "dtroismetres"

CAMPAIGNS: dict[str, CampaignSpec] = {
    "routeur_centre": CampaignSpec(_B121_ROOT / "routeurcentretableau", router_distance_m=3.0),
    "routeur_gauche": CampaignSpec(_B121_ROOT / "routeurgauche", router_distance_m=3.0),
    "routeur_droite": CampaignSpec(_B121_ROOT / "routeuradroitedutableau", router_distance_m=3.0),
}


def load_campaign(name: str, spec: CampaignSpec) -> pd.DataFrame:
    df = load_measurements([spec])
    df["campaign"] = name
    return df


def localization_error(y_true_cells: pd.Series, y_pred: np.ndarray, cell_lookup: pd.DataFrame) -> dict[str, float]:
    pred_coords = cell_lookup.loc[y_pred][["coord_x_m", "coord_y_m"]].to_numpy()
    true_coords = cell_lookup.loc[y_true_cells.to_numpy()][["coord_x_m", "coord_y_m"]].to_numpy()
    errors = np.linalg.norm(true_coords - pred_coords, axis=1)
    acc = float((y_pred == y_true_cells.to_numpy()).mean())
    return {
        "cell_accuracy": acc,
        "mean_error_m": float(errors.mean()),
        "p90_error_m": float(np.percentile(errors, 90)),
    }


def fit_nn_lknn(X_train: np.ndarray, y_train: pd.Series) -> EmbeddingKnnLocalizer:
    cfg = EmbeddingKnnConfig(
        hidden_layer_sizes=(48, 24),
        k_neighbors=5,
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=400,
        tol=1e-3,
        random_state=7,
    )
    model = EmbeddingKnnLocalizer(config=cfg)
    model.fit(X_train, y_train)
    return model


def run_loco() -> pd.DataFrame:
    print("Loading all B121 campaigns...")
    frames: dict[str, pd.DataFrame] = {
        name: load_campaign(name, spec) for name, spec in CAMPAIGNS.items()
    }

    all_df = pd.concat(frames.values(), ignore_index=True)
    cell_lookup = (
        all_df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )
    labels = sorted(all_df["grid_cell"].unique())
    campaign_names = list(CAMPAIGNS.keys())

    print(f"Total samples: {len(all_df)} | cells: {len(labels)} | campaigns: {len(campaign_names)}\n")

    rows: list[dict[str, float | int | str]] = []
    for held_out in campaign_names:
        train_df = pd.concat([df for name, df in frames.items() if name != held_out], ignore_index=True)
        test_df = frames[held_out]
        X_train = train_df[FEATURE_COLUMNS].to_numpy()
        X_test = test_df[FEATURE_COLUMNS].to_numpy()
        y_train = train_df["grid_cell"]
        y_test = test_df["grid_cell"]
        n_test = len(test_df)

        print(f"--- Held-out: {held_out} ({n_test} samples) ---")
        row: dict[str, float | int | str] = {"held_out": held_out, "n_test": int(n_test)}

        knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn.fit(X_train, y_train)
        m = localization_error(y_test, knn.predict(X_test), cell_lookup)
        row.update({f"knn_{k}": v for k, v in m.items()})
        print(f"  KNN:        acc={m['cell_accuracy']:.3f}  err={m['mean_error_m']:.3f}m  p90={m['p90_error_m']:.3f}m")

        bag = BaggingClassifier(
            estimator=KNeighborsClassifier(n_neighbors=7, weights="distance"),
            n_estimators=15,
            random_state=7,
            n_jobs=-1,
        )
        bag.fit(X_train, y_train)
        m = localization_error(y_test, bag.predict(X_test), cell_lookup)
        row.update({f"bag_knn_{k}": v for k, v in m.items()})
        print(f"  Bagging KNN:acc={m['cell_accuracy']:.3f}  err={m['mean_error_m']:.3f}m  p90={m['p90_error_m']:.3f}m")

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        knn_lda = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn_lda.fit(lda.transform(X_train), y_train)
        m = localization_error(y_test, knn_lda.predict(lda.transform(X_test)), cell_lookup)
        row.update({f"lda_knn_{k}": v for k, v in m.items()})
        print(f"  LDA+KNN:    acc={m['cell_accuracy']:.3f}  err={m['mean_error_m']:.3f}m  p90={m['p90_error_m']:.3f}m")

        rf = RandomForestClassifier(n_estimators=220, max_depth=14, min_samples_leaf=2, random_state=7, n_jobs=-1)
        rf.fit(X_train, y_train)
        m = localization_error(y_test, rf.predict(X_test), cell_lookup)
        row.update({f"rf_{k}": v for k, v in m.items()})
        print(f"  RF:         acc={m['cell_accuracy']:.3f}  err={m['mean_error_m']:.3f}m  p90={m['p90_error_m']:.3f}m")

        et = ExtraTreesClassifier(n_estimators=200, random_state=7, n_jobs=-1)
        et.fit(X_train, y_train)
        m = localization_error(y_test, et.predict(X_test), cell_lookup)
        row.update({f"et_{k}": v for k, v in m.items()})
        print(f"  ExtraTrees: acc={m['cell_accuracy']:.3f}  err={m['mean_error_m']:.3f}m  p90={m['p90_error_m']:.3f}m")

        hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=8, max_iter=200, random_state=7)
        hgb.fit(X_train, y_train)
        m = localization_error(y_test, hgb.predict(X_test), cell_lookup)
        row.update({f"hgb_{k}": v for k, v in m.items()})
        print(f"  HistGB:     acc={m['cell_accuracy']:.3f}  err={m['mean_error_m']:.3f}m  p90={m['p90_error_m']:.3f}m")

        nn_model = fit_nn_lknn(X_train, y_train)
        m = localization_error(y_test, np.array(nn_model.predict(X_test)), cell_lookup)
        row.update({f"nn_lknn_{k}": v for k, v in m.items()})
        print(f"  NN+L-KNN:   acc={m['cell_accuracy']:.3f}  err={m['mean_error_m']:.3f}m  p90={m['p90_error_m']:.3f}m")

        rows.append(row)
        print()

    return pd.DataFrame(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_loco()

    models = ["knn", "bag_knn", "lda_knn", "rf", "et", "hgb", "nn_lknn"]
    model_labels = ["KNN", "Bagging KNN", "LDA+KNN", "RandomForest", "ExtraTrees", "HistGB", "NN+L-KNN"]

    print("=" * 80)
    print("LOCO B121 — Cell accuracy per held-out campaign")
    print("=" * 80)
    acc_rows = []
    for _, row in df.iterrows():
        r = {"campaign": row["held_out"], "n_test": int(row["n_test"])}
        for m, label in zip(models, model_labels):
            r[label] = f"{row[f'{m}_cell_accuracy']:.3f}"
        acc_rows.append(r)
    acc_df = pd.DataFrame(acc_rows).set_index("campaign")
    print(acc_df.to_string())

    print("\n" + "=" * 80)
    print("LOCO B121 — Mean error (m) per held-out campaign")
    print("=" * 80)
    err_rows = []
    for _, row in df.iterrows():
        r = {"campaign": row["held_out"], "n_test": int(row["n_test"])}
        for m, label in zip(models, model_labels):
            r[label] = f"{row[f'{m}_mean_error_m']:.3f}"
        err_rows.append(r)
    err_df = pd.DataFrame(err_rows).set_index("campaign")
    print(err_df.to_string())

    print("\n" + "=" * 80)
    print("LOCO B121 — MEAN across all held-out campaigns")
    print("=" * 80)
    for m, label in zip(models, model_labels):
        acc = df[f"{m}_cell_accuracy"].mean()
        err = df[f"{m}_mean_error_m"].mean()
        p90 = df[f"{m}_p90_error_m"].mean()
        print(f"  {label:<14} acc={acc:.3f}  mean_err={err:.3f}m  p90={p90:.3f}m")

    out_path = REPORT_DIR / "loco_b121.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    summary: dict[str, dict[str, object]] = {}
    for m, label in zip(models, model_labels):
        summary[label] = {
            "mean_cell_accuracy": float(df[f"{m}_cell_accuracy"].mean()),
            "mean_error_m": float(df[f"{m}_mean_error_m"].mean()),
            "mean_p90_m": float(df[f"{m}_p90_error_m"].mean()),
            "per_campaign": {
                row["held_out"]: {
                    "cell_accuracy": float(row[f"{m}_cell_accuracy"]),
                    "mean_error_m": float(row[f"{m}_mean_error_m"]),
                }
                for _, row in df.iterrows()
            },
        }
    json_path = REPORT_DIR / "loco_b121_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to {json_path}")


if __name__ == "__main__":
    main()
