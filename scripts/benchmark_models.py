"""Benchmark NN+L-KNN variants and auxiliary classifiers across rooms.

This script mirrors the notebook scenarios:
- room-agnostic NN+L-KNN (leave-one-room-out),
- room-aware NN+L-KNN with room one-hot,
- LogisticRegression head for router distance (room-agnostic embedding),
- Room classification accuracy from the same embedding.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer  # noqa: E402
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Campaign folders grouped by room, excluding circular trajectories.
ROOM_CAMPAIGNS = {
    "D005": [
        PROJECT_ROOT / "data" / "D005" / "ddeuxmetres",
        PROJECT_ROOT / "data" / "D005" / "dquatremetres",
    ],
    "E101": [
        PROJECT_ROOT / "data" / "E101" / "dtroismetres",
        PROJECT_ROOT / "data" / "E101" / "dcinqmetres",
    ],
}

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"


@dataclass
class BenchmarkResult:
    cell_accuracy: float
    mean_error_m: float
    p90_error_m: float
    extra: dict


def load_cross_room() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for room, folders in ROOM_CAMPAIGNS.items():
        for folder in folders:
            if folder.exists():
                df_room = load_measurements([CampaignSpec(folder)])
                df_room["room"] = room
                df_room["campaign"] = f"{room}/{folder.name}"
                frames.append(df_room)
            else:
                missing.append(folder)

    if missing:
        raise FileNotFoundError(f"Missing campaign folders: {missing}")
    if not frames:
        raise RuntimeError("No measurement data found for any room.")

    df = pd.concat(frames, ignore_index=True)
    room_ohe = pd.get_dummies(df["room"], prefix="room")
    df = pd.concat([df, room_ohe], axis=1)
    return df


def build_features(df: pd.DataFrame, include_room: bool) -> np.ndarray:
    base = df[FEATURE_COLUMNS].reset_index(drop=True)
    if not include_room:
        return base.to_numpy()
    room_cols = [c for c in df.columns if c.startswith("room_")]
    if not room_cols:
        raise ValueError("Room one-hot columns missing from dataframe.")
    extras = df[room_cols].reset_index(drop=True)
    return pd.concat([base, extras], axis=1).to_numpy()


def save_confusion(y_true, y_pred, labels, path: Path, normalize: str = "true") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def localization_summary(df_true: pd.DataFrame, y_pred: Iterable[str], cell_lookup: pd.DataFrame) -> BenchmarkResult:
    y_pred = np.asarray(y_pred)
    pred_meta = cell_lookup.loc[y_pred]
    true_coords = df_true[["coord_x_m", "coord_y_m"]].to_numpy()
    pred_coords = pred_meta[["coord_x_m", "coord_y_m"]].to_numpy()
    errors_m = np.linalg.norm(true_coords - pred_coords, axis=1)
    return BenchmarkResult(
        cell_accuracy=float((y_pred == df_true["grid_cell"].to_numpy()).mean()),
        mean_error_m=float(errors_m.mean()),
        p90_error_m=float(np.percentile(errors_m, 90)),
        extra={},
    )


def fit_localizer(train_df: pd.DataFrame, include_room: bool, random_state: int = 7) -> EmbeddingKnnLocalizer:
    cfg = EmbeddingKnnConfig(
        hidden_layer_sizes=(48, 24),
        k_neighbors=5,
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=400,
        tol=1e-3,
        random_state=random_state,
    )
    model = EmbeddingKnnLocalizer(config=cfg)
    model.fit(build_features(train_df, include_room), train_df["grid_cell"])
    return model


def benchmark_room_agnostic(df: pd.DataFrame, cell_lookup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = sorted(df["grid_cell"].unique())
    for held_out in sorted(df["room"].unique()):
        train_df = df[df["room"] != held_out]
        test_df = df[df["room"] == held_out]

        model = fit_localizer(train_df, include_room=False)
        y_pred = model.predict(build_features(test_df, include_room=False))
        summary = localization_summary(test_df, y_pred, cell_lookup)
        save_confusion(
            test_df["grid_cell"],
            y_pred,
            labels,
            REPORT_DIR / f"confusion_room_agnostic_nn_{held_out}.csv",
        )

        # Baseline tree model trained directly on features (no embedding).
        rf = RandomForestClassifier(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=2,
            random_state=held_out.__hash__() % 10_000,
            n_jobs=-1,
        )
        X_train = build_features(train_df, include_room=False)
        X_test = build_features(test_df, include_room=False)
        rf.fit(X_train, train_df["grid_cell"])
        rf_pred = rf.predict(X_test)
        rf_summary = localization_summary(test_df, rf_pred, cell_lookup)
        save_confusion(
            test_df["grid_cell"],
            rf_pred,
            labels,
            REPORT_DIR / f"confusion_room_agnostic_rf_{held_out}.csv",
        )

        # KNN direct on RSSI.
        knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn.fit(X_train, train_df["grid_cell"])
        knn_pred = knn.predict(X_test)
        knn_summary = localization_summary(test_df, knn_pred, cell_lookup)
        save_confusion(
            test_df["grid_cell"],
            knn_pred,
            labels,
            REPORT_DIR / f"confusion_room_agnostic_knn_{held_out}.csv",
        )

        # ExtraTrees as alternative ensemble.
        et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            random_state=held_out.__hash__() % 10_000,
            n_jobs=-1,
        )
        et.fit(X_train, train_df["grid_cell"])
        et_pred = et.predict(X_test)
        et_summary = localization_summary(test_df, et_pred, cell_lookup)
        save_confusion(
            test_df["grid_cell"],
            et_pred,
            labels,
            REPORT_DIR / f"confusion_room_agnostic_extratrees_{held_out}.csv",
        )

        # Distance head only if all distances are present in train.
        router_acc = np.nan
        if set(test_df["router_distance_m"]).issubset(set(train_df["router_distance_m"])):
            train_emb = model.transform(build_features(train_df, include_room=False))
            test_emb = model.transform(build_features(test_df, include_room=False))
            dist_clf = LogisticRegression(max_iter=600)
            dist_clf.fit(train_emb, train_df["router_distance_m"])
            dist_pred = dist_clf.predict(test_emb)
            router_acc = float(accuracy_score(test_df["router_distance_m"], dist_pred))

        rows.append(
            {
                "held_out_room": held_out,
                "n_test": len(test_df),
                "cell_accuracy": summary.cell_accuracy,
                "mean_error_m": summary.mean_error_m,
                "p90_error_m": summary.p90_error_m,
                "rf_cell_accuracy": rf_summary.cell_accuracy,
                "rf_mean_error_m": rf_summary.mean_error_m,
                "rf_p90_error_m": rf_summary.p90_error_m,
                "knn_cell_accuracy": knn_summary.cell_accuracy,
                "knn_mean_error_m": knn_summary.mean_error_m,
                "knn_p90_error_m": knn_summary.p90_error_m,
                "et_cell_accuracy": et_summary.cell_accuracy,
                "et_mean_error_m": et_summary.mean_error_m,
                "et_p90_error_m": et_summary.p90_error_m,
                "router_distance_acc": router_acc,
            }
        )
    return pd.DataFrame(rows)


def benchmark_room_aware(df: pd.DataFrame, cell_lookup: pd.DataFrame) -> dict:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=21,
        stratify=df["grid_cell"],
    )
    labels = sorted(df["grid_cell"].unique())
    model = fit_localizer(train_df, include_room=True, random_state=21)
    y_pred = model.predict(build_features(test_df, include_room=True))
    summary = localization_summary(test_df, y_pred, cell_lookup)
    save_confusion(
        test_df["grid_cell"],
        y_pred,
        labels,
        REPORT_DIR / "confusion_room_aware_nn.csv",
    )

    # RandomForest with room feature.
    rf = RandomForestClassifier(
        n_estimators=220,
        max_depth=14,
        min_samples_leaf=2,
        random_state=21,
        n_jobs=-1,
    )
    X_train = build_features(train_df, include_room=True)
    X_test = build_features(test_df, include_room=True)
    rf.fit(X_train, train_df["grid_cell"])
    rf_pred = rf.predict(X_test)
    rf_summary = localization_summary(test_df, rf_pred, cell_lookup)
    save_confusion(
        test_df["grid_cell"],
        rf_pred,
        labels,
        REPORT_DIR / "confusion_room_aware_rf.csv",
    )

    # KNN with room feature.
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_train, train_df["grid_cell"])
    knn_pred = knn.predict(X_test)
    knn_summary = localization_summary(test_df, knn_pred, cell_lookup)
    save_confusion(
        test_df["grid_cell"],
        knn_pred,
        labels,
        REPORT_DIR / "confusion_room_aware_knn.csv",
    )

    # ExtraTrees with room feature.
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        random_state=21,
        n_jobs=-1,
    )
    et.fit(X_train, train_df["grid_cell"])
    et_pred = et.predict(X_test)
    et_summary = localization_summary(test_df, et_pred, cell_lookup)
    save_confusion(
        test_df["grid_cell"],
        et_pred,
        labels,
        REPORT_DIR / "confusion_room_aware_extratrees.csv",
    )

    # Distance head with room features.
    train_emb = model.transform(build_features(train_df, include_room=True))
    test_emb = model.transform(build_features(test_df, include_room=True))
    dist_clf = LogisticRegression(max_iter=600)
    dist_clf.fit(train_emb, train_df["router_distance_m"])
    dist_pred = dist_clf.predict(test_emb)
    dist_acc = float(accuracy_score(test_df["router_distance_m"], dist_pred))
    return {
        "nn_lknn_cell_accuracy": summary.cell_accuracy,
        "nn_lknn_mean_error_m": summary.mean_error_m,
        "nn_lknn_p90_error_m": summary.p90_error_m,
        "nn_lknn_router_distance_acc": dist_acc,
        "rf_cell_accuracy": rf_summary.cell_accuracy,
        "rf_mean_error_m": rf_summary.mean_error_m,
        "rf_p90_error_m": rf_summary.p90_error_m,
        "knn_cell_accuracy": knn_summary.cell_accuracy,
        "knn_mean_error_m": knn_summary.mean_error_m,
        "knn_p90_error_m": knn_summary.p90_error_m,
        "et_cell_accuracy": et_summary.cell_accuracy,
        "et_mean_error_m": et_summary.mean_error_m,
        "et_p90_error_m": et_summary.p90_error_m,
    }


def benchmark_distance_logreg(df: pd.DataFrame, save_path: Path | None = None) -> dict:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=33,
        stratify=df["router_distance_m"],
    )
    model = fit_localizer(train_df, include_room=False, random_state=33)
    train_emb = model.transform(build_features(train_df, include_room=False))
    test_emb = model.transform(build_features(test_df, include_room=False))

    clf = LogisticRegression(max_iter=600)
    clf.fit(train_emb, train_df["router_distance_m"])
    pred = clf.predict(test_emb)
    acc = float(accuracy_score(test_df["router_distance_m"], pred))
    cm = confusion_matrix(test_df["router_distance_m"], pred, normalize="true")
    if save_path is not None:
        save_confusion(test_df["router_distance_m"], pred, clf.classes_, save_path)
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "labels": [float(c) for c in clf.classes_],
    }


def benchmark_distance_rf(df: pd.DataFrame, save_path: Path | None = None) -> dict:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=44,
        stratify=df["router_distance_m"],
    )
    clf = RandomForestClassifier(
        n_estimators=220,
        max_depth=18,
        min_samples_leaf=2,
        random_state=44,
        n_jobs=-1,
    )
    X_train = build_features(train_df, include_room=False)
    X_test = build_features(test_df, include_room=False)
    clf.fit(X_train, train_df["router_distance_m"])
    pred = clf.predict(X_test)
    acc = float(accuracy_score(test_df["router_distance_m"], pred))
    cm = confusion_matrix(test_df["router_distance_m"], pred, normalize="true")
    if save_path is not None:
        save_confusion(test_df["router_distance_m"], pred, clf.classes_, save_path)
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "labels": [float(c) for c in clf.classes_],
    }


def benchmark_distance_extratrees(df: pd.DataFrame, save_path: Path | None = None) -> dict:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=52,
        stratify=df["router_distance_m"],
    )
    clf = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=24,
        min_samples_leaf=2,
        random_state=52,
        n_jobs=-1,
    )
    X_train = build_features(train_df, include_room=False)
    X_test = build_features(test_df, include_room=False)
    clf.fit(X_train, train_df["router_distance_m"])
    pred = clf.predict(X_test)
    acc = float(accuracy_score(test_df["router_distance_m"], pred))
    cm = confusion_matrix(test_df["router_distance_m"], pred, normalize="true")
    if save_path is not None:
        save_confusion(test_df["router_distance_m"], pred, clf.classes_, save_path)
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "labels": [float(c) for c in clf.classes_],
    }


def benchmark_multihead_embedding(
    df: pd.DataFrame,
    cell_lookup: pd.DataFrame,
    *,
    save_prefix: Path,
) -> dict:
    """Single model using RSSI only, predicting cell + distance + room from the same embedding."""
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=64,
        stratify=df["grid_cell"],
    )
    labels = sorted(df["grid_cell"].unique())
    # Fit embedding on cell labels only (no room/distance as features).
    model = fit_localizer(train_df, include_room=False, random_state=64)
    y_pred = model.predict(build_features(test_df, include_room=False))
    cell_metrics = localization_summary(test_df, y_pred, cell_lookup)
    save_confusion(
        test_df["grid_cell"],
        y_pred,
        labels,
        save_prefix.with_name("confusion_multihead_cell.csv"),
    )

    # Shared embedding reused for distance and room heads.
    train_emb = model.transform(build_features(train_df, include_room=False))
    test_emb = model.transform(build_features(test_df, include_room=False))

    dist_clf = LogisticRegression(max_iter=800)
    dist_clf.fit(train_emb, train_df["router_distance_m"])
    dist_pred = dist_clf.predict(test_emb)
    dist_acc = float(accuracy_score(test_df["router_distance_m"], dist_pred))
    save_confusion(
        test_df["router_distance_m"],
        dist_pred,
        dist_clf.classes_,
        save_prefix.with_name("confusion_multihead_distance.csv"),
    )

    room_clf = LogisticRegression(max_iter=800)
    room_clf.fit(train_emb, train_df["room"])
    room_pred = room_clf.predict(test_emb)
    room_acc = float(accuracy_score(test_df["room"], room_pred))
    save_confusion(
        test_df["room"],
        room_pred,
        room_clf.classes_,
        save_prefix.with_name("confusion_multihead_room.csv"),
    )

    return {
        "cell_acc": cell_metrics.cell_accuracy,
        "mean_error_m": cell_metrics.mean_error_m,
        "p90_error_m": cell_metrics.p90_error_m,
        "router_dist_acc": dist_acc,
        "room_acc": room_acc,
    }


def benchmark_room_classifier(df: pd.DataFrame, save_path: Path | None = None) -> dict:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=19,
        stratify=df["room"],
    )
    model = fit_localizer(train_df, include_room=False, random_state=19)
    train_emb = model.transform(build_features(train_df, include_room=False))
    test_emb = model.transform(build_features(test_df, include_room=False))

    clf = LogisticRegression(max_iter=600)
    clf.fit(train_emb, train_df["room"])
    pred = clf.predict(test_emb)
    acc = float(accuracy_score(test_df["room"], pred))
    cm = confusion_matrix(test_df["room"], pred, normalize="true")
    if save_path is not None:
        save_confusion(test_df["room"], pred, clf.classes_, save_path)
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "labels": clf.classes_.tolist(),
    }


def main():
    df = load_cross_room()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cell_lookup = (
        df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )

    print(
        f"Loaded {len(df)} samples | rooms={sorted(df['room'].unique())} | "
        f"cells={df['grid_cell'].nunique()} | distances={sorted(df['router_distance_m'].unique())}"
    )

    print("\n=== Room-agnostic (leave-one-room-out) ===")
    room_agnostic = benchmark_room_agnostic(df, cell_lookup)
    print(room_agnostic.to_string(index=False))
    print("Averages:", room_agnostic.mean(numeric_only=True).to_dict())
    print(f"Confusions enregistrées dans {REPORT_DIR}/confusion_room_agnostic_*.csv")
    room_agnostic_mean = room_agnostic.mean(numeric_only=True)

    print("\n=== Room-aware (one-hot room feature) ===")
    aware = benchmark_room_aware(df, cell_lookup)
    print(json.dumps(aware, indent=2))
    print(f"Confusions enregistrées dans {REPORT_DIR}/confusion_room_aware_*.csv")

    print("\n=== LogisticRegression on embedding: router distance ===")
    distance_res = benchmark_distance_logreg(df, REPORT_DIR / "confusion_distance_logreg.csv")
    print(json.dumps(distance_res, indent=2))
    print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_distance_logreg.csv")

    print("\n=== RandomForest on raw features: router distance ===")
    distance_rf = benchmark_distance_rf(df, REPORT_DIR / "confusion_distance_rf.csv")
    print(json.dumps(distance_rf, indent=2))
    print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_distance_rf.csv")

    print("\n=== ExtraTrees on raw features: router distance ===")
    distance_et = benchmark_distance_extratrees(df, REPORT_DIR / "confusion_distance_extratrees.csv")
    print(json.dumps(distance_et, indent=2))
    print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_distance_extratrees.csv")

    print("\n=== Multihead embedding (cell + distance + room from RSSI only) ===")
    multihead = benchmark_multihead_embedding(df, cell_lookup, save_prefix=REPORT_DIR / "multihead_placeholder.csv")
    print(json.dumps(multihead, indent=2))
    print("Confusions enregistrées dans reports/benchmarks/confusion_multihead_*.csv")

    print("\n=== LogisticRegression on embedding: room classification ===")
    room_res = benchmark_room_classifier(df, REPORT_DIR / "confusion_room_classifier.csv")
    print(json.dumps(room_res, indent=2))
    print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_room_classifier.csv")

    # Aggregate a compact score table for quick comparison.
    summary_rows = [
        {
            "algorithm": "NN+L-KNN (room-agnostic, LORO mean)",
            "cell_acc": room_agnostic_mean["cell_accuracy"],
            "mean_error_m": room_agnostic_mean["mean_error_m"],
            "p90_error_m": room_agnostic_mean["p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "RandomForest (room-agnostic, LORO mean)",
            "cell_acc": room_agnostic_mean["rf_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["rf_mean_error_m"],
            "p90_error_m": room_agnostic_mean["rf_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "KNN (room-agnostic, LORO mean)",
            "cell_acc": room_agnostic_mean["knn_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["knn_mean_error_m"],
            "p90_error_m": room_agnostic_mean["knn_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "ExtraTrees (room-agnostic, LORO mean)",
            "cell_acc": room_agnostic_mean["et_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["et_mean_error_m"],
            "p90_error_m": room_agnostic_mean["et_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "NN+L-KNN (room-aware, cell+distance)",
            "cell_acc": aware["nn_lknn_cell_accuracy"],
            "mean_error_m": aware["nn_lknn_mean_error_m"],
            "p90_error_m": aware["nn_lknn_p90_error_m"],
            "router_dist_acc": aware["nn_lknn_router_distance_acc"],
            "room_acc": np.nan,
        },
        {
            "algorithm": "RandomForest (room-aware, cell)",
            "cell_acc": aware["rf_cell_accuracy"],
            "mean_error_m": aware["rf_mean_error_m"],
            "p90_error_m": aware["rf_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "KNN (room-aware, cell)",
            "cell_acc": aware["knn_cell_accuracy"],
            "mean_error_m": aware["knn_mean_error_m"],
            "p90_error_m": aware["knn_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "ExtraTrees (room-aware, cell)",
            "cell_acc": aware["et_cell_accuracy"],
            "mean_error_m": aware["et_mean_error_m"],
            "p90_error_m": aware["et_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "LogReg (router distance on embedding)",
            "cell_acc": np.nan,
            "mean_error_m": np.nan,
            "p90_error_m": np.nan,
            "router_dist_acc": distance_res["accuracy"],
            "room_acc": np.nan,
        },
        {
            "algorithm": "RandomForest (router distance on raw RSSI)",
            "cell_acc": np.nan,
            "mean_error_m": np.nan,
            "p90_error_m": np.nan,
            "router_dist_acc": distance_rf["accuracy"],
            "room_acc": np.nan,
        },
        {
            "algorithm": "ExtraTrees (router distance on raw RSSI)",
            "cell_acc": np.nan,
            "mean_error_m": np.nan,
            "p90_error_m": np.nan,
            "router_dist_acc": distance_et["accuracy"],
            "room_acc": np.nan,
        },
        {
            "algorithm": "LogReg (room on embedding)",
            "cell_acc": np.nan,
            "mean_error_m": np.nan,
            "p90_error_m": np.nan,
            "router_dist_acc": np.nan,
            "room_acc": room_res["accuracy"],
        },
        {
            "algorithm": "Multihead (RSSI only: cell+distance+room)",
            "cell_acc": multihead["cell_acc"],
            "mean_error_m": multihead["mean_error_m"],
            "p90_error_m": multihead["p90_error_m"],
            "router_dist_acc": multihead["router_dist_acc"],
            "room_acc": multihead["room_acc"],
        },
    ]
    summary_df = pd.DataFrame(summary_rows).set_index("algorithm")
    summary_path = REPORT_DIR / "benchmark_summary.csv"
    summary_df.to_csv(summary_path)
    print("\n=== Benchmark summary (rows = algos, columns = tests) ===")
    print(summary_df.to_string())
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
