"""Benchmark NN+L-KNN variants and auxiliary classifiers across rooms.

This script mirrors the notebook scenarios:
- room-agnostic NN+L-KNN (leave-one-room-out),
- room-aware NN+L-KNN with room one-hot,
- LogisticRegression head for router distance (room-agnostic embedding),
- Room classification accuracy from the same embedding.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    GradientBoostingClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
try:
    from progress_table import ProgressTable
except ImportError:  # pragma: no cover - optional dependency
    class ProgressTable:
        """Minimal fallback when progress_table is not installed."""

        def __init__(self, columns, interactive=None, refresh_rate=None):
            self._columns = list(columns)
            self._rows: list[dict] = []

        def add_row(self, *values):
            row = dict(zip(self._columns, values))
            self._rows.append(row)

        def num_rows(self) -> int:
            return len(self._rows)

        def update(self, column, value, *, row: int):
            if row < 0 or row >= len(self._rows):
                return
            self._rows[row][column] = value

        def to_df(self):
            return pd.DataFrame(self._rows, columns=self._columns)

        def close(self):
            return None
import time

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, infer_router_distance, load_measurements  # noqa: E402
from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer  # noqa: E402
warnings.filterwarnings("ignore", category=ConvergenceWarning)
console = Console()

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


FAIL_LOG: list[dict] = []


def _normalize_room_name(room: str) -> str:
    return room.strip().upper()


def _distance_matches(distance: float, distance_filter: set[float]) -> bool:
    return any(abs(distance - target) < 1e-6 for target in distance_filter)


def _filter_room_campaigns(
    room_filter: list[str] | None,
    distance_filter: list[float] | None,
) -> dict[str, list[Path]]:
    normalized_rooms = None
    if room_filter:
        normalized_rooms = {_normalize_room_name(room) for room in room_filter if room and room.strip()}
        if not normalized_rooms:
            normalized_rooms = None

    distance_set = set(distance_filter) if distance_filter else None
    filtered: dict[str, list[Path]] = {}
    for room, folders in ROOM_CAMPAIGNS.items():
        if normalized_rooms and _normalize_room_name(room) not in normalized_rooms:
            continue
        selected: list[Path] = []
        for folder in folders:
            if distance_set:
                inferred = infer_router_distance(folder.name)
                if inferred is None or not _distance_matches(inferred, distance_set):
                    continue
            selected.append(folder)
        if selected:
            filtered[room] = selected
    return filtered


def _make_label_encoder(classes: Iterable[str]) -> tuple[dict[str, int], np.ndarray]:
    """Return (label_to_idx, idx_to_label) for consistent int encoding."""
    classes = list(classes)
    label_to_idx = {label: i for i, label in enumerate(classes)}
    idx_to_label = np.asarray(classes)
    return label_to_idx, idx_to_label


def _encode_labels(series: pd.Series, label_to_idx: dict[str, int]) -> np.ndarray:
    try:
        return series.map(label_to_idx).to_numpy(dtype=int)
    except KeyError as exc:  # pragma: no cover - defensive
        missing = set(series.unique()) - set(label_to_idx)
        raise ValueError(f"Unexpected labels encountered: {missing}") from exc


def _fit_predict_xgb(
    clf,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit XGBoost on a contiguous label encoding derived from the training fold."""
    train_labels = sorted(y_train.unique())
    local_label_to_idx = {label: i for i, label in enumerate(train_labels)}
    y_enc = y_train.map(local_label_to_idx).to_numpy(dtype=int)
    clf.set_params(num_class=len(train_labels))
    clf.fit(X_train, y_enc)
    pred_idx = np.asarray(clf.predict(X_test)).astype(int).ravel()
    return np.asarray(train_labels, dtype=object)[pred_idx]


def _as_feature_df(X: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame with stable column names to silence feature-name warnings."""
    X_arr = np.asarray(X, dtype=float)
    return pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])


def _ensure_benchmark_result(value, stage_name: str) -> "BenchmarkResult":
    if isinstance(value, BenchmarkResult):
        return value
    if callable(value):
        try:
            value = value()
        except Exception as exc:  # pragma: no cover - defensive
            return BenchmarkResult(
                np.nan,
                np.nan,
                np.nan,
                extra={"status": "failed", "reason": f"{stage_name} callable failed: {exc}"},
            )
        if isinstance(value, BenchmarkResult):
            return value
    return BenchmarkResult(
        np.nan,
        np.nan,
        np.nan,
        extra={"status": "failed", "reason": f"{stage_name} returned {type(value).__name__}"},
    )


def _maybe_subsample_df(
    df: pd.DataFrame,
    *,
    max_samples: int | None,
    label_col: str,
    random_state: int,
) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    stratify = df[label_col]
    if max_samples < df[label_col].nunique():
        stratify = None
    if stratify is not None:
        try:
            subset, _ = train_test_split(
                df,
                train_size=max_samples,
                random_state=random_state,
                stratify=stratify,
            )
            return subset
        except ValueError:
            return df.sample(n=max_samples, random_state=random_state)
    return df.sample(n=max_samples, random_state=random_state)


def _extract_status(result) -> tuple[str, str]:
    """Return (status, reason) from a BenchmarkResult, defaulting to done/empty."""
    if isinstance(result, BenchmarkResult) and result.extra:
        return result.extra.get("status", "done"), result.extra.get("reason", "")
    return "done", ""


def _record_stage_outcome(stage_name: str, status: str, context: dict | None, reason: str = "") -> None:
    """Persist non-successful stages so we can surface skipped/failed runs later."""
    if status == "done":
        return
    entry = {"stage": stage_name, "status": status}
    if context:
        entry["context"] = context
    if reason:
        entry["reason"] = reason
    FAIL_LOG.append(entry)


def _failed_result(reason: str) -> BenchmarkResult:
    return BenchmarkResult(np.nan, np.nan, np.nan, extra={"status": "failed", "reason": reason})


def load_cross_room(
    room_filter: list[str] | None = None,
    distance_filter: list[float] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    campaigns = _filter_room_campaigns(room_filter, distance_filter)
    if not campaigns:
        raise ValueError("No campaign folders match the provided filters.")
    for room, folders in campaigns.items():
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


def _mahalanobis_vi(X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Return inverse covariance matrix with ridge for Mahalanobis distance."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 2:
        raise np.linalg.LinAlgError("Mahalanobis requires a 2D array with at least 2 rows.")
    n_features = X.shape[1]
    cov = np.asarray(np.cov(X, rowvar=False))
    if cov.ndim == 0:  # degenerate case -> scalar
        cov = np.array([[float(cov)]], dtype=float)
    if cov.shape != (n_features, n_features):
        cov = np.eye(n_features) * float(np.nan_to_num(cov).max(initial=1.0))
    # Try progressively larger ridges to avoid singular matrices; fall back to identity.
    for penalty in (ridge, ridge * 10, ridge * 100, 1.0):
        try:
            cov_reg = cov + np.eye(n_features) * penalty
            return np.linalg.pinv(cov_reg)
        except Exception:  # pragma: no cover - defensive
            continue
    return np.eye(n_features)


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


def benchmark_room_agnostic(
    df: pd.DataFrame,
    cell_lookup: pd.DataFrame,
    *,
    skip_optional: bool = False,
    run_gpc: bool = False,
    gpc_max_samples: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    labels = sorted(df["grid_cell"].unique())
    label_to_idx, idx_to_label = _make_label_encoder(labels)
    rooms = sorted(df["room"].unique())
    optional_stages = {"KNN Mahalanobis", "GPC", "CatBoost", "LightGBM", "XGBoost"}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Room-agnostic LORO", total=len(rooms))
        for held_out in rooms:
            stage_table = ProgressTable(columns=["stage", "status", "elapsed_s"], interactive=2, refresh_rate=20)
            train_df = df[df["room"] != held_out]
            test_df = df[df["room"] == held_out]
            stage_context = {"mode": "room_agnostic", "held_out_room": held_out}

            def log_stage(stage_name: str, fn, *, fallback=None):
                if skip_optional and stage_name in optional_stages and not (stage_name == "GPC" and run_gpc):
                    stage_table.add_row(stage_name, "skipped", 0.0)
                    _record_stage_outcome(stage_name, "skipped", stage_context, "")
                    return fallback if fallback is not None else BenchmarkResult(np.nan, np.nan, np.nan, extra={"status": "skipped"})
                stage_table.add_row(stage_name, "running", 0.0)
                row_idx = stage_table.num_rows() - 1
                start = time.time()
                status = "done"
                reason = ""
                try:
                    result = fn()
                    status, reason = _extract_status(result)
                except Exception as exc:  # noqa: BLE001 - we want to log any failure
                    reason = f"{type(exc).__name__}: {exc}"
                    status = "failed"
                    result = fallback if fallback is not None else _failed_result(reason)
                    console.print(f"[red]{stage_name} failed: {reason}[/red]")
                _record_stage_outcome(stage_name, status, stage_context, reason)
                if reason and status != "done":
                    console.print(f"[yellow]{stage_name}: {reason}[/yellow]")
                stage_table.update("status", status, row=row_idx)
                stage_table.update("elapsed_s", time.time() - start, row=row_idx)
                return result

            model, summary, y_pred = log_stage(
                "NN+L-KNN",
                lambda: (
                    lambda m=fit_localizer(train_df, include_room=False), preds=None: (
                        m,
                        localization_summary(
                            test_df,
                            (preds := m.predict(build_features(test_df, include_room=False))),
                            cell_lookup,
                        ),
                        preds,
                    )
                )(),
            )
            save_confusion(
                test_df["grid_cell"],
                y_pred,
                labels,
                REPORT_DIR / f"confusion_room_agnostic_nn_{held_out}.csv",
            )

            X_train = build_features(train_df, include_room=False)
            X_test = build_features(test_df, include_room=False)
            X_train_df = _as_feature_df(X_train)
            X_test_df = _as_feature_df(X_test)

            rf_summary = log_stage(
                "RandomForest",
                lambda: (
                    lambda clf=RandomForestClassifier(
                        n_estimators=220,
                        max_depth=14,
                        min_samples_leaf=2,
                        random_state=held_out.__hash__() % 10_000,
                        n_jobs=-1,
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                        save_confusion(
                            test_df["grid_cell"],
                            clf.predict(X_test),
                            labels,
                            REPORT_DIR / f"confusion_room_agnostic_rf_{held_out}.csv",
                        ),
                    )
                )()[1],
            )

            knn_summary = log_stage(
                "KNN raw",
                lambda: (
                    lambda clf=KNeighborsClassifier(n_neighbors=7, weights="distance"): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                        save_confusion(
                            test_df["grid_cell"],
                            clf.predict(X_test),
                            labels,
                            REPORT_DIR / f"confusion_room_agnostic_knn_{held_out}.csv",
                        ),
                    )
                )()[1],
            )

            et_summary = log_stage(
                "ExtraTrees",
                lambda: (
                    lambda clf=ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=18,
                        min_samples_leaf=2,
                        random_state=held_out.__hash__() % 10_000,
                        n_jobs=-1,
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                        save_confusion(
                            test_df["grid_cell"],
                            clf.predict(X_test),
                            labels,
                            REPORT_DIR / f"confusion_room_agnostic_extratrees_{held_out}.csv",
                        ),
                    )
                )()[1],
            )

            router_acc = np.nan

            def fit_distance():
                nonlocal router_acc
                if train_df["router_distance_m"].nunique() < 2:
                    router_acc = np.nan
                    return
                train_emb = model.transform(build_features(train_df, include_room=False))
                test_emb = model.transform(build_features(test_df, include_room=False))
                dist_clf = LogisticRegression(max_iter=600)
                dist_clf.fit(train_emb, train_df["router_distance_m"])
                dist_pred = dist_clf.predict(test_emb)
                router_acc = float(accuracy_score(test_df["router_distance_m"], dist_pred))

            if set(test_df["router_distance_m"]).issubset(set(train_df["router_distance_m"])):
                log_stage("LogReg distance head", fit_distance, fallback=np.nan)

            svm_summary = log_stage(
                "SVM RBF",
                lambda: (
                    lambda clf=make_pipeline(StandardScaler(), SVC(kernel="rbf", C=8.0, gamma="scale")): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            hgb_summary = log_stage(
                "HistGB",
                lambda: (
                    lambda clf=HistGradientBoostingClassifier(
                        learning_rate=0.1,
                        max_depth=8,
                        max_iter=200,
                        random_state=held_out.__hash__() % 10_000,
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            gbdt_summary = log_stage(
                "GBDT",
                lambda: (
                    lambda clf=GradientBoostingClassifier(random_state=held_out.__hash__() % 10_000): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            def safe_mahalanobis():
                try:
                    vi = _mahalanobis_vi(X_train)
                    vi = np.asarray(vi, dtype=float)
                    if (
                        vi.ndim != 2
                        or vi.shape[0] != vi.shape[1]
                        or vi.shape[0] != X_train.shape[1]
                        or not np.isfinite(vi).all()
                    ):
                        return BenchmarkResult(
                            np.nan,
                            np.nan,
                            np.nan,
                            extra={"status": "skipped", "reason": "invalid Mahalanobis matrix"},
                        )
                    knn_maha = KNeighborsClassifier(
                        n_neighbors=7, weights="distance", metric="mahalanobis", metric_params={"VI": vi}
                    )
                    knn_maha.fit(X_train, train_df["grid_cell"])
                    return localization_summary(test_df, knn_maha.predict(X_test), cell_lookup)
                except np.linalg.LinAlgError:
                    return BenchmarkResult(
                        np.nan,
                        np.nan,
                        np.nan,
                        extra={"status": "failed", "reason": "mahalanobis covariance inversion failed"},
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    return BenchmarkResult(
                        np.nan,
                        np.nan,
                        np.nan,
                        extra={"status": "skipped", "reason": f"mahalanobis skipped: {exc}"},
                    )

            knn_maha_summary = log_stage("KNN Mahalanobis", safe_mahalanobis)

            mlp_summary = log_stage(
                "MLP",
                lambda: (
                    lambda clf=MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        max_iter=500,
                        alpha=5e-4,
                        learning_rate_init=5e-4,
                        random_state=held_out.__hash__() % 10_000,
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            gnb_summary = log_stage(
                "GaussianNB",
                lambda: (
                    lambda clf=GaussianNB(): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            bag_knn_summary = log_stage(
                "Bagging KNN",
                lambda: (
                    lambda clf=BaggingClassifier(
                        estimator=KNeighborsClassifier(n_neighbors=7, weights="distance"),
                        n_estimators=15,
                        random_state=held_out.__hash__() % 10_000,
                        n_jobs=-1,
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            svm_lin_summary = log_stage(
                "SVM lin",
                lambda: (
                    lambda clf=make_pipeline(StandardScaler(), LinearSVC()): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            svm_poly_summary = log_stage(
                "SVM poly",
                lambda: (
                    lambda clf=make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, C=6.0, gamma="scale")): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            if run_gpc:
                def fit_gpc():
                    gpc_train_df = _maybe_subsample_df(
                        train_df,
                        max_samples=gpc_max_samples,
                        label_col="grid_cell",
                        random_state=held_out.__hash__() % 10_000,
                    )
                    X_train_gpc = build_features(gpc_train_df, include_room=False)
                    X_test_gpc = build_features(test_df, include_room=False)
                    clf = make_pipeline(
                        StandardScaler(),
                        GaussianProcessClassifier(
                            kernel=RBF(length_scale=1.0),
                            optimizer=None,
                            random_state=held_out.__hash__() % 10_000,
                        ),
                    )
                    clf.fit(X_train_gpc, gpc_train_df["grid_cell"])
                    return localization_summary(test_df, clf.predict(X_test_gpc), cell_lookup)

                gpc_summary = log_stage("GPC", fit_gpc)
                gpc_summary = _ensure_benchmark_result(gpc_summary, "GPC")
            else:
                gpc_summary = log_stage(
                    "GPC",
                    lambda: BenchmarkResult(
                        np.nan,
                        np.nan,
                        np.nan,
                        extra={"status": "skipped", "reason": "skipped for speed"},
                    ),
                )

            qda_summary = log_stage(
                "QDA",
                lambda: (
                    lambda clf=QuadraticDiscriminantAnalysis(store_covariance=False, reg_param=1e-3): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            knn_lda_summary = log_stage(
                "LDA + KNN",
                lambda: (
                    lambda lda=LinearDiscriminantAnalysis(): (
                        lda.fit(X_train, train_df["grid_cell"]),
                        localization_summary(
                            test_df,
                            KNeighborsClassifier(n_neighbors=7, weights="distance").fit(
                                lda.transform(X_train), train_df["grid_cell"]
                            ).predict(lda.transform(X_test)),
                            cell_lookup,
                        ),
                    )
                )()[1],
            )

            cat_summary = log_stage(
                "CatBoost",
                lambda: BenchmarkResult(
                    np.nan,
                    np.nan,
                    np.nan,
                    extra={"status": "skipped", "reason": "catboost not installed"},
                )
                if CatBoostClassifier is None
                else (
                    lambda clf=CatBoostClassifier(
                        depth=8,
                        learning_rate=0.08,
                        iterations=220,
                        loss_function="MultiClass",
                        verbose=False,
                        random_seed=held_out.__hash__() % 10_000,
                    ): (
                        lambda y_train_enc=_encode_labels(train_df["grid_cell"], label_to_idx): (
                            clf.fit(X_train, y_train_enc),
                            localization_summary(
                                test_df,
                                idx_to_label[np.asarray(clf.predict(X_test)).astype(int).ravel()],
                                cell_lookup,
                            ),
                        )
                    )()[1]
                )(),
            )
            cat_summary = _ensure_benchmark_result(cat_summary, "CatBoost")

            lgbm_summary = log_stage(
                "LightGBM",
                lambda: BenchmarkResult(
                    np.nan,
                    np.nan,
                    np.nan,
                    extra={"status": "skipped", "reason": "lightgbm not installed"},
                )
                if LGBMClassifier is None
                else (
                    lambda clf=LGBMClassifier(
                        num_leaves=63,
                        learning_rate=0.06,
                        n_estimators=260,
                        max_depth=-1,
                        random_state=held_out.__hash__() % 10_000,
                        min_split_gain=0.0,
                        min_child_samples=5,
                        min_child_weight=1e-3,
                        verbosity=-1,
                    ): (
                        lambda y_train_enc=_encode_labels(train_df["grid_cell"], label_to_idx): (
                            clf.fit(X_train_df, y_train_enc),
                            localization_summary(
                                test_df,
                                idx_to_label[np.asarray(clf.predict(X_test_df)).astype(int).ravel()],
                                cell_lookup,
                            ),
                        )
                    )()[1]
                )(),
            )
            lgbm_summary = _ensure_benchmark_result(lgbm_summary, "LightGBM")

            xgb_summary = log_stage(
                "XGBoost",
                lambda: BenchmarkResult(
                    np.nan,
                    np.nan,
                    np.nan,
                    extra={"status": "skipped", "reason": "xgboost not installed"},
                )
                if XGBClassifier is None
                else (
                    lambda clf=XGBClassifier(
                        max_depth=8,
                        learning_rate=0.08,
                        n_estimators=360,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="multi:softmax",
                        random_state=held_out.__hash__() % 10_000,
                        eval_metric="mlogloss",
                    ): (
                        lambda: (
                            lambda preds=_fit_predict_xgb(clf, X_train, train_df["grid_cell"], X_test): (
                                clf,
                                localization_summary(test_df, preds, cell_lookup),
                            )
                        )()[1]
                    )()
                )(),
            )
            xgb_summary = _ensure_benchmark_result(xgb_summary, "XGBoost")

            stacking_summary = log_stage(
                "Stacking",
                lambda: (
                    lambda clf=StackingClassifier(
                        estimators=[
                            ("rf", RandomForestClassifier(
                                n_estimators=220,
                                max_depth=14,
                                min_samples_leaf=2,
                                random_state=held_out.__hash__() % 10_000,
                                n_jobs=-1,
                            )),
                            ("svm_rbf", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=8.0, gamma="scale", probability=True))),
                            ("mlp", MLPClassifier(
                                hidden_layer_sizes=(128, 64),
                                activation="relu",
                                max_iter=500,
                                alpha=5e-4,
                                learning_rate_init=5e-4,
                                random_state=held_out.__hash__() % 10_000,
                            )),
                        ],
                        final_estimator=LogisticRegression(max_iter=400),
                        stack_method="predict_proba",
                        n_jobs=1,
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

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
                    "svm_cell_accuracy": svm_summary.cell_accuracy,
                    "svm_mean_error_m": svm_summary.mean_error_m,
                    "svm_p90_error_m": svm_summary.p90_error_m,
                    "svm_lin_cell_accuracy": svm_lin_summary.cell_accuracy,
                    "svm_lin_mean_error_m": svm_lin_summary.mean_error_m,
                    "svm_lin_p90_error_m": svm_lin_summary.p90_error_m,
                    "svm_poly_cell_accuracy": svm_poly_summary.cell_accuracy,
                    "svm_poly_mean_error_m": svm_poly_summary.mean_error_m,
                    "svm_poly_p90_error_m": svm_poly_summary.p90_error_m,
                    "hgb_cell_accuracy": hgb_summary.cell_accuracy,
                    "hgb_mean_error_m": hgb_summary.mean_error_m,
                    "hgb_p90_error_m": hgb_summary.p90_error_m,
                    "gbdt_cell_accuracy": gbdt_summary.cell_accuracy,
                    "gbdt_mean_error_m": gbdt_summary.mean_error_m,
                    "gbdt_p90_error_m": gbdt_summary.p90_error_m,
                    "knn_maha_cell_accuracy": knn_maha_summary.cell_accuracy,
                    "knn_maha_mean_error_m": knn_maha_summary.mean_error_m,
                    "knn_maha_p90_error_m": knn_maha_summary.p90_error_m,
                    "knn_lda_cell_accuracy": knn_lda_summary.cell_accuracy,
                    "knn_lda_mean_error_m": knn_lda_summary.mean_error_m,
                    "knn_lda_p90_error_m": knn_lda_summary.p90_error_m,
                    "mlp_cell_accuracy": mlp_summary.cell_accuracy,
                    "mlp_mean_error_m": mlp_summary.mean_error_m,
                    "mlp_p90_error_m": mlp_summary.p90_error_m,
                    "gnb_cell_accuracy": gnb_summary.cell_accuracy,
                    "gnb_mean_error_m": gnb_summary.mean_error_m,
                    "gnb_p90_error_m": gnb_summary.p90_error_m,
                    "bag_knn_cell_accuracy": bag_knn_summary.cell_accuracy,
                    "bag_knn_mean_error_m": bag_knn_summary.mean_error_m,
                    "bag_knn_p90_error_m": bag_knn_summary.p90_error_m,
                    "gpc_cell_accuracy": gpc_summary.cell_accuracy,
                    "gpc_mean_error_m": gpc_summary.mean_error_m,
                    "gpc_p90_error_m": gpc_summary.p90_error_m,
                    "qda_cell_accuracy": qda_summary.cell_accuracy,
                    "qda_mean_error_m": qda_summary.mean_error_m,
                    "qda_p90_error_m": qda_summary.p90_error_m,
                    "cat_cell_accuracy": cat_summary.cell_accuracy,
                    "cat_mean_error_m": cat_summary.mean_error_m,
                    "cat_p90_error_m": cat_summary.p90_error_m,
                    "lgbm_cell_accuracy": lgbm_summary.cell_accuracy,
                    "lgbm_mean_error_m": lgbm_summary.mean_error_m,
                    "lgbm_p90_error_m": lgbm_summary.p90_error_m,
                    "xgb_cell_accuracy": xgb_summary.cell_accuracy,
                    "xgb_mean_error_m": xgb_summary.mean_error_m,
                    "xgb_p90_error_m": xgb_summary.p90_error_m,
                    "stacking_cell_accuracy": stacking_summary.cell_accuracy,
                    "stacking_mean_error_m": stacking_summary.mean_error_m,
                    "stacking_p90_error_m": stacking_summary.p90_error_m,
                    "router_distance_acc": router_acc,
                }
            )
            console.print(f"\n[bold]Stages for room {held_out}[/bold]")
            console.print(stage_table.to_df().round(3).to_string(index=False))
            stage_table.close()
            progress.advance(task)
    return pd.DataFrame(rows)



def benchmark_room_aware(
    df: pd.DataFrame,
    cell_lookup: pd.DataFrame,
    *,
    skip_optional: bool = False,
    run_gpc: bool = False,
    gpc_max_samples: int | None = None,
) -> dict:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Room-aware models", total=21)
        stage_table = ProgressTable(columns=["stage", "status", "elapsed_s"], interactive=2, refresh_rate=20)
        stage_context = {"mode": "room_aware"}
        labels = sorted(df["grid_cell"].unique())
        label_to_idx, idx_to_label = _make_label_encoder(labels)
        optional_stages = {"KNN Mahalanobis", "GPC", "CatBoost", "LightGBM", "XGBoost"}

        def log_stage(stage_name: str, fn, *, fallback=None):
            if skip_optional and stage_name in optional_stages and not (stage_name == "GPC" and run_gpc):
                stage_table.add_row(stage_name, "skipped", 0.0)
                _record_stage_outcome(stage_name, "skipped", stage_context, "")
                progress.advance(task)
                return fallback if fallback is not None else BenchmarkResult(np.nan, np.nan, np.nan, extra={"status": "skipped"})
            stage_table.add_row(stage_name, "running", 0.0)
            row_idx = stage_table.num_rows() - 1
            start_t = time.time()
            status = "done"
            reason = ""
            try:
                result = fn()
                status, reason = _extract_status(result)
            except Exception as exc:  # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"
                status = "failed"
                result = fallback if fallback is not None else _failed_result(reason)
                console.print(f"[red]{stage_name} failed: {reason}[/red]")
            _record_stage_outcome(stage_name, status, stage_context, reason)
            if reason and status != "done":
                console.print(f"[yellow]{stage_name}: {reason}[/yellow]")
            stage_table.update("status", status, row=row_idx)
            stage_table.update("elapsed_s", time.time() - start_t, row=row_idx)
            progress.advance(task)
            return result

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=21,
            stratify=df["grid_cell"],
        )
        labels = sorted(df["grid_cell"].unique())
        model, summary, y_pred = log_stage(
            "NN+L-KNN",
            lambda: (
                lambda m=fit_localizer(train_df, include_room=True, random_state=21), preds=None: (
                    m,
                    localization_summary(
                        test_df,
                        (preds := m.predict(build_features(test_df, include_room=True))),
                        cell_lookup,
                    ),
                    preds,
                )
            )(),
        )
        save_confusion(
            test_df["grid_cell"],
            y_pred,
            labels,
            REPORT_DIR / "confusion_room_aware_nn.csv",
        )

        X_train = build_features(train_df, include_room=True)
        X_test = build_features(test_df, include_room=True)
        X_train_df = _as_feature_df(X_train)
        X_test_df = _as_feature_df(X_test)

        rf_summary = log_stage(
            "RandomForest",
            lambda: (
                lambda clf=RandomForestClassifier(
                    n_estimators=220,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=21,
                    n_jobs=-1,
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                        save_confusion(
                            test_df["grid_cell"],
                            clf.predict(X_test),
                            labels,
                            REPORT_DIR / "confusion_room_aware_rf.csv",
                        ),
                    )
                )()[1],
        )

        knn_summary = log_stage(
            "KNN raw",
            lambda: (
                lambda clf=KNeighborsClassifier(n_neighbors=7, weights="distance"): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    save_confusion(
                        test_df["grid_cell"],
                        clf.predict(X_test),
                        labels,
                        REPORT_DIR / "confusion_room_aware_knn.csv",
                    ),
                )
            )()[1],
        )

        et_summary = log_stage(
            "ExtraTrees",
            lambda: (
                lambda clf=ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_leaf=2,
                    random_state=21,
                    n_jobs=-1,
                ): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    save_confusion(
                        test_df["grid_cell"],
                        clf.predict(X_test),
                        labels,
                        REPORT_DIR / "confusion_room_aware_extratrees.csv",
                    ),
                )
            )()[1],
        )

        def fit_distance():
            if train_df["router_distance_m"].nunique() < 2:
                return np.nan
            train_emb = model.transform(build_features(train_df, include_room=True))
            test_emb = model.transform(build_features(test_df, include_room=True))
            dist_clf = LogisticRegression(max_iter=600)
            dist_clf.fit(train_emb, train_df["router_distance_m"])
            dist_pred = dist_clf.predict(test_emb)
            return float(accuracy_score(test_df["router_distance_m"], dist_pred))

        dist_acc = log_stage("LogReg distance head", fit_distance, fallback=np.nan)

        svm_summary = log_stage(
            "SVM RBF",
            lambda: (
                lambda clf=make_pipeline(StandardScaler(), SVC(kernel="rbf", C=8.0, gamma="scale")): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        svm_lin_summary = log_stage(
            "SVM lin",
            lambda: (
                lambda clf=make_pipeline(StandardScaler(), LinearSVC()): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        svm_poly_summary = log_stage(
            "SVM poly",
            lambda: (
                lambda clf=make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, C=6.0, gamma="scale")): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        hgb_summary = log_stage(
            "HistGB",
            lambda: (
                lambda clf=HistGradientBoostingClassifier(
                    learning_rate=0.1,
                    max_depth=8,
                    max_iter=200,
                    random_state=21,
                ): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        gbdt_summary = log_stage(
            "GBDT",
            lambda: (
                lambda clf=GradientBoostingClassifier(random_state=21): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        def safe_mahalanobis():
            try:
                vi = _mahalanobis_vi(X_train)
                vi = np.asarray(vi, dtype=float)
                if (
                    vi.ndim != 2
                    or vi.shape[0] != vi.shape[1]
                    or vi.shape[0] != X_train.shape[1]
                    or not np.isfinite(vi).all()
                ):
                    return BenchmarkResult(
                        np.nan,
                        np.nan,
                        np.nan,
                        extra={"status": "skipped", "reason": "invalid Mahalanobis matrix"},
                    )
                knn_maha = KNeighborsClassifier(
                    n_neighbors=7, weights="distance", metric="mahalanobis", metric_params={"VI": vi}
                )
                knn_maha.fit(X_train, train_df["grid_cell"])
                return localization_summary(test_df, knn_maha.predict(X_test), cell_lookup)
            except np.linalg.LinAlgError:
                return BenchmarkResult(
                    np.nan,
                    np.nan,
                    np.nan,
                    extra={"status": "failed", "reason": "mahalanobis covariance inversion failed"},
                )
            except Exception as exc:  # pragma: no cover - defensive
                return BenchmarkResult(
                    np.nan,
                    np.nan,
                    np.nan,
                    extra={"status": "skipped", "reason": f"mahalanobis skipped: {exc}"},
                )

        knn_maha_summary = log_stage("KNN Mahalanobis", safe_mahalanobis)

        knn_lda_summary = log_stage(
            "LDA + KNN",
            lambda: (
                lambda lda=LinearDiscriminantAnalysis(): (
                    lda.fit(X_train, train_df["grid_cell"]),
                    localization_summary(
                        test_df,
                        KNeighborsClassifier(n_neighbors=7, weights="distance").fit(
                            lda.transform(X_train), train_df["grid_cell"]
                        ).predict(lda.transform(X_test)),
                        cell_lookup,
                    ),
                )
            )()[1],
        )

        mlp_summary = log_stage(
            "MLP",
            lambda: (
                lambda clf=MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=500,
                    alpha=5e-4,
                    learning_rate_init=5e-4,
                    random_state=21,
                ): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        gnb_summary = log_stage(
            "GaussianNB",
            lambda: (
                lambda clf=GaussianNB(): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        bag_knn_summary = log_stage(
            "Bagging KNN",
            lambda: (
                lambda clf=BaggingClassifier(
                    estimator=KNeighborsClassifier(n_neighbors=7, weights="distance"),
                    n_estimators=15,
                    random_state=21,
                    n_jobs=-1,
                ): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        if run_gpc:
            def fit_gpc():
                gpc_train_df = _maybe_subsample_df(
                    train_df, max_samples=gpc_max_samples, label_col="grid_cell", random_state=21
                )
                X_train_gpc = build_features(gpc_train_df, include_room=True)
                X_test_gpc = build_features(test_df, include_room=True)
                clf = make_pipeline(
                    StandardScaler(),
                    GaussianProcessClassifier(
                        kernel=RBF(length_scale=1.0),
                        optimizer=None,
                        random_state=21,
                    ),
                )
                clf.fit(X_train_gpc, gpc_train_df["grid_cell"])
                return localization_summary(test_df, clf.predict(X_test_gpc), cell_lookup)

            gpc_summary = log_stage("GPC", fit_gpc)
            gpc_summary = _ensure_benchmark_result(gpc_summary, "GPC")
        else:
            gpc_summary = log_stage(
                "GPC",
                lambda: BenchmarkResult(
                    np.nan,
                    np.nan,
                    np.nan,
                    extra={"status": "skipped", "reason": "skipped for speed"},
                ),
            )

        qda_summary = log_stage(
            "QDA",
            lambda: (
                lambda clf=QuadraticDiscriminantAnalysis(store_covariance=False, reg_param=1e-3): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        cat_summary = log_stage(
            "CatBoost",
            lambda: BenchmarkResult(
                np.nan,
                np.nan,
                np.nan,
                extra={"status": "skipped", "reason": "catboost not installed"},
            )
            if CatBoostClassifier is None
            else (
                lambda clf=CatBoostClassifier(
                    depth=8,
                    learning_rate=0.08,
                    iterations=220,
                    loss_function="MultiClass",
                    verbose=False,
                    random_seed=21,
                ): (
                    lambda y_train_enc=_encode_labels(train_df["grid_cell"], label_to_idx): (
                        clf.fit(X_train, y_train_enc),
                        localization_summary(
                            test_df,
                            idx_to_label[np.asarray(clf.predict(X_test)).astype(int).ravel()],
                            cell_lookup,
                        ),
                    )
                )()[1]
            )(),
        )
        cat_summary = _ensure_benchmark_result(cat_summary, "CatBoost")

        lgbm_summary = log_stage(
            "LightGBM",
            lambda: BenchmarkResult(
                np.nan,
                np.nan,
                np.nan,
                extra={"status": "skipped", "reason": "lightgbm not installed"},
            )
            if LGBMClassifier is None
            else (
                lambda clf=LGBMClassifier(
                    num_leaves=63,
                    learning_rate=0.06,
                    n_estimators=260,
                    max_depth=-1,
                    random_state=21,
                    min_split_gain=0.0,
                    min_child_samples=5,
                    min_child_weight=1e-3,
                    verbosity=-1,
                ): (
                    lambda y_train_enc=_encode_labels(train_df["grid_cell"], label_to_idx): (
                        clf.fit(X_train_df, y_train_enc),
                        localization_summary(
                            test_df,
                            idx_to_label[np.asarray(clf.predict(X_test_df)).astype(int).ravel()],
                            cell_lookup,
                        ),
                    )
                )()[1]
            )(),
        )
        lgbm_summary = _ensure_benchmark_result(lgbm_summary, "LightGBM")

        xgb_summary = log_stage(
            "XGBoost",
            lambda: BenchmarkResult(
                np.nan,
                np.nan,
                np.nan,
                extra={"status": "skipped", "reason": "xgboost not installed"},
            )
            if XGBClassifier is None
            else (
                lambda clf=XGBClassifier(
                    max_depth=8,
                    learning_rate=0.08,
                    n_estimators=360,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softmax",
                    random_state=21,
                    eval_metric="mlogloss",
                ): (
                    lambda: (
                        lambda preds=_fit_predict_xgb(clf, X_train, train_df["grid_cell"], X_test): (
                            clf,
                            localization_summary(test_df, preds, cell_lookup),
                        )
                    )()[1]
                )()
            )(),
        )
        xgb_summary = _ensure_benchmark_result(xgb_summary, "XGBoost")

        base_estimators = [
            ("rf", RandomForestClassifier(
                n_estimators=220,
                max_depth=14,
                min_samples_leaf=2,
                random_state=21,
                n_jobs=-1,
            )),
            ("svm_rbf", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=8.0, gamma="scale", probability=True))),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=500,
                alpha=5e-4,
                learning_rate_init=5e-4,
                random_state=21,
            )),
        ]
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=400),
            stack_method="predict_proba",
            n_jobs=1,
        )
        stacking_summary = log_stage(
            "Stacking",
            lambda: (
                stacking.fit(X_train, train_df["grid_cell"]),
                localization_summary(test_df, stacking.predict(X_test), cell_lookup),
            )[1],
        )

        console.print("\n[bold]Room-aware stage timings[/bold]")
        console.print(stage_table.to_df().round(3).to_string(index=False))
        stage_table.close()

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
        "svm_cell_accuracy": svm_summary.cell_accuracy,
        "svm_mean_error_m": svm_summary.mean_error_m,
        "svm_p90_error_m": svm_summary.p90_error_m,
        "svm_lin_cell_accuracy": svm_lin_summary.cell_accuracy,
        "svm_lin_mean_error_m": svm_lin_summary.mean_error_m,
        "svm_lin_p90_error_m": svm_lin_summary.p90_error_m,
        "svm_poly_cell_accuracy": svm_poly_summary.cell_accuracy,
        "svm_poly_mean_error_m": svm_poly_summary.mean_error_m,
        "svm_poly_p90_error_m": svm_poly_summary.p90_error_m,
        "hgb_cell_accuracy": hgb_summary.cell_accuracy,
        "hgb_mean_error_m": hgb_summary.mean_error_m,
        "hgb_p90_error_m": hgb_summary.p90_error_m,
        "gbdt_cell_accuracy": gbdt_summary.cell_accuracy,
        "gbdt_mean_error_m": gbdt_summary.mean_error_m,
        "gbdt_p90_error_m": gbdt_summary.p90_error_m,
        "knn_maha_cell_accuracy": knn_maha_summary.cell_accuracy,
        "knn_maha_mean_error_m": knn_maha_summary.mean_error_m,
        "knn_maha_p90_error_m": knn_maha_summary.p90_error_m,
        "knn_lda_cell_accuracy": knn_lda_summary.cell_accuracy,
        "knn_lda_mean_error_m": knn_lda_summary.mean_error_m,
        "knn_lda_p90_error_m": knn_lda_summary.p90_error_m,
        "mlp_cell_accuracy": mlp_summary.cell_accuracy,
        "mlp_mean_error_m": mlp_summary.mean_error_m,
        "mlp_p90_error_m": mlp_summary.p90_error_m,
        "gnb_cell_accuracy": gnb_summary.cell_accuracy,
        "gnb_mean_error_m": gnb_summary.mean_error_m,
        "gnb_p90_error_m": gnb_summary.p90_error_m,
        "bag_knn_cell_accuracy": bag_knn_summary.cell_accuracy,
        "bag_knn_mean_error_m": bag_knn_summary.mean_error_m,
        "bag_knn_p90_error_m": bag_knn_summary.p90_error_m,
        "gpc_cell_accuracy": gpc_summary.cell_accuracy,
        "gpc_mean_error_m": gpc_summary.mean_error_m,
        "gpc_p90_error_m": gpc_summary.p90_error_m,
        "qda_cell_accuracy": qda_summary.cell_accuracy,
        "qda_mean_error_m": qda_summary.mean_error_m,
        "qda_p90_error_m": qda_summary.p90_error_m,
        "cat_cell_accuracy": cat_summary.cell_accuracy,
        "cat_mean_error_m": cat_summary.mean_error_m,
        "cat_p90_error_m": cat_summary.p90_error_m,
        "lgbm_cell_accuracy": lgbm_summary.cell_accuracy,
        "lgbm_mean_error_m": lgbm_summary.mean_error_m,
        "lgbm_p90_error_m": lgbm_summary.p90_error_m,
        "xgb_cell_accuracy": xgb_summary.cell_accuracy,
        "xgb_mean_error_m": xgb_summary.mean_error_m,
        "xgb_p90_error_m": xgb_summary.p90_error_m,
        "stacking_cell_accuracy": stacking_summary.cell_accuracy,
        "stacking_mean_error_m": stacking_summary.mean_error_m,
        "stacking_p90_error_m": stacking_summary.p90_error_m,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark localization models across rooms.")
    parser.add_argument(
        "--room",
        action="append",
        help="Filter to a specific room (repeatable). Example: --room D005 --room E101",
    )
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        help="Filter to router distances in meters. Example: --distances 2 4",
    )
    parser.add_argument(
        "--holdout",
        choices=["auto", "room", "distance", "none"],
        default="auto",
        help="Holdout strategy for the group-split benchmarks (room or distance).",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional/unstable models (Mahalanobis, GPC, CatBoost, LightGBM, XGBoost).",
    )
    parser.add_argument(
        "--run-gpc",
        action="store_true",
        help="Run the Gaussian Process Classifier stages (very slow on large datasets).",
    )
    parser.add_argument(
        "--gpc-max-samples",
        type=int,
        help="Optional cap on training samples for GPC to keep runtime reasonable (default: 1400 when GPC is enabled).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all optional models, including GPC.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.all_models:
        args.skip_optional = False
        args.run_gpc = True
    if args.run_gpc and args.gpc_max_samples is None:
        args.gpc_max_samples = 1400
    df = load_cross_room(room_filter=args.room, distance_filter=args.distances)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cell_lookup = (
        df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )
    rooms = sorted(df["room"].unique())
    distances = sorted(df["router_distance_m"].unique())
    if args.room or args.distances:
        print(f"Filters -> rooms={args.room or 'all'} | distances={args.distances or 'all'}")
    holdout_mode = args.holdout
    if holdout_mode == "auto":
        if len(rooms) > 1:
            holdout_mode = "room"
        elif len(distances) > 1:
            holdout_mode = "distance"
        else:
            holdout_mode = "none"
    if holdout_mode == "room":
        holdout_label = "room-agnostic, LORO mean"
    elif holdout_mode == "distance":
        holdout_label = "distance-agnostic, LODO mean"
    else:
        holdout_label = "holdout skipped"
    holdout_suffix = f"({holdout_label})"
    print(f"Holdout strategy: {holdout_mode}")

    print(
        f"Loaded {len(df)} samples | rooms={rooms} | "
        f"cells={df['grid_cell'].nunique()} | distances={distances}"
    )

    print(f"\n=== Holdout benchmark ({holdout_label}) ===")
    can_room = len(rooms) > 1
    can_distance = len(distances) > 1
    room_agnostic_mean = defaultdict(lambda: np.nan)
    if holdout_mode == "room":
        if can_room:
            room_agnostic = benchmark_room_agnostic(
                df,
                cell_lookup,
                skip_optional=args.skip_optional,
                run_gpc=args.run_gpc,
                gpc_max_samples=args.gpc_max_samples,
            )
            print(room_agnostic.to_string(index=False))
            print("Averages:", room_agnostic.mean(numeric_only=True).to_dict())
            print(f"Confusions enregistrées dans {REPORT_DIR}/confusion_room_agnostic_*.csv")
            room_agnostic_mean.update(room_agnostic.mean(numeric_only=True).to_dict())
        else:
            print("Skipped: needs at least 2 rooms.")
    elif holdout_mode == "distance":
        if can_distance:
            df_holdout = df.copy()
            df_holdout["room"] = df_holdout["router_distance_m"].map(lambda d: f"dist_{d:g}")
            room_agnostic = benchmark_room_agnostic(
                df_holdout,
                cell_lookup,
                skip_optional=args.skip_optional,
                run_gpc=args.run_gpc,
                gpc_max_samples=args.gpc_max_samples,
            )
            print(room_agnostic.to_string(index=False))
            print("Averages:", room_agnostic.mean(numeric_only=True).to_dict())
            print(f"Confusions enregistrées dans {REPORT_DIR}/confusion_room_agnostic_*.csv")
            room_agnostic_mean.update(room_agnostic.mean(numeric_only=True).to_dict())
        else:
            print("Skipped: needs at least 2 distances.")
    else:
        print("Skipped: holdout disabled.")

    print("\n=== Room-aware (one-hot room feature) ===")
    aware = benchmark_room_aware(
        df,
        cell_lookup,
        skip_optional=args.skip_optional,
        run_gpc=args.run_gpc,
        gpc_max_samples=args.gpc_max_samples,
    )
    print(json.dumps(aware, indent=2))
    print(f"Confusions enregistrées dans {REPORT_DIR}/confusion_room_aware_*.csv")

    print("\n=== LogisticRegression on embedding: router distance ===")
    if can_distance:
        distance_res = benchmark_distance_logreg(df, REPORT_DIR / "confusion_distance_logreg.csv")
        print(json.dumps(distance_res, indent=2))
        print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_distance_logreg.csv")
    else:
        print("Skipped: needs at least 2 distances.")
        distance_res = {"accuracy": np.nan}

    print("\n=== RandomForest on raw features: router distance ===")
    if can_distance:
        distance_rf = benchmark_distance_rf(df, REPORT_DIR / "confusion_distance_rf.csv")
        print(json.dumps(distance_rf, indent=2))
        print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_distance_rf.csv")
    else:
        print("Skipped: needs at least 2 distances.")
        distance_rf = {"accuracy": np.nan}

    print("\n=== ExtraTrees on raw features: router distance ===")
    if can_distance:
        distance_et = benchmark_distance_extratrees(df, REPORT_DIR / "confusion_distance_extratrees.csv")
        print(json.dumps(distance_et, indent=2))
        print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_distance_extratrees.csv")
    else:
        print("Skipped: needs at least 2 distances.")
        distance_et = {"accuracy": np.nan}

    print("\n=== Multihead embedding (cell + distance + room from RSSI only) ===")
    if can_room and can_distance:
        multihead = benchmark_multihead_embedding(df, cell_lookup, save_prefix=REPORT_DIR / "multihead_placeholder.csv")
        print(json.dumps(multihead, indent=2))
        print("Confusions enregistrées dans reports/benchmarks/confusion_multihead_*.csv")
    else:
        print("Skipped: needs at least 2 rooms and 2 distances.")
        multihead = {
            "cell_acc": np.nan,
            "mean_error_m": np.nan,
            "p90_error_m": np.nan,
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        }

    print("\n=== LogisticRegression on embedding: room classification ===")
    if can_room:
        room_res = benchmark_room_classifier(df, REPORT_DIR / "confusion_room_classifier.csv")
        print(json.dumps(room_res, indent=2))
        print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_room_classifier.csv")
    else:
        print("Skipped: needs at least 2 rooms.")
        room_res = {"accuracy": np.nan}

    # Aggregate a compact score table for quick comparison.
    summary_rows = [
        {
            "algorithm": f"NN+L-KNN {holdout_suffix}",
            "cell_acc": room_agnostic_mean["cell_accuracy"],
            "mean_error_m": room_agnostic_mean["mean_error_m"],
            "p90_error_m": room_agnostic_mean["p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"SVM RBF {holdout_suffix}",
            "cell_acc": room_agnostic_mean["svm_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["svm_mean_error_m"],
            "p90_error_m": room_agnostic_mean["svm_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"SVM lin {holdout_suffix}",
            "cell_acc": room_agnostic_mean["svm_lin_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["svm_lin_mean_error_m"],
            "p90_error_m": room_agnostic_mean["svm_lin_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"SVM poly {holdout_suffix}",
            "cell_acc": room_agnostic_mean["svm_poly_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["svm_poly_mean_error_m"],
            "p90_error_m": room_agnostic_mean["svm_poly_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"HistGradientBoosting {holdout_suffix}",
            "cell_acc": room_agnostic_mean["hgb_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["hgb_mean_error_m"],
            "p90_error_m": room_agnostic_mean["hgb_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"GradientBoosting {holdout_suffix}",
            "cell_acc": room_agnostic_mean["gbdt_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["gbdt_mean_error_m"],
            "p90_error_m": room_agnostic_mean["gbdt_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"RandomForest {holdout_suffix}",
            "cell_acc": room_agnostic_mean["rf_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["rf_mean_error_m"],
            "p90_error_m": room_agnostic_mean["rf_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"KNN {holdout_suffix}",
            "cell_acc": room_agnostic_mean["knn_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["knn_mean_error_m"],
            "p90_error_m": room_agnostic_mean["knn_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"Bagging KNN {holdout_suffix}",
            "cell_acc": room_agnostic_mean["bag_knn_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["bag_knn_mean_error_m"],
            "p90_error_m": room_agnostic_mean["bag_knn_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"Mahalanobis KNN {holdout_suffix}",
            "cell_acc": room_agnostic_mean["knn_maha_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["knn_maha_mean_error_m"],
            "p90_error_m": room_agnostic_mean["knn_maha_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"LDA+KNN {holdout_suffix}",
            "cell_acc": room_agnostic_mean["knn_lda_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["knn_lda_mean_error_m"],
            "p90_error_m": room_agnostic_mean["knn_lda_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"ExtraTrees {holdout_suffix}",
            "cell_acc": room_agnostic_mean["et_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["et_mean_error_m"],
            "p90_error_m": room_agnostic_mean["et_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"Deep MLP softmax {holdout_suffix}",
            "cell_acc": room_agnostic_mean["mlp_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["mlp_mean_error_m"],
            "p90_error_m": room_agnostic_mean["mlp_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"GaussianNB {holdout_suffix}",
            "cell_acc": room_agnostic_mean["gnb_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["gnb_mean_error_m"],
            "p90_error_m": room_agnostic_mean["gnb_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"Gaussian Process {holdout_suffix}",
            "cell_acc": room_agnostic_mean["gpc_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["gpc_mean_error_m"],
            "p90_error_m": room_agnostic_mean["gpc_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"QDA {holdout_suffix}",
            "cell_acc": room_agnostic_mean["qda_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["qda_mean_error_m"],
            "p90_error_m": room_agnostic_mean["qda_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"CatBoost {holdout_suffix}",
            "cell_acc": room_agnostic_mean["cat_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["cat_mean_error_m"],
            "p90_error_m": room_agnostic_mean["cat_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"LightGBM {holdout_suffix}",
            "cell_acc": room_agnostic_mean["lgbm_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["lgbm_mean_error_m"],
            "p90_error_m": room_agnostic_mean["lgbm_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"XGBoost {holdout_suffix}",
            "cell_acc": room_agnostic_mean["xgb_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["xgb_mean_error_m"],
            "p90_error_m": room_agnostic_mean["xgb_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"Stacking (RF+SVM+MLP) {holdout_suffix}",
            "cell_acc": room_agnostic_mean["stacking_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["stacking_mean_error_m"],
            "p90_error_m": room_agnostic_mean["stacking_p90_error_m"],
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
            "algorithm": "SVM RBF (room-aware, cell)",
            "cell_acc": aware["svm_cell_accuracy"],
            "mean_error_m": aware["svm_mean_error_m"],
            "p90_error_m": aware["svm_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "SVM lin (room-aware, cell)",
            "cell_acc": aware["svm_lin_cell_accuracy"],
            "mean_error_m": aware["svm_lin_mean_error_m"],
            "p90_error_m": aware["svm_lin_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "SVM poly (room-aware, cell)",
            "cell_acc": aware["svm_poly_cell_accuracy"],
            "mean_error_m": aware["svm_poly_mean_error_m"],
            "p90_error_m": aware["svm_poly_p90_error_m"],
            "router_dist_acc": np.nan,
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
            "algorithm": "HistGradientBoosting (room-aware, cell)",
            "cell_acc": aware["hgb_cell_accuracy"],
            "mean_error_m": aware["hgb_mean_error_m"],
            "p90_error_m": aware["hgb_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "GradientBoosting (room-aware, cell)",
            "cell_acc": aware["gbdt_cell_accuracy"],
            "mean_error_m": aware["gbdt_mean_error_m"],
            "p90_error_m": aware["gbdt_p90_error_m"],
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
            "algorithm": "Bagging KNN (room-aware, cell)",
            "cell_acc": aware["bag_knn_cell_accuracy"],
            "mean_error_m": aware["bag_knn_mean_error_m"],
            "p90_error_m": aware["bag_knn_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "Mahalanobis KNN (room-aware, cell)",
            "cell_acc": aware["knn_maha_cell_accuracy"],
            "mean_error_m": aware["knn_maha_mean_error_m"],
            "p90_error_m": aware["knn_maha_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "LDA+KNN (room-aware, cell)",
            "cell_acc": aware["knn_lda_cell_accuracy"],
            "mean_error_m": aware["knn_lda_mean_error_m"],
            "p90_error_m": aware["knn_lda_p90_error_m"],
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
            "algorithm": "Deep MLP softmax (room-aware, cell)",
            "cell_acc": aware["mlp_cell_accuracy"],
            "mean_error_m": aware["mlp_mean_error_m"],
            "p90_error_m": aware["mlp_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "GaussianNB (room-aware, cell)",
            "cell_acc": aware["gnb_cell_accuracy"],
            "mean_error_m": aware["gnb_mean_error_m"],
            "p90_error_m": aware["gnb_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "Gaussian Process (room-aware, cell)",
            "cell_acc": aware["gpc_cell_accuracy"],
            "mean_error_m": aware["gpc_mean_error_m"],
            "p90_error_m": aware["gpc_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "QDA (room-aware, cell)",
            "cell_acc": aware["qda_cell_accuracy"],
            "mean_error_m": aware["qda_mean_error_m"],
            "p90_error_m": aware["qda_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "CatBoost (room-aware, cell)",
            "cell_acc": aware["cat_cell_accuracy"],
            "mean_error_m": aware["cat_mean_error_m"],
            "p90_error_m": aware["cat_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "LightGBM (room-aware, cell)",
            "cell_acc": aware["lgbm_cell_accuracy"],
            "mean_error_m": aware["lgbm_mean_error_m"],
            "p90_error_m": aware["lgbm_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "XGBoost (room-aware, cell)",
            "cell_acc": aware["xgb_cell_accuracy"],
            "mean_error_m": aware["xgb_mean_error_m"],
            "p90_error_m": aware["xgb_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "Stacking (RF+SVM+MLP, room-aware, cell)",
            "cell_acc": aware["stacking_cell_accuracy"],
            "mean_error_m": aware["stacking_mean_error_m"],
            "p90_error_m": aware["stacking_p90_error_m"],
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

    failure_path = REPORT_DIR / "benchmark_failures.json"
    if FAIL_LOG:
        failure_path.write_text(json.dumps(FAIL_LOG, indent=2))
        print(f"Stages skipped/failed logged in {failure_path}")
    else:
        print("No skipped/failed stages recorded.")


if __name__ == "__main__":
    main()
