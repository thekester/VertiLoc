"""Benchmark NN+L-KNN variants and auxiliary classifiers across rooms.

This script mirrors the notebook scenarios:
- room-agnostic NN+L-KNN (leave-one-room-out),
- room-aware NN+L-KNN with room one-hot,
- LogisticRegression head for router distance (room-agnostic embedding),
- Room classification accuracy from the same embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from scipy.stats import binomtest, ttest_rel, wilcoxon
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

from localization.catalog import (  # noqa: E402
    BENCHMARK_REPORT_DIR,
    FEATURE_COLUMNS,
    ROOM_CAMPAIGNS,
    filter_room_campaigns,
)
from localization.data import infer_router_distance, load_measurements  # noqa: E402
from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer  # noqa: E402
warnings.filterwarnings("ignore", category=ConvergenceWarning)
console = Console()

REPORT_DIR = BENCHMARK_REPORT_DIR
HEATMAP_WINDOW = 5
GLOBAL_BASE_SEED = 21
ANTI_LEAKAGE_GROUP_COLS = ("room", "campaign", "grid_cell")
ANTI_LEAKAGE_QUANT_STEP_DB = 0.5


@dataclass
class BenchmarkResult:
    cell_accuracy: float
    mean_error_m: float
    p90_error_m: float
    extra: dict


FAIL_LOG: list[dict] = []
SPATIAL_TOP_K = 3


def _seed_value(*parts: object) -> int:
    raw = "::".join([str(GLOBAL_BASE_SEED), *(str(part) for part in parts)])
    digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _parse_seed_spec(seed_spec: str) -> list[int]:
    values: list[int] = []
    for raw in str(seed_spec).replace(",", " ").split():
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid seed range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise ValueError("No valid seeds provided.")
    return sorted(set(values))


def _parse_float_spec(values: str | None) -> list[float]:
    if not values:
        return []
    parsed: list[float] = []
    for raw in str(values).replace(",", " ").split():
        token = raw.strip()
        if not token:
            continue
        parsed.append(float(token))
    return parsed


def _t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
        27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(n - 1, 1.96)


def _aggregate_seed_summaries(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "cell_acc",
        "mean_error_m",
        "p90_error_m",
        "router_dist_acc",
        "room_acc",
        "top1_acc",
        "top3_acc",
        "top3_best_error_m",
        "top3_best_gain_m",
    ]
    rows: list[dict] = []
    for algo, group in per_seed_df.groupby("algorithm", sort=False):
        row: dict[str, float | int | str] = {"algorithm": algo, "n_seeds": int(group["seed"].nunique())}
        n = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci95"] = np.nan
                row[f"{metric}_ci95_low"] = np.nan
                row[f"{metric}_ci95_high"] = np.nan
                continue
            m = float(values.mean())
            s = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            sem = s / math.sqrt(len(values)) if len(values) > 1 else 0.0
            ci = float(_t_critical_95(len(values)) * sem) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
            row[f"{metric}_ci95"] = ci
            row[f"{metric}_ci95_low"] = m - ci
            row[f"{metric}_ci95_high"] = m + ci
        rows.append(row)
    return pd.DataFrame(rows)


def _group_keys(df: pd.DataFrame, group_cols: tuple[str, ...]) -> set[tuple]:
    return set(df.loc[:, list(group_cols)].itertuples(index=False, name=None))


def _quantized_signature_keys(
    df: pd.DataFrame,
    *,
    quant_step_db: float,
    group_cols: tuple[str, ...],
) -> set[tuple]:
    if quant_step_db <= 0:
        raise ValueError("quant_step_db must be > 0.")
    quantized = np.rint(df[FEATURE_COLUMNS].to_numpy(dtype=float) / float(quant_step_db)).astype(np.int32)
    base = df.loc[:, list(group_cols)].reset_index(drop=True)
    keys: set[tuple] = set()
    for row, q in zip(base.itertuples(index=False, name=None), quantized):
        keys.add((*row, *q.tolist()))
    return keys


def _strict_anti_leakage_audit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    quant_step_db: float,
    group_cols: tuple[str, ...] = ANTI_LEAKAGE_GROUP_COLS,
) -> dict[str, int | float]:
    train_groups = _group_keys(train_df, group_cols)
    test_groups = _group_keys(test_df, group_cols)
    overlap_groups = train_groups.intersection(test_groups)

    train_sig = _quantized_signature_keys(train_df, quant_step_db=quant_step_db, group_cols=group_cols)
    test_sig = _quantized_signature_keys(test_df, quant_step_db=quant_step_db, group_cols=group_cols)
    overlap_sig = train_sig.intersection(test_sig)

    return {
        "n_groups_train": len(train_groups),
        "n_groups_test": len(test_groups),
        "n_overlap_groups": len(overlap_groups),
        "n_overlap_signatures": len(overlap_sig),
        "quant_step_db": float(quant_step_db),
    }


def _strict_group_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    quant_step_db: float,
    context: str,
    group_cols: tuple[str, ...] = ANTI_LEAKAGE_GROUP_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing group columns for anti-leakage split: {missing}")
    groups = df.loc[:, list(group_cols)].astype(str).agg("::".join, axis=1)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    audit = _strict_anti_leakage_audit(
        train_df,
        test_df,
        quant_step_db=quant_step_db,
        group_cols=group_cols,
    )
    if audit["n_overlap_groups"] > 0 or audit["n_overlap_signatures"] > 0:
        raise RuntimeError(
            f"{context}: anti-leakage audit failed: "
            f"n_overlap_groups={audit['n_overlap_groups']}, "
            f"n_overlap_signatures={audit['n_overlap_signatures']}"
        )
    print(
        f"[anti-leakage:{context}] train_groups={audit['n_groups_train']} "
        f"test_groups={audit['n_groups_test']} overlap_groups=0 overlap_signatures=0 "
        f"(quant={audit['quant_step_db']:.3f} dB)"
    )
    return train_df, test_df, audit


def _filter_room_campaigns(
    room_filter: list[str] | None,
    distance_filter: list[float] | None,
):
    return filter_room_campaigns(room_filter, distance_filter)


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
    for room, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df_room = load_measurements([spec])
                df_room["room"] = room
                df_room["campaign"] = f"{room}/{spec.path.name}"
                frames.append(df_room)
            else:
                missing.append(spec.path)

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


def _topk_labels_from_proba(classes: np.ndarray, proba: np.ndarray, k: int) -> np.ndarray:
    """Return labels sorted by descending probability for each sample."""
    probs = np.asarray(proba, dtype=float)
    labels = np.asarray(classes)
    if probs.ndim != 2:
        raise ValueError("proba must be a 2-D array.")
    if probs.shape[1] != labels.shape[0]:
        raise ValueError("classes/proba shape mismatch.")
    k_eff = max(1, min(int(k), probs.shape[1]))
    topk_idx = np.argpartition(probs, -k_eff, axis=1)[:, -k_eff:]
    topk_scores = np.take_along_axis(probs, topk_idx, axis=1)
    order = np.argsort(-topk_scores, axis=1)
    topk_idx = np.take_along_axis(topk_idx, order, axis=1)
    return labels[topk_idx]


def compute_spatial_topk_metrics(
    df_true: pd.DataFrame,
    proba: np.ndarray,
    classes: np.ndarray,
    cell_lookup: pd.DataFrame,
    *,
    top_k: int = SPATIAL_TOP_K,
) -> dict[str, float | int]:
    """Compute top-1/top-k accuracy and mean best spatial distance among top-k."""
    y_true = df_true["grid_cell"].to_numpy()
    true_coords = df_true[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)

    probs = np.asarray(proba, dtype=float)
    labels = np.asarray(classes)
    pred_idx = np.argmax(probs, axis=1)
    y_top1 = labels[pred_idx]
    top1_hit = y_top1 == y_true
    pred_coords = cell_lookup.loc[y_top1, ["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    top1_errors = np.linalg.norm(true_coords - pred_coords, axis=1)

    topk_labels = _topk_labels_from_proba(labels, probs, top_k)
    topk_hit = np.any(topk_labels == y_true[:, None], axis=1)
    best_topk = np.zeros(len(y_true), dtype=float)
    for i in range(len(y_true)):
        cand_coords = cell_lookup.loc[topk_labels[i], ["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
        best_topk[i] = float(np.min(np.linalg.norm(cand_coords - true_coords[i], axis=1)))

    return {
        "top_k": int(topk_labels.shape[1]),
        "top1_acc": float(np.mean(top1_hit)),
        "topk_acc": float(np.mean(topk_hit)),
        "top1_error_m": float(np.mean(top1_errors)),
        "topk_best_error_m": float(np.mean(best_topk)),
        "topk_best_gain_m": float(np.mean(top1_errors) - np.mean(best_topk)),
    }


def _coverage_risk_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    max_points: int = 201,
) -> pd.DataFrame:
    conf = np.asarray(confidences, dtype=float).ravel()
    ok = np.asarray(correct, dtype=bool).ravel()
    if conf.shape[0] != ok.shape[0]:
        raise ValueError("confidences and correct must have the same length.")
    n = conf.shape[0]
    if n == 0:
        return pd.DataFrame(
            columns=[
                "n_selected",
                "coverage",
                "rejection_rate",
                "conditional_accuracy",
                "risk",
                "confidence_threshold",
            ]
        )

    order = np.argsort(-conf)
    conf_sorted = conf[order]
    ok_sorted = ok[order].astype(np.float64)
    k = np.arange(1, n + 1, dtype=np.int64)
    cum_correct = np.cumsum(ok_sorted)
    coverage = k / float(n)
    cond_acc = cum_correct / k
    risk = 1.0 - cond_acc
    rejection = 1.0 - coverage

    curve = pd.DataFrame(
        {
            "n_selected": k,
            "coverage": coverage,
            "rejection_rate": rejection,
            "conditional_accuracy": cond_acc,
            "risk": risk,
            "confidence_threshold": conf_sorted,
        }
    )
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
        idx[-1] = n - 1
        curve = curve.iloc[idx].reset_index(drop=True)
    return curve


def _operating_point_for_target_risk(curve: pd.DataFrame, target_risk: float) -> tuple[float, bool]:
    eligible = curve[curve["risk"] <= float(target_risk)]
    if not eligible.empty:
        return float(eligible.iloc[-1]["confidence_threshold"]), True
    best = curve.sort_values(["risk", "coverage"], ascending=[True, False]).iloc[0]
    return float(best["confidence_threshold"]), False


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
    skip_stacking: bool = False,
    run_gpc: bool = False,
    gpc_max_samples: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    labels = sorted(df["grid_cell"].unique())
    label_to_idx, idx_to_label = _make_label_encoder(labels)
    rooms = sorted(df["room"].unique())
    optional_stages = {"KNN Mahalanobis", "GPC", "CatBoost", "LightGBM", "XGBoost"}
    if skip_stacking:
        optional_stages.add("Stacking")

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
                    lambda m=fit_localizer(
                        train_df,
                        include_room=False,
                        random_state=_seed_value("room_agnostic", held_out, "nn_lknn"),
                    ), preds=None: (
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
                        random_state=_seed_value("room_agnostic", held_out, "rf"),
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

            knn_dist_scaled_summary = log_stage(
                "KNN distance (scaled)",
                lambda: (
                    lambda clf=make_pipeline(
                        StandardScaler(),
                        KNeighborsClassifier(n_neighbors=9, weights="distance"),
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                        save_confusion(
                            test_df["grid_cell"],
                            clf.predict(X_test),
                            labels,
                            REPORT_DIR / f"confusion_room_agnostic_knn_dist_scaled_{held_out}.csv",
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
                        random_state=_seed_value("room_agnostic", held_out, "et"),
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

            cal_logreg_summary = log_stage(
                "Calibrated LogReg",
                lambda: (
                    lambda clf=CalibratedClassifierCV(
                        estimator=make_pipeline(
                            StandardScaler(),
                            LogisticRegression(
                                max_iter=1000,
                                C=2.0,
                                multi_class="auto",
                            ),
                        ),
                        method="sigmoid",
                        cv=3,
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                        save_confusion(
                            test_df["grid_cell"],
                            clf.predict(X_test),
                            labels,
                            REPORT_DIR / f"confusion_room_agnostic_logreg_cal_{held_out}.csv",
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
                        random_state=_seed_value("room_agnostic", held_out, "hgb"),
                    ): (
                        clf.fit(X_train, train_df["grid_cell"]),
                        localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    )
                )()[1],
            )

            gbdt_summary = log_stage(
                "GBDT",
                lambda: (
                    lambda clf=GradientBoostingClassifier(
                        random_state=_seed_value("room_agnostic", held_out, "gbdt")
                    ): (
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
                        random_state=_seed_value("room_agnostic", held_out, "mlp"),
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
                        random_state=_seed_value("room_agnostic", held_out, "bag_knn"),
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
                        random_state=_seed_value("room_agnostic", held_out, "gpc_subsample"),
                    )
                    X_train_gpc = build_features(gpc_train_df, include_room=False)
                    X_test_gpc = build_features(test_df, include_room=False)
                    clf = make_pipeline(
                        StandardScaler(),
                        GaussianProcessClassifier(
                            kernel=RBF(length_scale=1.0),
                            optimizer=None,
                            random_state=_seed_value("room_agnostic", held_out, "gpc"),
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
                        random_seed=_seed_value("room_agnostic", held_out, "catboost"),
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
                        random_state=_seed_value("room_agnostic", held_out, "lgbm"),
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
                        random_state=_seed_value("room_agnostic", held_out, "xgb"),
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
                                random_state=_seed_value("room_agnostic", held_out, "stacking_rf"),
                                n_jobs=-1,
                            )),
                            ("svm_rbf", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=8.0, gamma="scale", probability=True))),
                            ("mlp", MLPClassifier(
                                hidden_layer_sizes=(128, 64),
                                activation="relu",
                                max_iter=500,
                                alpha=5e-4,
                                learning_rate_init=5e-4,
                                random_state=_seed_value("room_agnostic", held_out, "stacking_mlp"),
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
                    "knn_dist_scaled_cell_accuracy": knn_dist_scaled_summary.cell_accuracy,
                    "knn_dist_scaled_mean_error_m": knn_dist_scaled_summary.mean_error_m,
                    "knn_dist_scaled_p90_error_m": knn_dist_scaled_summary.p90_error_m,
                    "cal_logreg_cell_accuracy": cal_logreg_summary.cell_accuracy,
                    "cal_logreg_mean_error_m": cal_logreg_summary.mean_error_m,
                    "cal_logreg_p90_error_m": cal_logreg_summary.p90_error_m,
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
    skip_stacking: bool = False,
    run_gpc: bool = False,
    gpc_max_samples: int | None = None,
    strict_anti_leakage: bool = False,
    quant_step_db: float = ANTI_LEAKAGE_QUANT_STEP_DB,
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
        task = progress.add_task("Room-aware models", total=23)
        stage_table = ProgressTable(columns=["stage", "status", "elapsed_s"], interactive=2, refresh_rate=20)
        stage_context = {"mode": "room_aware"}
        labels = sorted(df["grid_cell"].unique())
        label_to_idx, idx_to_label = _make_label_encoder(labels)
        optional_stages = {"KNN Mahalanobis", "GPC", "CatBoost", "LightGBM", "XGBoost"}
        if skip_stacking:
            optional_stages.add("Stacking")

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

        if strict_anti_leakage:
            train_df, test_df, _ = _strict_group_train_test_split(
                df,
                test_size=0.2,
                random_state=_seed_value("room_aware", "split"),
                quant_step_db=quant_step_db,
                context="room_aware",
            )
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=_seed_value("room_aware", "split"),
                stratify=df["grid_cell"],
            )
        labels = sorted(df["grid_cell"].unique())
        model, summary, y_pred = log_stage(
            "NN+L-KNN",
            lambda: (
                lambda m=fit_localizer(
                    train_df, include_room=True, random_state=_seed_value("room_aware", "nn_lknn")
                ), preds=None: (
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
        nn_lknn_topk = compute_spatial_topk_metrics(
            test_df,
            model.predict_proba(X_test),
            model.knn_.classes_,
            cell_lookup,
            top_k=SPATIAL_TOP_K,
        )

        rf_summary = log_stage(
            "RandomForest",
            lambda: (
                lambda clf=RandomForestClassifier(
                    n_estimators=220,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=_seed_value("room_aware", "rf"),
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

        knn_dist_scaled_summary = log_stage(
            "KNN distance (scaled)",
            lambda: (
                lambda clf=make_pipeline(
                    StandardScaler(),
                    KNeighborsClassifier(n_neighbors=9, weights="distance"),
                ): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    save_confusion(
                        test_df["grid_cell"],
                        clf.predict(X_test),
                        labels,
                        REPORT_DIR / "confusion_room_aware_knn_dist_scaled.csv",
                    ),
                )
            )()[1],
        )

        cal_logreg_summary = log_stage(
            "Calibrated LogReg",
            lambda: (
                lambda clf=CalibratedClassifierCV(
                    estimator=make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=1000,
                            C=2.0,
                            multi_class="auto",
                        ),
                    ),
                    method="sigmoid",
                    cv=3,
                ): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                    save_confusion(
                        test_df["grid_cell"],
                        clf.predict(X_test),
                        labels,
                        REPORT_DIR / "confusion_room_aware_logreg_cal.csv",
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
                    random_state=_seed_value("room_aware", "et"),
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
                    random_state=_seed_value("room_aware", "hgb"),
                ): (
                    clf.fit(X_train, train_df["grid_cell"]),
                    localization_summary(test_df, clf.predict(X_test), cell_lookup),
                )
            )()[1],
        )

        gbdt_summary = log_stage(
            "GBDT",
            lambda: (
                lambda clf=GradientBoostingClassifier(random_state=_seed_value("room_aware", "gbdt")): (
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
                    random_state=_seed_value("room_aware", "mlp"),
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
                    random_state=_seed_value("room_aware", "bag_knn"),
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
                    train_df,
                    max_samples=gpc_max_samples,
                    label_col="grid_cell",
                    random_state=_seed_value("room_aware", "gpc_subsample"),
                )
                X_train_gpc = build_features(gpc_train_df, include_room=True)
                X_test_gpc = build_features(test_df, include_room=True)
                clf = make_pipeline(
                    StandardScaler(),
                    GaussianProcessClassifier(
                        kernel=RBF(length_scale=1.0),
                        optimizer=None,
                        random_state=_seed_value("room_aware", "gpc"),
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
                    random_seed=_seed_value("room_aware", "catboost"),
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
                    random_state=_seed_value("room_aware", "lgbm"),
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
                    random_state=_seed_value("room_aware", "xgb"),
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
                random_state=_seed_value("room_aware", "stacking_rf"),
                n_jobs=-1,
            )),
            ("svm_rbf", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=8.0, gamma="scale", probability=True))),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=500,
                alpha=5e-4,
                learning_rate_init=5e-4,
                random_state=_seed_value("room_aware", "stacking_mlp"),
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
        "nn_lknn_top1_acc": nn_lknn_topk["top1_acc"],
        "nn_lknn_top3_acc": nn_lknn_topk["topk_acc"],
        "nn_lknn_top3_best_error_m": nn_lknn_topk["topk_best_error_m"],
        "nn_lknn_top3_best_gain_m": nn_lknn_topk["topk_best_gain_m"],
        "rf_cell_accuracy": rf_summary.cell_accuracy,
        "rf_mean_error_m": rf_summary.mean_error_m,
        "rf_p90_error_m": rf_summary.p90_error_m,
        "knn_cell_accuracy": knn_summary.cell_accuracy,
        "knn_mean_error_m": knn_summary.mean_error_m,
        "knn_p90_error_m": knn_summary.p90_error_m,
        "knn_dist_scaled_cell_accuracy": knn_dist_scaled_summary.cell_accuracy,
        "knn_dist_scaled_mean_error_m": knn_dist_scaled_summary.mean_error_m,
        "knn_dist_scaled_p90_error_m": knn_dist_scaled_summary.p90_error_m,
        "cal_logreg_cell_accuracy": cal_logreg_summary.cell_accuracy,
        "cal_logreg_mean_error_m": cal_logreg_summary.mean_error_m,
        "cal_logreg_p90_error_m": cal_logreg_summary.p90_error_m,
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

def benchmark_distance_logreg(
    df: pd.DataFrame,
    save_path: Path | None = None,
    *,
    strict_anti_leakage: bool = False,
    quant_step_db: float = ANTI_LEAKAGE_QUANT_STEP_DB,
) -> dict:
    if strict_anti_leakage:
        train_df, test_df, _ = _strict_group_train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("distance_logreg", "split"),
            quant_step_db=quant_step_db,
            context="distance_logreg",
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("distance_logreg", "split"),
            stratify=df["router_distance_m"],
        )
    model = fit_localizer(train_df, include_room=False, random_state=_seed_value("distance_logreg", "nn_lknn"))
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


def benchmark_distance_rf(
    df: pd.DataFrame,
    save_path: Path | None = None,
    *,
    strict_anti_leakage: bool = False,
    quant_step_db: float = ANTI_LEAKAGE_QUANT_STEP_DB,
) -> dict:
    if strict_anti_leakage:
        train_df, test_df, _ = _strict_group_train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("distance_rf", "split"),
            quant_step_db=quant_step_db,
            context="distance_rf",
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("distance_rf", "split"),
            stratify=df["router_distance_m"],
        )
    clf = RandomForestClassifier(
        n_estimators=220,
        max_depth=18,
        min_samples_leaf=2,
        random_state=_seed_value("distance_rf", "rf"),
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


def benchmark_distance_extratrees(
    df: pd.DataFrame,
    save_path: Path | None = None,
    *,
    strict_anti_leakage: bool = False,
    quant_step_db: float = ANTI_LEAKAGE_QUANT_STEP_DB,
) -> dict:
    if strict_anti_leakage:
        train_df, test_df, _ = _strict_group_train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("distance_et", "split"),
            quant_step_db=quant_step_db,
            context="distance_et",
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("distance_et", "split"),
            stratify=df["router_distance_m"],
        )
    clf = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=24,
        min_samples_leaf=2,
        random_state=_seed_value("distance_et", "et"),
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
    strict_anti_leakage: bool = False,
    quant_step_db: float = ANTI_LEAKAGE_QUANT_STEP_DB,
) -> dict:
    """Single model using RSSI only, predicting cell + distance + room from the same embedding."""
    if strict_anti_leakage:
        train_df, test_df, _ = _strict_group_train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("multihead", "split"),
            quant_step_db=quant_step_db,
            context="multihead",
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("multihead", "split"),
            stratify=df["grid_cell"],
        )
    labels = sorted(df["grid_cell"].unique())
    # Fit embedding on cell labels only (no room/distance as features).
    model = fit_localizer(train_df, include_room=False, random_state=_seed_value("multihead", "nn_lknn"))
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


def benchmark_room_classifier(
    df: pd.DataFrame,
    save_path: Path | None = None,
    *,
    strict_anti_leakage: bool = False,
    quant_step_db: float = ANTI_LEAKAGE_QUANT_STEP_DB,
) -> dict:
    if strict_anti_leakage:
        train_df, test_df, _ = _strict_group_train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("room_classifier", "split"),
            quant_step_db=quant_step_db,
            context="room_classifier",
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("room_classifier", "split"),
            stratify=df["room"],
        )
    model = fit_localizer(train_df, include_room=False, random_state=_seed_value("room_classifier", "nn_lknn"))
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


def benchmark_calibrated_rejection(
    df: pd.DataFrame,
    *,
    output_csv: Path,
    output_png: Path,
    output_json: Path,
    include_room: bool = True,
    target_risks: list[float] | None = None,
    strict_anti_leakage: bool = False,
    quant_step_db: float = ANTI_LEAKAGE_QUANT_STEP_DB,
) -> dict:
    """Compute a calibrated reject option curve (coverage-risk) for production thresholding."""
    if target_risks is None:
        target_risks = [0.05, 0.10, 0.15, 0.20]
    target_risks = sorted(set(float(v) for v in target_risks if float(v) >= 0.0))
    if not target_risks:
        raise ValueError("At least one non-negative target risk is required.")

    if strict_anti_leakage:
        train_df, temp_df, _ = _strict_group_train_test_split(
            df,
            test_size=0.4,
            random_state=_seed_value("coverage_risk", "train_temp"),
            quant_step_db=quant_step_db,
            context="coverage_risk_train_temp",
        )
        cal_df, test_df, _ = _strict_group_train_test_split(
            temp_df,
            test_size=0.5,
            random_state=_seed_value("coverage_risk", "cal_test"),
            quant_step_db=quant_step_db,
            context="coverage_risk_cal_test",
        )
    else:
        strat = df["grid_cell"] if df["grid_cell"].nunique() > 1 else None
        try:
            train_df, temp_df = train_test_split(
                df,
                test_size=0.4,
                random_state=_seed_value("coverage_risk", "train_temp"),
                stratify=strat,
            )
        except ValueError:
            train_df, temp_df = train_test_split(
                df,
                test_size=0.4,
                random_state=_seed_value("coverage_risk", "train_temp"),
                stratify=None,
            )
        strat_temp = temp_df["grid_cell"] if temp_df["grid_cell"].nunique() > 1 else None
        try:
            cal_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=_seed_value("coverage_risk", "cal_test"),
                stratify=strat_temp,
            )
        except ValueError:
            cal_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=_seed_value("coverage_risk", "cal_test"),
                stratify=None,
            )

    clf = RandomForestClassifier(
        n_estimators=260,
        max_depth=16,
        min_samples_leaf=2,
        random_state=_seed_value("coverage_risk", "rf"),
        n_jobs=-1,
    )
    X_train = build_features(train_df, include_room=include_room)
    X_cal = build_features(cal_df, include_room=include_room)
    X_test = build_features(test_df, include_room=include_room)
    clf.fit(X_train, train_df["grid_cell"])

    y_cal_true = cal_df["grid_cell"].to_numpy()
    y_test_true = test_df["grid_cell"].to_numpy()

    proba_cal = clf.predict_proba(X_cal)
    proba_test = clf.predict_proba(X_test)
    classes = np.asarray(clf.classes_)

    cal_idx = np.argmax(proba_cal, axis=1)
    test_idx = np.argmax(proba_test, axis=1)
    y_cal_pred = classes[cal_idx]
    y_test_pred = classes[test_idx]
    conf_cal = np.max(proba_cal, axis=1)
    conf_test = np.max(proba_test, axis=1)
    cal_correct = y_cal_pred == y_cal_true
    test_correct = y_test_pred == y_test_true

    curve_cal = _coverage_risk_curve(conf_cal, cal_correct)
    curve_cal["split"] = "calibration"
    curve_test = _coverage_risk_curve(conf_test, test_correct)
    curve_test["split"] = "test"
    curve_df = pd.concat([curve_cal, curve_test], ignore_index=True)

    operating_rows: list[dict] = []
    for target in target_risks:
        threshold, attained = _operating_point_for_target_risk(curve_cal, target)
        kept = conf_test >= threshold
        coverage = float(np.mean(kept))
        rejected = 1.0 - coverage
        if np.any(kept):
            cond_acc = float(np.mean(test_correct[kept]))
            risk = 1.0 - cond_acc
        else:
            cond_acc = float("nan")
            risk = float("nan")
        operating_rows.append(
            {
                "target_risk": float(target),
                "threshold_from_calibration": float(threshold),
                "target_attained_on_calibration": bool(attained),
                "coverage_test": coverage,
                "rejection_rate_test": rejected,
                "conditional_accuracy_test": cond_acc,
                "risk_test": risk,
                "n_selected_test": int(np.sum(kept)),
                "n_test": int(len(test_df)),
            }
        )
    operating_df = pd.DataFrame(operating_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(output_csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        curve_cal["rejection_rate"],
        curve_cal["conditional_accuracy"],
        label="Calibration",
        linewidth=2.0,
        color="#1f77b4",
    )
    ax.plot(
        curve_test["rejection_rate"],
        curve_test["conditional_accuracy"],
        label="Test",
        linewidth=2.0,
        color="#ff7f0e",
    )
    if not operating_df.empty:
        ax.scatter(
            operating_df["rejection_rate_test"],
            operating_df["conditional_accuracy_test"],
            color="#d62728",
            s=36,
            label="Points calibres (test)",
            zorder=3,
        )
    ax.set_xlabel("Taux de rejet")
    ax.set_ylabel("Accuracy conditionnelle")
    ax.set_title("Courbe coverage-risk (rejet calibre)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    summary = {
        "baseline_test_accuracy_no_reject": float(np.mean(test_correct)),
        "n_train": int(len(train_df)),
        "n_calibration": int(len(cal_df)),
        "n_test": int(len(test_df)),
        "include_room_feature": bool(include_room),
        "target_risks": target_risks,
        "operating_points_test": operating_rows,
        "artifacts": {
            "curve_csv": str(output_csv),
            "curve_png": str(output_png),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2))
    return summary


def _pairwise_significance_from_predictions(
    top_models_df: pd.DataFrame,
    model_outputs: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict] = []
    selected = top_models_df["model"].tolist()
    for i, model_a in enumerate(selected):
        for model_b in selected[i + 1 :]:
            out_a = model_outputs[model_a]
            out_b = model_outputs[model_b]
            correct_a = out_a["correct"]
            correct_b = out_b["correct"]

            a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
            a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
            discordant = a_correct_b_wrong + a_wrong_b_correct
            if discordant == 0:
                mcnemar_p = 1.0
                mcnemar_note = "no_discordant_samples"
            else:
                mcnemar_p = float(binomtest(min(a_correct_b_wrong, a_wrong_b_correct), discordant, p=0.5).pvalue)
                mcnemar_note = "exact_binomial"

            err_a = out_a["error_m"]
            err_b = out_b["error_m"]
            delta = err_a - err_b
            delta_mean = float(np.mean(delta))

            try:
                wilcoxon_res = wilcoxon(err_a, err_b, zero_method="wilcox", alternative="two-sided", mode="auto")
                wilcoxon_p = float(wilcoxon_res.pvalue)
                wilcoxon_note = "ok"
            except ValueError:
                wilcoxon_p = 1.0
                wilcoxon_note = "all_differences_zero_or_invalid"

            t_res = ttest_rel(err_a, err_b, nan_policy="omit")
            ttest_p = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else np.nan

            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_samples": int(correct_a.shape[0]),
                    "acc_a": float(out_a["acc"]),
                    "acc_b": float(out_b["acc"]),
                    "delta_acc_a_minus_b": float(out_a["acc"] - out_b["acc"]),
                    "mean_error_m_a": float(np.mean(err_a)),
                    "mean_error_m_b": float(np.mean(err_b)),
                    "delta_mean_error_m_a_minus_b": delta_mean,
                    "mcnemar_a_correct_b_wrong": a_correct_b_wrong,
                    "mcnemar_a_wrong_b_correct": a_wrong_b_correct,
                    "mcnemar_p_value": mcnemar_p,
                    "mcnemar_note": mcnemar_note,
                    "wilcoxon_p_value": wilcoxon_p,
                    "wilcoxon_note": wilcoxon_note,
                    "paired_ttest_p_value": ttest_p,
                }
            )
    return pd.DataFrame(rows)


def run_significance_tests_room_aware(
    df: pd.DataFrame,
    cell_lookup: pd.DataFrame,
    *,
    top_k: int,
    output_dir: Path,
    strict_anti_leakage: bool = False,
    quant_step_db: float = ANTI_LEAKAGE_QUANT_STEP_DB,
) -> dict:
    if strict_anti_leakage:
        train_df, test_df, _ = _strict_group_train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("significance_room_aware", "split"),
            quant_step_db=quant_step_db,
            context="significance_room_aware",
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("significance_room_aware", "split"),
            stratify=df["grid_cell"],
        )
    true_cells = test_df["grid_cell"].to_numpy()

    def compute_error_m(y_pred: np.ndarray) -> np.ndarray:
        pred_meta = cell_lookup.loc[y_pred]
        true_coords = test_df[["coord_x_m", "coord_y_m"]].to_numpy()
        pred_coords = pred_meta[["coord_x_m", "coord_y_m"]].to_numpy()
        return np.linalg.norm(true_coords - pred_coords, axis=1)

    model_predictors = {
        "NN+L-KNN": lambda: fit_localizer(
            train_df, include_room=True, random_state=_seed_value("significance_room_aware", "nn_lknn")
        ).predict(
            build_features(test_df, include_room=True)
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=2,
            random_state=_seed_value("significance_room_aware", "rf"),
            n_jobs=-1,
        ).fit(build_features(train_df, include_room=True), train_df["grid_cell"]).predict(build_features(test_df, include_room=True)),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            random_state=_seed_value("significance_room_aware", "et"),
            n_jobs=-1,
        ).fit(build_features(train_df, include_room=True), train_df["grid_cell"]).predict(build_features(test_df, include_room=True)),
        "SVM RBF": lambda: make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=8.0, gamma="scale"),
        ).fit(build_features(train_df, include_room=True), train_df["grid_cell"]).predict(build_features(test_df, include_room=True)),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance")
        .fit(build_features(train_df, include_room=True), train_df["grid_cell"])
        .predict(build_features(test_df, include_room=True)),
        "HistGB": lambda: HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=8,
            max_iter=200,
            random_state=_seed_value("significance_room_aware", "hgb"),
        ).fit(build_features(train_df, include_room=True), train_df["grid_cell"]).predict(build_features(test_df, include_room=True)),
        "GBDT": lambda: GradientBoostingClassifier(random_state=_seed_value("significance_room_aware", "gbdt"))
        .fit(build_features(train_df, include_room=True), train_df["grid_cell"])
        .predict(build_features(test_df, include_room=True)),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            alpha=5e-4,
            learning_rate_init=5e-4,
            random_state=_seed_value("significance_room_aware", "mlp"),
        ).fit(build_features(train_df, include_room=True), train_df["grid_cell"]).predict(build_features(test_df, include_room=True)),
    }

    model_outputs: dict[str, dict[str, np.ndarray]] = {}
    top_rows: list[dict] = []
    failed: list[dict] = []

    for model_name, predictor in model_predictors.items():
        try:
            y_pred = np.asarray(predictor())
            correct = y_pred == true_cells
            errors_m = compute_error_m(y_pred)
            acc = float(np.mean(correct))
            mean_error = float(np.mean(errors_m))
            p90_error = float(np.percentile(errors_m, 90))
            model_outputs[model_name] = {
                "pred": y_pred,
                "correct": correct,
                "error_m": errors_m,
                "acc": np.asarray(acc),
            }
            top_rows.append(
                {
                    "model": model_name,
                    "cell_acc": acc,
                    "mean_error_m": mean_error,
                    "p90_error_m": p90_error,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep benchmark robust
            failed.append({"model": model_name, "reason": f"{type(exc).__name__}: {exc}"})

    if len(top_rows) < 2:
        raise RuntimeError("Not enough successful models to run significance tests.")

    max_top = min(3, len(top_rows))
    selected_k = max(2, min(top_k, max_top))
    top_models_df = (
        pd.DataFrame(top_rows)
        .sort_values(["cell_acc", "mean_error_m"], ascending=[False, True], ignore_index=True)
        .head(selected_k)
    )
    pairwise_df = _pairwise_significance_from_predictions(top_models_df, model_outputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    top_path = output_dir / "significance_room_aware_top_models.csv"
    pair_path = output_dir / "significance_room_aware_pairs.csv"
    json_path = output_dir / "significance_room_aware.json"
    top_models_df.to_csv(top_path, index=False)
    pairwise_df.to_csv(pair_path, index=False)
    json_payload = {
        "split": {
            "mode": "room_aware",
            "test_size": 0.2,
            "random_state": _seed_value("significance_room_aware", "split"),
            "strict_anti_leakage": bool(strict_anti_leakage),
            "quant_step_db": float(quant_step_db),
        },
        "n_test_samples": int(len(test_df)),
        "selected_top_k": int(selected_k),
        "top_models": top_models_df.to_dict(orient="records"),
        "pairwise_tests": pairwise_df.to_dict(orient="records"),
        "failed_models": failed,
        "paths": {
            "top_models_csv": str(top_path),
            "pairwise_csv": str(pair_path),
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2))
    return {
        "summary": {
            "n_test_samples": int(len(test_df)),
            "selected_top_k": int(selected_k),
            "top_models": top_models_df["model"].tolist(),
            "n_pairwise_tests": int(len(pairwise_df)),
        },
        "paths": {
            "top_models_csv": str(top_path),
            "pairwise_csv": str(pair_path),
            "json": str(json_path),
        },
        "failed_models": failed,
    }


def _rssi_to_heatmaps(X_raw: np.ndarray) -> np.ndarray:
    """Project raw RSSI vectors into a compact 2x3 spatial heatmap."""
    X = np.asarray(X_raw, dtype=float)
    signal = X[:, 0]
    noise = X[:, 1]
    a1 = X[:, 2]
    a2 = X[:, 3]
    a3 = X[:, 4]
    ant_mean = (a1 + a2 + a3) / 3.0
    heatmaps = np.stack(
        [
            signal,
            a1,
            a2,
            noise,
            a3,
            ant_mean,
        ],
        axis=1,
    ).reshape(-1, 2, 3)
    return heatmaps


def _build_heatmap_sequences(df: pd.DataFrame, window: int = HEATMAP_WINDOW) -> tuple[np.ndarray, pd.DataFrame]:
    """Build fixed-length sequence samples from each (room,campaign,cell) trajectory."""
    seqs: list[np.ndarray] = []
    meta_rows: list[pd.Series] = []
    group_cols = ["room", "campaign", "grid_cell"]
    for _, grp in df.groupby(group_cols, sort=False):
        grp = grp.reset_index(drop=True)
        raw = grp[FEATURE_COLUMNS].to_numpy(dtype=float)
        if len(raw) < window:
            continue
        hm = _rssi_to_heatmaps(raw)
        for i in range(len(raw) - window + 1):
            seqs.append(hm[i : i + window])
            # Use the final timestamp as supervision row for localization metrics.
            meta_rows.append(grp.iloc[i + window - 1])
    if not seqs:
        raise RuntimeError(f"No sequence built with window={window}.")
    return np.asarray(seqs, dtype=float), pd.DataFrame(meta_rows).reset_index(drop=True)


def benchmark_heatmap_fingerprinting(
    df: pd.DataFrame,
    cell_lookup: pd.DataFrame,
    *,
    save_prefix: Path,
    strict_anti_leakage: bool = False,
    quant_step_db: float = ANTI_LEAKAGE_QUANT_STEP_DB,
) -> dict:
    """Benchmark heatmap-based RSSI encodings against raw-vector baselines."""
    if strict_anti_leakage:
        train_df, test_df, _ = _strict_group_train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("heatmap", "split"),
            quant_step_db=quant_step_db,
            context="heatmap_main",
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=_seed_value("heatmap", "split"),
            stratify=df["grid_cell"],
        )
    labels = sorted(df["grid_cell"].unique())

    # Baseline on raw RSSI vectors.
    X_train_raw = build_features(train_df, include_room=False)
    X_test_raw = build_features(test_df, include_room=False)
    raw_scaler = StandardScaler()
    X_train_raw_z = raw_scaler.fit_transform(X_train_raw)
    X_test_raw_z = raw_scaler.transform(X_test_raw)

    raw_knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    raw_knn.fit(X_train_raw_z, train_df["grid_cell"])
    raw_knn_pred = raw_knn.predict(X_test_raw_z)
    raw_knn_summary = localization_summary(test_df, raw_knn_pred, cell_lookup)
    save_confusion(
        test_df["grid_cell"],
        raw_knn_pred,
        labels,
        save_prefix.with_name("confusion_heatmap_raw_knn.csv"),
    )

    raw_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=80,
        alpha=5e-4,
        learning_rate_init=5e-4,
        random_state=_seed_value("heatmap", "raw_mlp"),
    )
    raw_mlp.fit(X_train_raw_z, train_df["grid_cell"])
    raw_pred = raw_mlp.predict(X_test_raw_z)
    raw_summary = localization_summary(test_df, raw_pred, cell_lookup)
    save_confusion(
        test_df["grid_cell"],
        raw_pred,
        labels,
        save_prefix.with_name("confusion_heatmap_raw_mlp.csv"),
    )

    # Single-frame heatmap representation.
    X_train_hm = _rssi_to_heatmaps(X_train_raw).reshape(len(train_df), -1)
    X_test_hm = _rssi_to_heatmaps(X_test_raw).reshape(len(test_df), -1)
    hm_scaler = StandardScaler()
    X_train_hm_z = hm_scaler.fit_transform(X_train_hm)
    X_test_hm_z = hm_scaler.transform(X_test_hm)

    hm_knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    hm_knn.fit(X_train_hm_z, train_df["grid_cell"])
    hm_knn_pred = hm_knn.predict(X_test_hm_z)
    hm_knn_summary = localization_summary(test_df, hm_knn_pred, cell_lookup)
    save_confusion(
        test_df["grid_cell"],
        hm_knn_pred,
        labels,
        save_prefix.with_name("confusion_heatmap_knn.csv"),
    )

    hm_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=80,
        alpha=5e-4,
        learning_rate_init=5e-4,
        random_state=_seed_value("heatmap", "hm_mlp"),
    )
    hm_mlp.fit(X_train_hm_z, train_df["grid_cell"])
    hm_mlp_pred = hm_mlp.predict(X_test_hm_z)
    hm_mlp_summary = localization_summary(test_df, hm_mlp_pred, cell_lookup)
    save_confusion(
        test_df["grid_cell"],
        hm_mlp_pred,
        labels,
        save_prefix.with_name("confusion_heatmap_mlp.csv"),
    )

    # Sequence heatmaps (proxy for CNN-LSTM without requiring deep-learning runtimes).
    seq_X, seq_meta = _build_heatmap_sequences(df, window=HEATMAP_WINDOW)
    if strict_anti_leakage:
        seq_train_meta, seq_test_meta, _ = _strict_group_train_test_split(
            seq_meta,
            test_size=0.2,
            random_state=_seed_value("heatmap", "seq_split"),
            quant_step_db=quant_step_db,
            context="heatmap_sequence",
        )
        seq_train_idx = seq_train_meta.index.to_numpy()
        seq_test_idx = seq_test_meta.index.to_numpy()
    else:
        seq_y = seq_meta["grid_cell"]
        seq_train_idx, seq_test_idx = train_test_split(
            np.arange(len(seq_meta)),
            test_size=0.2,
            random_state=_seed_value("heatmap", "seq_split"),
            stratify=seq_y,
        )
        seq_test_meta = seq_meta.iloc[seq_test_idx].reset_index(drop=True)
    seq_train_X = seq_X[seq_train_idx].reshape(len(seq_train_idx), -1)
    seq_test_X = seq_X[seq_test_idx].reshape(len(seq_test_idx), -1)
    seq_scaler = StandardScaler()
    seq_train_X_z = seq_scaler.fit_transform(seq_train_X)
    seq_test_X_z = seq_scaler.transform(seq_test_X)

    seq_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=100,
        alpha=1e-3,
        learning_rate_init=5e-4,
        random_state=_seed_value("heatmap", "seq_mlp"),
    )
    seq_train_y = seq_meta.iloc[seq_train_idx]["grid_cell"]
    if strict_anti_leakage:
        seq_test_meta = seq_test_meta.reset_index(drop=True)
    seq_mlp.fit(seq_train_X_z, seq_train_y)
    seq_pred = seq_mlp.predict(seq_test_X_z)
    seq_summary = localization_summary(seq_test_meta, seq_pred, cell_lookup)
    save_confusion(
        seq_test_meta["grid_cell"],
        seq_pred,
        labels,
        save_prefix.with_name("confusion_heatmap_sequence_mlp.csv"),
    )

    return {
        "raw_knn_cell_acc": raw_knn_summary.cell_accuracy,
        "raw_knn_mean_error_m": raw_knn_summary.mean_error_m,
        "raw_knn_p90_error_m": raw_knn_summary.p90_error_m,
        "raw_mlp_cell_acc": raw_summary.cell_accuracy,
        "raw_mlp_mean_error_m": raw_summary.mean_error_m,
        "raw_mlp_p90_error_m": raw_summary.p90_error_m,
        "heatmap_knn_cell_acc": hm_knn_summary.cell_accuracy,
        "heatmap_knn_mean_error_m": hm_knn_summary.mean_error_m,
        "heatmap_knn_p90_error_m": hm_knn_summary.p90_error_m,
        "heatmap_mlp_cell_acc": hm_mlp_summary.cell_accuracy,
        "heatmap_mlp_mean_error_m": hm_mlp_summary.mean_error_m,
        "heatmap_mlp_p90_error_m": hm_mlp_summary.p90_error_m,
        "heatmap_seq_mlp_cell_acc": seq_summary.cell_accuracy,
        "heatmap_seq_mlp_mean_error_m": seq_summary.mean_error_m,
        "heatmap_seq_mlp_p90_error_m": seq_summary.p90_error_m,
        "sequence_window": HEATMAP_WINDOW,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark localization models across rooms.")
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Base random seed for splits/models (single-seed mode).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds to run from --seed (multi-seed mode).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help="Explicit seed list/range (e.g. '1,7,21' or '1-10'). Overrides --num-seeds.",
    )
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
    parser.add_argument(
        "--no-stacking",
        action="store_true",
        help="Skip the Stacking classifier (very slow on large datasets).",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip heatmap-based fingerprinting benchmarks.",
    )
    parser.add_argument(
        "--strict-anti-leakage",
        action="store_true",
        help="Use a strict split by (room,campaign,grid_cell) and fail on train/test quasi-duplicate overlaps.",
    )
    parser.add_argument(
        "--quant-step-db",
        type=float,
        default=ANTI_LEAKAGE_QUANT_STEP_DB,
        help="Quantization step (dB) used by strict anti-leakage signature checks.",
    )
    parser.add_argument(
        "--significance-top-k",
        type=int,
        default=3,
        help="Number of top room-aware models used for pairwise significance tests (McNemar + Wilcoxon/paired t-test).",
    )
    parser.add_argument(
        "--skip-significance",
        action="store_true",
        help="Skip significance tests on top room-aware models.",
    )
    parser.add_argument(
        "--skip-coverage-risk",
        action="store_true",
        help="Skip calibrated reject-option benchmark (coverage-risk curve).",
    )
    parser.add_argument(
        "--coverage-risk-targets",
        type=str,
        default="0.05,0.10,0.15,0.20",
        help="Target risk levels used to calibrate rejection thresholds (comma/space-separated).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    global GLOBAL_BASE_SEED
    args = parse_args(argv)
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1.")
    if args.seeds:
        seed_values = _parse_seed_spec(args.seeds)
    else:
        seed_values = [int(args.seed) + i for i in range(int(args.num_seeds))]
    seed_values = sorted(dict.fromkeys(seed_values))

    if len(seed_values) > 1:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = REPORT_DIR / "benchmark_summary.csv"
        per_seed_path = REPORT_DIR / "benchmark_summary_per_seed.csv"
        stats_path = REPORT_DIR / "benchmark_summary_stats.csv"
        script_path = Path(__file__).resolve()
        per_seed_frames: list[pd.DataFrame] = []

        for idx, seed in enumerate(seed_values, start=1):
            cmd = [
                sys.executable,
                str(script_path),
                "--seed",
                str(seed),
                "--num-seeds",
                "1",
                "--holdout",
                args.holdout,
                "--significance-top-k",
                str(args.significance_top_k),
            ]
            if args.skip_significance:
                cmd += ["--skip-significance"]
            for room in args.room or []:
                cmd += ["--room", room]
            if args.distances:
                cmd += ["--distances", *[f"{float(d):g}" for d in args.distances]]
            if args.skip_optional:
                cmd += ["--skip-optional"]
            if args.run_gpc:
                cmd += ["--run-gpc"]
            if args.gpc_max_samples is not None:
                cmd += ["--gpc-max-samples", str(args.gpc_max_samples)]
            if args.all_models:
                cmd += ["--all-models"]
            if args.no_stacking:
                cmd += ["--no-stacking"]
            if args.skip_heatmap:
                cmd += ["--skip-heatmap"]
            if args.strict_anti_leakage:
                cmd += ["--strict-anti-leakage"]
            if args.skip_coverage_risk:
                cmd += ["--skip-coverage-risk"]
            if args.coverage_risk_targets:
                cmd += ["--coverage-risk-targets", args.coverage_risk_targets]
            if float(args.quant_step_db) != float(ANTI_LEAKAGE_QUANT_STEP_DB):
                cmd += ["--quant-step-db", f"{float(args.quant_step_db):g}"]

            print(f"\n=== Multi-seed run {idx}/{len(seed_values)} (seed={seed}) ===")
            subprocess.run(cmd, check=True)
            if not summary_path.exists():
                raise FileNotFoundError(f"Expected summary file not found after seed {seed}: {summary_path}")
            seed_df = pd.read_csv(summary_path, index_col=0).reset_index().rename(columns={"index": "algorithm"})
            seed_df["seed"] = seed
            per_seed_frames.append(seed_df)

        per_seed_df = pd.concat(per_seed_frames, ignore_index=True)
        cols = [
            "seed",
            "algorithm",
            "cell_acc",
            "mean_error_m",
            "p90_error_m",
            "router_dist_acc",
            "room_acc",
            "top1_acc",
            "top3_acc",
            "top3_best_error_m",
            "top3_best_gain_m",
        ]
        per_seed_df = per_seed_df[cols].sort_values(["algorithm", "seed"]).reset_index(drop=True)
        per_seed_df.to_csv(per_seed_path, index=False)

        stats_df = _aggregate_seed_summaries(per_seed_df).sort_values("algorithm").reset_index(drop=True)
        stats_df.to_csv(stats_path, index=False)

        summary_mean_df = stats_df[
            [
                "algorithm",
                "cell_acc_mean",
                "mean_error_m_mean",
                "p90_error_m_mean",
                "router_dist_acc_mean",
                "room_acc_mean",
                "top1_acc_mean",
                "top3_acc_mean",
                "top3_best_error_m_mean",
                "top3_best_gain_m_mean",
            ]
        ].rename(
            columns={
                "cell_acc_mean": "cell_acc",
                "mean_error_m_mean": "mean_error_m",
                "p90_error_m_mean": "p90_error_m",
                "router_dist_acc_mean": "router_dist_acc",
                "room_acc_mean": "room_acc",
                "top1_acc_mean": "top1_acc",
                "top3_acc_mean": "top3_acc",
                "top3_best_error_m_mean": "top3_best_error_m",
                "top3_best_gain_m_mean": "top3_best_gain_m",
            }
        ).set_index("algorithm")
        summary_mean_df.to_csv(summary_path)

        print("\n=== Multi-seed aggregation complete ===")
        print(f"Per-seed summary: {per_seed_path}")
        print(f"Stats (mean/std/CI95): {stats_path}")
        print(f"Compatibility summary (means): {summary_path}")
        return

    GLOBAL_BASE_SEED = int(seed_values[0])
    FAIL_LOG.clear()
    print(f"Using base seed: {GLOBAL_BASE_SEED}")
    if args.all_models:
        args.skip_optional = False
        args.run_gpc = True
    if args.run_gpc and args.gpc_max_samples is None:
        args.gpc_max_samples = 1400
    if args.quant_step_db <= 0:
        raise ValueError("--quant-step-db must be > 0.")
    coverage_risk_targets = _parse_float_spec(args.coverage_risk_targets)
    if any((v < 0.0 or v > 1.0) for v in coverage_risk_targets):
        raise ValueError("--coverage-risk-targets values must be in [0, 1].")
    if not coverage_risk_targets:
        coverage_risk_targets = [0.05, 0.10, 0.15, 0.20]
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
        "Strict anti-leakage: "
        f"{'enabled' if args.strict_anti_leakage else 'disabled'} "
        f"(quant_step={float(args.quant_step_db):g} dB)"
    )

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
                skip_stacking=args.no_stacking,
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
                skip_stacking=args.no_stacking,
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
        skip_stacking=args.no_stacking,
        run_gpc=args.run_gpc,
        gpc_max_samples=args.gpc_max_samples,
        strict_anti_leakage=args.strict_anti_leakage,
        quant_step_db=float(args.quant_step_db),
    )
    print(json.dumps(aware, indent=2))
    print(f"Confusions enregistrées dans {REPORT_DIR}/confusion_room_aware_*.csv")

    print("\n=== Significance tests on top room-aware models ===")
    if args.skip_significance:
        significance = None
        print("Skipped by --skip-significance.")
    else:
        try:
            significance = run_significance_tests_room_aware(
                df,
                cell_lookup,
                top_k=args.significance_top_k,
                output_dir=REPORT_DIR,
                strict_anti_leakage=args.strict_anti_leakage,
                quant_step_db=float(args.quant_step_db),
            )
            print(json.dumps(significance["summary"], indent=2))
            print(f"Top models CSV: {significance['paths']['top_models_csv']}")
            print(f"Pairwise tests CSV: {significance['paths']['pairwise_csv']}")
            print(f"JSON summary: {significance['paths']['json']}")
            if significance["failed_models"]:
                print(f"Some models were skipped in significance run: {len(significance['failed_models'])}")
        except Exception as exc:  # noqa: BLE001
            significance = None
            print(f"Significance tests skipped due to error: {type(exc).__name__}: {exc}")

    print("\n=== LogisticRegression on embedding: router distance ===")
    if can_distance:
        distance_res = benchmark_distance_logreg(
            df,
            REPORT_DIR / "confusion_distance_logreg.csv",
            strict_anti_leakage=args.strict_anti_leakage,
            quant_step_db=float(args.quant_step_db),
        )
        print(json.dumps(distance_res, indent=2))
        print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_distance_logreg.csv")
    else:
        print("Skipped: needs at least 2 distances.")
        distance_res = {"accuracy": np.nan}

    print("\n=== RandomForest on raw features: router distance ===")
    if can_distance:
        distance_rf = benchmark_distance_rf(
            df,
            REPORT_DIR / "confusion_distance_rf.csv",
            strict_anti_leakage=args.strict_anti_leakage,
            quant_step_db=float(args.quant_step_db),
        )
        print(json.dumps(distance_rf, indent=2))
        print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_distance_rf.csv")
    else:
        print("Skipped: needs at least 2 distances.")
        distance_rf = {"accuracy": np.nan}

    print("\n=== ExtraTrees on raw features: router distance ===")
    if can_distance:
        distance_et = benchmark_distance_extratrees(
            df,
            REPORT_DIR / "confusion_distance_extratrees.csv",
            strict_anti_leakage=args.strict_anti_leakage,
            quant_step_db=float(args.quant_step_db),
        )
        print(json.dumps(distance_et, indent=2))
        print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_distance_extratrees.csv")
    else:
        print("Skipped: needs at least 2 distances.")
        distance_et = {"accuracy": np.nan}

    print("\n=== Multihead embedding (cell + distance + room from RSSI only) ===")
    if can_room and can_distance:
        multihead = benchmark_multihead_embedding(
            df,
            cell_lookup,
            save_prefix=REPORT_DIR / "multihead_placeholder.csv",
            strict_anti_leakage=args.strict_anti_leakage,
            quant_step_db=float(args.quant_step_db),
        )
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
        room_res = benchmark_room_classifier(
            df,
            REPORT_DIR / "confusion_room_classifier.csv",
            strict_anti_leakage=args.strict_anti_leakage,
            quant_step_db=float(args.quant_step_db),
        )
        print(json.dumps(room_res, indent=2))
        print(f"Confusion enregistrée dans {REPORT_DIR}/confusion_room_classifier.csv")
    else:
        print("Skipped: needs at least 2 rooms.")
        room_res = {"accuracy": np.nan}

    print("\n=== Rejet calibre: courbe coverage-risk (RandomForest) ===")
    if args.skip_coverage_risk:
        print("Skipped by --skip-coverage-risk.")
        coverage_risk = None
    else:
        coverage_risk = benchmark_calibrated_rejection(
            df,
            output_csv=REPORT_DIR / "coverage_risk_curve.csv",
            output_png=REPORT_DIR / "coverage_risk_curve.png",
            output_json=REPORT_DIR / "coverage_risk_operating_points.json",
            include_room=True,
            target_risks=coverage_risk_targets,
            strict_anti_leakage=args.strict_anti_leakage,
            quant_step_db=float(args.quant_step_db),
        )
        print(json.dumps(coverage_risk, indent=2))
        print("Artefacts coverage-risk: reports/benchmarks/coverage_risk_curve.{csv,png}")
        print("Points de fonctionnement calibres: reports/benchmarks/coverage_risk_operating_points.json")

    print("\n=== Heatmap fingerprinting benchmarks ===")
    if args.skip_heatmap:
        print("Skipped by --skip-heatmap.")
        heatmap_res = {
            "raw_knn_cell_acc": np.nan,
            "raw_knn_mean_error_m": np.nan,
            "raw_knn_p90_error_m": np.nan,
            "raw_mlp_cell_acc": np.nan,
            "raw_mlp_mean_error_m": np.nan,
            "raw_mlp_p90_error_m": np.nan,
            "heatmap_knn_cell_acc": np.nan,
            "heatmap_knn_mean_error_m": np.nan,
            "heatmap_knn_p90_error_m": np.nan,
            "heatmap_mlp_cell_acc": np.nan,
            "heatmap_mlp_mean_error_m": np.nan,
            "heatmap_mlp_p90_error_m": np.nan,
            "heatmap_seq_mlp_cell_acc": np.nan,
            "heatmap_seq_mlp_mean_error_m": np.nan,
            "heatmap_seq_mlp_p90_error_m": np.nan,
            "sequence_window": HEATMAP_WINDOW,
        }
    else:
        heatmap_res = benchmark_heatmap_fingerprinting(
            df,
            cell_lookup,
            save_prefix=REPORT_DIR / "heatmap_placeholder.csv",
            strict_anti_leakage=args.strict_anti_leakage,
            quant_step_db=float(args.quant_step_db),
        )
        print(json.dumps(heatmap_res, indent=2))
        print("Confusions enregistrées dans reports/benchmarks/confusion_heatmap_*.csv")

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
            "algorithm": f"KNN distance+scaled {holdout_suffix}",
            "cell_acc": room_agnostic_mean["knn_dist_scaled_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["knn_dist_scaled_mean_error_m"],
            "p90_error_m": room_agnostic_mean["knn_dist_scaled_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"Calibrated LogReg {holdout_suffix}",
            "cell_acc": room_agnostic_mean["cal_logreg_cell_accuracy"],
            "mean_error_m": room_agnostic_mean["cal_logreg_mean_error_m"],
            "p90_error_m": room_agnostic_mean["cal_logreg_p90_error_m"],
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
            "top1_acc": aware["nn_lknn_top1_acc"],
            "top3_acc": aware["nn_lknn_top3_acc"],
            "top3_best_error_m": aware["nn_lknn_top3_best_error_m"],
            "top3_best_gain_m": aware["nn_lknn_top3_best_gain_m"],
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
            "algorithm": "KNN distance+scaled (room-aware, cell)",
            "cell_acc": aware["knn_dist_scaled_cell_accuracy"],
            "mean_error_m": aware["knn_dist_scaled_mean_error_m"],
            "p90_error_m": aware["knn_dist_scaled_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "Calibrated LogReg (room-aware, cell)",
            "cell_acc": aware["cal_logreg_cell_accuracy"],
            "mean_error_m": aware["cal_logreg_mean_error_m"],
            "p90_error_m": aware["cal_logreg_p90_error_m"],
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
        {
            "algorithm": "Raw KNN baseline (RSSI vector only)",
            "cell_acc": heatmap_res["raw_knn_cell_acc"],
            "mean_error_m": heatmap_res["raw_knn_mean_error_m"],
            "p90_error_m": heatmap_res["raw_knn_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "Raw MLP baseline (RSSI vector only)",
            "cell_acc": heatmap_res["raw_mlp_cell_acc"],
            "mean_error_m": heatmap_res["raw_mlp_mean_error_m"],
            "p90_error_m": heatmap_res["raw_mlp_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "Heatmap KNN (2x3 RSSI map)",
            "cell_acc": heatmap_res["heatmap_knn_cell_acc"],
            "mean_error_m": heatmap_res["heatmap_knn_mean_error_m"],
            "p90_error_m": heatmap_res["heatmap_knn_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": "Heatmap MLP (2x3 RSSI map)",
            "cell_acc": heatmap_res["heatmap_mlp_cell_acc"],
            "mean_error_m": heatmap_res["heatmap_mlp_mean_error_m"],
            "p90_error_m": heatmap_res["heatmap_mlp_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
        {
            "algorithm": f"Heatmap sequence MLP (window={HEATMAP_WINDOW})",
            "cell_acc": heatmap_res["heatmap_seq_mlp_cell_acc"],
            "mean_error_m": heatmap_res["heatmap_seq_mlp_mean_error_m"],
            "p90_error_m": heatmap_res["heatmap_seq_mlp_p90_error_m"],
            "router_dist_acc": np.nan,
            "room_acc": np.nan,
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    for col in ("top1_acc", "top3_acc", "top3_best_error_m", "top3_best_gain_m"):
        if col not in summary_df.columns:
            summary_df[col] = np.nan
    summary_df["top1_acc"] = summary_df["top1_acc"].fillna(summary_df["cell_acc"])
    summary_df = summary_df.set_index("algorithm")
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
