#!/usr/bin/env python3
"""Embedded-oriented profiling for localization models.

Measures model footprint (disk/flash proxy), warm-up and steady-state latency,
throughput, and energy-per-inference estimates.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import benchmark_models as bm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.embedding_knn import EmbeddingKnnLocalizer  # noqa: E402


def _rss_mb() -> float:
    # Linux returns ru_maxrss in KB, macOS in bytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss <= 0:
        return float("nan")
    # Heuristic: values above this threshold are very likely bytes.
    if rss > 10_000_000:
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _iter_batches(X: np.ndarray, batch_size: int):
    n = len(X)
    if n == 0:
        return
    for i in range(0, n, batch_size):
        yield X[i : i + batch_size]


def _collect_numpy_arrays(obj: Any, *, seen: set[int] | None = None) -> list[np.ndarray]:
    if seen is None:
        seen = set()

    arrays: list[np.ndarray] = []

    def visit(value: Any) -> None:
        obj_id = id(value)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(value, np.ndarray):
            arrays.append(value)
            return
        if isinstance(value, (str, bytes, bytearray, int, float, bool, type(None), Path)):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)
            return
        if hasattr(value, "__dict__"):
            visit(vars(value))

    visit(obj)
    return arrays


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _dense_param_stats_from_mlp(mlp: MLPClassifier) -> dict[str, float | int]:
    coefs = getattr(mlp, "coefs_", None) or []
    intercepts = getattr(mlp, "intercepts_", None) or []
    weight_count = int(sum(np.size(arr) for arr in coefs))
    bias_count = int(sum(np.size(arr) for arr in intercepts))
    total_params = weight_count + bias_count
    total_bytes = int(sum(arr.nbytes for arr in [*coefs, *intercepts]))
    hidden_layers = tuple(int(x) for x in getattr(mlp, "hidden_layer_sizes", ()) or ())
    return {
        "param_count": total_params,
        "weight_count": weight_count,
        "bias_count": bias_count,
        "param_bytes": total_bytes,
        "input_dim": int(coefs[0].shape[0]) if coefs else 0,
        "output_dim": int(coefs[-1].shape[1]) if coefs else 0,
        "hidden_layer_count": int(len(hidden_layers)),
        "hidden_units_total": int(sum(hidden_layers)) if hidden_layers else 0,
    }


def _model_structure_metrics(name: str, model) -> dict[str, object]:
    metrics: dict[str, object] = {"model_family": name}

    arrays = _collect_numpy_arrays(model)
    unique_arrays: dict[int, np.ndarray] = {id(arr): arr for arr in arrays}
    total_array_bytes = int(sum(arr.nbytes for arr in unique_arrays.values()))
    total_array_values = int(sum(arr.size for arr in unique_arrays.values()))
    metrics.update(
        {
            "numpy_array_count": int(len(unique_arrays)),
            "numpy_values_count": total_array_values,
            "numpy_array_bytes": total_array_bytes,
            "numpy_array_kb": total_array_bytes / 1024.0,
            "numpy_array_mb": total_array_bytes / (1024.0 * 1024.0),
        }
    )

    if isinstance(model, EmbeddingKnnLocalizer):
        encoder = getattr(model, "encoder_", None)
        knn = getattr(model, "knn_", None)
        scaler = getattr(model, "scaler_", None)
        distance_clf = getattr(model, "distance_clf_", None)
        train_embeddings = getattr(model, "train_embeddings_", None)
        train_scaled = getattr(model, "train_scaled_", None)
        train_labels = getattr(model, "train_labels_", None)

        encoder_stats = _dense_param_stats_from_mlp(encoder) if isinstance(encoder, MLPClassifier) else {}
        scaler_bytes = 0
        if scaler is not None:
            scaler_bytes = int(
                sum(
                    getattr(scaler, attr).nbytes
                    for attr in ("mean_", "scale_", "var_")
                    if hasattr(scaler, attr) and isinstance(getattr(scaler, attr), np.ndarray)
                )
            )
        distance_param_count = 0
        distance_param_bytes = 0
        if distance_clf is not None:
            for attr in ("coef_", "intercept_"):
                arr = getattr(distance_clf, attr, None)
                if isinstance(arr, np.ndarray):
                    distance_param_count += int(arr.size)
                    distance_param_bytes += int(arr.nbytes)

        metrics.update(
            {
                "model_family": "nn_lknn",
                "encoder_param_count": int(encoder_stats.get("param_count", 0)),
                "encoder_param_bytes": int(encoder_stats.get("param_bytes", 0)),
                "encoder_weight_count": int(encoder_stats.get("weight_count", 0)),
                "encoder_bias_count": int(encoder_stats.get("bias_count", 0)),
                "encoder_input_dim": int(encoder_stats.get("input_dim", 0)),
                "encoder_output_dim": int(encoder_stats.get("output_dim", 0)),
                "encoder_hidden_layer_count": int(encoder_stats.get("hidden_layer_count", 0)),
                "encoder_hidden_units_total": int(encoder_stats.get("hidden_units_total", 0)),
                "knn_neighbors": _safe_int(getattr(knn, "n_neighbors", None)),
                "knn_train_samples": int(len(train_labels)) if train_labels is not None else 0,
                "knn_embedding_dim": int(train_embeddings.shape[1]) if isinstance(train_embeddings, np.ndarray) and train_embeddings.ndim == 2 else 0,
                "knn_embedding_bytes": int(train_embeddings.nbytes) if isinstance(train_embeddings, np.ndarray) else 0,
                "train_scaled_bytes": int(train_scaled.nbytes) if isinstance(train_scaled, np.ndarray) else 0,
                "train_labels_bytes": int(train_labels.nbytes) if isinstance(train_labels, np.ndarray) else 0,
                "scaler_bytes": scaler_bytes,
                "distance_head_param_count": int(distance_param_count),
                "distance_head_param_bytes": int(distance_param_bytes),
                "ood_enabled": bool(
                    getattr(model, "ood_energy_threshold_", None) is not None
                    and getattr(model, "ood_distance_threshold_", None) is not None
                ),
            }
        )
        metrics["learned_param_count"] = (
            int(metrics["encoder_param_count"]) + int(metrics["distance_head_param_count"])
        )
        metrics["learned_param_bytes"] = (
            int(metrics["encoder_param_bytes"]) + int(metrics["distance_head_param_bytes"])
        )
        metrics["runtime_state_bytes"] = (
            int(metrics["knn_embedding_bytes"])
            + int(metrics["train_scaled_bytes"])
            + int(metrics["train_labels_bytes"])
            + int(metrics["scaler_bytes"])
        )
        return metrics

    if hasattr(model, "estimators_") and isinstance(model.estimators_, (list, np.ndarray)):
        estimators = list(model.estimators_)
        node_count = 0
        max_depth = 0
        for estimator in estimators:
            tree = getattr(estimator, "tree_", None)
            if tree is None:
                continue
            node_count += int(getattr(tree, "node_count", 0))
            max_depth = max(max_depth, int(getattr(tree, "max_depth", 0)))
        metrics.update(
            {
                "estimator_count": int(len(estimators)),
                "tree_total_nodes": int(node_count),
                "tree_max_depth": int(max_depth),
                "learned_param_count": total_array_values,
                "learned_param_bytes": total_array_bytes,
            }
        )
        return metrics

    if isinstance(model, KNeighborsClassifier):
        fit_x = getattr(model, "_fit_X", None)
        fit_y = getattr(model, "_y", None)
        metrics.update(
            {
                "knn_neighbors": _safe_int(getattr(model, "n_neighbors", None)),
                "knn_train_samples": int(fit_x.shape[0]) if isinstance(fit_x, np.ndarray) and fit_x.ndim == 2 else 0,
                "knn_feature_dim": int(fit_x.shape[1]) if isinstance(fit_x, np.ndarray) and fit_x.ndim == 2 else 0,
                "knn_fit_x_bytes": int(fit_x.nbytes) if isinstance(fit_x, np.ndarray) else 0,
                "knn_fit_y_bytes": int(fit_y.nbytes) if isinstance(fit_y, np.ndarray) else 0,
                "learned_param_count": 0,
                "learned_param_bytes": 0,
                "runtime_state_bytes": int(total_array_bytes),
            }
        )
        return metrics

    if hasattr(model, "named_steps"):
        scaler = getattr(model.named_steps.get("standardscaler"), "scale_", None)
        clf = model.named_steps.get("mlpclassifier")
        if isinstance(clf, MLPClassifier):
            encoder_stats = _dense_param_stats_from_mlp(clf)
            metrics.update(
                {
                    "model_family": "pipeline_mlp",
                    "encoder_param_count": int(encoder_stats.get("param_count", 0)),
                    "encoder_param_bytes": int(encoder_stats.get("param_bytes", 0)),
                    "encoder_input_dim": int(encoder_stats.get("input_dim", 0)),
                    "encoder_output_dim": int(encoder_stats.get("output_dim", 0)),
                    "encoder_hidden_layer_count": int(encoder_stats.get("hidden_layer_count", 0)),
                    "encoder_hidden_units_total": int(encoder_stats.get("hidden_units_total", 0)),
                    "scaler_feature_dim": int(scaler.shape[0]) if isinstance(scaler, np.ndarray) else 0,
                    "learned_param_count": int(encoder_stats.get("param_count", 0)),
                    "learned_param_bytes": int(encoder_stats.get("param_bytes", 0)),
                }
            )
            return metrics

        clf = model.named_steps.get("kneighborsclassifier")
        if isinstance(clf, KNeighborsClassifier):
            fit_x = getattr(clf, "_fit_X", None)
            fit_y = getattr(clf, "_y", None)
            metrics.update(
                {
                    "model_family": "pipeline_knn",
                    "knn_neighbors": _safe_int(getattr(clf, "n_neighbors", None)),
                    "knn_train_samples": int(fit_x.shape[0]) if isinstance(fit_x, np.ndarray) and fit_x.ndim == 2 else 0,
                    "knn_feature_dim": int(fit_x.shape[1]) if isinstance(fit_x, np.ndarray) and fit_x.ndim == 2 else 0,
                    "knn_fit_x_bytes": int(fit_x.nbytes) if isinstance(fit_x, np.ndarray) else 0,
                    "knn_fit_y_bytes": int(fit_y.nbytes) if isinstance(fit_y, np.ndarray) else 0,
                    "scaler_feature_dim": int(scaler.shape[0]) if isinstance(scaler, np.ndarray) else 0,
                    "learned_param_count": 0,
                    "learned_param_bytes": 0,
                    "runtime_state_bytes": int(total_array_bytes),
                }
            )
            return metrics

    metrics.update(
        {
            "learned_param_count": total_array_values,
            "learned_param_bytes": total_array_bytes,
        }
    )
    return metrics


def _benchmark_predict(predict_fn, X_eval: np.ndarray, *, warmup_runs: int, repeats: int, batch_size: int) -> dict:
    if len(X_eval) == 0:
        raise ValueError("Empty evaluation matrix.")

    # Warm-up to stabilize caches/allocators.
    warmup_times_ms: list[float] = []
    warmup_batch = X_eval[: max(1, min(batch_size, len(X_eval)))]
    for _ in range(max(0, warmup_runs)):
        t0 = time.perf_counter_ns()
        _ = predict_fn(warmup_batch)
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        warmup_times_ms.append(float(dt_ms))

    # Timed runs.
    per_inference_ms: list[float] = []
    total_samples = 0
    total_time_s = 0.0
    for _ in range(max(1, repeats)):
        for batch in _iter_batches(X_eval, batch_size=batch_size):
            t0 = time.perf_counter_ns()
            _ = predict_fn(batch)
            dt_s = (time.perf_counter_ns() - t0) / 1e9
            total_time_s += dt_s
            n_batch = len(batch)
            total_samples += n_batch
            one_ms = (dt_s * 1e3) / max(1, n_batch)
            per_inference_ms.extend([one_ms] * n_batch)

    mean_ms = float(np.mean(per_inference_ms)) if per_inference_ms else float("nan")
    throughput = float(total_samples / total_time_s) if total_time_s > 0 else float("nan")

    return {
        "warmup_mean_ms": float(np.mean(warmup_times_ms)) if warmup_times_ms else float("nan"),
        "warmup_p95_ms": _percentile(warmup_times_ms, 95),
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": _percentile(per_inference_ms, 50),
        "latency_p90_ms": _percentile(per_inference_ms, 90),
        "latency_p95_ms": _percentile(per_inference_ms, 95),
        "latency_p99_ms": _percentile(per_inference_ms, 99),
        "throughput_inf_per_s": throughput,
        "n_profiled_inferences": int(total_samples),
    }


def _fit_model(name: str, train_df: pd.DataFrame, *, seed: int):
    X_train = bm.build_features(train_df, include_room=False)
    y_train = train_df["grid_cell"].to_numpy()

    if name == "nn_lknn":
        model = bm.fit_localizer(train_df, include_room=False, random_state=seed)
        return model, model.predict

    if name == "knn":
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))
        model.fit(X_train, y_train)
        return model, model.predict

    if name == "rf":
        model = RandomForestClassifier(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, model.predict

    if name == "et":
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, model.predict

    if name == "mlp":
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=400,
                alpha=5e-4,
                learning_rate_init=5e-4,
                random_state=seed,
            ),
        )
        model.fit(X_train, y_train)
        return model, model.predict

    raise ValueError(f"Unknown model key: {name}")


def _model_size_and_cold_start(model, sample_batch: np.ndarray) -> dict:
    with tempfile.TemporaryDirectory(prefix="embedded_profile_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "model.joblib"
        joblib.dump(model, tmp_path)
        size_bytes = int(tmp_path.stat().st_size)

        gc.collect()
        rss_before = _rss_mb()
        t0 = time.perf_counter_ns()
        loaded = joblib.load(tmp_path)
        t1 = time.perf_counter_ns()
        _ = loaded.predict(sample_batch)
        t2 = time.perf_counter_ns()
        rss_after_warm = _rss_mb()

        load_ms = (t1 - t0) / 1e6
        first_inf_ms = (t2 - t1) / 1e6
        cold_start_total_ms = (t2 - t0) / 1e6

        del loaded
        gc.collect()

    return {
        "model_size_bytes": size_bytes,
        "model_size_kb": size_bytes / 1024.0,
        "model_size_mb": size_bytes / (1024.0 * 1024.0),
        "storage_size_bytes": size_bytes,
        "storage_size_kb": size_bytes / 1024.0,
        "storage_size_mb": size_bytes / (1024.0 * 1024.0),
        "flash_proxy_kb": size_bytes / 1024.0,
        "load_time_ms": float(load_ms),
        "first_inference_ms": float(first_inf_ms),
        "cold_start_total_ms": float(cold_start_total_ms),
        "rss_after_warmup_mb": float(rss_after_warm),
        "rss_baseline_before_load_mb": float(rss_before),
        "rss_delta_load_warmup_mb": float(rss_after_warm - rss_before)
        if math.isfinite(rss_before) and math.isfinite(rss_after_warm)
        else float("nan"),
    }


def _estimate_energy(
    latency_mean_ms: float,
    *,
    host_power_w: float,
    esp32_voltage_v: float,
    esp32_current_ma: float | None,
    esp32_inference_ms: float | None,
    esp32_latency_scale: float,
) -> dict:
    mean_s = latency_mean_ms / 1e3 if math.isfinite(latency_mean_ms) else float("nan")
    host_j = host_power_w * mean_s if math.isfinite(mean_s) else float("nan")

    out = {
        "host_power_w_assumed": float(host_power_w),
        "host_energy_j_per_inference": float(host_j),
        "host_energy_mj_per_inference": float(host_j * 1e3) if math.isfinite(host_j) else float("nan"),
        "host_energy_wh_per_inference": float(host_j / 3600.0) if math.isfinite(host_j) else float("nan"),
    }

    if esp32_current_ma is None:
        out.update(
            {
                "esp32_profile_mode": "not_provided",
                "esp32_power_w_assumed": float("nan"),
                "esp32_inference_ms_assumed": float("nan"),
                "esp32_energy_j_per_inference": float("nan"),
                "esp32_energy_uj_per_inference": float("nan"),
            }
        )
        return out

    esp32_power_w = esp32_voltage_v * (esp32_current_ma / 1000.0)
    if esp32_inference_ms is not None:
        esp32_inf_ms = float(esp32_inference_ms)
        mode = "explicit_ms"
    else:
        esp32_inf_ms = float(latency_mean_ms) * float(esp32_latency_scale)
        mode = "scaled_from_host"

    esp32_j = esp32_power_w * (esp32_inf_ms / 1e3)
    out.update(
        {
            "esp32_profile_mode": mode,
            "esp32_voltage_v_assumed": float(esp32_voltage_v),
            "esp32_current_ma_assumed": float(esp32_current_ma),
            "esp32_power_w_assumed": float(esp32_power_w),
            "esp32_inference_ms_assumed": float(esp32_inf_ms),
            "esp32_energy_j_per_inference": float(esp32_j),
            "esp32_energy_uj_per_inference": float(esp32_j * 1e6),
            "esp32_energy_wh_per_inference": float(esp32_j / 3600.0),
        }
    )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedded profiling for localization models.")
    parser.add_argument("--room", action="append", help="Room filter (repeatable).")
    parser.add_argument("--distances", type=float, nargs="+", help="Distance filter in meters.")
    parser.add_argument(
        "--models",
        default="nn_lknn,knn,rf,et,mlp",
        help="Comma-separated model keys among: nn_lknn,knn,rf,et,mlp",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation split ratio.")
    parser.add_argument("--seed", type=int, default=21, help="Random seed.")
    parser.add_argument("--max-train-samples", type=int, help="Optional cap on train-set size.")
    parser.add_argument("--max-test-samples", type=int, default=2000, help="Optional cap on eval-set size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for timed predict loops.")
    parser.add_argument("--warmup-runs", type=int, default=100, help="Number of warm-up predict calls.")
    parser.add_argument("--repeats", type=int, default=30, help="Number of full timed passes over eval batches.")
    parser.add_argument(
        "--host-power-w",
        type=float,
        default=6.0,
        help="Assumed host power in watts to estimate energy/inference.",
    )
    parser.add_argument(
        "--esp32-voltage-v",
        type=float,
        default=3.3,
        help="ESP32 supply voltage for energy estimate.",
    )
    parser.add_argument(
        "--esp32-current-ma",
        type=float,
        help="ESP32 average active current in mA (enables ESP32 energy estimate).",
    )
    parser.add_argument(
        "--esp32-inference-ms",
        type=float,
        help="Measured ESP32 inference time in ms (if omitted, host latency is scaled).",
    )
    parser.add_argument(
        "--esp32-latency-scale",
        type=float,
        default=1.0,
        help="Scale factor from host latency to ESP32 latency when --esp32-inference-ms is not set.",
    )
    parser.add_argument(
        "--output-prefix",
        default="embedded_profile",
        help="Output prefix in reports/benchmarks (without extension).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    model_keys = [token.strip().lower() for token in str(args.models).split(",") if token.strip()]
    valid = {"nn_lknn", "knn", "rf", "et", "mlp"}
    invalid = [m for m in model_keys if m not in valid]
    if invalid:
        raise ValueError(f"Unknown --models entries: {invalid}. Valid: {sorted(valid)}")

    df = bm.load_cross_room(room_filter=args.room, distance_filter=args.distances)
    if not 0.0 < float(args.test_size) < 1.0:
        raise ValueError("--test-size must be in (0,1).")

    train_df, test_df = train_test_split(
        df,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=df["grid_cell"],
    )

    if args.max_train_samples and len(train_df) > int(args.max_train_samples):
        n = int(args.max_train_samples)
        stratify = train_df["grid_cell"] if n >= train_df["grid_cell"].nunique() else None
        train_df, _ = train_test_split(train_df, train_size=n, random_state=int(args.seed), stratify=stratify)

    if args.max_test_samples and len(test_df) > int(args.max_test_samples):
        n = int(args.max_test_samples)
        stratify = test_df["grid_cell"] if n >= test_df["grid_cell"].nunique() else None
        test_df, _ = train_test_split(test_df, train_size=n, random_state=int(args.seed), stratify=stratify)

    X_test = bm.build_features(test_df, include_room=False)
    y_test = test_df["grid_cell"].to_numpy()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for idx, model_key in enumerate(model_keys, start=1):
        print(f"[{idx}/{len(model_keys)}] Profiling model={model_key}")
        model, predict_fn = _fit_model(model_key, train_df, seed=int(args.seed))

        y_pred = predict_fn(X_test)
        cell_acc = float(accuracy_score(y_test, y_pred))

        size_res = _model_size_and_cold_start(model, sample_batch=X_test[: max(1, min(8, len(X_test)))])
        structure_res = _model_structure_metrics(model_key, model)
        perf_res = _benchmark_predict(
            predict_fn,
            X_test,
            warmup_runs=int(args.warmup_runs),
            repeats=int(args.repeats),
            batch_size=int(args.batch_size),
        )
        energy_res = _estimate_energy(
            perf_res["latency_mean_ms"],
            host_power_w=float(args.host_power_w),
            esp32_voltage_v=float(args.esp32_voltage_v),
            esp32_current_ma=args.esp32_current_ma,
            esp32_inference_ms=args.esp32_inference_ms,
            esp32_latency_scale=float(args.esp32_latency_scale),
        )

        row = {
            "model": model_key,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_cells": int(df["grid_cell"].nunique()),
            "cell_accuracy": cell_acc,
            **size_res,
            **structure_res,
            **perf_res,
            **energy_res,
        }
        storage_kb = _float_or_nan(row.get("storage_size_kb"))
        ram_kb = _float_or_nan(row.get("numpy_array_kb"))
        learned_param_bytes = _float_or_nan(row.get("learned_param_bytes"))
        row["storage_vs_numpy_ram_ratio"] = (
            float(storage_kb / ram_kb) if math.isfinite(storage_kb) and math.isfinite(ram_kb) and ram_kb > 0 else float("nan")
        )
        row["learned_param_kb"] = learned_param_bytes / 1024.0 if math.isfinite(learned_param_bytes) else float("nan")
        row["learned_param_mb"] = learned_param_bytes / (1024.0 * 1024.0) if math.isfinite(learned_param_bytes) else float("nan")
        row["flash_kb"] = row["storage_size_kb"]
        row["flash_mb"] = row["storage_size_mb"]
        row["ram_kb"] = row["numpy_array_kb"]
        row["ram_mb"] = row["numpy_array_mb"]
        row["weights_kb"] = row["learned_param_kb"]
        row["weights_mb"] = row["learned_param_mb"]
        row["model_loaded_rss_delta_mb"] = row["rss_delta_load_warmup_mb"]
        row["cold_start_ms"] = row["cold_start_total_ms"]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("latency_mean_ms").reset_index(drop=True)

    csv_path = REPORT_DIR / f"{args.output_prefix}.csv"
    json_path = REPORT_DIR / f"{args.output_prefix}.json"

    out_df.to_csv(csv_path, index=False)
    payload = {
        "generated_at_unix_s": int(time.time()),
        "config": {
            "room_filter": args.room,
            "distance_filter": args.distances,
            "models": model_keys,
            "test_size": float(args.test_size),
            "seed": int(args.seed),
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "batch_size": int(args.batch_size),
            "warmup_runs": int(args.warmup_runs),
            "repeats": int(args.repeats),
            "host_power_w": float(args.host_power_w),
            "esp32_voltage_v": float(args.esp32_voltage_v),
            "esp32_current_ma": args.esp32_current_ma,
            "esp32_inference_ms": args.esp32_inference_ms,
            "esp32_latency_scale": float(args.esp32_latency_scale),
            "cwd": os.getcwd(),
        },
        "results": out_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print("\n=== Embedded profile summary ===")
    cols = [
        "model",
        "cell_accuracy",
        "flash_kb",
        "ram_kb",
        "weights_kb",
        "latency_p50_ms",
        "latency_p95_ms",
        "throughput_inf_per_s",
        "host_energy_mj_per_inference",
        "esp32_energy_uj_per_inference",
    ]
    print(out_df[cols].round(6).to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
