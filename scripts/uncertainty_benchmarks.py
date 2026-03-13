"""Uncertainty benchmarks for indoor RSSI localization.

Evaluates:
- probability calibration (temperature, Platt/sigmoid, isotonic),
- top-k accuracy and top-k spatial proximity,
- reject-option behavior via confidence thresholding.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_models import FEATURE_COLUMNS, load_cross_room

OUT_DIR = ROOT / "reports" / "benchmarks"
CALIBRATION_MODES = ("none", "temperature", "sigmoid", "isotonic")


@dataclass
class FoldSpec:
    protocol: str
    fold_id: str
    include_room_onehot: bool
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def _subsample_stratified(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    strat = df["grid_cell"] if df["grid_cell"].nunique() <= max_samples else None
    if strat is not None:
        try:
            sampled, _ = train_test_split(
                df,
                train_size=max_samples,
                random_state=seed,
                stratify=strat,
            )
            return sampled.reset_index(drop=True)
        except ValueError:
            pass
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def _build_features(df: pd.DataFrame, include_room_onehot: bool) -> np.ndarray:
    base = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not include_room_onehot:
        return base
    room_cols = sorted([c for c in df.columns if c.startswith("room_")])
    if not room_cols:
        raise ValueError("Room one-hot columns are missing from dataframe.")
    room_extra = df[room_cols].to_numpy(dtype=float)
    return np.concatenate([base, room_extra], axis=1)


def _make_model(model_name: str, random_state: int):
    key = model_name.lower()
    if key == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
        )
    if key in {"histgb", "hgb", "histgradientboosting"}:
        return HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=8,
            max_iter=200,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model '{model_name}'. Expected one of: KNN, HistGB.")


def _topk_accuracy(y_true: np.ndarray, classes: np.ndarray, proba: np.ndarray, k: int) -> float:
    k_eff = min(k, proba.shape[1])
    topk_idx = np.argpartition(proba, -k_eff, axis=1)[:, -k_eff:]
    topk_labels = classes[topk_idx]
    hit = np.any(topk_labels == y_true[:, None], axis=1)
    return float(np.mean(hit))


def _multiclass_brier(y_true: np.ndarray, classes: np.ndarray, proba: np.ndarray) -> float:
    class_to_idx = {label: i for i, label in enumerate(classes)}
    y_idx = np.array([class_to_idx[label] for label in y_true], dtype=int)
    one_hot = np.zeros_like(proba, dtype=float)
    one_hot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _ece(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if not np.any(in_bin):
            continue
        bin_acc = float(np.mean(y_pred[in_bin] == y_true[in_bin]))
        bin_conf = float(np.mean(confidence[in_bin]))
        ece += (np.sum(in_bin) / len(confidence)) * abs(bin_acc - bin_conf)
    return float(ece)


def _temperature_scale(prob: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(prob, 1e-12, 1.0)
    scaled = clipped ** (1.0 / max(temperature, 1e-6))
    scaled /= np.sum(scaled, axis=1, keepdims=True)
    return scaled


def _fit_temperature(prob_cal: np.ndarray, y_cal: np.ndarray, classes: np.ndarray) -> float:
    candidates = np.geomspace(0.05, 5.0, num=40)
    best_t = 1.0
    best_nll = float("inf")
    for t in candidates:
        scaled = _temperature_scale(prob_cal, float(t))
        nll = float(log_loss(y_cal, scaled, labels=classes))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def _topk_min_distance_m(
    y_true: np.ndarray,
    classes: np.ndarray,
    proba: np.ndarray,
    true_coords: np.ndarray,
    coord_lookup: pd.DataFrame,
    k: int,
) -> float:
    k_eff = min(k, proba.shape[1])
    topk_idx = np.argpartition(proba, -k_eff, axis=1)[:, -k_eff:]
    class_to_col = {label: i for i, label in enumerate(classes)}
    class_coords = np.zeros((len(classes), 2), dtype=float)
    for label, idx in class_to_col.items():
        class_coords[idx] = coord_lookup.loc[label, ["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    topk_coords = class_coords[topk_idx]  # (n_samples, k, 2)
    deltas = topk_coords - true_coords[:, None, :]
    dists = np.linalg.norm(deltas, axis=2)
    mins = np.min(dists, axis=1)
    return float(np.mean(mins))


def _selective_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    errors_m: np.ndarray,
    confidence: np.ndarray,
    threshold_points: int,
) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, threshold_points)
    rows: list[dict] = []
    for thr in thresholds:
        accepted = confidence >= thr
        coverage = float(np.mean(accepted))
        if np.any(accepted):
            sel_acc = float(np.mean(y_pred[accepted] == y_true[accepted]))
            sel_err = float(np.mean(errors_m[accepted]))
        else:
            sel_acc = np.nan
            sel_err = np.nan
        rows.append(
            {
                "threshold": float(thr),
                "coverage": coverage,
                "reject_rate": float(1.0 - coverage),
                "selective_accuracy": sel_acc,
                "selective_mean_error_m": sel_err,
            }
        )
    return pd.DataFrame(rows)


def _metrics_from_proba(
    test_df: pd.DataFrame,
    proba: np.ndarray,
    classes: np.ndarray,
    coord_lookup: pd.DataFrame,
    *,
    ece_bins: int,
    threshold_points: int,
) -> tuple[dict, pd.DataFrame]:
    y_true = test_df["grid_cell"].to_numpy()
    pred_idx = np.argmax(proba, axis=1)
    y_pred = classes[pred_idx]
    confidence = np.max(proba, axis=1)

    true_coords = test_df[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    pred_coords = coord_lookup.loc[y_pred, ["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    errors_m = np.linalg.norm(true_coords - pred_coords, axis=1)

    metrics = {
        "n_test": int(len(test_df)),
        "top1_acc": float(accuracy_score(y_true, y_pred)),
        "top3_acc": _topk_accuracy(y_true, classes, proba, k=3),
        "top5_acc": _topk_accuracy(y_true, classes, proba, k=5),
        "mean_error_m": float(np.mean(errors_m)),
        "p90_error_m": float(np.percentile(errors_m, 90)),
        "top3_min_error_m": _topk_min_distance_m(y_true, classes, proba, true_coords, coord_lookup, k=3),
        "top5_min_error_m": _topk_min_distance_m(y_true, classes, proba, true_coords, coord_lookup, k=5),
        "nll": float(log_loss(y_true, proba, labels=classes)),
        "brier_multiclass": _multiclass_brier(y_true, classes, proba),
        "ece": _ece(y_true, y_pred, confidence, n_bins=ece_bins),
        "avg_confidence": float(np.mean(confidence)),
    }
    curve = _selective_curve(y_true, y_pred, errors_m, confidence, threshold_points=threshold_points)
    return metrics, curve


def _apply_calibration(
    mode: str,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_room_onehot: bool,
    *,
    random_state: int,
    calibration_size: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    X_train = _build_features(train_df, include_room_onehot)
    y_train = train_df["grid_cell"].to_numpy()
    X_test = _build_features(test_df, include_room_onehot)

    model = _make_model(model_name, random_state=random_state)

    if mode == "none":
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        return proba, model.classes_, {}

    if mode == "temperature":
        fit_df, cal_df = train_test_split(
            train_df,
            test_size=calibration_size,
            random_state=random_state,
            stratify=train_df["grid_cell"],
        )
        X_fit = _build_features(fit_df, include_room_onehot)
        y_fit = fit_df["grid_cell"].to_numpy()
        X_cal = _build_features(cal_df, include_room_onehot)
        y_cal = cal_df["grid_cell"].to_numpy()

        model.fit(X_fit, y_fit)
        raw_cal = model.predict_proba(X_cal)
        temp = _fit_temperature(raw_cal, y_cal, model.classes_)
        raw_test = model.predict_proba(X_test)
        scaled = _temperature_scale(raw_test, temp)
        return scaled, model.classes_, {"temperature": temp}

    if mode in {"sigmoid", "isotonic"}:
        calibrated = CalibratedClassifierCV(
            estimator=clone(model),
            method=mode,
            cv=3,
        )
        calibrated.fit(X_train, y_train)
        proba = calibrated.predict_proba(X_test)
        return proba, calibrated.classes_, {}

    raise ValueError(f"Unknown calibration mode '{mode}'.")


def _make_folds(protocol: str, seed: int) -> tuple[pd.DataFrame, list[FoldSpec]]:
    if protocol == "room_aware":
        df = load_cross_room()
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=seed,
            stratify=df["grid_cell"],
        )
        folds = [
            FoldSpec(
                protocol="room_aware",
                fold_id="split_80_20",
                include_room_onehot=True,
                train_df=train_df.reset_index(drop=True),
                test_df=test_df.reset_index(drop=True),
            )
        ]
        return df, folds

    if protocol == "loro_room":
        df = load_cross_room()
        folds: list[FoldSpec] = []
        for room in sorted(df["room"].unique()):
            train_df = df[df["room"] != room].reset_index(drop=True)
            test_df = df[df["room"] == room].reset_index(drop=True)
            folds.append(
                FoldSpec(
                    protocol="loro_room",
                    fold_id=f"heldout_{room}",
                    include_room_onehot=False,
                    train_df=train_df,
                    test_df=test_df,
                )
            )
        return df, folds

    if protocol == "loco_e102":
        df = load_cross_room(room_filter=["E102"])
        folds = []
        for campaign in sorted(df["campaign"].unique()):
            train_df = df[df["campaign"] != campaign].reset_index(drop=True)
            test_df = df[df["campaign"] == campaign].reset_index(drop=True)
            folds.append(
                FoldSpec(
                    protocol="loco_e102",
                    fold_id=f"heldout_{campaign.split('/')[-1]}",
                    include_room_onehot=False,
                    train_df=train_df,
                    test_df=test_df,
                )
            )
        return df, folds

    raise ValueError(f"Unsupported protocol '{protocol}'.")


def run_uncertainty_benchmarks(
    protocols: list[str],
    models: list[str],
    calibrations: list[str],
    *,
    seed: int,
    calibration_size: float,
    ece_bins: int,
    threshold_points: int,
    target_selective_acc: float,
    output_prefix: str,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    curve_rows: list[dict] = []

    for protocol in protocols:
        full_df, folds = _make_folds(protocol, seed=seed)
        coord_lookup = (
            full_df[["grid_cell", "coord_x_m", "coord_y_m"]]
            .drop_duplicates("grid_cell")
            .set_index("grid_cell")
        )

        for fold in folds:
            fold_train = _subsample_stratified(fold.train_df, max_train_samples, seed)
            fold_test = _subsample_stratified(fold.test_df, max_test_samples, seed)
            known_labels = set(fold_train["grid_cell"].unique())
            fold_test = fold_test[fold_test["grid_cell"].isin(known_labels)].reset_index(drop=True)
            if len(fold_test) == 0:
                print(f"[{protocol}] {fold.fold_id} skipped: no test sample with seen labels", flush=True)
                continue
            print(f"[{protocol}] {fold.fold_id} | n_train={len(fold.train_df)} n_test={len(fold.test_df)}")
            for model_name in models:
                for calib in calibrations:
                    print(
                        f"  -> model={model_name} calib={calib} "
                        f"(train={len(fold_train)} test={len(fold_test)})",
                        flush=True,
                    )
                    row = {
                        "protocol": protocol,
                        "fold_id": fold.fold_id,
                        "model": model_name,
                        "calibration": calib,
                        "status": "ok",
                        "reason": "",
                    }
                    try:
                        proba, classes, extra = _apply_calibration(
                            calib,
                            model_name=model_name,
                            train_df=fold_train,
                            test_df=fold_test,
                            include_room_onehot=fold.include_room_onehot,
                            random_state=seed,
                            calibration_size=calibration_size,
                        )
                        metrics, curve = _metrics_from_proba(
                            fold_test,
                            proba,
                            classes,
                            coord_lookup,
                            ece_bins=ece_bins,
                            threshold_points=threshold_points,
                        )
                        row.update(metrics)
                        row.update(extra)
                        curve = curve.copy()
                        curve.insert(0, "calibration", calib)
                        curve.insert(0, "model", model_name)
                        curve.insert(0, "fold_id", fold.fold_id)
                        curve.insert(0, "protocol", protocol)
                        valid = curve.dropna(subset=["selective_accuracy"])
                        kept = valid[valid["selective_accuracy"] >= target_selective_acc]
                        row["coverage_at_target_acc"] = float(kept["coverage"].max()) if not kept.empty else 0.0
                        curve_rows.extend(curve.to_dict(orient="records"))
                        print(
                            f"     done: top1={row['top1_acc']:.4f} ece={row['ece']:.4f} nll={row['nll']:.4f}",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        row["status"] = "failed"
                        row["reason"] = f"{type(exc).__name__}: {exc}"
                        for key in (
                            "n_test",
                            "top1_acc",
                            "top3_acc",
                            "top5_acc",
                            "mean_error_m",
                            "p90_error_m",
                            "top3_min_error_m",
                            "top5_min_error_m",
                            "nll",
                            "brier_multiclass",
                            "ece",
                            "avg_confidence",
                            "coverage_at_target_acc",
                        ):
                            row[key] = np.nan
                    metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    curves_df = pd.DataFrame(curve_rows)

    metrics_path = OUT_DIR / f"{output_prefix}_metrics.csv"
    curves_path = OUT_DIR / f"{output_prefix}_selective_curves.csv"
    summary_path = OUT_DIR / f"{output_prefix}_summary.json"

    metrics_df.to_csv(metrics_path, index=False)
    curves_df.to_csv(curves_path, index=False)

    ok_metrics = metrics_df[metrics_df["status"] == "ok"].copy()
    grouped = (
        ok_metrics.groupby(["protocol", "model", "calibration"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    summary = {
        "date": "2026-03-13",
        "seed": seed,
        "protocols": protocols,
        "models": models,
        "calibration_modes": list(CALIBRATION_MODES),
        "calibrations_run": calibrations,
        "rows_metrics": int(len(metrics_df)),
        "rows_curves": int(len(curves_df)),
        "failed_runs": int(np.sum(metrics_df["status"] != "ok")),
        "aggregate_mean": grouped.to_dict(orient="records"),
        "target_selective_acc": target_selective_acc,
        "files": {
            "metrics_csv": str(metrics_path.relative_to(ROOT)),
            "selective_curves_csv": str(curves_path.relative_to(ROOT)),
            "summary_json": str(summary_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {metrics_path}")
    print(f"Saved: {curves_path}")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uncertainty + calibration benchmarks for RSSI localization.")
    parser.add_argument(
        "--protocol",
        action="append",
        choices=["room_aware", "loro_room", "loco_e102", "all"],
        default=None,
        help="Evaluation protocol(s). Repeatable. Default: all.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=["KNN", "HistGB"],
        default=None,
        help="Model(s) to evaluate. Repeatable.",
    )
    parser.add_argument(
        "--calibration",
        action="append",
        choices=list(CALIBRATION_MODES) + ["all"],
        default=None,
        help="Calibration mode(s) to run. Repeatable. Default: all.",
    )
    parser.add_argument("--seed", type=int, default=21, help="Random seed.")
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=0.2,
        help="Calibration split ratio used for temperature scaling.",
    )
    parser.add_argument("--ece-bins", type=int, default=15, help="Number of bins for ECE.")
    parser.add_argument(
        "--threshold-points",
        type=int,
        default=21,
        help="Number of confidence thresholds in selective curve.",
    )
    parser.add_argument(
        "--target-selective-acc",
        type=float,
        default=0.95,
        help="Target selective accuracy to report max coverage.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="uncertainty_eval",
        help="Prefix for generated files in reports/benchmarks.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for train samples per fold (for faster exploratory runs).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap for test samples per fold (for faster exploratory runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_args = args.protocol or ["all"]
    protocols = [p for p in protocol_args if p != "all"]
    if not protocols:
        protocols = ["room_aware", "loro_room", "loco_e102"]
    models = args.model or ["KNN", "HistGB"]
    calibration_args = args.calibration or ["all"]
    calibrations = [c for c in calibration_args if c != "all"]
    if not calibrations:
        calibrations = list(CALIBRATION_MODES)
    run_uncertainty_benchmarks(
        protocols=protocols,
        models=models,
        calibrations=calibrations,
        seed=args.seed,
        calibration_size=args.calibration_size,
        ece_bins=args.ece_bins,
        threshold_points=args.threshold_points,
        target_selective_acc=args.target_selective_acc,
        output_prefix=args.output_prefix,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )


if __name__ == "__main__":
    main()
