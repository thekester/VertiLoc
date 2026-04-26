#!/usr/bin/env python3
"""XGBoost polar-regression experiment for 360-degree RSSI captures.

The script follows the proposed three-head strategy:
1. predict radial distance to a board-grid origin;
2. predict sin(theta);
3. predict cos(theta).

It evaluates XGBoost against a RandomForest baseline and two lightweight fusion
rules. Input CSV files are sliced into short windows so each 5 s capture yields
multiple training examples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.catalog import BENCHMARK_REPORT_DIR, FEATURE_COLUMNS, ORIENTATION_CAMPAIGNS  # noqa: E402
from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.orientation import orientation_label_to_degrees  # noqa: E402

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - optional runtime dependency
    XGBClassifier = None
    XGBRegressor = None


@dataclass(frozen=True)
class PolarOrigin:
    x_m: float
    y_m: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="E101", help="Comma-separated datasets: E101,E102.")
    parser.add_argument(
        "--include-e102-elevation",
        action="store_true",
        help="Include E102 exp5/exp6 front elevation campaigns.",
    )
    parser.add_argument("--window-size", type=int, default=5, help="Rows per short capture window.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Random test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--grid-search", action="store_true", help="Run a small GridSearchCV for XGBoost heads.")
    parser.add_argument("--n-estimators", type=int, default=180, help="Default trees per XGBoost head.")
    parser.add_argument("--max-depth", type=int, default=3, help="Default max_depth per XGBoost head.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Default learning_rate.")
    parser.add_argument("--rf-estimators", type=int, default=240, help="Trees per RandomForest head.")
    parser.add_argument("--fusion-weight-xgb", type=float, default=0.7, help="Weighted fusion share for XGBoost.")
    parser.add_argument("--agreement-threshold-m", type=float, default=0.10, help="Agreement threshold for arbitration.")
    parser.add_argument(
        "--output-prefix",
        default="xgb_polar_360",
        help="Output file prefix under reports/benchmarks.",
    )
    return parser.parse_args(argv)


def _selected_campaigns(datasets: Iterable[str], *, include_e102_elevation: bool) -> list[tuple[str, CampaignSpec, str, str]]:
    selected: list[tuple[str, CampaignSpec, str, str]] = []
    for dataset in datasets:
        key = dataset.strip().upper()
        if not key:
            continue
        if key not in ORIENTATION_CAMPAIGNS:
            raise ValueError(f"Unknown dataset {dataset!r}. Expected one of {sorted(ORIENTATION_CAMPAIGNS)}.")
        for campaign_name, spec, orientation_label in ORIENTATION_CAMPAIGNS[key]:
            if key == "E102" and not include_e102_elevation and campaign_name.endswith(("exp5_ground", "exp6_1m50")):
                continue
            selected.append((key, spec, campaign_name, orientation_label))
    if not selected:
        raise RuntimeError("No orientation campaigns selected.")
    return selected


def load_360_dataframe(datasets: Iterable[str], *, include_e102_elevation: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for room, spec, campaign_name, orientation_label in _selected_campaigns(
        datasets,
        include_e102_elevation=include_e102_elevation,
    ):
        df = load_measurements([spec]).copy()
        df["room"] = room
        df["campaign"] = campaign_name
        df["orientation_label"] = orientation_label
        df["orientation_deg"] = orientation_label_to_degrees(orientation_label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_windowed_dataset(df: pd.DataFrame, *, window_size: int) -> pd.DataFrame:
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")

    work = df.copy()
    group_cols = ["room", "campaign", "orientation_label", "grid_cell"]
    work["sample_index"] = work.groupby(group_cols).cumcount()
    work["window_id"] = work["sample_index"] // int(window_size)

    agg_spec = {col: ["mean", "std", "median", "min", "max"] for col in FEATURE_COLUMNS}
    meta_cols = [
        "room",
        "campaign",
        "orientation_label",
        "orientation_deg",
        "grid_cell",
        "grid_x",
        "grid_y",
        "coord_x_m",
        "coord_y_m",
        "router_distance_m",
        "window_id",
    ]
    grouped = work.groupby(meta_cols, dropna=False, sort=False)
    features = grouped.agg(agg_spec)
    features.columns = [f"{col}_{stat}" for col, stat in features.columns]
    out = features.reset_index()
    out = out.fillna(0.0)
    return enrich_features(out)


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    antenna_cols = ["signal_A1_median", "signal_A2_median", "signal_A3_median"]
    antenna_values = out[antenna_cols].to_numpy(dtype=float)

    out["ant_median_mean"] = antenna_values.mean(axis=1)
    out["ant_median_std"] = antenna_values.std(axis=1)
    out["ant_median_min"] = antenna_values.min(axis=1)
    out["ant_median_max"] = antenna_values.max(axis=1)
    out["ant_median_range"] = out["ant_median_max"] - out["ant_median_min"]
    out["signal_minus_ant_mean"] = out["Signal_median"] - out["ant_median_mean"]
    out["noise_minus_signal"] = out["Noise_median"] - out["Signal_median"]

    pairs = [
        ("a1_minus_a2", "signal_A1_median", "signal_A2_median"),
        ("a1_minus_a3", "signal_A1_median", "signal_A3_median"),
        ("a2_minus_a3", "signal_A2_median", "signal_A3_median"),
        ("signal_minus_a1", "Signal_median", "signal_A1_median"),
        ("signal_minus_a2", "Signal_median", "signal_A2_median"),
        ("signal_minus_a3", "Signal_median", "signal_A3_median"),
    ]
    for name, left, right in pairs:
        delta = out[left] - out[right]
        out[name] = delta
        out[f"{name}_abs"] = np.abs(delta)
        out[f"{name}_sq"] = np.square(delta)

    eps = 1e-12
    a1_mw = np.power(10.0, out["signal_A1_median"].to_numpy(dtype=float) / 10.0)
    a2_mw = np.power(10.0, out["signal_A2_median"].to_numpy(dtype=float) / 10.0)
    a3_mw = np.power(10.0, out["signal_A3_median"].to_numpy(dtype=float) / 10.0)
    out["a1_over_a2_mw"] = a1_mw / (a2_mw + eps)
    out["a1_over_a3_mw"] = a1_mw / (a3_mw + eps)
    out["a2_over_a3_mw"] = a2_mw / (a3_mw + eps)

    for col in FEATURE_COLUMNS:
        median_col = f"{col}_median"
        out[f"{col}_median_log_abs"] = np.log1p(np.abs(out[median_col].to_numpy(dtype=float)))
        out[f"{col}_median_mw"] = np.power(10.0, out[median_col].to_numpy(dtype=float) / 10.0)

    out["orientation_sin_input"] = np.sin(np.deg2rad(out["orientation_deg"].to_numpy(dtype=float)))
    out["orientation_cos_input"] = np.cos(np.deg2rad(out["orientation_deg"].to_numpy(dtype=float)))
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "room",
        "campaign",
        "orientation_label",
        "orientation_deg",
        "grid_cell",
        "grid_x",
        "grid_y",
        "coord_x_m",
        "coord_y_m",
        "router_distance_m",
        "window_id",
        "target_radius_m",
        "target_sin",
        "target_cos",
        "target_angle_deg",
        "orientation_sin_input",
        "orientation_cos_input",
    }
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def add_polar_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, PolarOrigin]:
    out = df.copy()
    origin = PolarOrigin(
        x_m=float((out["coord_x_m"].min() + out["coord_x_m"].max()) / 2.0),
        y_m=float((out["coord_y_m"].min() + out["coord_y_m"].max()) / 2.0),
    )
    dx = out["coord_x_m"].to_numpy(dtype=float) - origin.x_m
    dy = out["coord_y_m"].to_numpy(dtype=float) - origin.y_m
    radius = np.sqrt(np.square(dx) + np.square(dy))
    angle = np.rad2deg(np.arctan2(dy, dx)) % 360.0
    out["target_radius_m"] = radius
    out["target_sin"] = np.sin(np.deg2rad(angle))
    out["target_cos"] = np.cos(np.deg2rad(angle))
    out["target_angle_deg"] = angle
    return out, origin


def _xgb_base(seed: int, *, n_estimators: int, max_depth: int, learning_rate: float):
    if XGBRegressor is None:
        raise RuntimeError(
            "xgboost is not installed in this Python environment. "
            "Use the project venv or install optional dependency `xgboost`."
        )
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )


def fit_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    kind: str,
    seed: int,
    grid_search: bool,
    args: argparse.Namespace,
):
    if kind == "xgb":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    _xgb_base(
                        seed,
                        n_estimators=int(args.n_estimators),
                        max_depth=int(args.max_depth),
                        learning_rate=float(args.learning_rate),
                    ),
                ),
            ]
        )
        if grid_search:
            search = GridSearchCV(
                estimator,
                param_grid={
                    "model__n_estimators": [120, 220],
                    "model__max_depth": [2, 3],
                    "model__learning_rate": [0.04, 0.08],
                },
                scoring="neg_mean_absolute_error",
                cv=3,
                n_jobs=1,
            )
            search.fit(X_train, y_train)
            return search.best_estimator_, {"best_params": search.best_params_}
        estimator.fit(X_train, y_train)
        return estimator, {}

    if kind == "rf":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(args.rf_estimators),
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        estimator.fit(X_train, y_train)
        return estimator, {}
    raise ValueError(f"Unknown model kind: {kind}")


def fit_three_heads(
    X_train: np.ndarray,
    targets: pd.DataFrame,
    *,
    kind: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    heads: dict[str, object] = {}
    metadata: dict[str, object] = {}
    for offset, target in enumerate(["target_radius_m", "target_sin", "target_cos"]):
        model, info = fit_head(
            X_train,
            targets[target].to_numpy(dtype=float),
            kind=kind,
            seed=seed + offset,
            grid_search=bool(args.grid_search and kind == "xgb"),
            args=args,
        )
        heads[target] = model
        if info:
            metadata[target] = info
    return heads, metadata


def fit_cell_classifier(
    X_train: np.ndarray,
    y_train: pd.Series,
    *,
    kind: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Pipeline, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train.astype(str))
    if kind == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed in this Python environment.")
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=len(encoder.classes_),
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            eval_metric="mlogloss",
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=int(args.rf_estimators),
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier kind: {kind}")
    model = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    model.fit(X_train, y_encoded)
    return model, encoder


def predict_cell_positions(model: Pipeline, encoder: LabelEncoder, X: np.ndarray, lookup: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred_idx = np.asarray(model.predict(X), dtype=int)
    pred_cell = encoder.inverse_transform(pred_idx)
    coords = lookup.set_index("grid_cell").loc[pred_cell, ["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    return pred_cell.astype(object), coords


def predict_positions(heads: dict[str, object], X: np.ndarray, origin: PolarOrigin) -> np.ndarray:
    radius = np.asarray(heads["target_radius_m"].predict(X), dtype=float)
    sin_value = np.asarray(heads["target_sin"].predict(X), dtype=float)
    cos_value = np.asarray(heads["target_cos"].predict(X), dtype=float)
    norm = np.sqrt(np.square(sin_value) + np.square(cos_value))
    norm = np.where(norm < 1e-9, 1.0, norm)
    sin_value = sin_value / norm
    cos_value = cos_value / norm
    x = origin.x_m + radius * cos_value
    y = origin.y_m + radius * sin_value
    return np.column_stack([x, y])


def weighted_average(pred_xgb: np.ndarray, pred_rf: np.ndarray, *, weight_xgb: float) -> np.ndarray:
    weight_xgb = float(np.clip(weight_xgb, 0.0, 1.0))
    return weight_xgb * pred_xgb + (1.0 - weight_xgb) * pred_rf


def arbitrate_sequence(
    pred_xgb: np.ndarray,
    pred_rf: np.ndarray,
    *,
    threshold_m: float,
    high_alpha: float = 0.8,
    low_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(pred_xgb) == 0:
        return pred_xgb.copy(), np.array([], dtype=float), np.array([], dtype=object)
    out = np.zeros_like(pred_xgb, dtype=float)
    agreement = np.linalg.norm(pred_xgb - pred_rf, axis=1)
    states = np.where(agreement <= threshold_m, "STABLE", "PERTURBED").astype(object)
    out[0] = (pred_xgb[0] + pred_rf[0]) / 2.0
    for idx in range(1, len(pred_xgb)):
        alpha = high_alpha if agreement[idx] <= threshold_m else low_alpha
        mean_pos = (pred_xgb[idx] + pred_rf[idx]) / 2.0
        out[idx] = (1.0 - alpha) * out[idx - 1] + alpha * mean_pos
    return out, agreement, states


def nearest_cells(pred_xy: np.ndarray, lookup: pd.DataFrame) -> np.ndarray:
    cell_xy = lookup[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    labels = lookup["grid_cell"].to_numpy(dtype=object)
    distances = np.linalg.norm(pred_xy[:, None, :] - cell_xy[None, :, :], axis=2)
    return labels[np.argmin(distances, axis=1)]


def evaluate_positions(
    name: str,
    pred_xy: np.ndarray,
    test_df: pd.DataFrame,
    lookup: pd.DataFrame,
) -> dict[str, object]:
    true_xy = test_df[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    errors = np.linalg.norm(pred_xy - true_xy, axis=1)
    lateral = np.abs(pred_xy[:, 0] - true_xy[:, 0])
    depth = np.abs(pred_xy[:, 1] - true_xy[:, 1])
    pred_cell = nearest_cells(pred_xy, lookup)
    cell_acc = pred_cell == test_df["grid_cell"].to_numpy(dtype=object)
    return {
        "model": name,
        "n_test": int(len(test_df)),
        "mean_error_m": float(errors.mean()),
        "median_error_m": float(np.median(errors)),
        "p80_error_m": float(np.percentile(errors, 80)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "within_0_10m": float((errors <= 0.10).mean()),
        "within_0_22m": float((errors <= 0.22).mean()),
        "cell_accuracy": float(cell_acc.mean()),
        "mean_lateral_error_m": float(lateral.mean()),
        "mean_depth_error_m": float(depth.mean()),
        "pred_cell": pred_cell,
        "error_m": errors,
    }


def model_feature_importance(heads: dict[str, object], columns: list[str], *, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, pipeline in heads.items():
        model = pipeline.named_steps["model"]
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            continue
        for feature, value in zip(columns, importances, strict=False):
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "feature": feature,
                    "importance": float(value),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["model", "target", "feature", "importance", "importance_mean"])
    df = pd.DataFrame(rows)
    mean_df = df.groupby(["model", "feature"], as_index=False)["importance"].mean()
    mean_df = mean_df.rename(columns={"importance": "importance_mean"})
    return df.merge(mean_df, on=["model", "feature"], how="left").sort_values(
        ["importance_mean", "importance"],
        ascending=[False, False],
    )


def classifier_feature_importance(model: Pipeline, columns: list[str], *, model_name: str) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["model", "target", "feature", "importance", "importance_mean"])
    rows = [
        {
            "model": model_name,
            "target": "grid_cell",
            "feature": feature,
            "importance": float(value),
            "importance_mean": float(value),
        }
        for feature, value in zip(columns, importances, strict=False)
    ]
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def save_outputs(
    *,
    args: argparse.Namespace,
    summary: dict[str, object],
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    xgb_heads: dict[str, object],
    rf_heads: dict[str, object],
    xgb_cell_model: Pipeline,
    rf_cell_model: Pipeline,
) -> None:
    BENCHMARK_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)
    metrics_path = BENCHMARK_REPORT_DIR / f"{prefix}_metrics.csv"
    predictions_path = BENCHMARK_REPORT_DIR / f"{prefix}_predictions.csv"
    importance_path = BENCHMARK_REPORT_DIR / f"{prefix}_feature_importance.csv"
    summary_path = BENCHMARK_REPORT_DIR / f"{prefix}_summary.json"
    model_path = BENCHMARK_REPORT_DIR / f"{prefix}_models.joblib"
    snippet_path = BENCHMARK_REPORT_DIR / f"{prefix}_snippet.md"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "xgb": xgb_heads,
            "rf": rf_heads,
            "xgb_cell": xgb_cell_model,
            "rf_cell": rf_cell_model,
            "summary": summary,
        },
        model_path,
    )

    best = metrics_df.sort_values("p80_error_m").iloc[0]
    top_features = importance_df.drop_duplicates("feature").head(8)["feature"].tolist()
    snippet = (
        "### XGBoost polar 360 benchmark\n\n"
        f"- Dataset: `{summary['datasets']}`; samples: `{summary['n_windows']}` windows.\n"
        f"- Best p80 model: `{best['model']}` with p50 `{best['median_error_m']:.3f} m`, "
        f"p80 `{best['p80_error_m']:.3f} m`, p90 `{best['p90_error_m']:.3f} m`.\n"
        f"- XGB/RF agreement <= {summary['agreement_threshold_m']:.2f} m: "
        f"`{summary['agreement_rate']:.3f}`.\n"
        f"- Top features: `{', '.join(top_features)}`.\n"
        "- Note: arbitration smoothing is evaluated on a sorted capture proxy, not a real movement trajectory.\n"
    )
    snippet_path.write_text(snippet, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = [part.strip().upper() for part in str(args.datasets).split(",") if part.strip()]
    df_raw = load_360_dataframe(datasets, include_e102_elevation=bool(args.include_e102_elevation))
    df = build_windowed_dataset(df_raw, window_size=int(args.window_size))
    df, origin = add_polar_targets(df)
    columns = feature_columns(df)

    train_df, test_df = train_test_split(
        df,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=df["grid_cell"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = train_df[columns].to_numpy(dtype=float)
    X_test = test_df[columns].to_numpy(dtype=float)

    xgb_heads, xgb_meta = fit_three_heads(X_train, train_df, kind="xgb", seed=int(args.seed), args=args)
    rf_heads, _ = fit_three_heads(X_train, train_df, kind="rf", seed=int(args.seed) + 100, args=args)

    pred_xgb = predict_positions(xgb_heads, X_test, origin)
    pred_rf = predict_positions(rf_heads, X_test, origin)
    pred_weighted = weighted_average(pred_xgb, pred_rf, weight_xgb=float(args.fusion_weight_xgb))

    ordered = test_df.sort_values(["campaign", "grid_x", "grid_y", "window_id"]).index.to_numpy()
    pred_arb_ordered, agreement_ordered, state_ordered = arbitrate_sequence(
        pred_xgb[ordered],
        pred_rf[ordered],
        threshold_m=float(args.agreement_threshold_m),
    )
    pred_arbitrated = np.zeros_like(pred_xgb)
    agreement = np.zeros(len(test_df), dtype=float)
    states = np.empty(len(test_df), dtype=object)
    pred_arbitrated[ordered] = pred_arb_ordered
    agreement[ordered] = agreement_ordered
    states[ordered] = state_ordered

    lookup = (
        df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .reset_index(drop=True)
    )
    xgb_cell_model, xgb_cell_encoder = fit_cell_classifier(
        X_train,
        train_df["grid_cell"],
        kind="xgb",
        seed=int(args.seed) + 200,
        args=args,
    )
    rf_cell_model, rf_cell_encoder = fit_cell_classifier(
        X_train,
        train_df["grid_cell"],
        kind="rf",
        seed=int(args.seed) + 300,
        args=args,
    )
    xgb_cell_pred, pred_xgb_cell = predict_cell_positions(xgb_cell_model, xgb_cell_encoder, X_test, lookup)
    rf_cell_pred, pred_rf_cell = predict_cell_positions(rf_cell_model, rf_cell_encoder, X_test, lookup)

    evals = [
        evaluate_positions("xgb_polar", pred_xgb, test_df, lookup),
        evaluate_positions("rf_polar", pred_rf, test_df, lookup),
        evaluate_positions("weighted_xgb_rf", pred_weighted, test_df, lookup),
        evaluate_positions("arbitrated_proxy", pred_arbitrated, test_df, lookup),
        evaluate_positions("xgb_cell_classifier", pred_xgb_cell, test_df, lookup),
        evaluate_positions("rf_cell_classifier", pred_rf_cell, test_df, lookup),
    ]
    metrics_rows = []
    predictions = test_df[
        ["room", "campaign", "orientation_label", "grid_cell", "grid_x", "grid_y", "coord_x_m", "coord_y_m", "window_id"]
    ].copy()
    predictions["xgb_rf_agreement_m"] = agreement
    predictions["arbitration_state"] = states
    for item in evals:
        row = {key: value for key, value in item.items() if key not in {"pred_cell", "error_m"}}
        metrics_rows.append(row)
        name = str(item["model"])
        pred_xy = {
            "xgb_polar": pred_xgb,
            "rf_polar": pred_rf,
            "weighted_xgb_rf": pred_weighted,
            "arbitrated_proxy": pred_arbitrated,
            "xgb_cell_classifier": pred_xgb_cell,
            "rf_cell_classifier": pred_rf_cell,
        }[name]
        predictions[f"{name}_x_m"] = pred_xy[:, 0]
        predictions[f"{name}_y_m"] = pred_xy[:, 1]
        predictions[f"{name}_pred_cell"] = item["pred_cell"]
        predictions[f"{name}_error_m"] = item["error_m"]

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["p80_error_m", "median_error_m"])
    importance_df = pd.concat(
        [
            model_feature_importance(xgb_heads, columns, model_name="xgb"),
            model_feature_importance(rf_heads, columns, model_name="rf"),
            classifier_feature_importance(xgb_cell_model, columns, model_name="xgb_cell"),
            classifier_feature_importance(rf_cell_model, columns, model_name="rf_cell"),
        ],
        ignore_index=True,
    )

    xgb_eval = next(item for item in evals if item["model"] == "xgb_polar")
    xgb_cell_eval = next(item for item in evals if item["model"] == "xgb_cell_classifier")
    labels = sorted(lookup["grid_cell"].astype(str).unique())
    xgb_cm = confusion_matrix(test_df["grid_cell"].astype(str), xgb_eval["pred_cell"].astype(str), labels=labels)
    cm_path = BENCHMARK_REPORT_DIR / f"{args.output_prefix}_xgb_confusion_matrix.csv"
    pd.DataFrame(xgb_cm, index=labels, columns=labels).to_csv(cm_path)
    xgb_cell_cm = confusion_matrix(
        test_df["grid_cell"].astype(str),
        xgb_cell_eval["pred_cell"].astype(str),
        labels=labels,
    )
    cell_cm_path = BENCHMARK_REPORT_DIR / f"{args.output_prefix}_xgb_cell_confusion_matrix.csv"
    pd.DataFrame(xgb_cell_cm, index=labels, columns=labels).to_csv(cell_cm_path)

    summary = {
        "datasets": datasets,
        "n_raw_samples": int(len(df_raw)),
        "n_windows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "window_size": int(args.window_size),
        "test_size": float(args.test_size),
        "origin": {"x_m": origin.x_m, "y_m": origin.y_m},
        "feature_count": int(len(columns)),
        "feature_columns": columns,
        "grid_search": bool(args.grid_search),
        "xgb_grid_search": xgb_meta,
        "fusion_weight_xgb": float(args.fusion_weight_xgb),
        "agreement_threshold_m": float(args.agreement_threshold_m),
        "agreement_rate": float((agreement <= float(args.agreement_threshold_m)).mean()),
        "metrics": metrics_df.to_dict(orient="records"),
        "confusion_matrix_csv": str(cm_path),
        "cell_confusion_matrix_csv": str(cell_cm_path),
    }

    save_outputs(
        args=args,
        summary=summary,
        metrics_df=metrics_df,
        predictions_df=predictions,
        importance_df=importance_df,
        xgb_heads=xgb_heads,
        rf_heads=rf_heads,
        xgb_cell_model=xgb_cell_model,
        rf_cell_model=rf_cell_model,
    )

    print(metrics_df.to_string(index=False))
    print("\nTop XGBoost polar features:")
    print(
        importance_df[importance_df["model"] == "xgb"]
        .drop_duplicates("feature")
        .head(12)[["feature", "importance_mean"]]
        .to_string(index=False)
    )
    print("\nTop XGBoost cell-classifier features:")
    print(
        importance_df[importance_df["model"] == "xgb_cell"]
        .drop_duplicates("feature")
        .head(12)[["feature", "importance_mean"]]
        .to_string(index=False)
    )
    print(f"\nSaved outputs with prefix: {BENCHMARK_REPORT_DIR / str(args.output_prefix)}")


if __name__ == "__main__":
    main()
