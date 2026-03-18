#!/usr/bin/env python3
"""Train and evaluate a circular orientation model for ESP32 pose labels.

The target angle is encoded on the unit circle so the model respects wrap-around:
0 deg == back, 90 deg == right, 180 deg == front, 270 deg == left.
Intermediate modes such as back_right are handled naturally (45 deg).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.orientation import (  # noqa: E402
    angles_to_unit_circle,
    circular_distance_degrees,
    nearest_orientation_label,
    orientation_label_to_degrees,
    unit_circle_to_angle,
)

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"

ORIENTATION_CAMPAIGNS: dict[str, list[tuple[str, CampaignSpec, str]]] = {
    "E101": [
        ("E101/back", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "back", router_distance_m=3.0), "back"),
        ("E101/right", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "right", router_distance_m=3.0), "right"),
        ("E101/front", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "front", router_distance_m=3.0), "front"),
        ("E101/left", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "left", router_distance_m=3.0), "left"),
    ],
    "E102": [
        ("E102/exp1_back_right", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp1", router_distance_m=4.0), "back_right"),
        ("E102/exp2_front_right", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp2", router_distance_m=4.0), "front_right"),
        ("E102/exp3_front_left", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp3", router_distance_m=4.0), "front_left"),
        ("E102/exp4_back_left", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp4", router_distance_m=4.0), "back_left"),
        ("E102/exp5_ground", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "elevation" / "exp5", router_distance_m=4.0), "front"),
        ("E102/exp6_1m50", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "elevation" / "exp6", router_distance_m=4.0), "front"),
    ],
}


@dataclass
class EvalMetrics:
    n_samples: int
    n_test: int
    mean_angular_error_deg: float
    median_angular_error_deg: float
    p90_angular_error_deg: float
    snapped_label_accuracy: float
    snapped_label_f1_macro: float


def build_feature_frame(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Build the input feature matrix used by the orientation models."""
    base = df.loc[:, FEATURE_COLUMNS].astype(float).copy()
    antennas = base[["signal_A1", "signal_A2", "signal_A3"]]
    work = base.copy()

    work["a1_minus_a2"] = base["signal_A1"] - base["signal_A2"]
    work["a1_minus_a3"] = base["signal_A1"] - base["signal_A3"]
    work["a2_minus_a3"] = base["signal_A2"] - base["signal_A3"]
    work["signal_minus_a1"] = base["Signal"] - base["signal_A1"]
    work["signal_minus_a2"] = base["Signal"] - base["signal_A2"]
    work["signal_minus_a3"] = base["Signal"] - base["signal_A3"]
    work["noise_minus_signal"] = base["Noise"] - base["Signal"]
    work["ant_mean"] = antennas.mean(axis=1)
    work["ant_std"] = antennas.std(axis=1, ddof=0)
    work["ant_min"] = antennas.min(axis=1)
    work["ant_max"] = antennas.max(axis=1)
    work["ant_range"] = work["ant_max"] - work["ant_min"]
    work["signal_minus_ant_mean"] = base["Signal"] - work["ant_mean"]
    work["noise_minus_ant_mean"] = base["Noise"] - work["ant_mean"]

    feature_map = {
        "raw": FEATURE_COLUMNS,
        "signal_vs_ant": ["signal_minus_a1", "signal_minus_a2", "signal_minus_a3", "noise_minus_signal"],
        "raw_plus_deltas": FEATURE_COLUMNS + ["a1_minus_a2", "a1_minus_a3", "a2_minus_a3"],
        "engineered": list(work.columns),
    }
    try:
        return work.loc[:, feature_map[feature_set]].copy()
    except KeyError as exc:
        raise ValueError(f"Unknown feature_set: {feature_set}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="E101,E102",
        help="Comma-separated datasets among E101,E102.",
    )
    parser.add_argument(
        "--exclude-e102-elevation",
        action="store_true",
        help="Exclude E102 exp5/exp6, which are both mapped to front (180 deg).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Group split ratio by cell.")
    parser.add_argument("--seed", type=int, default=21, help="Random seed for the group split.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Trees per regressor.")
    parser.add_argument(
        "--model-family",
        choices=["rf", "extratrees", "histgb"],
        default="rf",
        help="Classifier family used for circular classification.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["raw", "signal_vs_ant", "raw_plus_deltas", "engineered"],
        default="raw",
        help="Feature family used as model input.",
    )
    parser.add_argument(
        "--approach",
        choices=["circular_classifier", "circular_regression"],
        default="circular_classifier",
        help="Training approach. The classifier computes a circular mean from class probabilities.",
    )
    parser.add_argument(
        "--model-output",
        default=str(REPORT_DIR / "orientation_circular_model.joblib"),
        help="Path to the serialized model bundle.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(REPORT_DIR / "orientation_circular_metrics.json"),
        help="Path to the JSON metrics report.",
    )
    parser.add_argument(
        "--predictions-output",
        default=str(REPORT_DIR / "orientation_circular_predictions.csv"),
        help="Path to the per-sample prediction CSV.",
    )
    return parser.parse_args()


def _load_orientation_dataframe(selected_datasets: list[str], *, include_e102_elevation: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_name in selected_datasets:
        for campaign_name, spec, orientation_label in ORIENTATION_CAMPAIGNS[dataset_name]:
            if dataset_name == "E102" and not include_e102_elevation and campaign_name.endswith(("exp5_ground", "exp6_1m50")):
                continue
            df = load_measurements([spec]).copy()
            df["room"] = dataset_name
            df["campaign"] = campaign_name
            df["orientation_label"] = orientation_label
            df["orientation_deg"] = orientation_label_to_degrees(orientation_label)
            frames.append(df)
    if not frames:
        raise RuntimeError("No orientation campaigns selected.")
    combined = pd.concat(frames, ignore_index=True)
    combined["group_id"] = combined["room"] + "::" + combined["grid_cell"]
    return combined


def _build_regression_model(seed: int, n_estimators: int) -> Pipeline:
    regressor = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )


def _build_classifier_model(seed: int, n_estimators: int, model_family: str) -> Pipeline:
    if model_family == "rf":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    elif model_family == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    elif model_family == "histgb":
        clf = HistGradientBoostingClassifier(
            max_iter=max(100, n_estimators),
            random_state=seed,
            min_samples_leaf=10,
        )
    else:
        raise ValueError(f"Unknown model_family: {model_family}")
    return Pipeline(steps=[("scaler", StandardScaler()), ("classifier", clf)])


def _predict_angles_regression(model: Pipeline, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_unit = np.asarray(model.predict(X), dtype=float)
    pred_angles = np.asarray([unit_circle_to_angle(c, s) for c, s in pred_unit], dtype=float)
    pred_labels = np.asarray([nearest_orientation_label(angle) for angle in pred_angles], dtype=object)
    return pred_angles, pred_labels


def _predict_angles_classifier(
    model: Pipeline,
    encoder: LabelEncoder,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = encoder.inverse_transform(np.arange(proba.shape[1]))
    class_angles = np.asarray([orientation_label_to_degrees(label) for label in classes], dtype=float)
    class_radians = np.deg2rad(class_angles)
    x_coord = (proba * np.cos(class_radians)).sum(axis=1)
    y_coord = (proba * np.sin(class_radians)).sum(axis=1)
    pred_angles = np.rad2deg(np.arctan2(y_coord, x_coord)) % 360.0
    pred_labels = np.asarray(
        [nearest_orientation_label(angle, labels=classes) for angle in pred_angles],
        dtype=object,
    )
    return pred_angles, pred_labels


def _evaluate(df_true: pd.DataFrame, pred_angles: np.ndarray, pred_labels: np.ndarray) -> EvalMetrics:
    true_angles = df_true["orientation_deg"].to_numpy(dtype=float)
    errors = np.asarray(
        [circular_distance_degrees(true_angle, pred_angle) for true_angle, pred_angle in zip(true_angles, pred_angles)],
        dtype=float,
    )
    return EvalMetrics(
        n_samples=int(len(df_true)),
        n_test=int(len(df_true)),
        mean_angular_error_deg=float(errors.mean()),
        median_angular_error_deg=float(np.median(errors)),
        p90_angular_error_deg=float(np.percentile(errors, 90)),
        snapped_label_accuracy=float(accuracy_score(df_true["orientation_label"], pred_labels)),
        snapped_label_f1_macro=float(f1_score(df_true["orientation_label"], pred_labels, average="macro")),
    )


def main() -> None:
    args = parse_args()
    selected = [token.strip().upper() for token in str(args.datasets).split(",") if token.strip()]
    unknown = sorted(set(selected) - set(ORIENTATION_CAMPAIGNS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    df = _load_orientation_dataframe(selected, include_e102_elevation=not args.exclude_e102_elevation)
    feature_df = build_feature_frame(df, args.feature_set)
    X = feature_df.to_numpy(dtype=float)
    y_unit = angles_to_unit_circle(df["orientation_deg"].to_numpy(dtype=float))
    encoder = LabelEncoder()
    y_label = encoder.fit_transform(df["orientation_label"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, groups=df["group_id"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    if args.approach == "circular_regression":
        model = _build_regression_model(seed=args.seed, n_estimators=args.n_estimators)
        model.fit(build_feature_frame(train_df, args.feature_set).to_numpy(dtype=float), y_unit[train_idx])
        pred_angles, pred_labels = _predict_angles_regression(
            model,
            build_feature_frame(test_df, args.feature_set).to_numpy(dtype=float),
        )
    else:
        model = _build_classifier_model(
            seed=args.seed,
            n_estimators=args.n_estimators,
            model_family=args.model_family,
        )
        model.fit(build_feature_frame(train_df, args.feature_set).to_numpy(dtype=float), y_label[train_idx])
        pred_angles, pred_labels = _predict_angles_classifier(
            model,
            encoder,
            build_feature_frame(test_df, args.feature_set).to_numpy(dtype=float),
        )
    metrics = _evaluate(test_df, pred_angles, pred_labels)

    errors = [
        circular_distance_degrees(true_angle, pred_angle)
        for true_angle, pred_angle in zip(test_df["orientation_deg"], pred_angles)
    ]
    pred_df = test_df[
        ["room", "campaign", "grid_cell", "router_distance_m", "orientation_label", "orientation_deg"]
    ].copy()
    pred_df["pred_orientation_deg"] = pred_angles
    pred_df["pred_orientation_label"] = pred_labels
    pred_df["angular_error_deg"] = errors

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.predictions_output)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)

    # Refit on all available samples before serializing the final model.
    if args.approach == "circular_regression":
        final_model = _build_regression_model(seed=args.seed, n_estimators=args.n_estimators)
        final_model.fit(X, y_unit)
    else:
        final_model = _build_classifier_model(
            seed=args.seed,
            n_estimators=args.n_estimators,
            model_family=args.model_family,
        )
        final_model.fit(X, y_label)
    model_bundle = {
        "model": final_model,
        "approach": args.approach,
        "model_family": args.model_family,
        "feature_set": args.feature_set,
        "feature_columns": feature_df.columns.tolist(),
        "label_degrees": {label: orientation_label_to_degrees(label) for label in sorted(df["orientation_label"].unique())},
        "datasets": selected,
        "include_e102_elevation": bool(not args.exclude_e102_elevation),
    }
    if args.approach == "circular_classifier":
        model_bundle["label_encoder_classes"] = encoder.classes_.tolist()
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)

    summary = {
        "datasets": selected,
        "approach": args.approach,
        "model_family": args.model_family,
        "feature_set": args.feature_set,
        "include_e102_elevation": bool(not args.exclude_e102_elevation),
        "feature_columns": feature_df.columns.tolist(),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "group_split": {
            "seed": int(args.seed),
            "test_size": float(args.test_size),
            "n_train_groups": int(train_df["group_id"].nunique()),
            "n_test_groups": int(test_df["group_id"].nunique()),
        },
        "metrics": asdict(metrics),
        "orientation_distribution": (
            df.groupby(["room", "orientation_label"]).size().rename("samples").reset_index().to_dict(orient="records")
        ),
    }
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Predictions saved to {pred_path}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
