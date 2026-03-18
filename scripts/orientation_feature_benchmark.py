#!/usr/bin/env python3
"""Benchmark orientation feature/model variants across multiple seeds."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.orientation import (  # noqa: E402
    circular_distance_degrees,
    nearest_orientation_label,
    orientation_label_to_degrees,
)
from scripts.train_orientation_model import build_feature_frame  # noqa: E402

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
SEEDS = [7, 17, 27, 37, 47]

CAMPAIGNS = [
    ("E102/exp1_back_right", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp1", router_distance_m=4.0), "back_right"),
    ("E102/exp2_front_right", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp2", router_distance_m=4.0), "front_right"),
    ("E102/exp3_front_left", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp3", router_distance_m=4.0), "front_left"),
    ("E102/exp4_back_left", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp4", router_distance_m=4.0), "back_left"),
]

CONFIGS = [
    ("raw_rf", "raw", "rf"),
    ("signal_vs_ant_rf", "signal_vs_ant", "rf"),
    ("raw_plus_deltas_rf", "raw_plus_deltas", "rf"),
    ("signal_vs_ant_extratrees", "signal_vs_ant", "extratrees"),
]


@dataclass
class BenchmarkRow:
    config: str
    feature_set: str
    model_family: str
    seed: int
    accuracy: float
    f1_macro: float
    mean_angular_error_deg: float
    median_angular_error_deg: float
    p90_angular_error_deg: float


def load_dataset() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for campaign_name, spec, orientation_label in CAMPAIGNS:
        df = load_measurements([spec]).copy()
        df["room"] = "E102"
        df["campaign"] = campaign_name
        df["orientation_label"] = orientation_label
        df["orientation_deg"] = orientation_label_to_degrees(orientation_label)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["group_id"] = combined["room"] + "::" + combined["grid_cell"]
    return combined


def build_classifier(model_family: str, seed: int) -> Pipeline:
    if model_family == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    elif model_family == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=300,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    elif model_family == "histgb":
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            random_state=seed,
            min_samples_leaf=10,
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


def evaluate_config(df: pd.DataFrame, *, feature_set: str, model_family: str, seed: int) -> BenchmarkRow:
    feature_df = build_feature_frame(df, feature_set)
    X = feature_df.to_numpy(dtype=float)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["orientation_label"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["group_id"]))
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    test_df = df.iloc[test_idx].reset_index(drop=True)

    model = build_classifier(model_family, seed)
    model.fit(X_train, y_train)
    proba = np.asarray(model.predict_proba(X_test), dtype=float)
    classes = encoder.inverse_transform(np.arange(proba.shape[1]))
    class_angles = np.asarray([orientation_label_to_degrees(label) for label in classes], dtype=float)
    class_radians = np.deg2rad(class_angles)
    x_coord = (proba * np.cos(class_radians)).sum(axis=1)
    y_coord = (proba * np.sin(class_radians)).sum(axis=1)
    pred_angles = np.rad2deg(np.arctan2(y_coord, x_coord)) % 360.0
    pred_labels = np.asarray([nearest_orientation_label(angle, labels=classes) for angle in pred_angles], dtype=object)

    true_angles = test_df["orientation_deg"].to_numpy(dtype=float)
    angular_errors = np.asarray(
        [circular_distance_degrees(true_angle, pred_angle) for true_angle, pred_angle in zip(true_angles, pred_angles)],
        dtype=float,
    )
    return BenchmarkRow(
        config=f"{feature_set}_{model_family}",
        feature_set=feature_set,
        model_family=model_family,
        seed=seed,
        accuracy=float(accuracy_score(test_df["orientation_label"], pred_labels)),
        f1_macro=float(f1_score(test_df["orientation_label"], pred_labels, average="macro")),
        mean_angular_error_deg=float(angular_errors.mean()),
        median_angular_error_deg=float(np.median(angular_errors)),
        p90_angular_error_deg=float(np.percentile(angular_errors, 90)),
    )


def main() -> None:
    df = load_dataset()
    rows: list[BenchmarkRow] = []
    for _, feature_set, model_family in CONFIGS:
        for seed in SEEDS:
            rows.append(evaluate_config(df, feature_set=feature_set, model_family=model_family, seed=seed))

    result_df = pd.DataFrame([asdict(row) for row in rows])
    summary_df = (
        result_df.groupby(["config", "feature_set", "model_family"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            mean_angular_error_deg_mean=("mean_angular_error_deg", "mean"),
            mean_angular_error_deg_std=("mean_angular_error_deg", "std"),
            median_angular_error_deg_mean=("median_angular_error_deg", "mean"),
            p90_angular_error_deg_mean=("p90_angular_error_deg", "mean"),
        )
        .sort_values(["accuracy_mean", "f1_macro_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_path = REPORT_DIR / "orientation_feature_benchmark_e102_per_seed.csv"
    summary_path = REPORT_DIR / "orientation_feature_benchmark_e102_summary.csv"
    json_path = REPORT_DIR / "orientation_feature_benchmark_e102_summary.json"
    snippet_path = REPORT_DIR / "orientation_feature_benchmark_e102_snippet.md"

    result_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "dataset": "E102",
                "seeds": SEEDS,
                "configs": [config for config, _, _ in CONFIGS],
                "ranking": summary_df.to_dict(orient="records"),
            },
            indent=2,
        )
    )

    best = summary_df.iloc[0]
    baseline = summary_df.loc[summary_df["config"] == "raw_rf"].iloc[0]
    snippet = (
        "### Orientation benchmark E102\n\n"
        f"- Best config: `{best['config']}` with accuracy `{best['accuracy_mean']:.3f}` and mean angular error `{best['mean_angular_error_deg_mean']:.1f}` deg.\n"
        f"- Baseline `raw_rf`: accuracy `{baseline['accuracy_mean']:.3f}`, mean angular error `{baseline['mean_angular_error_deg_mean']:.1f}` deg.\n"
        f"- Absolute gain vs baseline: `{best['accuracy_mean'] - baseline['accuracy_mean']:+.3f}` accuracy points.\n"
    )
    snippet_path.write_text(snippet)

    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
