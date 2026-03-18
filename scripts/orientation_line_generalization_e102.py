#!/usr/bin/env python3
"""Evaluate E102 orientation models by holding out physical grid rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
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

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
CONFIGS = [
    ("raw_rf", "raw"),
    ("signal_vs_ant_rf", "signal_vs_ant"),
]
CAMPAIGNS = [
    ("E102/exp1_back_right", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp1", router_distance_m=4.0), "back_right"),
    ("E102/exp2_front_right", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp2", router_distance_m=4.0), "front_right"),
    ("E102/exp3_front_left", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp3", router_distance_m=4.0), "front_left"),
    ("E102/exp4_back_left", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp4", router_distance_m=4.0), "back_left"),
]


def load_dataset() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for campaign_name, spec, orientation_label in CAMPAIGNS:
        df = load_measurements([spec]).copy()
        df["campaign"] = campaign_name
        df["orientation_label"] = orientation_label
        df["orientation_deg"] = orientation_label_to_degrees(orientation_label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=seed,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def predict_circular_labels(model: Pipeline, encoder: LabelEncoder, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = encoder.inverse_transform(np.arange(proba.shape[1]))
    class_angles = np.asarray([orientation_label_to_degrees(label) for label in classes], dtype=float)
    class_radians = np.deg2rad(class_angles)
    x_coord = (proba * np.cos(class_radians)).sum(axis=1)
    y_coord = (proba * np.sin(class_radians)).sum(axis=1)
    pred_angles = np.rad2deg(np.arctan2(y_coord, x_coord)) % 360.0
    pred_labels = np.asarray([nearest_orientation_label(angle, labels=classes) for angle in pred_angles], dtype=object)
    return pred_labels, pred_angles


def evaluate_config(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, dict]:
    encoder = LabelEncoder().fit(df["orientation_label"])
    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    for held_out_row in sorted(df["grid_x"].unique()):
        train_df = df[df["grid_x"] != held_out_row].reset_index(drop=True)
        test_df = df[df["grid_x"] == held_out_row].reset_index(drop=True)
        X_train = build_feature_frame(train_df, feature_set).to_numpy(dtype=float)
        X_test = build_feature_frame(test_df, feature_set).to_numpy(dtype=float)
        y_train = encoder.transform(train_df["orientation_label"])

        model = build_model(seed=held_out_row + 101)
        model.fit(X_train, y_train)
        pred_labels, pred_angles = predict_circular_labels(model, encoder, X_test)
        angular_errors = np.asarray(
            [
                circular_distance_degrees(true_angle, pred_angle)
                for true_angle, pred_angle in zip(test_df["orientation_deg"], pred_angles)
            ],
            dtype=float,
        )
        rows.append(
            {
                "held_out_grid_x": int(held_out_row),
                "n_test": int(len(test_df)),
                "accuracy": float(accuracy_score(test_df["orientation_label"], pred_labels)),
                "f1_macro": float(f1_score(test_df["orientation_label"], pred_labels, average="macro")),
                "mean_angular_error_deg": float(angular_errors.mean()),
                "median_angular_error_deg": float(np.median(angular_errors)),
                "p90_angular_error_deg": float(np.percentile(angular_errors, 90)),
            }
        )
        fold_pred = test_df[["campaign", "grid_cell", "grid_x", "orientation_label", "orientation_deg"]].copy()
        fold_pred["pred_orientation_label"] = pred_labels
        fold_pred["pred_orientation_deg"] = pred_angles
        fold_pred["angular_error_deg"] = angular_errors
        fold_pred["held_out_grid_x"] = int(held_out_row)
        pred_rows.append(fold_pred)

    fold_df = pd.DataFrame(rows).sort_values("held_out_grid_x").reset_index(drop=True)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    summary = {
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
        "f1_macro_mean": float(fold_df["f1_macro"].mean()),
        "mean_angular_error_deg_mean": float(fold_df["mean_angular_error_deg"].mean()),
        "mean_angular_error_deg_std": float(fold_df["mean_angular_error_deg"].std(ddof=0)),
        "per_row": fold_df.to_dict(orient="records"),
    }
    return pred_df, summary


def main() -> None:
    df = load_dataset()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ranking_rows: list[dict] = []
    for config_name, feature_set in CONFIGS:
        pred_df, summary = evaluate_config(df, feature_set)
        pred_path = REPORT_DIR / f"{config_name}_line_generalization_predictions.csv"
        json_path = REPORT_DIR / f"{config_name}_line_generalization_summary.json"
        pred_df.to_csv(pred_path, index=False)
        json_path.write_text(json.dumps(summary, indent=2))
        ranking_rows.append(
            {
                "config": config_name,
                "feature_set": feature_set,
                "accuracy_mean": summary["accuracy_mean"],
                "accuracy_std": summary["accuracy_std"],
                "f1_macro_mean": summary["f1_macro_mean"],
                "mean_angular_error_deg_mean": summary["mean_angular_error_deg_mean"],
                "mean_angular_error_deg_std": summary["mean_angular_error_deg_std"],
            }
        )

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        ["accuracy_mean", "f1_macro_mean"],
        ascending=[False, False],
    )
    ranking_path = REPORT_DIR / "orientation_line_generalization_e102_ranking.csv"
    snippet_path = REPORT_DIR / "orientation_line_generalization_e102_snippet.md"
    ranking_df.to_csv(ranking_path, index=False)

    best = ranking_df.iloc[0]
    snippet = (
        "### Orientation line generalization E102\n\n"
        f"- Best config: `{best['config']}` with row-held-out accuracy `{best['accuracy_mean']:.3f}` and mean angular error `{best['mean_angular_error_deg_mean']:.1f}` deg.\n"
        f"- Ranking saved to `reports/benchmarks/orientation_line_generalization_e102_ranking.csv`.\n"
    )
    snippet_path.write_text(snippet)

    print(ranking_df.to_string(index=False))
    print(f"\nSaved ranking to {ranking_path}")


if __name__ == "__main__":
    main()
