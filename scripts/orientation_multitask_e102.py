#!/usr/bin/env python3
"""Benchmark a shared RSSI encoder with an auxiliary orientation head on E102.

The protocol compares:
1. an orientation-only MLP trained from the 5 RSSI features;
2. a multi-task MLP trained on the same inputs with two heads:
   - grid cell classification,
   - orientation mode classification.

Orientation labels are mapped to circular degrees:
  - exp1 -> back_right  (45 deg)
  - exp2 -> front_right (135 deg)
  - exp3 -> front_left  (225 deg)
  - exp4 -> back_left   (315 deg)
"""

from __future__ import annotations

import json
import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.orientation import (  # noqa: E402
    circular_distance_degrees,
    nearest_orientation_label,
    orientation_label_to_degrees,
)

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"

CAMPAIGNS: dict[str, tuple[CampaignSpec, str]] = {
    "exp1_back_right": (CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp1", router_distance_m=4.0), "back_right"),
    "exp2_front_right": (CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp2", router_distance_m=4.0), "front_right"),
    "exp3_front_left": (CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp3", router_distance_m=4.0), "front_left"),
    "exp4_back_left": (CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp4", router_distance_m=4.0), "back_left"),
}
ORIENTATION_LABELS = ["back_right", "front_right", "front_left", "back_left"]


@dataclass
class SplitResult:
    seed: int
    orient_only_acc: float
    orient_only_f1_macro: float
    orient_only_mean_ang_err: float
    multitask_orient_acc: float
    multitask_orient_f1_macro: float
    multitask_mean_ang_err: float
    multitask_cell_acc: float
    multitask_cell_f1_macro: float
    predictions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-weight", type=float, default=0.5, help="Relative weight of the cell loss.")
    parser.add_argument(
        "--seeds",
        default="7,17,27,37,47",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--output-prefix",
        default="e102_orientation_multitask",
        help="Prefix used for CSV/JSON/PNG outputs inside reports/benchmarks.",
    )
    return parser.parse_args()


class SharedMLP:
    def __init__(self, input_dim: int, cell_classes: int | None, orientation_classes: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, 0.15, size=(input_dim, 48)),
            "b1": np.zeros(48, dtype=np.float64),
            "W2": rng.normal(0.0, 0.15, size=(48, 24)),
            "b2": np.zeros(24, dtype=np.float64),
            "Wo": rng.normal(0.0, 0.15, size=(24, orientation_classes)),
            "bo": np.zeros(orientation_classes, dtype=np.float64),
        }
        if cell_classes is not None:
            self.params["Wc"] = rng.normal(0.0, 0.15, size=(24, cell_classes))
            self.params["bc"] = np.zeros(cell_classes, dtype=np.float64)

    def copy_state(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.params = {k: v.copy() for k, v in state.items()}

    def forward(self, X: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray]:
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.maximum(z2, 0.0)
        cell_logits = None
        if "Wc" in self.params:
            cell_logits = a2 @ self.params["Wc"] + self.params["bc"]
        orient_logits = a2 @ self.params["Wo"] + self.params["bo"]
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return cache, cell_logits, orient_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_e102_dataframe() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for campaign_name, (spec, orientation_label) in CAMPAIGNS.items():
        df = load_measurements([spec]).copy()
        df["campaign"] = campaign_name
        df["orientation_label"] = orientation_label
        df["orientation_deg"] = orientation_label_to_degrees(orientation_label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_and_grad(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
    probs = softmax(logits)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    loss = -np.mean(np.sum(y_onehot * np.log(np.clip(probs, 1e-9, 1.0)), axis=1))
    grad = (probs - y_onehot) / len(y_true)
    return float(loss), grad


def backward_shared(
    model: SharedMLP,
    cache: dict[str, np.ndarray],
    grad_cell_logits: np.ndarray | None,
    grad_orient_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    grads: dict[str, np.ndarray] = {}
    a2 = cache["a2"]
    grads["Wo"] = a2.T @ grad_orient_logits
    grads["bo"] = grad_orient_logits.sum(axis=0)
    grad_a2 = grad_orient_logits @ model.params["Wo"].T

    if grad_cell_logits is not None and "Wc" in model.params:
        grads["Wc"] = a2.T @ grad_cell_logits
        grads["bc"] = grad_cell_logits.sum(axis=0)
        grad_a2 = grad_a2 + grad_cell_logits @ model.params["Wc"].T

    grad_z2 = grad_a2 * (cache["z2"] > 0.0)
    grads["W2"] = cache["a1"].T @ grad_z2
    grads["b2"] = grad_z2.sum(axis=0)
    grad_a1 = grad_z2 @ model.params["W2"].T
    grad_z1 = grad_a1 * (cache["z1"] > 0.0)
    grads["W1"] = cache["X"].T @ grad_z1
    grads["b1"] = grad_z1.sum(axis=0)
    return grads


def adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    opt_state: dict[str, dict[str, np.ndarray] | int],
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    opt_state["step"] = int(opt_state["step"]) + 1
    t = int(opt_state["step"])
    for name, grad in grads.items():
        m = opt_state["m"][name]
        v = opt_state["v"][name]
        m[:] = beta1 * m + (1.0 - beta1) * grad
        v[:] = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_model(
    X_train: np.ndarray,
    y_orient_train: np.ndarray,
    y_cell_train: np.ndarray | None,
    *,
    seed: int,
    multitask: bool,
    cell_weight: float = 0.5,
    max_epochs: int = 140,
    patience: int = 18,
    batch_size: int = 256,
) -> tuple[SharedMLP, StandardScaler]:
    set_seed(seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    stratify_target = y_orient_train if y_cell_train is None else np.array(
        [f"{o}_{c}" for o, c in zip(y_orient_train, y_cell_train, strict=False)]
    )
    idx = np.arange(len(X_train_scaled))
    idx_train, idx_val = train_test_split(
        idx,
        test_size=0.1,
        random_state=seed,
        stratify=stratify_target,
    )

    model = SharedMLP(
        input_dim=X_train_scaled.shape[1],
        cell_classes=int(np.max(y_cell_train) + 1) if y_cell_train is not None else None,
        orientation_classes=int(np.max(y_orient_train) + 1),
        seed=seed,
    )
    opt_state: dict[str, dict[str, np.ndarray] | int] = {
        "step": 0,
        "m": {name: np.zeros_like(value) for name, value in model.params.items()},
        "v": {name: np.zeros_like(value) for name, value in model.params.items()},
    }

    X_fit = X_train_scaled[idx_train]
    y_orient_fit = y_orient_train[idx_train]
    y_cell_fit = y_cell_train[idx_train] if y_cell_train is not None else None
    X_val = X_train_scaled[idx_val]
    y_orient_val = y_orient_train[idx_val]
    y_cell_val = y_cell_train[idx_val] if y_cell_train is not None else None

    best_state: dict[str, np.ndarray] | None = None
    best_val = float("inf")
    wait = 0

    for _ in range(max_epochs):
        order = np.random.permutation(len(X_fit))
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            xb = X_fit[batch_ids]
            yb_orient = y_orient_fit[batch_ids]
            yb_cell = y_cell_fit[batch_ids] if y_cell_fit is not None else None

            cache, cell_logits, orient_logits = model.forward(xb)
            _, grad_orient = cross_entropy_and_grad(orient_logits, yb_orient)
            grad_cell = None
            if multitask and yb_cell is not None and cell_logits is not None:
                _, grad_cell_raw = cross_entropy_and_grad(cell_logits, yb_cell)
                grad_cell = cell_weight * grad_cell_raw
            grads = backward_shared(model, cache, grad_cell, grad_orient)
            adam_update(model.params, grads, opt_state)

        _, cell_logits_val, orient_logits_val = model.forward(X_val)
        orient_val_loss, _ = cross_entropy_and_grad(orient_logits_val, y_orient_val)
        val_value = orient_val_loss
        if multitask and y_cell_val is not None and cell_logits_val is not None:
            cell_val_loss, _ = cross_entropy_and_grad(cell_logits_val, y_cell_val)
            val_value = val_value + cell_weight * cell_val_loss

        if val_value < best_val - 1e-4:
            best_val = val_value
            best_state = model.copy_state()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")
    model.load_state(best_state)
    return model, scaler


def predict_model(
    model: SharedMLP,
    scaler: StandardScaler,
    X: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    X_scaled = scaler.transform(X)
    _, cell_logits, orient_logits = model.forward(X_scaled)
    orient_probs = softmax(orient_logits)
    cell_pred = cell_logits.argmax(axis=1) if cell_logits is not None else None
    orient_pred = orient_logits.argmax(axis=1)
    return cell_pred, orient_pred, orient_probs


def orientation_probs_to_angles(
    orient_probs: np.ndarray,
    orient_encoder: LabelEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    class_labels = orient_encoder.inverse_transform(np.arange(orient_probs.shape[1]))
    class_angles = np.asarray([orientation_label_to_degrees(label) for label in class_labels], dtype=float)
    class_radians = np.deg2rad(class_angles)
    x_coord = (orient_probs * np.cos(class_radians)).sum(axis=1)
    y_coord = (orient_probs * np.sin(class_radians)).sum(axis=1)
    pred_angles = np.rad2deg(np.arctan2(y_coord, x_coord)) % 360.0
    pred_labels = np.asarray(
        [nearest_orientation_label(angle, labels=class_labels) for angle in pred_angles],
        dtype=object,
    )
    return pred_angles, pred_labels


def mean_angular_error_deg(true_angles: np.ndarray, pred_angles: np.ndarray) -> float:
    return float(
        np.mean(
            [
                circular_distance_degrees(true_angle, pred_angle)
                for true_angle, pred_angle in zip(true_angles, pred_angles, strict=False)
            ]
        )
    )


def run_seed_with_weight(
    df: pd.DataFrame,
    *,
    seed: int,
    cell_weight: float,
    cell_encoder: LabelEncoder,
    orient_encoder: LabelEncoder,
) -> SplitResult:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    stratify_labels = df["grid_cell"] + "|" + df["orientation_label"]
    train_idx, test_idx = next(splitter.split(df, stratify_labels))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_cell_train = cell_encoder.transform(train_df["grid_cell"])
    y_cell_test = cell_encoder.transform(test_df["grid_cell"])
    y_orient_train = orient_encoder.transform(train_df["orientation_label"])
    y_orient_test = orient_encoder.transform(test_df["orientation_label"])
    true_angles = test_df["orientation_deg"].to_numpy(dtype=float)

    orient_only_model, orient_only_scaler = train_model(
        X_train,
        y_orient_train,
        None,
        seed=seed,
        multitask=False,
        cell_weight=cell_weight,
    )
    multitask_model, multitask_scaler = train_model(
        X_train,
        y_orient_train,
        y_cell_train,
        seed=seed,
        multitask=True,
        cell_weight=cell_weight,
    )

    _, orient_only_pred, orient_only_probs = predict_model(orient_only_model, orient_only_scaler, X_test)
    multitask_cell_pred, multitask_orient_pred, multitask_probs = predict_model(multitask_model, multitask_scaler, X_test)
    if multitask_cell_pred is None:
        raise RuntimeError("Multi-task model did not return cell predictions.")

    orient_only_angles, orient_only_labels = orientation_probs_to_angles(orient_only_probs, orient_encoder)
    multitask_angles, multitask_labels = orientation_probs_to_angles(multitask_probs, orient_encoder)

    pred_df = test_df[
        ["campaign", "grid_cell", "orientation_label", "orientation_deg"]
    ].copy()
    pred_df["seed"] = seed
    pred_df["orient_only_pred_label"] = orient_encoder.inverse_transform(orient_only_pred)
    pred_df["orient_only_pred_angle_deg"] = orient_only_angles
    pred_df["orient_only_pred_label_circular"] = orient_only_labels
    pred_df["multitask_pred_label"] = orient_encoder.inverse_transform(multitask_orient_pred)
    pred_df["multitask_pred_angle_deg"] = multitask_angles
    pred_df["multitask_pred_label_circular"] = multitask_labels
    pred_df["multitask_pred_cell"] = cell_encoder.inverse_transform(multitask_cell_pred)

    return SplitResult(
        seed=seed,
        orient_only_acc=float(accuracy_score(y_orient_test, orient_only_pred)),
        orient_only_f1_macro=float(f1_score(y_orient_test, orient_only_pred, average="macro")),
        orient_only_mean_ang_err=mean_angular_error_deg(true_angles, orient_only_angles),
        multitask_orient_acc=float(accuracy_score(y_orient_test, multitask_orient_pred)),
        multitask_orient_f1_macro=float(f1_score(y_orient_test, multitask_orient_pred, average="macro")),
        multitask_mean_ang_err=mean_angular_error_deg(true_angles, multitask_angles),
        multitask_cell_acc=float(accuracy_score(y_cell_test, multitask_cell_pred)),
        multitask_cell_f1_macro=float(f1_score(y_cell_test, multitask_cell_pred, average="macro")),
        predictions=pred_df,
    )


def run_seed(df: pd.DataFrame, *, seed: int, cell_encoder: LabelEncoder, orient_encoder: LabelEncoder) -> SplitResult:
    return run_seed_with_weight(
        df,
        seed=seed,
        cell_weight=0.5,
        cell_encoder=cell_encoder,
        orient_encoder=orient_encoder,
    )


def save_confusion_plot(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted orientation")
    ax.set_ylabel("True orientation")
    ax.set_title("E102 orientation confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarise(results: list[SplitResult], pred_df: pd.DataFrame) -> dict:
    orient_only_acc = np.array([r.orient_only_acc for r in results], dtype=float)
    orient_only_f1 = np.array([r.orient_only_f1_macro for r in results], dtype=float)
    orient_only_ang = np.array([r.orient_only_mean_ang_err for r in results], dtype=float)
    multitask_acc = np.array([r.multitask_orient_acc for r in results], dtype=float)
    multitask_f1 = np.array([r.multitask_orient_f1_macro for r in results], dtype=float)
    multitask_ang = np.array([r.multitask_mean_ang_err for r in results], dtype=float)
    cell_acc = np.array([r.multitask_cell_acc for r in results], dtype=float)
    cell_f1 = np.array([r.multitask_cell_f1_macro for r in results], dtype=float)

    cm = confusion_matrix(
        pred_df["orientation_label"],
        pred_df["multitask_pred_label_circular"],
        labels=ORIENTATION_LABELS,
    )

    campaign_rows = []
    for campaign, group in pred_df.groupby("campaign", sort=True):
        campaign_rows.append(
            {
                "campaign": campaign,
                "samples": int(len(group)),
                "orient_only_accuracy": float(
                    accuracy_score(group["orientation_label"], group["orient_only_pred_label_circular"])
                ),
                "multitask_orientation_accuracy": float(
                    accuracy_score(group["orientation_label"], group["multitask_pred_label_circular"])
                ),
                "multitask_cell_accuracy": float(
                    accuracy_score(group["grid_cell"], group["multitask_pred_cell"])
                ),
                "orient_only_mean_angular_error_deg": float(
                    np.mean(
                        [
                            circular_distance_degrees(t, p)
                            for t, p in zip(group["orientation_deg"], group["orient_only_pred_angle_deg"], strict=False)
                        ]
                    )
                ),
                "multitask_mean_angular_error_deg": float(
                    np.mean(
                        [
                            circular_distance_degrees(t, p)
                            for t, p in zip(group["orientation_deg"], group["multitask_pred_angle_deg"], strict=False)
                        ]
                    )
                ),
            }
        )

    return {
        "dataset": {
            "samples": int(len(pred_df) / len(results)),
            "test_predictions_total": int(len(pred_df)),
            "seeds": [int(r.seed) for r in results],
            "orientation_labels": ORIENTATION_LABELS,
            "orientation_degrees": {label: orientation_label_to_degrees(label) for label in ORIENTATION_LABELS},
        },
        "orientation_task": {
            "orient_only_accuracy_mean": float(orient_only_acc.mean()),
            "orient_only_accuracy_std": float(orient_only_acc.std(ddof=0)),
            "orient_only_f1_macro_mean": float(orient_only_f1.mean()),
            "orient_only_mean_angular_error_deg": float(orient_only_ang.mean()),
            "multitask_accuracy_mean": float(multitask_acc.mean()),
            "multitask_accuracy_std": float(multitask_acc.std(ddof=0)),
            "multitask_f1_macro_mean": float(multitask_f1.mean()),
            "multitask_mean_angular_error_deg": float(multitask_ang.mean()),
            "accuracy_delta_abs_mean": float((multitask_acc - orient_only_acc).mean()),
            "angular_error_delta_deg_mean": float((multitask_ang - orient_only_ang).mean()),
            "confusion_matrix_multitask_circular": cm.tolist(),
        },
        "cell_task": {
            "multitask_cell_accuracy_mean": float(cell_acc.mean()),
            "multitask_cell_accuracy_std": float(cell_acc.std(ddof=0)),
            "multitask_cell_f1_macro_mean": float(cell_f1.mean()),
        },
        "per_campaign": campaign_rows,
    }


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_e102_dataframe()
    cell_encoder = LabelEncoder().fit(df["grid_cell"])
    orient_encoder = LabelEncoder().fit(ORIENTATION_LABELS)
    seeds = [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]

    print(
        f"Loaded E102 orientation: {len(df)} samples | cells={df['grid_cell'].nunique()} | "
        f"campaigns={df['campaign'].nunique()} | orientations={sorted(df['orientation_label'].unique())}"
    )

    results: list[SplitResult] = []
    pred_frames: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"Running seed {seed}...")
        split_result = run_seed_with_weight(
            df,
            seed=seed,
            cell_weight=float(args.cell_weight),
            cell_encoder=cell_encoder,
            orient_encoder=orient_encoder,
        )
        results.append(split_result)
        pred_frames.append(split_result.predictions)
        print(
            "  "
            f"orient_only_acc={split_result.orient_only_acc:.3f} "
            f"multitask_orient_acc={split_result.multitask_orient_acc:.3f} "
            f"multitask_cell_acc={split_result.multitask_cell_acc:.3f} "
            f"orient_only_mae={split_result.orient_only_mean_ang_err:.1f}deg "
            f"multitask_mae={split_result.multitask_mean_ang_err:.1f}deg"
        )

    pred_df = pd.concat(pred_frames, ignore_index=True)
    summary = summarise(results, pred_df)
    summary["cell_weight"] = float(args.cell_weight)

    prefix = str(args.output_prefix).strip() or "e102_orientation_multitask"
    pred_path = REPORT_DIR / f"{prefix}_predictions.csv"
    json_path = REPORT_DIR / f"{prefix}_summary.json"
    cm_csv_path = REPORT_DIR / f"{prefix}_confusion_matrix.csv"
    cm_png_path = REPORT_DIR / f"{prefix}_confusion_matrix.png"
    campaign_csv_path = REPORT_DIR / f"{prefix}_per_campaign.csv"

    pred_df.to_csv(pred_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))

    cm_df = pd.DataFrame(
        summary["orientation_task"]["confusion_matrix_multitask_circular"],
        index=ORIENTATION_LABELS,
        columns=ORIENTATION_LABELS,
    )
    cm_df.to_csv(cm_csv_path)
    save_confusion_plot(cm_df.to_numpy(), ORIENTATION_LABELS, cm_png_path)

    pd.DataFrame(summary["per_campaign"]).sort_values("campaign").to_csv(campaign_csv_path, index=False)

    print("\nSummary")
    print(
        f"  Orientation-only acc={summary['orientation_task']['orient_only_accuracy_mean']:.3f} "
        f"| Multitask acc={summary['orientation_task']['multitask_accuracy_mean']:.3f} "
        f"| Delta={summary['orientation_task']['accuracy_delta_abs_mean']:+.3f}"
    )
    print(
        f"  Orientation-only mean angular error={summary['orientation_task']['orient_only_mean_angular_error_deg']:.1f}deg "
        f"| Multitask mean angular error={summary['orientation_task']['multitask_mean_angular_error_deg']:.1f}deg "
        f"| Delta={summary['orientation_task']['angular_error_delta_deg_mean']:+.1f}deg"
    )
    print(
        f"  Multitask cell acc={summary['cell_task']['multitask_cell_accuracy_mean']:.3f} "
        f"| cell F1={summary['cell_task']['multitask_cell_f1_macro_mean']:.3f}"
    )
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
