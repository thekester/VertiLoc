#!/usr/bin/env python3
"""Benchmark a fused cell+orientation+geometric-height model on E101."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.board_geometry import BoardGeometry, add_board_geometry  # noqa: E402
from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.orientation import (  # noqa: E402
    circular_distance_degrees,
    nearest_orientation_label,
    orientation_label_to_degrees,
)

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
ROUTER_HEIGHT_M = 0.75
ORIENTATION_LABELS = ["back", "right", "front", "left"]
CAMPAIGNS: dict[str, tuple[CampaignSpec, str]] = {
    "E101/back": (CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "back", router_distance_m=3.0), "back"),
    "E101/front": (CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "front", router_distance_m=3.0), "front"),
    "E101/left": (CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "left", router_distance_m=3.0), "left"),
    "E101/right": (CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "right", router_distance_m=3.0), "right"),
}


@dataclass
class SplitResult:
    seed: int
    cell_accuracy: float
    cell_f1_macro: float
    cell_mean_distance_m: float
    orientation_accuracy: float
    orientation_f1_macro: float
    orientation_mean_angular_error_deg: float
    elevation_mae_raw_m: float
    elevation_mae_clamped_m: float
    predictions: pd.DataFrame


class SharedMLP:
    def __init__(self, input_dim: int, cell_classes: int, orient_classes: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, 0.15, size=(input_dim, 64)),
            "b1": np.zeros(64, dtype=np.float64),
            "W2": rng.normal(0.0, 0.15, size=(64, 32)),
            "b2": np.zeros(32, dtype=np.float64),
            "Wc": rng.normal(0.0, 0.15, size=(32, cell_classes)),
            "bc": np.zeros(cell_classes, dtype=np.float64),
            "Wo": rng.normal(0.0, 0.15, size=(32, orient_classes)),
            "bo": np.zeros(orient_classes, dtype=np.float64),
            "Wh": rng.normal(0.0, 0.15, size=(32, 1)),
            "bh": np.zeros(1, dtype=np.float64),
        }

    def copy_state(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.params.items()}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.params = {name: value.copy() for name, value in state.items()}

    def forward(self, X: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.maximum(z2, 0.0)
        cell_logits = a2 @ self.params["Wc"] + self.params["bc"]
        orient_logits = a2 @ self.params["Wo"] + self.params["bo"]
        height_pred = a2 @ self.params["Wh"] + self.params["bh"]
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return cache, cell_logits, orient_logits, height_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-weight", type=float, default=0.2, help="Relative weight of the cell loss.")
    parser.add_argument(
        "--orientation-weight", type=float, default=1.0, help="Relative weight of the orientation loss."
    )
    parser.add_argument("--height-weight", type=float, default=0.35, help="Relative weight of the height regression loss.")
    parser.add_argument("--seeds", default="7,17,27,37,47", help="Comma-separated random seeds.")
    parser.add_argument("--output-prefix", default="e101_fused_cell_orientation_geometry", help="Artifact prefix.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_e101_dataframe() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    geometry = BoardGeometry()
    for campaign_name, (spec, orientation_label) in CAMPAIGNS.items():
        df = load_measurements([spec]).copy()
        df["campaign"] = campaign_name
        df["orientation_label"] = orientation_label
        df["orientation_deg"] = orientation_label_to_degrees(orientation_label)
        df = add_board_geometry(
            df,
            geometry=geometry,
            router_height_m=ROUTER_HEIGHT_M,
            grid_x_top_is_zero=True,
            coordinate_mode="fit_to_data",
        )
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


def mse_and_grad(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    diff = pred.reshape(-1) - target.reshape(-1)
    loss = float(np.mean(diff * diff))
    grad = (2.0 * diff / len(diff)).reshape(-1, 1)
    return loss, grad


def backward_shared(
    model: SharedMLP,
    cache: dict[str, np.ndarray],
    grad_cell_logits: np.ndarray,
    grad_orient_logits: np.ndarray,
    grad_height: np.ndarray,
) -> dict[str, np.ndarray]:
    grads: dict[str, np.ndarray] = {}
    a2 = cache["a2"]
    grads["Wc"] = a2.T @ grad_cell_logits
    grads["bc"] = grad_cell_logits.sum(axis=0)
    grads["Wo"] = a2.T @ grad_orient_logits
    grads["bo"] = grad_orient_logits.sum(axis=0)
    grads["Wh"] = a2.T @ grad_height
    grads["bh"] = grad_height.sum(axis=0)
    grad_a2 = (
        grad_cell_logits @ model.params["Wc"].T
        + grad_orient_logits @ model.params["Wo"].T
        + grad_height @ model.params["Wh"].T
    )
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
    y_cell_train: np.ndarray,
    y_orient_train: np.ndarray,
    y_height_train: np.ndarray,
    *,
    seed: int,
    cell_weight: float,
    orient_weight: float,
    height_weight: float,
    max_epochs: int = 150,
    patience: int = 18,
    batch_size: int = 256,
) -> tuple[SharedMLP, StandardScaler]:
    set_seed(seed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    stratify_target = np.array(
        [f"{c}|{o}|{round(h, 2)}" for c, o, h in zip(y_cell_train, y_orient_train, y_height_train, strict=False)]
    )
    idx = np.arange(len(X_scaled))
    idx_train, idx_val = train_test_split(
        idx, test_size=0.1, random_state=seed, stratify=stratify_target
    )

    model = SharedMLP(
        input_dim=X_scaled.shape[1],
        cell_classes=int(np.max(y_cell_train) + 1),
        orient_classes=int(np.max(y_orient_train) + 1),
        seed=seed,
    )
    opt_state: dict[str, dict[str, np.ndarray] | int] = {
        "step": 0,
        "m": {name: np.zeros_like(value) for name, value in model.params.items()},
        "v": {name: np.zeros_like(value) for name, value in model.params.items()},
    }

    X_fit = X_scaled[idx_train]
    y_cell_fit = y_cell_train[idx_train]
    y_orient_fit = y_orient_train[idx_train]
    y_height_fit = y_height_train[idx_train]
    X_val = X_scaled[idx_val]
    y_cell_val = y_cell_train[idx_val]
    y_orient_val = y_orient_train[idx_val]
    y_height_val = y_height_train[idx_val]

    best_state: dict[str, np.ndarray] | None = None
    best_val = float("inf")
    wait = 0

    for _ in range(max_epochs):
        order = np.random.permutation(len(X_fit))
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            xb = X_fit[batch_ids]
            yb_cell = y_cell_fit[batch_ids]
            yb_orient = y_orient_fit[batch_ids]
            yb_height = y_height_fit[batch_ids]

            cache, cell_logits, orient_logits, height_pred = model.forward(xb)
            _, grad_cell_raw = cross_entropy_and_grad(cell_logits, yb_cell)
            _, grad_orient_raw = cross_entropy_and_grad(orient_logits, yb_orient)
            _, grad_height_raw = mse_and_grad(height_pred, yb_height)
            grads = backward_shared(
                model,
                cache,
                cell_weight * grad_cell_raw,
                orient_weight * grad_orient_raw,
                height_weight * grad_height_raw,
            )
            adam_update(model.params, grads, opt_state)

        _, cell_val_logits, orient_val_logits, height_val_pred = model.forward(X_val)
        cell_val_loss, _ = cross_entropy_and_grad(cell_val_logits, y_cell_val)
        orient_val_loss, _ = cross_entropy_and_grad(orient_val_logits, y_orient_val)
        height_val_loss, _ = mse_and_grad(height_val_pred, y_height_val)
        val = cell_weight * cell_val_loss + orient_weight * orient_val_loss + height_weight * height_val_loss

        if val < best_val - 1e-4:
            best_val = val
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_scaled = scaler.transform(X)
    _, cell_logits, orient_logits, height_pred = model.forward(X_scaled)
    cell_pred = cell_logits.argmax(axis=1)
    orient_probs = softmax(orient_logits)
    return cell_pred, orient_probs, height_pred.reshape(-1)


def orientation_probs_to_angles(orient_probs: np.ndarray, orient_encoder: LabelEncoder) -> tuple[np.ndarray, np.ndarray]:
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
    return float(np.mean([circular_distance_degrees(t, p) for t, p in zip(true_angles, pred_angles, strict=False)]))


def clamp_heights(height_m: np.ndarray, geometry: BoardGeometry) -> np.ndarray:
    return np.clip(height_m, geometry.board_bottom_height_m, geometry.board_top_height_m)


def run_seed(
    df: pd.DataFrame,
    *,
    seed: int,
    cell_weight: float,
    orient_weight: float,
    height_weight: float,
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
    y_height_train = train_df["esp_height_m"].to_numpy(dtype=np.float32)

    model, scaler = train_model(
        X_train,
        y_cell_train,
        y_orient_train,
        y_height_train,
        seed=seed,
        cell_weight=cell_weight,
        orient_weight=orient_weight,
        height_weight=height_weight,
    )
    cell_pred, orient_probs, height_pred_raw = predict_model(model, scaler, X_test)
    pred_cells = cell_encoder.inverse_transform(cell_pred)
    pred_angles, pred_orient_labels = orientation_probs_to_angles(orient_probs, orient_encoder)

    geometry = BoardGeometry()
    true_height_raw = test_df["esp_height_m"].to_numpy(dtype=float)
    true_height_clamped = test_df["board_z_clamped_m"].to_numpy(dtype=float)
    pred_height_clamped = clamp_heights(height_pred_raw, geometry)

    cell_meta = (
        train_df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )
    pred_coords = cell_meta.loc[pred_cells, ["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    true_coords = test_df[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    cell_errors_m = np.linalg.norm(true_coords - pred_coords, axis=1)

    pred_df = test_df[
        [
            "campaign",
            "grid_cell",
            "grid_x",
            "grid_y",
            "coord_x_m",
            "coord_y_m",
            "orientation_label",
            "orientation_deg",
            "esp_height_m",
            "board_z_clamped_m",
            "is_within_board",
        ]
    ].copy()
    pred_df["seed"] = seed
    pred_df["pred_cell"] = pred_cells
    pred_df["pred_orientation_label"] = pred_orient_labels
    pred_df["pred_orientation_deg"] = pred_angles
    pred_df["pred_height_raw_m"] = height_pred_raw
    pred_df["pred_height_clamped_m"] = pred_height_clamped
    pred_df["cell_error_m"] = cell_errors_m
    pred_df["angular_error_deg"] = [
        circular_distance_degrees(t, p)
        for t, p in zip(pred_df["orientation_deg"], pred_df["pred_orientation_deg"], strict=False)
    ]
    pred_df["height_error_raw_m"] = np.abs(true_height_raw - height_pred_raw)
    pred_df["height_error_clamped_m"] = np.abs(true_height_clamped - pred_height_clamped)

    return SplitResult(
        seed=seed,
        cell_accuracy=float(accuracy_score(y_cell_test, cell_pred)),
        cell_f1_macro=float(f1_score(y_cell_test, cell_pred, average="macro")),
        cell_mean_distance_m=float(cell_errors_m.mean()),
        orientation_accuracy=float(accuracy_score(test_df["orientation_label"], pred_orient_labels)),
        orientation_f1_macro=float(f1_score(test_df["orientation_label"], pred_orient_labels, average="macro")),
        orientation_mean_angular_error_deg=mean_angular_error_deg(
            test_df["orientation_deg"].to_numpy(dtype=float),
            pred_angles,
        ),
        elevation_mae_raw_m=float(pred_df["height_error_raw_m"].mean()),
        elevation_mae_clamped_m=float(pred_df["height_error_clamped_m"].mean()),
        predictions=pred_df,
    )


def summarise(results: list[SplitResult], pred_df: pd.DataFrame) -> dict:
    cell_acc = np.asarray([r.cell_accuracy for r in results], dtype=float)
    cell_f1 = np.asarray([r.cell_f1_macro for r in results], dtype=float)
    cell_dist = np.asarray([r.cell_mean_distance_m for r in results], dtype=float)
    orient_acc = np.asarray([r.orientation_accuracy for r in results], dtype=float)
    orient_f1 = np.asarray([r.orientation_f1_macro for r in results], dtype=float)
    orient_ang = np.asarray([r.orientation_mean_angular_error_deg for r in results], dtype=float)
    elev_raw = np.asarray([r.elevation_mae_raw_m for r in results], dtype=float)
    elev_clamped = np.asarray([r.elevation_mae_clamped_m for r in results], dtype=float)
    orient_cm = confusion_matrix(
        pred_df["orientation_label"],
        pred_df["pred_orientation_label"],
        labels=ORIENTATION_LABELS,
    )

    per_campaign = []
    for campaign, group in pred_df.groupby("campaign", sort=True):
        per_campaign.append(
            {
                "campaign": campaign,
                "samples": int(len(group)),
                "cell_accuracy": float(accuracy_score(group["grid_cell"], group["pred_cell"])),
                "cell_mean_distance_m": float(group["cell_error_m"].mean()),
                "orientation_accuracy": float(accuracy_score(group["orientation_label"], group["pred_orientation_label"])),
                "orientation_f1_macro": float(
                    f1_score(group["orientation_label"], group["pred_orientation_label"], labels=ORIENTATION_LABELS, average="macro", zero_division=0)
                ),
                "orientation_mean_angular_error_deg": float(group["angular_error_deg"].mean()),
                "height_mae_raw_m": float(group["height_error_raw_m"].mean()),
                "height_mae_clamped_m": float(group["height_error_clamped_m"].mean()),
                "within_board_ratio": float(group["is_within_board"].mean()),
            }
        )

    return {
        "dataset": {
            "room": "E101",
            "samples": int(len(pred_df) / len(results)),
            "test_predictions_total": int(len(pred_df)),
            "seeds": [int(r.seed) for r in results],
            "campaigns": sorted(CAMPAIGNS),
            "router_height_m": ROUTER_HEIGHT_M,
            "orientation_labels": ORIENTATION_LABELS,
            "orientation_degrees": {label: orientation_label_to_degrees(label) for label in ORIENTATION_LABELS},
            "height_note": "Elevation target is geometric ESP height derived from board projection; both raw and clamped errors are reported.",
        },
        "cell_task": {
            "accuracy_mean": float(cell_acc.mean()),
            "accuracy_std": float(cell_acc.std(ddof=0)),
            "f1_macro_mean": float(cell_f1.mean()),
            "f1_macro_std": float(cell_f1.std(ddof=0)),
            "mean_distance_m": float(cell_dist.mean()),
            "mean_distance_std_m": float(cell_dist.std(ddof=0)),
        },
        "orientation_task": {
            "accuracy_mean": float(orient_acc.mean()),
            "accuracy_std": float(orient_acc.std(ddof=0)),
            "f1_macro_mean": float(orient_f1.mean()),
            "f1_macro_std": float(orient_f1.std(ddof=0)),
            "mean_angular_error_deg": float(orient_ang.mean()),
            "mean_angular_error_std_deg": float(orient_ang.std(ddof=0)),
            "confusion_matrix": orient_cm.tolist(),
        },
        "height_task": {
            "mae_raw_m_mean": float(elev_raw.mean()),
            "mae_raw_m_std": float(elev_raw.std(ddof=0)),
            "mae_clamped_m_mean": float(elev_clamped.mean()),
            "mae_clamped_m_std": float(elev_clamped.std(ddof=0)),
        },
        "per_campaign": per_campaign,
    }


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_e101_dataframe()
    cell_encoder = LabelEncoder().fit(df["grid_cell"])
    orient_encoder = LabelEncoder().fit(ORIENTATION_LABELS)
    seeds = [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]

    print(
        f"Loaded E101 fused benchmark: samples={len(df)} | cells={df['grid_cell'].nunique()} | "
        f"campaigns={df['campaign'].nunique()} | orientations={sorted(df['orientation_label'].unique())} | "
        f"router_height={ROUTER_HEIGHT_M:.2f}m"
    )

    results: list[SplitResult] = []
    pred_frames: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"Running seed {seed}...")
        split_result = run_seed(
            df,
            seed=seed,
            cell_weight=float(args.cell_weight),
            orient_weight=float(args.orientation_weight),
            height_weight=float(args.height_weight),
            cell_encoder=cell_encoder,
            orient_encoder=orient_encoder,
        )
        results.append(split_result)
        pred_frames.append(split_result.predictions)
        print(
            "  "
            f"cell_acc={split_result.cell_accuracy:.3f} "
            f"orient_acc={split_result.orientation_accuracy:.3f} "
            f"orient_err={split_result.orientation_mean_angular_error_deg:.1f}deg "
            f"height_mae_raw={split_result.elevation_mae_raw_m:.3f}m "
            f"height_mae_clamped={split_result.elevation_mae_clamped_m:.3f}m"
        )

    pred_df = pd.concat(pred_frames, ignore_index=True)
    summary = summarise(results, pred_df)
    summary["weights"] = {
        "cell_weight": float(args.cell_weight),
        "orientation_weight": float(args.orientation_weight),
        "height_weight": float(args.height_weight),
    }

    prefix = str(args.output_prefix).strip() or "e101_fused_cell_orientation_geometry"
    pred_path = REPORT_DIR / f"{prefix}_predictions.csv"
    summary_path = REPORT_DIR / f"{prefix}_summary.json"
    orient_cm_path = REPORT_DIR / f"{prefix}_orientation_confusion_matrix.csv"
    campaign_path = REPORT_DIR / f"{prefix}_per_campaign.csv"
    snippet_path = REPORT_DIR / f"{prefix}_snippet.md"

    pred_df.to_csv(pred_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        summary["orientation_task"]["confusion_matrix"],
        index=ORIENTATION_LABELS,
        columns=ORIENTATION_LABELS,
    ).to_csv(orient_cm_path)
    pd.DataFrame(summary["per_campaign"]).sort_values("campaign").to_csv(campaign_path, index=False)

    snippet = (
        "### Benchmark fusionne E101: cellule + orientation + hauteur geometrique\n\n"
        f"- Cellule: accuracy={summary['cell_task']['accuracy_mean']:.3f}, "
        f"F1 macro={summary['cell_task']['f1_macro_mean']:.3f}, "
        f"erreur moyenne={summary['cell_task']['mean_distance_m']:.3f} m.\n"
        f"- Orientation: accuracy={summary['orientation_task']['accuracy_mean']:.3f}, "
        f"F1 macro={summary['orientation_task']['f1_macro_mean']:.3f}, "
        f"erreur angulaire moyenne={summary['orientation_task']['mean_angular_error_deg']:.1f} deg.\n"
        f"- Hauteur geometrique: MAE brute={summary['height_task']['mae_raw_m_mean']:.3f} m, "
        f"MAE clamp={summary['height_task']['mae_clamped_m_mean']:.3f} m.\n"
        f"- Routeur fixe a {ROUTER_HEIGHT_M:.2f} m au-dessus du sol.\n"
    )
    snippet_path.write_text(snippet)

    print("\nSummary")
    print(json.dumps(summary["cell_task"], indent=2))
    print(json.dumps(summary["orientation_task"], indent=2))
    print(json.dumps(summary["height_task"], indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
