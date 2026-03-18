#!/usr/bin/env python3
"""Benchmark a shared RSSI encoder with an auxiliary elevation head on E102.

The protocol compares:
1. a cell-only MLP trained from the 5 RSSI features;
2. a multi-task MLP trained on the same inputs with two heads:
   - grid cell classification,
   - elevation class classification.

Elevation labels are defined from the acquisition campaign:
  - exp5 -> 0.00 m (router on the floor)
  - exp6 -> 1.50 m
  - exp1..exp4 -> 0.75 m (reference height)
"""

from __future__ import annotations

import json
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

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"

CAMPAIGNS: dict[str, CampaignSpec] = {
    "exp1_back_right": CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp1", router_distance_m=4.0),
    "exp2_front_right": CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp2", router_distance_m=4.0),
    "exp3_front_left": CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp3", router_distance_m=4.0),
    "exp4_back_left": CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp4", router_distance_m=4.0),
    "exp5_ground": CampaignSpec(PROJECT_ROOT / "data" / "E102" / "elevation" / "exp5", router_distance_m=4.0),
    "exp6_1m50": CampaignSpec(PROJECT_ROOT / "data" / "E102" / "elevation" / "exp6", router_distance_m=4.0),
}

ELEVATION_BY_CAMPAIGN = {
    "exp1_back_right": 0.75,
    "exp2_front_right": 0.75,
    "exp3_front_left": 0.75,
    "exp4_back_left": 0.75,
    "exp5_ground": 0.00,
    "exp6_1m50": 1.50,
}
ELEVATION_LABELS = ["0.00m", "0.75m", "1.50m"]


@dataclass
class SplitResult:
    seed: int
    baseline_cell_acc: float
    multitask_cell_acc: float
    elevation_acc: float
    elevation_f1_macro: float
    baseline_cell_f1_macro: float
    multitask_cell_f1_macro: float
    predictions: pd.DataFrame


class SharedMLP:
    def __init__(self, input_dim: int, cell_classes: int, elevation_classes: int | None, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, 0.15, size=(input_dim, 32)),
            "b1": np.zeros(32, dtype=np.float64),
            "W2": rng.normal(0.0, 0.15, size=(32, 16)),
            "b2": np.zeros(16, dtype=np.float64),
            "Wc": rng.normal(0.0, 0.15, size=(16, cell_classes)),
            "bc": np.zeros(cell_classes, dtype=np.float64),
        }
        if elevation_classes is not None:
            self.params["We"] = rng.normal(0.0, 0.15, size=(16, elevation_classes))
            self.params["be"] = np.zeros(elevation_classes, dtype=np.float64)

    def copy_state(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.params = {k: v.copy() for k, v in state.items()}

    def forward(self, X: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray | None]:
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.maximum(z2, 0.0)
        cell_logits = a2 @ self.params["Wc"] + self.params["bc"]
        elevation_logits = None
        if "We" in self.params:
            elevation_logits = a2 @ self.params["We"] + self.params["be"]
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return cache, cell_logits, elevation_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_e102_dataframe() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for campaign_name, spec in CAMPAIGNS.items():
        df = load_measurements([spec])
        df["campaign"] = campaign_name
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    df_all["elevation_m"] = df_all["campaign"].map(ELEVATION_BY_CAMPAIGN)
    df_all["elevation_class"] = df_all["elevation_m"].map(
        {0.0: "0.00m", 0.75: "0.75m", 1.5: "1.50m"}
    )
    return df_all


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_and_grad(
    logits: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, np.ndarray]:
    probs = softmax(logits)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    loss = -np.mean(np.sum(y_onehot * np.log(np.clip(probs, 1e-9, 1.0)), axis=1))
    grad = (probs - y_onehot) / len(y_true)
    return float(loss), grad


def backward_shared(
    model: SharedMLP,
    cache: dict[str, np.ndarray],
    grad_cell_logits: np.ndarray,
    grad_elev_logits: np.ndarray | None,
) -> dict[str, np.ndarray]:
    grads: dict[str, np.ndarray] = {}
    a2 = cache["a2"]
    grads["Wc"] = a2.T @ grad_cell_logits
    grads["bc"] = grad_cell_logits.sum(axis=0)
    grad_a2 = grad_cell_logits @ model.params["Wc"].T

    if grad_elev_logits is not None and "We" in model.params:
        grads["We"] = a2.T @ grad_elev_logits
        grads["be"] = grad_elev_logits.sum(axis=0)
        grad_a2 = grad_a2 + grad_elev_logits @ model.params["We"].T

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
    y_elev_train: np.ndarray | None,
    *,
    seed: int,
    multitask: bool,
    elevation_weight: float = 0.4,
    max_epochs: int = 120,
    patience: int = 15,
    batch_size: int = 256,
) -> tuple[SharedMLP, StandardScaler]:
    set_seed(seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    stratify_target = y_cell_train if y_elev_train is None else np.array(
        [f"{c}_{e}" for c, e in zip(y_cell_train, y_elev_train, strict=False)]
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
        cell_classes=int(np.max(y_cell_train) + 1),
        elevation_classes=int(np.max(y_elev_train) + 1) if y_elev_train is not None else None,
        seed=seed,
    )
    opt_state: dict[str, dict[str, np.ndarray] | int] = {
        "step": 0,
        "m": {name: np.zeros_like(value) for name, value in model.params.items()},
        "v": {name: np.zeros_like(value) for name, value in model.params.items()},
    }

    X_fit = X_train_scaled[idx_train]
    y_cell_fit = y_cell_train[idx_train]
    y_elev_fit = y_elev_train[idx_train] if y_elev_train is not None else None
    X_val = X_train_scaled[idx_val]
    y_cell_val = y_cell_train[idx_val]
    y_elev_val = y_elev_train[idx_val] if y_elev_train is not None else None

    best_state: dict[str, np.ndarray] | None = None
    best_val = float("inf")
    wait = 0

    for _ in range(max_epochs):
        order = np.random.permutation(len(X_fit))
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            xb = X_fit[batch_ids]
            yb_cell = y_cell_fit[batch_ids]
            yb_elev = y_elev_fit[batch_ids] if y_elev_fit is not None else None

            cache, cell_logits, elev_logits = model.forward(xb)
            _, grad_cell = cross_entropy_and_grad(cell_logits, yb_cell)
            grad_elev = None
            if multitask and yb_elev is not None and elev_logits is not None:
                _, grad_elev_raw = cross_entropy_and_grad(elev_logits, yb_elev)
                grad_elev = elevation_weight * grad_elev_raw
            grads = backward_shared(model, cache, grad_cell, grad_elev)
            adam_update(model.params, grads, opt_state)

        _, cell_logits_val, elev_logits_val = model.forward(X_val)
        cell_val_loss, _ = cross_entropy_and_grad(cell_logits_val, y_cell_val)
        val_value = cell_val_loss
        if multitask and y_elev_val is not None and elev_logits_val is not None:
            elev_val_loss, _ = cross_entropy_and_grad(elev_logits_val, y_elev_val)
            val_value = val_value + elevation_weight * elev_val_loss

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
) -> tuple[np.ndarray, np.ndarray | None]:
    X_scaled = scaler.transform(X)
    _, cell_logits, elev_logits = model.forward(X_scaled)
    cell_pred = cell_logits.argmax(axis=1)
    elev_pred = elev_logits.argmax(axis=1) if elev_logits is not None else None
    return cell_pred, elev_pred


def run_seed(
    df: pd.DataFrame,
    *,
    seed: int,
    cell_encoder: LabelEncoder,
    elev_encoder: LabelEncoder,
) -> SplitResult:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    stratify_labels = df["grid_cell"] + "|" + df["elevation_class"]
    train_idx, test_idx = next(splitter.split(df, stratify_labels))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_cell_train = cell_encoder.transform(train_df["grid_cell"])
    y_cell_test = cell_encoder.transform(test_df["grid_cell"])
    y_elev_train = elev_encoder.transform(train_df["elevation_class"])
    y_elev_test = elev_encoder.transform(test_df["elevation_class"])

    baseline_model, baseline_scaler = train_model(
        X_train,
        y_cell_train,
        None,
        seed=seed,
        multitask=False,
    )
    multitask_model, multitask_scaler = train_model(
        X_train,
        y_cell_train,
        y_elev_train,
        seed=seed,
        multitask=True,
    )

    baseline_cell_pred, _ = predict_model(baseline_model, baseline_scaler, X_test)
    multitask_cell_pred, multitask_elev_pred = predict_model(multitask_model, multitask_scaler, X_test)
    if multitask_elev_pred is None:
        raise RuntimeError("Multi-task model did not return elevation predictions.")

    pred_df = test_df[
        ["campaign", "grid_cell", "elevation_class", "elevation_m"]
    ].copy()
    pred_df["seed"] = seed
    pred_df["baseline_pred_cell"] = cell_encoder.inverse_transform(baseline_cell_pred)
    pred_df["multitask_pred_cell"] = cell_encoder.inverse_transform(multitask_cell_pred)
    pred_df["pred_elevation_class"] = elev_encoder.inverse_transform(multitask_elev_pred)

    return SplitResult(
        seed=seed,
        baseline_cell_acc=float(accuracy_score(y_cell_test, baseline_cell_pred)),
        multitask_cell_acc=float(accuracy_score(y_cell_test, multitask_cell_pred)),
        elevation_acc=float(accuracy_score(y_elev_test, multitask_elev_pred)),
        elevation_f1_macro=float(f1_score(y_elev_test, multitask_elev_pred, average="macro")),
        baseline_cell_f1_macro=float(f1_score(y_cell_test, baseline_cell_pred, average="macro")),
        multitask_cell_f1_macro=float(f1_score(y_cell_test, multitask_cell_pred, average="macro")),
        predictions=pred_df,
    )


def save_confusion_plot(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted elevation")
    ax.set_ylabel("True elevation")
    ax.set_title("E102 elevation confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarise(results: list[SplitResult], pred_df: pd.DataFrame) -> dict:
    baseline_acc = np.array([r.baseline_cell_acc for r in results], dtype=float)
    multitask_acc = np.array([r.multitask_cell_acc for r in results], dtype=float)
    baseline_f1 = np.array([r.baseline_cell_f1_macro for r in results], dtype=float)
    multitask_f1 = np.array([r.multitask_cell_f1_macro for r in results], dtype=float)
    elev_acc = np.array([r.elevation_acc for r in results], dtype=float)
    elev_f1 = np.array([r.elevation_f1_macro for r in results], dtype=float)

    cm = confusion_matrix(
        pred_df["elevation_class"],
        pred_df["pred_elevation_class"],
        labels=ELEVATION_LABELS,
    )

    campaign_rows = []
    for campaign, group in pred_df.groupby("campaign", sort=True):
        campaign_rows.append(
            {
                "campaign": campaign,
                "samples": int(len(group)),
                "elevation_accuracy": float(
                    accuracy_score(group["elevation_class"], group["pred_elevation_class"])
                ),
                "elevation_f1_macro": float(
                    f1_score(
                        group["elevation_class"],
                        group["pred_elevation_class"],
                        labels=ELEVATION_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "baseline_cell_accuracy": float(
                    accuracy_score(group["grid_cell"], group["baseline_pred_cell"])
                ),
                "multitask_cell_accuracy": float(
                    accuracy_score(group["grid_cell"], group["multitask_pred_cell"])
                ),
            }
        )

    return {
        "dataset": {
            "samples": int(len(pred_df) / len(results)),
            "test_predictions_total": int(len(pred_df)),
            "seeds": [int(r.seed) for r in results],
            "elevation_labels": ELEVATION_LABELS,
        },
        "elevation_task": {
            "accuracy_mean": float(elev_acc.mean()),
            "accuracy_std": float(elev_acc.std(ddof=0)),
            "f1_macro_mean": float(elev_f1.mean()),
            "f1_macro_std": float(elev_f1.std(ddof=0)),
            "confusion_matrix": cm.tolist(),
        },
        "cell_task_impact": {
            "baseline_cell_accuracy_mean": float(baseline_acc.mean()),
            "baseline_cell_accuracy_std": float(baseline_acc.std(ddof=0)),
            "multitask_cell_accuracy_mean": float(multitask_acc.mean()),
            "multitask_cell_accuracy_std": float(multitask_acc.std(ddof=0)),
            "cell_accuracy_delta_abs_mean": float((multitask_acc - baseline_acc).mean()),
            "baseline_cell_f1_macro_mean": float(baseline_f1.mean()),
            "multitask_cell_f1_macro_mean": float(multitask_f1.mean()),
            "cell_f1_delta_abs_mean": float((multitask_f1 - baseline_f1).mean()),
        },
        "per_campaign": campaign_rows,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_e102_dataframe()
    cell_encoder = LabelEncoder().fit(df["grid_cell"])
    elev_encoder = LabelEncoder().fit(ELEVATION_LABELS)
    seeds = [7, 17, 27, 37, 47]

    print(
        f"Loaded E102: {len(df)} samples | cells={df['grid_cell'].nunique()} | "
        f"campaigns={df['campaign'].nunique()} | elevations={sorted(df['elevation_class'].unique())}"
    )

    results: list[SplitResult] = []
    pred_frames: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"Running seed {seed}...")
        split_result = run_seed(df, seed=seed, cell_encoder=cell_encoder, elev_encoder=elev_encoder)
        results.append(split_result)
        pred_frames.append(split_result.predictions)
        print(
            "  "
            f"elev_acc={split_result.elevation_acc:.3f} "
            f"elev_f1={split_result.elevation_f1_macro:.3f} "
            f"cell_base={split_result.baseline_cell_acc:.3f} "
            f"cell_mt={split_result.multitask_cell_acc:.3f}"
        )

    pred_df = pd.concat(pred_frames, ignore_index=True)
    summary = summarise(results, pred_df)

    pred_path = REPORT_DIR / "e102_elevation_multitask_predictions.csv"
    json_path = REPORT_DIR / "e102_elevation_multitask_summary.json"
    cm_csv_path = REPORT_DIR / "e102_elevation_confusion_matrix.csv"
    cm_png_path = REPORT_DIR / "e102_elevation_confusion_matrix.png"
    campaign_csv_path = REPORT_DIR / "e102_elevation_per_campaign.csv"
    snippet_path = REPORT_DIR / "e102_elevation_report_snippet.md"

    pred_df.to_csv(pred_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))

    cm_df = pd.DataFrame(
        summary["elevation_task"]["confusion_matrix"],
        index=ELEVATION_LABELS,
        columns=ELEVATION_LABELS,
    )
    cm_df.to_csv(cm_csv_path)
    save_confusion_plot(cm_df.to_numpy(), ELEVATION_LABELS, cm_png_path)

    campaign_df = pd.DataFrame(summary["per_campaign"]).sort_values("campaign")
    campaign_df.to_csv(campaign_csv_path, index=False)

    snippet = (
        f"### Benchmark auxiliaire elevation (E102)\n\n"
        f"- Tache elevation (3 classes : 0.00 m, 0.75 m, 1.50 m) : "
        f"accuracy moyenne = {summary['elevation_task']['accuracy_mean']:.3f} "
        f"(std {summary['elevation_task']['accuracy_std']:.3f}), "
        f"F1 macro = {summary['elevation_task']['f1_macro_mean']:.3f} "
        f"(std {summary['elevation_task']['f1_macro_std']:.3f}).\n"
        f"- Impact sur la cellule : baseline cell-only = "
        f"{summary['cell_task_impact']['baseline_cell_accuracy_mean']:.3f}, "
        f"multi-task cell+elevation = "
        f"{summary['cell_task_impact']['multitask_cell_accuracy_mean']:.3f}, "
        f"delta absolu moyen = {summary['cell_task_impact']['cell_accuracy_delta_abs_mean']:+.3f}.\n"
        f"- `exp5_ground` (0.00 m) et `exp6_1m50` (1.50 m) sont detailles dans "
        f"`reports/benchmarks/e102_elevation_per_campaign.csv`.\n"
    )
    snippet_path.write_text(snippet)

    print("\nSummary")
    print(
        f"  Elevation accuracy={summary['elevation_task']['accuracy_mean']:.3f} "
        f"| F1 macro={summary['elevation_task']['f1_macro_mean']:.3f}"
    )
    print(
        f"  Cell baseline={summary['cell_task_impact']['baseline_cell_accuracy_mean']:.3f} "
        f"| Cell multitask={summary['cell_task_impact']['multitask_cell_accuracy_mean']:.3f} "
        f"| Delta={summary['cell_task_impact']['cell_accuracy_delta_abs_mean']:+.3f}"
    )
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
