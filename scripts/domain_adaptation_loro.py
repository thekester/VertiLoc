"""Domain adaptation for LORO (Leave-One-Room-Out) generalization.

Implements and compares three approaches:
  1. Vanilla baseline (KNN/RF with no adaptation)
  2. CORAL (Correlation Alignment) — aligns source covariance to target
  3. DANN  (Domain Adversarial Neural Network) — learns room-invariant features

All tested under LORO protocol: train on N-1 rooms, test on the held-out room
using ONLY unlabeled test-room samples for adaptation (unsupervised DA).

References:
  - CORAL: Sun & Saenko, "Return of Frustratingly Easy Domain Adaptation", AAAI 2016
  - DANN: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localization.catalog import BENCHMARK_REPORT_DIR, FEATURE_COLUMNS, filter_room_campaigns
from localization.data import load_measurements

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings("ignore")
REPORT_DIR = BENCHMARK_REPORT_DIR
SEED = 42


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_multiroom() -> pd.DataFrame:
    campaigns = filter_room_campaigns()
    frames = []
    for room, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df = load_measurements([spec])
                df["room"] = room
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _cell_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("grid_cell")[["coord_x_m", "coord_y_m"]].first()


def _localization_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lookup: pd.DataFrame,
) -> dict:
    acc = float((y_true == y_pred).mean())
    pred_coords = lookup.reindex(y_pred)[["coord_x_m", "coord_y_m"]].to_numpy()
    true_coords = lookup.reindex(y_true)[["coord_x_m", "coord_y_m"]].to_numpy()
    mask = ~(np.isnan(pred_coords).any(axis=1) | np.isnan(true_coords).any(axis=1))
    if mask.sum() == 0:
        return {"cell_acc": acc, "mean_error_m": float("nan"), "p90_error_m": float("nan")}
    errors = np.linalg.norm(pred_coords[mask] - true_coords[mask], axis=1)
    return {
        "cell_acc": acc,
        "mean_error_m": float(errors.mean()),
        "p90_error_m": float(np.percentile(errors, 90)),
    }


# ---------------------------------------------------------------------------
# CORAL alignment
# ---------------------------------------------------------------------------

def coral_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Align source features to target covariance/mean (CORAL).

    Maps source distribution N(mu_s, Sigma_s) -> N(mu_t, Sigma_t).
    X_source is modified; X_target is used for statistics only.
    Returns transformed X_source suitable for training on X_target test data.
    """
    mu_s = X_source.mean(axis=0)
    mu_t = X_target.mean(axis=0)
    d = X_source.shape[1]
    eye = np.eye(d)

    cov_s = np.cov(X_source, rowvar=False) + ridge * eye
    cov_t = np.cov(X_target, rowvar=False) + ridge * eye

    # Whitening transform: X -> (X - mu_s) @ Cs^{-1/2}
    L_s = np.linalg.cholesky(cov_s)          # lower triangular
    L_s_inv = np.linalg.inv(L_s)

    # Coloring transform: X_white -> X_white @ Lt^{1/2}
    L_t = np.linalg.cholesky(cov_t)

    # Full transform: whiten source then color with target
    W = L_s_inv.T @ L_t                       # shape (d, d)
    X_aligned = (X_source - mu_s) @ W + mu_t
    return X_aligned


def _coral_predict_with_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 7,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Apply CORAL on training features (aligned to test distribution), then KNN."""
    try:
        X_train_aligned = coral_transform(X_train, X_test, ridge=ridge)
    except np.linalg.LinAlgError:
        X_train_aligned = X_train  # fallback: no alignment
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train_aligned, y_train)
    return knn.predict(X_test)


def _coral_predict_with_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    ridge: float = 1e-3,
    random_state: int = SEED,
) -> np.ndarray:
    try:
        X_train_aligned = coral_transform(X_train, X_test, ridge=ridge)
    except np.linalg.LinAlgError:
        X_train_aligned = X_train
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
    rf.fit(X_train_aligned, y_train)
    return rf.predict(X_test)


# ---------------------------------------------------------------------------
# DANN (Domain Adversarial Neural Network)
# ---------------------------------------------------------------------------

class _GradientReversal(torch.autograd.Function if HAS_TORCH else object):
    """Gradient reversal layer for domain adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None


def _grad_reverse(x, alpha: float = 1.0):
    return _GradientReversal.apply(x, alpha)


if HAS_TORCH:
    class DANNModel(nn.Module):
        def __init__(
            self,
            n_features: int,
            n_classes: int,
            n_domains: int,
            hidden: int = 64,
        ):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.BatchNorm1d(hidden // 2),
                nn.ReLU(),
            )
            self.label_classifier = nn.Sequential(
                nn.Linear(hidden // 2, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, n_classes),
            )
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden // 2, 32),
                nn.ReLU(),
                nn.Linear(32, n_domains),
            )

        def forward(self, x, alpha: float = 1.0):
            features = self.feature_extractor(x)
            label_output = self.label_classifier(features)
            domain_output = self.domain_classifier(_grad_reverse(features, alpha))
            return label_output, domain_output, features


def dann_train_predict(
    X_source: np.ndarray,
    y_source: np.ndarray,
    domain_source: np.ndarray,
    X_target: np.ndarray,
    domain_target: np.ndarray,
    *,
    epochs: int = 150,
    lr: float = 1e-3,
    hidden: int = 64,
    random_state: int = SEED,
    device: str = "cpu",
) -> np.ndarray:
    """Train a DANN on labeled source + unlabeled target, predict on target."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for DANN. Install with: pip install torch")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Encode labels
    label_classes = np.unique(y_source)
    label_to_idx = {lbl: i for i, lbl in enumerate(label_classes)}
    y_enc = np.array([label_to_idx[l] for l in y_source], dtype=np.int64)

    n_features = X_source.shape[1]
    n_classes = len(label_classes)
    n_domains = int(max(np.max(domain_source), np.max(domain_target)) + 1)

    # Standardize
    scaler = StandardScaler()
    X_src_scaled = scaler.fit_transform(X_source).astype(np.float32)
    X_tgt_scaled = scaler.transform(X_target).astype(np.float32)

    model = DANNModel(n_features, n_classes, n_domains, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    label_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    X_src_t = torch.from_numpy(X_src_scaled).to(device)
    y_src_t = torch.from_numpy(y_enc).to(device)
    dom_src_t = torch.from_numpy(domain_source.astype(np.int64)).to(device)
    X_tgt_t = torch.from_numpy(X_tgt_scaled).to(device)
    dom_tgt_t = torch.from_numpy(domain_target.astype(np.int64)).to(device)

    model.train()
    for epoch in range(epochs):
        # Annealed gradient reversal coefficient
        p = float(epoch) / epochs
        alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        optimizer.zero_grad()

        # Source: label loss + domain loss
        label_out, dom_out_src, _ = model(X_src_t, alpha)
        label_loss = label_criterion(label_out, y_src_t)
        domain_loss_src = domain_criterion(dom_out_src, dom_src_t)

        # Target: domain loss only (no labels)
        _, dom_out_tgt, _ = model(X_tgt_t, alpha)
        domain_loss_tgt = domain_criterion(dom_out_tgt, dom_tgt_t)

        loss = label_loss + 0.5 * (domain_loss_src + domain_loss_tgt)
        loss.backward()
        optimizer.step()

    # Inference on target
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(X_tgt_t, alpha=0.0)
        pred_idx = logits.argmax(dim=1).cpu().numpy()

    return label_classes[pred_idx]


# ---------------------------------------------------------------------------
# LORO benchmark
# ---------------------------------------------------------------------------

def run_loro_benchmark(args: argparse.Namespace) -> dict:
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)

    all_results: dict[str, dict] = {}

    for held_out in rooms:
        train_df = df[df["room"] != held_out].copy()
        test_df = df[df["room"] == held_out].copy()

        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_df["grid_cell"].to_numpy()
        y_test = test_df["grid_cell"].to_numpy()

        fold: dict[str, dict] = {}

        # --- Baseline: KNN raw ---
        knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn.fit(X_train, y_train)
        fold["KNN_vanilla"] = _localization_metrics(y_test, knn.predict(X_test), lookup)

        # --- Baseline: RF raw ---
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
        rf.fit(X_train, y_train)
        fold["RF_vanilla"] = _localization_metrics(y_test, rf.predict(X_test), lookup)

        # --- CORAL + KNN ---
        pred_coral_knn = _coral_predict_with_knn(X_train, y_train, X_test)
        fold["CORAL+KNN"] = _localization_metrics(y_test, pred_coral_knn, lookup)

        # --- CORAL + RF ---
        pred_coral_rf = _coral_predict_with_rf(X_train, y_train, X_test)
        fold["CORAL+RF"] = _localization_metrics(y_test, pred_coral_rf, lookup)

        # --- DANN ---
        if HAS_TORCH and not args.skip_dann:
            # Assign domain IDs: each source room = separate domain, target = last domain
            source_rooms = sorted(r for r in rooms if r != held_out)
            room_to_domain = {r: i for i, r in enumerate(source_rooms)}
            room_to_domain[held_out] = len(source_rooms)

            dom_source = np.array([room_to_domain[r] for r in train_df["room"]], dtype=np.int64)
            dom_target = np.full(len(test_df), room_to_domain[held_out], dtype=np.int64)

            try:
                pred_dann = dann_train_predict(
                    X_train, y_train, dom_source,
                    X_test, dom_target,
                    epochs=args.dann_epochs,
                    lr=args.dann_lr,
                    hidden=args.dann_hidden,
                    random_state=SEED,
                )
                fold["DANN"] = _localization_metrics(y_test, pred_dann, lookup)
            except Exception as exc:
                fold["DANN"] = {"cell_acc": float("nan"), "error": str(exc)}
                print(f"    DANN failed for {held_out}: {exc}")
        elif not HAS_TORCH:
            fold["DANN"] = {"cell_acc": float("nan"), "note": "torch not installed"}

        all_results[held_out] = fold

        line = f"  held_out={held_out:<6}"
        for m, v in fold.items():
            line += f"  {m}={v['cell_acc']:.4f}"
        print(line)

    # Aggregate means
    model_names = list(next(iter(all_results.values())).keys())
    averages: dict[str, dict] = {}
    for m in model_names:
        accs = [all_results[r][m]["cell_acc"] for r in rooms if m in all_results[r]]
        errs = [all_results[r][m].get("mean_error_m", float("nan")) for r in rooms if m in all_results[r]]
        valid_accs = [a for a in accs if not np.isnan(a)]
        valid_errs = [e for e in errs if not np.isnan(e)]
        averages[m] = {
            "cell_acc_mean": float(np.mean(valid_accs)) if valid_accs else float("nan"),
            "cell_acc_std": float(np.std(valid_accs)) if valid_accs else float("nan"),
            "mean_error_m_mean": float(np.mean(valid_errs)) if valid_errs else float("nan"),
        }

    print("\n  LORO averages (vs vanilla KNN baseline):")
    knn_base = averages.get("KNN_vanilla", {}).get("cell_acc_mean", float("nan"))
    for m, v in averages.items():
        delta = v["cell_acc_mean"] - knn_base
        sign = "+" if delta >= 0 else ""
        print(
            f"    {m:<15} acc={v['cell_acc_mean']:.4f} ± {v['cell_acc_std']:.4f}"
            f"  err={v['mean_error_m_mean']:.3f}m  Δ_vs_KNN={sign}{delta:.4f}"
        )

    return {"folds": all_results, "averages": averages}


def main():
    parser = argparse.ArgumentParser(
        description="CORAL and DANN domain adaptation for LORO RSSI localization"
    )
    parser.add_argument("--skip-dann", action="store_true",
                        help="Skip DANN (requires PyTorch).")
    parser.add_argument("--dann-epochs", type=int, default=150,
                        help="Training epochs for DANN.")
    parser.add_argument("--dann-lr", type=float, default=1e-3,
                        help="Learning rate for DANN optimizer.")
    parser.add_argument("--dann-hidden", type=int, default=64,
                        help="Hidden layer size for DANN feature extractor.")
    args = parser.parse_args()

    if not HAS_TORCH and not args.skip_dann:
        print("[WARNING] PyTorch not found — DANN will be skipped. "
              "Install: pip install torch")

    print("=== Domain Adaptation LORO Benchmark ===")
    print(f"  CORAL: enabled")
    print(f"  DANN:  {'enabled' if HAS_TORCH and not args.skip_dann else 'disabled'}")

    results = run_loro_benchmark(args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "domain_adaptation_loro.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
