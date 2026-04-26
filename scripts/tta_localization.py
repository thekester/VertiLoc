"""Test-Time Adaptation (TTA/Tent) for RSSI indoor localization.

Trains a small MLP with BatchNorm on source rooms, then at test time
adapts the BN statistics (or all params) to the target room distribution
using entropy minimization — without any labels from the target room.

Three adaptation modes:
  - none    : vanilla inference (baseline)
  - bn_stats: update BN running mean/var from test batch (BN adaptation)
  - tent    : minimize prediction entropy over BN params (Tent, Wang et al. ICLR 2021)
  - full    : entropy minimization over all params (aggressive)

Tested under LORO: each room is held out once as the unseen test room.

Reference: Tent — Wang et al., "Tent: Fully Test-time Adaptation by Entropy Minimization",
           ICLR 2021
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localization.catalog import BENCHMARK_REPORT_DIR, FEATURE_COLUMNS, filter_room_campaigns
from localization.data import load_measurements

REPORT_DIR = BENCHMARK_REPORT_DIR
SEED = 42


# ---------------------------------------------------------------------------
# Data
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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, lookup: pd.DataFrame) -> dict:
    acc = float((y_true == y_pred).mean())
    pred_c = lookup.reindex(y_pred)[["coord_x_m", "coord_y_m"]].to_numpy()
    true_c = lookup.reindex(y_true)[["coord_x_m", "coord_y_m"]].to_numpy()
    mask = ~(np.isnan(pred_c).any(1) | np.isnan(true_c).any(1))
    err = float(np.linalg.norm(pred_c[mask] - true_c[mask], axis=1).mean()) if mask.sum() else float("nan")
    return {"cell_acc": acc, "mean_error_m": err}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LocalizationMLP(nn.Module):
    """Small MLP with BatchNorm for TTA — BN layers are the adaptation targets."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training on source
# ---------------------------------------------------------------------------

def train_mlp(
    X_train: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    hidden: int = 128,
    random_state: int = SEED,
    device: str = "cpu",
) -> LocalizationMLP:
    torch.manual_seed(random_state)
    model = LocalizationMLP(X_train.shape[1], n_classes, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_enc).to(device)

    n = len(X_t)
    batch = min(512, n)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            logits = model(X_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


# ---------------------------------------------------------------------------
# TTA / Tent adaptation
# ---------------------------------------------------------------------------

def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return -(probs * (probs + 1e-8).log()).sum(dim=1).mean()


def adapt_bn_stats(model: LocalizationMLP, X_test_t: torch.Tensor) -> LocalizationMLP:
    """Update BN running statistics using one forward pass on test data (no grad)."""
    m = copy.deepcopy(model)
    for module in m.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()           # use batch stats, update running stats
            module.momentum = None   # cumulative moving average
    with torch.no_grad():
        m(X_test_t)
    for module in m.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    return m


def adapt_tent(
    model: LocalizationMLP,
    X_test_t: torch.Tensor,
    *,
    n_steps: int = 20,
    lr: float = 1e-3,
    mode: str = "tent",  # "tent" (BN only) or "full" (all params)
) -> LocalizationMLP:
    """Entropy minimisation adaptation (Tent style)."""
    m = copy.deepcopy(model)
    m.eval()

    if mode == "tent":
        # Adapt only BN affine params (scale + bias)
        for module in m.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train()
                module.requires_grad_(True)
            else:
                module.eval()
                module.requires_grad_(False)
        params = [p for module in m.modules()
                  if isinstance(module, nn.BatchNorm1d)
                  for p in module.parameters()]
    else:
        m.train()
        params = list(m.parameters())

    optimizer = optim.Adam(params, lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        logits = m(X_test_t)
        loss = _entropy(logits)
        loss.backward()
        optimizer.step()

    m.eval()
    return m


# ---------------------------------------------------------------------------
# LORO benchmark
# ---------------------------------------------------------------------------

def run_loro_tta(args: argparse.Namespace) -> dict:
    print("\n=== TTA/Tent LORO Benchmark ===")
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = df["grid_cell"].to_numpy()
    device = "cpu"

    all_folds: dict[str, dict] = {}

    for held_out in rooms:
        print(f"\n  Fold: held_out={held_out}")
        train_mask = df["room"].to_numpy() != held_out
        test_mask  = df["room"].to_numpy() == held_out

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_all[test_mask]
        y_test  = y_all[test_mask]

        # Encode labels using source classes only (OOD at test time)
        le = LabelEncoder().fit(y_train)
        y_enc = le.transform(y_train)
        n_classes = len(le.classes_)

        scaler = StandardScaler().fit(X_train)
        X_tr_s = scaler.transform(X_train).astype(np.float32)
        X_te_s = scaler.transform(X_test).astype(np.float32)

        print(f"    Training MLP on {X_tr_s.shape[0]} source samples, {n_classes} source classes...")
        model = train_mlp(
            X_tr_s, y_enc, n_classes,
            epochs=args.epochs,
            lr=args.lr,
            hidden=args.hidden,
            random_state=SEED,
            device=device,
        )

        X_te_t = torch.from_numpy(X_te_s).to(device)
        fold: dict[str, dict] = {}

        def _predict(m: LocalizationMLP) -> np.ndarray:
            m.eval()
            with torch.no_grad():
                logits = m(X_te_t)
            pred_enc = logits.argmax(dim=1).cpu().numpy()
            # Map back to string cell labels (only source classes known)
            pred_labels = le.inverse_transform(pred_enc)
            return pred_labels

        # 1. Vanilla (no adaptation)
        pred_vanilla = _predict(model)
        fold["vanilla"] = _metrics(y_test, pred_vanilla, lookup)
        print(f"    vanilla   acc={fold['vanilla']['cell_acc']:.4f}")

        # 2. BN stats update
        model_bn = adapt_bn_stats(model, X_te_t)
        pred_bn = _predict(model_bn)
        fold["bn_stats"] = _metrics(y_test, pred_bn, lookup)
        print(f"    bn_stats  acc={fold['bn_stats']['cell_acc']:.4f}")

        # 3. Tent (BN affine params entropy minimisation)
        model_tent = adapt_tent(model, X_te_t, n_steps=args.tent_steps, lr=args.tent_lr, mode="tent")
        pred_tent = _predict(model_tent)
        fold["tent"] = _metrics(y_test, pred_tent, lookup)
        print(f"    tent      acc={fold['tent']['cell_acc']:.4f}  ({args.tent_steps} steps, lr={args.tent_lr})")

        # 4. Full entropy minimisation
        model_full = adapt_tent(model, X_te_t, n_steps=args.tent_steps, lr=args.tent_lr * 0.1, mode="full")
        pred_full = _predict(model_full)
        fold["full_ent"] = _metrics(y_test, pred_full, lookup)
        print(f"    full_ent  acc={fold['full_ent']['cell_acc']:.4f}")

        all_folds[held_out] = fold

    # Averages
    methods = list(next(iter(all_folds.values())).keys())
    print("\n  LORO averages:")
    averages = {}
    for m in methods:
        accs = [all_folds[r][m]["cell_acc"] for r in rooms]
        valid = [a for a in accs if not np.isnan(a)]
        avg = float(np.mean(valid)) if valid else float("nan")
        std = float(np.std(valid)) if valid else float("nan")
        averages[m] = {"cell_acc_mean": avg, "cell_acc_std": std}
        print(f"    {m:<12} acc={avg:.4f} ± {std:.4f}")

    return {"folds": all_folds, "averages": averages}


def main():
    parser = argparse.ArgumentParser(description="TTA/Tent adaptation for RSSI LORO")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Source training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--tent-steps", type=int, default=30,
                        help="TTA gradient steps at test time.")
    parser.add_argument("--tent-lr", type=float, default=1e-3)
    args = parser.parse_args()

    results = run_loro_tta(args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "tta_loro.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
