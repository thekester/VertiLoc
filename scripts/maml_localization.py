"""MAML-style meta-learning for rapid adaptation to new rooms/campaigns.

Goal: learn an initialization θ* such that after K gradient steps on a few
labeled examples from a new room/campaign, the model reaches high accuracy.

MAML (Model-Agnostic Meta-Learning, Finn et al., ICML 2017):
  - Meta-train: for each episode (task), compute inner-loop gradient on support
    set, evaluate on query set, backprop through the inner update (2nd-order
    gradient). Accumulate meta-gradient across tasks.
  - Meta-test: given K labeled examples from a new room, run K inner steps,
    evaluate on remaining data.

Applied to RSSI localization:
  - Task = one room (leave-one-room-out split)
  - Support = K random samples per cell (like ProtoNet)
  - Query = remaining samples

We use first-order MAML (FOMAML) to avoid the expensive 2nd-order computation:
  - Same update direction as full MAML, slightly less accurate.

Reference: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of
           Deep Networks", ICML 2017.
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
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_multiroom(rooms: list[str] | None = None) -> pd.DataFrame:
    campaigns = filter_room_campaigns(room_filter=rooms)
    frames = []
    for room, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df = load_measurements([spec])
                df["room"] = room
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def _cell_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("grid_cell")[["coord_x_m", "coord_y_m"]].first()


def _metrics(y_true, y_pred, lookup) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    pc = lookup.reindex(y_pred)[["coord_x_m", "coord_y_m"]].to_numpy()
    tc = lookup.reindex(y_true)[["coord_x_m", "coord_y_m"]].to_numpy()
    mask = ~(np.isnan(pc).any(1) | np.isnan(tc).any(1))
    err = float(np.linalg.norm(pc[mask] - tc[mask], axis=1).mean()) if mask.sum() else float("nan")
    return {"cell_acc": acc, "mean_error_m": err}


def build_X_raw(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------

class MAMLEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MAMLClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64, emb_dim: int = 64):
        super().__init__()
        self.encoder = MAMLEncoder(n_features, hidden=hidden, emb_dim=emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Task/episode sampler
# ---------------------------------------------------------------------------

def sample_task(
    df: pd.DataFrame,
    scaler: StandardScaler,
    le: LabelEncoder,
    n_support: int,
    n_query: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (x_support, y_support, x_query, y_query) for one meta-task."""
    X = scaler.transform(build_X_raw(df)).astype(np.float32)
    y = df["grid_cell"].to_numpy()

    # Only cells with enough samples
    cells = [c for c in np.unique(y) if (y == c).sum() >= n_support + n_query]
    if len(cells) < 2:
        cells = list(np.unique(y))

    support_idx, query_idx = [], []
    for cell in cells:
        cell_idx = np.where(y == cell)[0]
        chosen = rng.choice(len(cell_idx), size=min(n_support + n_query, len(cell_idx)), replace=False)
        support_idx.extend(cell_idx[chosen[:n_support]])
        query_idx.extend(cell_idx[chosen[n_support:n_support + n_query]])

    def to_tensors(idx):
        X_i = torch.from_numpy(X[idx]).to(DEVICE)
        y_i = torch.from_numpy(le.transform(y[idx]).astype(np.int64)).to(DEVICE)
        return X_i, y_i

    return (*to_tensors(support_idx), *to_tensors(query_idx))


# ---------------------------------------------------------------------------
# FOMAML training
# ---------------------------------------------------------------------------

def fomaml_inner_update(
    model: MAMLClassifier,
    x_s: torch.Tensor,
    y_s: torch.Tensor,
    inner_lr: float,
    n_inner_steps: int,
) -> MAMLClassifier:
    """Clone model, run n inner gradient steps on support set, return adapted model."""
    adapted = copy.deepcopy(model)
    inner_opt = optim.SGD(adapted.parameters(), lr=inner_lr)
    adapted.train()
    for _ in range(n_inner_steps):
        loss = F.cross_entropy(adapted(x_s), y_s)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step()
    return adapted


def meta_train(
    source_dfs: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    le: LabelEncoder,
    n_features: int,
    n_classes: int,
    *,
    n_epochs: int = 100,
    n_tasks_per_epoch: int = 8,
    n_support: int = 5,
    n_query: int = 15,
    inner_lr: float = 0.01,
    meta_lr: float = 1e-3,
    n_inner_steps: int = 5,
    hidden: int = 64,
    emb_dim: int = 64,
    random_state: int = SEED,
) -> MAMLClassifier:
    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)
    rooms = list(source_dfs.keys())

    model = MAMLClassifier(n_features, n_classes, hidden=hidden, emb_dim=emb_dim).to(DEVICE)
    meta_opt = optim.Adam(model.parameters(), lr=meta_lr, weight_decay=1e-4)

    for epoch in range(n_epochs):
        meta_grads = [torch.zeros_like(p) for p in model.parameters()]
        total_query_loss = 0.0

        for _ in range(n_tasks_per_epoch):
            room = rng.choice(rooms)
            try:
                x_s, y_s, x_q, y_q = sample_task(
                    source_dfs[room], scaler, le, n_support, n_query, rng
                )
            except Exception:
                continue

            # Inner loop: adapt to support set
            adapted = fomaml_inner_update(model, x_s, y_s, inner_lr, n_inner_steps)

            # Query loss on adapted model
            query_loss = F.cross_entropy(adapted(x_q), y_q)
            total_query_loss += query_loss.item()

            # FOMAML: compute grad of query_loss w.r.t. adapted params,
            # but treat adapted params as leaf (first-order approx)
            grads = torch.autograd.grad(query_loss, adapted.parameters())
            for g_acc, g in zip(meta_grads, grads):
                g_acc += g.detach() / n_tasks_per_epoch

        # Apply meta-gradient
        meta_opt.zero_grad()
        for p, g in zip(model.parameters(), meta_grads):
            p.grad = g
        meta_opt.step()

        if (epoch + 1) % 25 == 0:
            avg_loss = total_query_loss / n_tasks_per_epoch
            print(f"    Meta epoch {epoch+1}/{n_epochs}  query_loss={avg_loss:.4f}")

    return model


# ---------------------------------------------------------------------------
# Meta-testing: adapt to new room with K shots
# ---------------------------------------------------------------------------

def meta_test(
    model: MAMLClassifier,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    le: LabelEncoder,
    lookup: pd.DataFrame,
    k_shot: int,
    *,
    inner_lr: float = 0.01,
    n_inner_steps: int = 10,
    rng: np.random.Generator,
) -> dict:
    X_test = scaler.transform(build_X_raw(test_df)).astype(np.float32)
    y_test = test_df["grid_cell"].to_numpy()

    # Sample k support examples per cell (if available)
    support_idx = []
    for cell in np.unique(y_test):
        cell_idx = np.where(y_test == cell)[0]
        n = min(k_shot, len(cell_idx))
        chosen = rng.choice(len(cell_idx), size=n, replace=False)
        support_idx.extend(cell_idx[chosen])

    if not support_idx:
        return {"cell_acc": float("nan"), "mean_error_m": float("nan")}

    support_idx = np.array(support_idx)
    query_mask = np.ones(len(X_test), dtype=bool)
    query_mask[support_idx] = False
    if query_mask.sum() == 0:
        return {"cell_acc": float("nan"), "mean_error_m": float("nan")}

    # Filter to known classes in le
    known_mask = np.isin(y_test, le.classes_)
    query_mask &= known_mask

    x_s = torch.from_numpy(X_test[support_idx]).to(DEVICE)
    y_s_raw = y_test[support_idx]
    y_s_valid = np.isin(y_s_raw, le.classes_)
    if y_s_valid.sum() < 2:
        return {"cell_acc": float("nan"), "mean_error_m": float("nan")}
    x_s = x_s[y_s_valid]
    y_s = torch.from_numpy(le.transform(y_s_raw[y_s_valid]).astype(np.int64)).to(DEVICE)

    adapted = fomaml_inner_update(model, x_s, y_s, inner_lr, n_inner_steps)
    adapted.eval()

    x_q = torch.from_numpy(X_test[query_mask]).to(DEVICE)
    with torch.no_grad():
        logits = adapted(x_q)
    pred_enc = logits.argmax(1).cpu().numpy()
    pred = le.inverse_transform(np.clip(pred_enc, 0, len(le.classes_) - 1))
    return _metrics(y_test[query_mask], pred, lookup)


# ---------------------------------------------------------------------------
# Main: LORO protocol
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAML meta-learning for RSSI localization")
    parser.add_argument("--meta-epochs", type=int, default=100)
    parser.add_argument("--tasks-per-epoch", type=int, default=8)
    parser.add_argument("--n-support", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=15)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--meta-lr", type=float, default=1e-3)
    parser.add_argument("--n-inner-steps", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--k-shots", nargs="+", type=int, default=[1, 5, 10, 25])
    args = parser.parse_args()

    df = load_multiroom()
    lookup = _cell_lookup(df)
    rooms = sorted(df["room"].unique())

    # Shared scaler + label encoder fitted on all data
    le = LabelEncoder().fit(df["grid_cell"].to_numpy())
    scaler = StandardScaler().fit(build_X_raw(df))
    n_features = len(FEATURE_COLUMNS)
    n_classes = len(le.classes_)

    rng = np.random.default_rng(SEED)
    all_results: dict = {}

    print(f"\nMAML LORO: {len(rooms)} rooms, {n_classes} cells, {n_features} features")

    loro_results: dict = {}
    for held_out in rooms:
        print(f"\n=== LORO: hold out {held_out} ===")
        source_dfs = {r: df[df["room"] == r] for r in rooms if r != held_out}
        test_df_room = df[df["room"] == held_out]

        print(f"  Meta-training ({args.meta_epochs} epochs, {args.tasks_per_epoch} tasks/epoch)...")
        model = meta_train(
            source_dfs, scaler, le, n_features, n_classes,
            n_epochs=args.meta_epochs,
            n_tasks_per_epoch=args.tasks_per_epoch,
            n_support=args.n_support,
            n_query=args.n_query,
            inner_lr=args.inner_lr,
            meta_lr=args.meta_lr,
            n_inner_steps=args.n_inner_steps,
            hidden=args.hidden,
            emb_dim=args.emb_dim,
        )

        fold_results: dict = {}
        for k in args.k_shots:
            r = meta_test(
                model, test_df_room, scaler, le, lookup,
                k_shot=k, inner_lr=args.inner_lr,
                n_inner_steps=args.n_inner_steps * 2,
                rng=rng,
            )
            fold_results[f"k{k}"] = r
            print(f"    k={k:2d}  acc={r['cell_acc']:.4f}  err={r['mean_error_m']:.3f}m")

        loro_results[held_out] = fold_results

    # Averages per k
    print("\n  LORO averages per k:")
    avgs: dict = {}
    for k in args.k_shots:
        key = f"k{k}"
        accs = [loro_results[r][key]["cell_acc"] for r in rooms if key in loro_results[r]]
        valid = [a for a in accs if not np.isnan(a)]
        avg = float(np.mean(valid)) if valid else float("nan")
        avgs[key] = {"cell_acc_mean": avg}
        print(f"    k={k:2d}  avg_acc={avg:.4f}")

    all_results["loro"] = {"folds": loro_results, "averages": avgs}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "maml_localization.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
