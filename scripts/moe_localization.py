"""Mixture-of-Experts (MoE) for RSSI localization.

Hypothesis: RSSI fingerprints cluster into several "radio regimes" depending
on router position, room geometry, or measurement campaign. A single MLP may
be learning an average policy. MoE learns specialized experts + a gating
network that routes each query to the most relevant expert.

Architecture (Sparse MoE, top-k routing):
  - Gating network: small MLP → softmax over K experts
  - K expert MLPs (each: 2 hidden layers, same width)
  - Top-k routing: select k=2 experts per sample, weighted average output
  - Load balancing loss to prevent expert collapse

Tested on:
  - Room-aware (80/20 stratified)
  - E102 intra-room
  - LORO (Leave-One-Room-Out)

References:
  - Shazeer et al., "Outrageously Large Neural Networks" (MoE), ICLR 2017
  - Fedus et al., Switch Transformer, JMLR 2022
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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
    room_ohe = pd.get_dummies(df["room"], prefix="room")
    return pd.concat([df, room_ohe], axis=1)


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


def build_X(df: pd.DataFrame, include_room: bool = True) -> np.ndarray:
    base = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not include_room:
        return base
    cols = sorted(c for c in df.columns if c.startswith("room_"))
    return np.hstack([base, df[cols].to_numpy(dtype=float)]) if cols else base


# ---------------------------------------------------------------------------
# MoE model
# ---------------------------------------------------------------------------

class ExpertMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatingNetwork(nn.Module):
    def __init__(self, n_features: int, n_experts: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits (not normalized)


class SparseMoE(nn.Module):
    """Top-k routing Mixture-of-Experts."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_experts: int = 4,
        top_k: int = 2,
        hidden: int = 64,
        gate_hidden: int = 32,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = GatingNetwork(n_features, n_experts, hidden=gate_hidden)
        self.experts = nn.ModuleList([
            ExpertMLP(n_features, hidden, n_classes)
            for _ in range(n_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, load_balance_loss)."""
        gate_logits = self.gate(x)  # (B, E)
        gate_probs = F.softmax(gate_logits, dim=1)  # (B, E)

        # Top-k selection
        topk_vals, topk_idx = gate_probs.topk(self.top_k, dim=1)  # (B, k)
        topk_weights = topk_vals / topk_vals.sum(dim=1, keepdim=True)  # normalize

        # Collect expert outputs
        B, n_classes = x.shape[0], self.experts[0].net[-1].out_features
        output = torch.zeros(B, n_classes, device=x.device)

        for i, expert in enumerate(self.experts):
            # Mask: samples routed to expert i
            mask = (topk_idx == i).any(dim=1)
            if mask.sum() == 0:
                continue
            x_e = x[mask]
            logits_e = expert(x_e)
            # Weight by gate probability for this expert
            w_e = torch.zeros(mask.sum(), device=x.device)
            for k in range(self.top_k):
                where_k = (topk_idx[mask, k] == i)
                w_e[where_k] = topk_weights[mask, k][where_k]
            output[mask] += logits_e * w_e.unsqueeze(1)

        # Load balancing: encourage uniform expert usage
        # Auxiliary loss: mean(gate_probs) should be uniform
        expert_load = gate_probs.mean(dim=0)  # (E,)
        target_load = torch.ones_like(expert_load) / self.n_experts
        load_loss = F.mse_loss(expert_load, target_load)

        return output, load_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_moe(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    *,
    n_experts: int = 4,
    top_k: int = 2,
    hidden: int = 64,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 512,
    lambda_lb: float = 0.01,
    random_state: int = SEED,
) -> tuple[SparseMoE, StandardScaler]:
    torch.manual_seed(random_state)
    y_enc = le.transform(y)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X).astype(np.float32)
    n_classes = len(le.classes_)

    model = SparseMoE(
        X_s.shape[1], n_classes,
        n_experts=n_experts, top_k=top_k, hidden=hidden,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_s).to(DEVICE)
    y_t = torch.from_numpy(y_enc.astype(np.int64)).to(DEVICE)
    n = len(X_t)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            logits, lb_loss = model(X_t[idx])
            ce_loss = F.cross_entropy(logits, y_t[idx])
            loss = ce_loss + lambda_lb * lb_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model, scaler


@torch.no_grad()
def moe_predict(
    model: SparseMoE,
    scaler: StandardScaler,
    X: np.ndarray,
    le: LabelEncoder,
) -> np.ndarray:
    model.eval()
    X_s = torch.from_numpy(scaler.transform(X).astype(np.float32)).to(DEVICE)
    logits, _ = model(X_s)
    return le.inverse_transform(logits.argmax(1).cpu().numpy())


# ---------------------------------------------------------------------------
# Plain MLP baseline (same architecture, single expert)
# ---------------------------------------------------------------------------

class PlainMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y_enc, n_classes, *, epochs=300, lr=1e-3, hidden=128, batch_size=512, random_state=SEED):
    torch.manual_seed(random_state)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X).astype(np.float32)
    model = PlainMLP(X_s.shape[1], n_classes, hidden=hidden).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    X_t = torch.from_numpy(X_s).to(DEVICE)
    y_t = torch.from_numpy(y_enc.astype(np.int64)).to(DEVICE)
    n = len(X_t)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            loss = F.cross_entropy(model(X_t[idx]), y_t[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
    return model, scaler


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(
    label: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    df_full: pd.DataFrame,
    include_room: bool,
    args: argparse.Namespace,
) -> dict:
    lookup = _cell_lookup(df_full)
    X_train = build_X(train_df, include_room=include_room)
    X_test  = build_X(test_df,  include_room=include_room)
    y_train = train_df["grid_cell"].to_numpy()
    y_test  = test_df["grid_cell"].to_numpy()

    print(f"\n  [{label}]  n_train={len(X_train)}")
    results = {}
    le = LabelEncoder().fit(y_train)
    y_enc = le.transform(y_train)
    n_classes = len(le.classes_)

    # KNN baseline
    scaler0 = StandardScaler()
    X_tr_s0 = scaler0.fit_transform(X_train)
    X_te_s0 = scaler0.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_tr_s0, y_train)
    results["KNN"] = _metrics(y_test, knn.predict(X_te_s0), lookup)
    print(f"    KNN          acc={results['KNN']['cell_acc']:.4f}  err={results['KNN']['mean_error_m']:.3f}m")

    # Plain MLP
    mlp, scaler_mlp = train_mlp(X_train, y_enc, n_classes, epochs=args.epochs, lr=args.lr, hidden=args.hidden)
    with torch.no_grad():
        X_te_t = torch.from_numpy(scaler_mlp.transform(X_test).astype(np.float32)).to(DEVICE)
        pred_mlp = le.inverse_transform(mlp(X_te_t).argmax(1).cpu().numpy())
    results["MLP"] = _metrics(y_test, pred_mlp, lookup)
    print(f"    MLP          acc={results['MLP']['cell_acc']:.4f}  err={results['MLP']['mean_error_m']:.3f}m")

    # MoE with various expert counts
    for n_exp in args.n_experts:
        try:
            moe, scaler_moe = train_moe(
                X_train, y_train, le,
                n_experts=n_exp, top_k=min(args.top_k, n_exp),
                hidden=args.hidden, epochs=args.epochs, lr=args.lr,
                lambda_lb=args.lambda_lb,
            )
            pred_moe = moe_predict(moe, scaler_moe, X_test, le)
            results[f"MoE_{n_exp}experts"] = _metrics(y_test, pred_moe, lookup)
            r = results[f"MoE_{n_exp}experts"]
            print(f"    MoE_{n_exp}exp     acc={r['cell_acc']:.4f}  err={r['mean_error_m']:.3f}m")
        except Exception as exc:
            print(f"    MoE_{n_exp}exp: FAILED — {exc}")
            results[f"MoE_{n_exp}experts"] = {"cell_acc": float("nan"), "error": str(exc)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Mixture-of-Experts for RSSI localization")
    parser.add_argument("--n-experts", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--top-k", type=int, default=2,
                        help="Number of experts activated per sample")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lambda-lb", type=float, default=0.01,
                        help="Load balancing loss weight")
    parser.add_argument("--skip-loro", action="store_true", default=True)
    args = parser.parse_args()

    all_results: dict = {}

    print("\n=== Room-aware (multi-room, 80/20) ===")
    df = load_multiroom()
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["grid_cell"])
    all_results["room_aware"] = run_protocol("room_aware", tr, te, df, include_room=True, args=args)

    print("\n=== E102 intra-room (80/20) ===")
    df_e = load_multiroom(["E102"])
    tr_e, te_e = train_test_split(df_e, test_size=0.2, random_state=SEED, stratify=df_e["grid_cell"])
    all_results["e102"] = run_protocol("E102", tr_e, te_e, df_e, include_room=False, args=args)

    if not args.skip_loro:
        print("\n=== LORO ===")
        df = load_multiroom()
        rooms = sorted(df["room"].unique())
        folds, avgs = {}, {}
        for held_out in rooms:
            tr_l = df[df["room"] != held_out]
            te_l = df[df["room"] == held_out]
            folds[held_out] = run_protocol(
                f"LORO_{held_out}", tr_l, te_l, df, include_room=False, args=args
            )
        model_names = list(next(iter(folds.values())).keys())
        for m in model_names:
            accs = [folds[r][m]["cell_acc"] for r in rooms if m in folds[r]]
            valid = [a for a in accs if not np.isnan(a)]
            avgs[m] = {"cell_acc_mean": float(np.mean(valid)) if valid else float("nan")}
        print("\n  LORO averages:")
        for m, v in avgs.items():
            print(f"    {m:<25} acc={v['cell_acc_mean']:.4f}")
        all_results["loro"] = {"folds": folds, "averages": avgs}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "moe_localization.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    print("\n=== SUMMARY ===")
    for proto, res in all_results.items():
        if proto == "loro":
            print(f"\n{proto} averages:")
            for m, v in res.get("averages", {}).items():
                print(f"  {m:<25} acc={v['cell_acc_mean']:.4f}")
        else:
            print(f"\n{proto}:")
            for m, v in res.items():
                print(f"  {m:<25} acc={v.get('cell_acc', float('nan')):.4f}  "
                      f"err={v.get('mean_error_m', float('nan')):.3f}m")


if __name__ == "__main__":
    main()
