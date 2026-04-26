"""Metric learning for RSSI localization: SupCon + Triplet loss.

Replaces NN+L-KNN's MSE-based embedding with a proper metric learning
objective that explicitly pulls same-cell samples together and pushes
different cells apart.

Methods tested:
  1. SupCon  — Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
             → most stable, works on arbitrary batch composition
  2. Triplet — Triplet Margin Loss with hard-negative mining
             → classic metric learning, sensitive to mining strategy
  3. Proxy-NCA — one learnable proxy per class, efficient for many classes
             → scales to 100+ classes better than pairwise losses

All methods: encoder → embedding → KNN in embedding space (same as NN+L-KNN,
but the embedding is trained with metric loss instead of CE).

Protocols: room-aware (80/20), E102 intra-room, LORO.
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
from sklearn.neighbors import KNeighborsClassifier
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
# Encoder
# ---------------------------------------------------------------------------

class MetricEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)  # unit sphere for cosine metric


# ---------------------------------------------------------------------------
# SupCon loss
# ---------------------------------------------------------------------------

def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
    device = features.device
    n = features.shape[0]

    # Cosine similarity matrix
    sim = torch.mm(features, features.T) / temperature  # (n, n)

    # Mask: positives = same label, excluding diagonal
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (n, n)
    diag_mask = ~torch.eye(n, dtype=torch.bool, device=device)
    pos_mask = labels_eq & diag_mask
    neg_mask = (~labels_eq) & diag_mask

    # For numerical stability
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    sim = sim - sim_max

    # Exp similarities (excluding self)
    exp_sim = torch.exp(sim) * diag_mask.float()

    # Sum over positives / sum over all negatives+positives
    pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
    all_sum = exp_sim.sum(dim=1)

    # Only include anchors that have at least one positive
    has_pos = pos_mask.any(dim=1)
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)

    log_prob = torch.log(pos_sum.clamp(min=1e-8) / all_sum.clamp(min=1e-8))
    loss = -(log_prob[has_pos] / n_pos[has_pos]).mean()
    return loss


# ---------------------------------------------------------------------------
# Triplet loss with hard mining
# ---------------------------------------------------------------------------

def triplet_loss_hard(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Batch-hard triplet loss (Hermans et al., 2017)."""
    n = features.shape[0]
    dist_mat = torch.cdist(features, features, p=2)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag_mask = ~torch.eye(n, dtype=torch.bool, device=features.device)

    # Hardest positive: max distance among same-class pairs
    pos_mask = labels_eq & diag_mask
    ap_dist = (dist_mat * pos_mask.float()).max(dim=1).values

    # Hardest negative: min distance among different-class pairs
    neg_mask = (~labels_eq) & diag_mask
    an_dist = dist_mat.masked_fill(~neg_mask, float("inf")).min(dim=1).values

    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid = has_pos & has_neg

    loss = F.relu(ap_dist[valid] - an_dist[valid] + margin).mean()
    return loss


# ---------------------------------------------------------------------------
# Proxy-NCA loss
# ---------------------------------------------------------------------------

class ProxyNCA(nn.Module):
    """One proxy (learnable anchor) per class. Scales to many classes."""

    def __init__(self, n_classes: int, emb_dim: int):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(n_classes, emb_dim))

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
    ) -> torch.Tensor:
        proxies = F.normalize(self.proxies, dim=1)
        sim = torch.mm(features, proxies.T) / temperature  # (n, n_classes)
        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_metric_encoder(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "supcon",
    epochs: int = 200,
    lr: float = 1e-3,
    hidden: int = 64,
    emb_dim: int = 32,
    temperature: float = 0.07,
    batch_size: int = 512,
    random_state: int = SEED,
) -> tuple[MetricEncoder, StandardScaler, LabelEncoder]:
    torch.manual_seed(random_state)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X).astype(np.float32)

    encoder = MetricEncoder(X_s.shape[1], hidden=hidden, emb_dim=emb_dim).to(DEVICE)
    params = list(encoder.parameters())

    proxy_head = None
    if method == "proxy":
        proxy_head = ProxyNCA(len(le.classes_), emb_dim).to(DEVICE)
        params += list(proxy_head.parameters())

    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_s).to(DEVICE)
    y_t = torch.from_numpy(y_enc.astype(np.int64)).to(DEVICE)
    n = len(X_t)

    encoder.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            z = encoder(X_t[idx])
            labels_b = y_t[idx]

            if method == "supcon":
                loss = supcon_loss(z, labels_b, temperature=temperature)
            elif method == "triplet":
                loss = triplet_loss_hard(z, labels_b)
            elif method == "proxy":
                loss = proxy_head(z, labels_b, temperature=temperature)
            else:
                raise ValueError(f"Unknown method: {method}")

            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return encoder, scaler, le


def encode(encoder: MetricEncoder, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    encoder.eval()
    X_s = torch.from_numpy(scaler.transform(X).astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        return encoder(X_s).cpu().numpy()


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(
    label: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lookup: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    print(f"\n  [{label}]  n_train={len(X_train)}")
    results = {}

    # KNN baseline (raw features)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_train, y_train)
    results["KNN_raw"] = _metrics(y_test, knn.predict(X_test), lookup)
    print(f"    KNN_raw    acc={results['KNN_raw']['cell_acc']:.4f}  err={results['KNN_raw']['mean_error_m']:.3f}m")

    for method in args.methods:
        try:
            enc, scaler, _ = train_metric_encoder(
                X_train, y_train,
                method=method,
                epochs=args.epochs,
                lr=args.lr,
                hidden=args.hidden,
                emb_dim=args.emb_dim,
                temperature=args.temperature,
                batch_size=args.batch_size,
            )
            Z_tr = encode(enc, scaler, X_train)
            Z_te = encode(enc, scaler, X_test)
            knn_emb = KNeighborsClassifier(n_neighbors=7, weights="distance")
            knn_emb.fit(Z_tr, y_train)
            pred = knn_emb.predict(Z_te)
            results[f"KNN+{method}"] = _metrics(y_test, pred, lookup)
            r = results[f"KNN+{method}"]
            print(f"    KNN+{method:<8} acc={r['cell_acc']:.4f}  err={r['mean_error_m']:.3f}m")
        except Exception as exc:
            print(f"    KNN+{method}: FAILED — {exc}")
            results[f"KNN+{method}"] = {"cell_acc": float("nan"), "error": str(exc)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Metric learning benchmark for RSSI localization")
    parser.add_argument("--methods", nargs="+", default=["supcon", "triplet", "proxy"],
                        choices=["supcon", "triplet", "proxy"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-loro", action="store_true")
    args = parser.parse_args()

    all_results: dict = {}

    print("\n=== Room-aware (80/20) ===")
    df = load_multiroom()
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["grid_cell"])
    lookup = _cell_lookup(df)
    all_results["room_aware"] = run_protocol(
        "room_aware", build_X(tr), tr["grid_cell"].to_numpy(),
        build_X(te), te["grid_cell"].to_numpy(), lookup, args
    )

    print("\n=== E102 intra-room (80/20) ===")
    df_e = load_multiroom(["E102"])
    tr_e, te_e = train_test_split(df_e, test_size=0.2, random_state=SEED, stratify=df_e["grid_cell"])
    lookup_e = _cell_lookup(df_e)
    all_results["e102"] = run_protocol(
        "E102", build_X(tr_e, include_room=False), tr_e["grid_cell"].to_numpy(),
        build_X(te_e, include_room=False), te_e["grid_cell"].to_numpy(), lookup_e, args
    )

    if not args.skip_loro:
        print("\n=== LORO ===")
        rooms = sorted(df["room"].unique())
        folds, avgs = {}, {}
        for held_out in rooms:
            tr_l = df[df["room"] != held_out]
            te_l = df[df["room"] == held_out]
            fold = run_protocol(
                f"LORO_{held_out}",
                build_X(tr_l, include_room=False), tr_l["grid_cell"].to_numpy(),
                build_X(te_l, include_room=False), te_l["grid_cell"].to_numpy(),
                lookup, args,
            )
            folds[held_out] = fold
        model_names = list(next(iter(folds.values())).keys())
        print("\n  LORO averages:")
        for m in model_names:
            accs = [folds[r][m]["cell_acc"] for r in rooms if m in folds[r]]
            valid = [a for a in accs if not np.isnan(a)]
            avg = float(np.mean(valid)) if valid else float("nan")
            avgs[m] = {"cell_acc_mean": avg}
            print(f"    {m:<18} acc={avg:.4f}")
        all_results["loro"] = {"folds": folds, "averages": avgs}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "metric_learning_benchmark.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
