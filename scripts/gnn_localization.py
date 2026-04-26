"""GNN on cell adjacency graph for RSSI localization.

Cells form a regular grid — physical neighbors are meaningful. A GNN that
propagates information along the grid graph can:
  1. Smooth noisy RSSI embeddings with spatially coherent neighbors
  2. Learn position-aware representations (nearby cells have similar embeddings)

Architecture:
  - Node features: per-cell prototype (mean RSSI over training samples)
  - Edge: connect cells within `edge_radius_m` metres
  - Graph Conv: 2 rounds of message passing (mean aggregation)
  - Classification: softmax over node logits

At test time: embed the query RSSI with an MLP, find the nearest node in
embedding space, output that node's cell label.

We implement a lightweight version using pure PyTorch (no torch_geometric
dependency) with explicit sparse adjacency matmul.

Protocols: room-aware and E102 intra-room.
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
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    le: LabelEncoder,
    lookup: pd.DataFrame,
    edge_radius_m: float = 0.6,
) -> torch.Tensor:
    """Normalized adjacency matrix A_hat = D^{-1/2} (A + I) D^{-1/2}."""
    cells = le.classes_
    coords = lookup.reindex(cells)[["coord_x_m", "coord_y_m"]].to_numpy(dtype=np.float32)
    n = len(cells)

    # Euclidean distance matrix
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)  # (n, n)
    adj = (dist <= edge_radius_m).astype(np.float32)
    np.fill_diagonal(adj, 1.0)  # self-loops (A + I)

    # Symmetric normalization
    deg = adj.sum(axis=1, keepdims=True)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    a_hat = (deg_inv_sqrt * adj * deg_inv_sqrt.T).astype(np.float32)
    return torch.from_numpy(a_hat).float()


# ---------------------------------------------------------------------------
# Cell prototype features
# ---------------------------------------------------------------------------

def build_cell_prototypes(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
) -> np.ndarray:
    """Mean RSSI per cell as node features."""
    n_classes = len(le.classes_)
    n_features = X.shape[1]
    protos = np.zeros((n_classes, n_features), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int32)
    y_enc = le.transform(y)
    for i, xi in zip(y_enc, X):
        protos[i] += xi.astype(np.float32)
        counts[i] += 1
    counts = np.maximum(counts, 1)
    return (protos / counts[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# GNN model (pure PyTorch, no torch_geometric)
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Single graph conv layer: H' = A_hat @ H @ W (+ bias)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(a_hat @ h))


class CellGNN(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(n_features, hidden))
        for _ in range(n_layers - 1):
            self.layers.append(GCNLayer(hidden, hidden))
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, a_hat)
        return self.classifier(h)  # (n_cells, n_classes)

    def node_embeddings(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, a_hat)
        return h  # (n_cells, hidden)


class QueryEncoder(nn.Module):
    """Encodes a raw RSSI query into the same embedding space as GNN nodes."""

    def __init__(self, n_features: int, hidden: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Joint training
# ---------------------------------------------------------------------------

def train_gnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    le: LabelEncoder,
    lookup: pd.DataFrame,
    *,
    edge_radius_m: float = 0.6,
    hidden: int = 64,
    emb_dim: int = 64,
    n_gcn_layers: int = 2,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 512,
    random_state: int = SEED,
) -> tuple[CellGNN, QueryEncoder, torch.Tensor, StandardScaler]:
    torch.manual_seed(random_state)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train).astype(np.float32)
    y_enc = le.transform(y_train)
    n_classes = len(le.classes_)
    n_features = X_s.shape[1]

    # Build graph
    a_hat = build_graph(le, lookup, edge_radius_m=edge_radius_m).to(DEVICE)

    # Cell prototypes as initial node features
    protos = build_cell_prototypes(X_s, y_train, le)
    proto_t = torch.from_numpy(protos).float().to(DEVICE)

    # Models
    gnn = CellGNN(n_features, hidden, n_classes, n_layers=n_gcn_layers).to(DEVICE)
    q_enc = QueryEncoder(n_features, hidden, emb_dim).to(DEVICE)

    # The GNN also needs a node-level encoder with emb_dim output for matching
    # We reuse CellGNN but add a separate embedding GNN for the node side
    gnn_emb = CellGNN(n_features, hidden, emb_dim, n_layers=n_gcn_layers).to(DEVICE)

    params = list(gnn.parameters()) + list(q_enc.parameters()) + list(gnn_emb.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_s).float().to(DEVICE)
    y_t = torch.from_numpy(y_enc.astype(np.int64)).to(DEVICE)
    n = len(X_t)

    # We train GNN on cell prototypes (transductive) and query encoder on instances
    for epoch in range(epochs):
        gnn.train()
        gnn_emb.train()
        q_enc.train()

        # --- Node classification loss (GNN on cell prototypes) ---
        node_logits = gnn(proto_t, a_hat)  # (n_classes, n_classes)
        # Self-supervised: each node should predict its own class
        node_labels = torch.arange(n_classes, device=DEVICE)
        loss_node = F.cross_entropy(node_logits, node_labels)

        # --- Instance classification: query encoder + nearest GNN node ---
        perm = torch.randperm(n, device=DEVICE)[:batch_size]
        z_query = q_enc(X_t[perm])    # (B, emb_dim)
        z_nodes = gnn_emb(proto_t, a_hat)  # (n_classes, emb_dim)
        z_nodes_norm = F.normalize(z_nodes, dim=1)
        z_query_norm = F.normalize(z_query, dim=1)
        sim = z_query_norm @ z_nodes_norm.T  # (B, n_classes)
        loss_inst = F.cross_entropy(sim * 10, y_t[perm])

        loss = loss_node + loss_inst
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return gnn_emb, q_enc, a_hat, scaler, proto_t


@torch.no_grad()
def gnn_predict(
    gnn_emb: CellGNN,
    q_enc: QueryEncoder,
    a_hat: torch.Tensor,
    proto_t: torch.Tensor,
    scaler: StandardScaler,
    X_test: np.ndarray,
    le: LabelEncoder,
) -> np.ndarray:
    gnn_emb.eval()
    q_enc.eval()

    X_s = torch.from_numpy(scaler.transform(X_test).astype(np.float32)).float().to(DEVICE)
    z_query = q_enc(X_s)
    z_nodes = gnn_emb(proto_t, a_hat)

    z_q_norm = F.normalize(z_query, dim=1)
    z_n_norm = F.normalize(z_nodes, dim=1)
    sim = z_q_norm @ z_n_norm.T  # (N_test, n_classes)
    pred_idx = sim.argmax(dim=1).cpu().numpy()
    return le.inverse_transform(pred_idx)


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

    # KNN baseline
    scaler0 = StandardScaler()
    X_tr_s = scaler0.fit_transform(X_train)
    X_te_s = scaler0.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_tr_s, y_train)
    results["KNN"] = _metrics(y_test, knn.predict(X_te_s), lookup)
    print(f"    KNN          acc={results['KNN']['cell_acc']:.4f}  err={results['KNN']['mean_error_m']:.3f}m")

    # GNN
    try:
        gnn_emb, q_enc, a_hat, scaler_gnn, proto_t = train_gnn(
            X_train, y_train, le, lookup,
            edge_radius_m=args.edge_radius,
            hidden=args.hidden,
            emb_dim=args.emb_dim,
            n_gcn_layers=args.n_layers,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        pred_gnn = gnn_predict(gnn_emb, q_enc, a_hat, proto_t, scaler_gnn, X_test, le)
        results["GNN_cell"] = _metrics(y_test, pred_gnn, lookup)
        r = results["GNN_cell"]
        print(f"    GNN_cell     acc={r['cell_acc']:.4f}  err={r['mean_error_m']:.3f}m")
    except Exception as exc:
        print(f"    GNN_cell: FAILED — {exc}")
        results["GNN_cell"] = {"cell_acc": float("nan"), "error": str(exc)}

    return results


def main():
    parser = argparse.ArgumentParser(description="GNN on cell adjacency graph")
    parser.add_argument("--edge-radius", type=float, default=0.6,
                        help="Max distance in metres for adjacency edge")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
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

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "gnn_localization.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    print("\n=== SUMMARY ===")
    for proto, res in all_results.items():
        print(f"\n{proto}:")
        for m, v in res.items():
            print(f"  {m:<20} acc={v.get('cell_acc', float('nan')):.4f}  "
                  f"err={v.get('mean_error_m', float('nan')):.3f}m")


if __name__ == "__main__":
    main()
