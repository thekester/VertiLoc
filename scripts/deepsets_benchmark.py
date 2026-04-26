"""DeepSets / Set Transformer for RSSI localization.

Standard approach: each sample is one row of RSSI values, trained
independently. But each physical cell has 25 repeated measurements — the
*set* of those scans carries richer information about the cell's distribution
than a single average.

DeepSets (Zaheer et al., NeurIPS 2017):
    f(X_set) = rho( sum_i phi(x_i) )
  where phi encodes each scan and rho decodes the aggregated embedding.
  Permutation-invariant by construction.

Two inference modes tested:
  A) Set-level: group all measurements from the same (cell, campaign) position
     into a set, pass the set through DeepSets → one prediction per set.
  B) Instance-level: at test time, use all available scans for the same
     physical location as the support set. One-shot calibration variant of
     proto-nets.

Protocols: room-aware (80/20 on sets), E102 intra-room.
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


def build_X_raw(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Group scans into sets per (grid_cell, campaign)
# ---------------------------------------------------------------------------

def build_sets(
    df: pd.DataFrame,
    scaler: StandardScaler,
    set_size: int = 10,
    include_room: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group rows by (grid_cell, campaign), sample `set_size` scans per group.
    Returns:
      X_sets : (N_sets, set_size, n_features)
      y_sets : (N_sets,) cell labels
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    feat_cols = FEATURE_COLUMNS
    if include_room:
        room_cols = sorted(c for c in df.columns if c.startswith("room_"))
        feat_cols = feat_cols + room_cols

    X_sets, y_sets = [], []
    for (cell, camp), group in df.groupby(["grid_cell", "campaign"]):
        feats = group[feat_cols].to_numpy(dtype=np.float32)
        feats = scaler.transform(feats).astype(np.float32)
        n = len(feats)
        if n < 2:
            continue
        # Sample multiple overlapping sets from this group
        n_sets = max(1, n // set_size)
        for _ in range(n_sets):
            chosen = rng.choice(n, size=set_size, replace=True)
            X_sets.append(feats[chosen])
            y_sets.append(cell)

    return np.array(X_sets, dtype=np.float32), np.array(y_sets)


# ---------------------------------------------------------------------------
# DeepSets model
# ---------------------------------------------------------------------------

class DeepSets(nn.Module):
    """phi (per-element) + rho (aggregated) architecture."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        phi_hidden: int = 64,
        rho_hidden: int = 128,
        emb_dim: int = 64,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.aggregation = aggregation
        self.phi = nn.Sequential(
            nn.Linear(n_features, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, emb_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(emb_dim, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden // 2),
            nn.ReLU(),
            nn.Linear(rho_hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F) — B sets, S elements, F features
        B, S, F = x.shape
        phi_out = self.phi(x.view(B * S, F)).view(B, S, -1)  # (B, S, emb_dim)

        if self.aggregation == "mean":
            agg = phi_out.mean(dim=1)
        elif self.aggregation == "max":
            agg = phi_out.max(dim=1).values
        elif self.aggregation == "sum":
            agg = phi_out.sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return self.rho(agg)  # (B, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F = x.shape
        phi_out = self.phi(x.view(B * S, F)).view(B, S, -1)
        return phi_out.mean(dim=1)  # (B, emb_dim)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_deepsets(
    X_sets: np.ndarray,
    y_sets: np.ndarray,
    le: LabelEncoder,
    *,
    aggregation: str = "mean",
    epochs: int = 200,
    lr: float = 1e-3,
    phi_hidden: int = 64,
    rho_hidden: int = 128,
    emb_dim: int = 64,
    batch_size: int = 64,
    random_state: int = SEED,
) -> DeepSets:
    torch.manual_seed(random_state)
    y_enc = le.transform(y_sets)
    n_classes = len(le.classes_)
    n_features = X_sets.shape[2]

    model = DeepSets(
        n_features, n_classes,
        phi_hidden=phi_hidden, rho_hidden=rho_hidden, emb_dim=emb_dim,
        aggregation=aggregation,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_sets).to(DEVICE)
    y_t = torch.from_numpy(y_enc.astype(np.int64)).to(DEVICE)
    n = len(X_t)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            logits = model(X_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


@torch.no_grad()
def deepsets_predict(
    model: DeepSets,
    X_sets: np.ndarray,
    le: LabelEncoder,
) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X_sets).to(DEVICE)
    logits = model(X_t)
    return le.inverse_transform(logits.argmax(1).cpu().numpy())


@torch.no_grad()
def deepsets_embed(model: DeepSets, X_sets: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X_sets).to(DEVICE)
    return model.embed(X_t).cpu().numpy()


# ---------------------------------------------------------------------------
# Protocol
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
    print(f"\n  [{label}]  n_train={len(train_df)}  n_test={len(test_df)}")
    results = {}

    # Fit scaler on raw training features
    feat_cols = FEATURE_COLUMNS
    if include_room:
        room_cols = sorted(c for c in train_df.columns if c.startswith("room_"))
        feat_cols = feat_cols + room_cols

    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols].to_numpy(dtype=np.float32))

    y_train_raw = train_df["grid_cell"].to_numpy()
    y_test_raw  = test_df["grid_cell"].to_numpy()
    le = LabelEncoder().fit(y_train_raw)

    rng = np.random.default_rng(SEED)

    # ---- KNN baseline (instance-level) ----
    X_tr_raw = scaler.transform(train_df[feat_cols].to_numpy(dtype=np.float32))
    X_te_raw = scaler.transform(test_df[feat_cols].to_numpy(dtype=np.float32))
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_tr_raw, y_train_raw)
    results["KNN_instance"] = _metrics(y_test_raw, knn.predict(X_te_raw), lookup)
    print(f"    KNN_instance     acc={results['KNN_instance']['cell_acc']:.4f}  "
          f"err={results['KNN_instance']['mean_error_m']:.3f}m")

    # ---- Build sets ----
    X_tr_sets, y_tr_sets = build_sets(train_df, scaler, args.set_size, include_room, rng)
    X_te_sets, y_te_sets = build_sets(test_df,  scaler, args.set_size, include_room, rng)

    print(f"    Sets: train={len(X_tr_sets)} test={len(X_te_sets)}")

    if len(X_tr_sets) < 10 or len(X_te_sets) < 2:
        print("    Insufficient sets — skipping DeepSets")
        return results

    for agg in args.aggregations:
        try:
            model = train_deepsets(
                X_tr_sets, y_tr_sets, le,
                aggregation=agg,
                epochs=args.epochs,
                lr=args.lr,
                phi_hidden=args.phi_hidden,
                rho_hidden=args.rho_hidden,
                emb_dim=args.emb_dim,
                batch_size=args.batch_size,
            )
            pred = deepsets_predict(model, X_te_sets, le)
            results[f"DeepSets_{agg}"] = _metrics(y_te_sets, pred, lookup)
            r = results[f"DeepSets_{agg}"]
            print(f"    DeepSets_{agg:<6} acc={r['cell_acc']:.4f}  err={r['mean_error_m']:.3f}m")

            # KNN in DeepSets embedding space
            Z_tr = deepsets_embed(model, X_tr_sets)
            Z_te = deepsets_embed(model, X_te_sets)
            knn_emb = KNeighborsClassifier(n_neighbors=7, weights="distance")
            knn_emb.fit(Z_tr, y_tr_sets)
            pred_knn = knn_emb.predict(Z_te)
            results[f"KNN+DeepSets_{agg}"] = _metrics(y_te_sets, pred_knn, lookup)
            r2 = results[f"KNN+DeepSets_{agg}"]
            print(f"    KNN+DS_{agg:<7} acc={r2['cell_acc']:.4f}  err={r2['mean_error_m']:.3f}m")

        except Exception as exc:
            print(f"    DeepSets_{agg}: FAILED — {exc}")
            results[f"DeepSets_{agg}"] = {"cell_acc": float("nan"), "error": str(exc)}

    return results


def main():
    parser = argparse.ArgumentParser(description="DeepSets benchmark for RSSI localization")
    parser.add_argument("--aggregations", nargs="+", default=["mean", "max"],
                        choices=["mean", "max", "sum"])
    parser.add_argument("--set-size", type=int, default=10,
                        help="Number of scans per set")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--phi-hidden", type=int, default=64)
    parser.add_argument("--rho-hidden", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    all_results: dict = {}

    print("\n=== Room-aware (multi-room, 80/20) ===")
    df = load_multiroom()
    # Split at the set level to avoid leakage: split by unique (cell, campaign) groups
    groups = df.groupby(["grid_cell", "campaign"]).ngroup()
    unique_groups = groups.unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_groups)
    split = int(len(unique_groups) * 0.8)
    train_groups = set(unique_groups[:split])
    tr = df[groups.isin(train_groups)]
    te = df[~groups.isin(train_groups)]
    all_results["room_aware"] = run_protocol("room_aware", tr, te, df, include_room=True, args=args)

    print("\n=== E102 intra-room (80/20) ===")
    df_e = load_multiroom(["E102"])
    groups_e = df_e.groupby(["grid_cell", "campaign"]).ngroup()
    ug_e = groups_e.unique()
    rng2 = np.random.default_rng(SEED)
    rng2.shuffle(ug_e)
    split_e = int(len(ug_e) * 0.8)
    tr_g_e = set(ug_e[:split_e])
    tr_e = df_e[groups_e.isin(tr_g_e)]
    te_e = df_e[~groups_e.isin(tr_g_e)]
    all_results["e102"] = run_protocol("E102", tr_e, te_e, df_e, include_room=False, args=args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "deepsets_benchmark.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    print("\n=== SUMMARY ===")
    for proto, res in all_results.items():
        print(f"\n{proto}:")
        for m, v in res.items():
            print(f"  {m:<25} acc={v.get('cell_acc', float('nan')):.4f}  "
                  f"err={v.get('mean_error_m', float('nan')):.3f}m")


if __name__ == "__main__":
    main()
