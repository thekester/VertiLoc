"""Spatially-aware loss + Hierarchical localization benchmark.

Two complementary ideas tested here:

1. Spatial Cross-Entropy Loss
   Standard CE treats every wrong cell equally. Here we weight the loss by the
   spatial distance between the predicted and true cell, so the model is
   penalised more for large spatial errors. This aligns training with the
   mean_error_m metric used at evaluation time.

   loss_spatial(i) = sum_j  P(j|x_i) * dist(cell_j, cell_true_i)  (differentiable)
   Final loss = alpha * CE + (1-alpha) * mean spatial penalty

2. Hierarchical Localization (Zone → Cell)
   Partition the grid into coarse zones (e.g. 2×2 super-cells), then:
     a) Train a zone classifier (cheap)
     b) For each sample, restrict the cell search to its predicted zone
   Motivation: misclassifications are often large jumps; restricting the
   search space at inference reduces catastrophic errors.

Protocols: room-aware (80/20) and E102 intra-room (80/20).
LORO is skipped by default (--skip-loro flag) for speed.
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
# Data helpers
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
# Spatial distance matrix between cells
# ---------------------------------------------------------------------------

def build_cell_dist_matrix(
    le: LabelEncoder,
    lookup: pd.DataFrame,
) -> torch.Tensor:
    """(n_classes, n_classes) pairwise Euclidean distance in metres."""
    coords = lookup.reindex(le.classes_)[["coord_x_m", "coord_y_m"]].to_numpy(dtype=np.float32)
    coords_t = torch.from_numpy(coords)
    return torch.cdist(coords_t, coords_t, p=2)  # (C, C)


# ---------------------------------------------------------------------------
# MLP classifier
# ---------------------------------------------------------------------------

class LocalizationMLP(nn.Module):
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
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Spatial CE loss
# ---------------------------------------------------------------------------

def spatial_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    dist_matrix: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """alpha * CE + (1-alpha) * E[dist(pred, true)]."""
    ce = F.cross_entropy(logits, labels)

    probs = F.softmax(logits, dim=1)  # (B, C)
    # dist_matrix[labels[i]] gives distances from true cell to all cells
    true_dists = dist_matrix[labels]   # (B, C)
    spatial_penalty = (probs * true_dists).sum(dim=1).mean()

    return alpha * ce + (1 - alpha) * spatial_penalty


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    lookup: pd.DataFrame,
    *,
    use_spatial_loss: bool = False,
    alpha: float = 0.5,
    epochs: int = 300,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 512,
    random_state: int = SEED,
) -> tuple[LocalizationMLP, StandardScaler]:
    torch.manual_seed(random_state)
    y_enc = le.transform(y)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X).astype(np.float32)
    n_classes = len(le.classes_)

    model = LocalizationMLP(X_s.shape[1], n_classes, hidden=hidden).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_s).to(DEVICE)
    y_t = torch.from_numpy(y_enc.astype(np.int64)).to(DEVICE)
    dist_mat = build_cell_dist_matrix(le, lookup).to(DEVICE) if use_spatial_loss else None

    n = len(X_t)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            logits = model(X_t[idx])
            if use_spatial_loss and dist_mat is not None:
                loss = spatial_ce_loss(logits, y_t[idx], dist_mat, alpha=alpha)
            else:
                loss = F.cross_entropy(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model, scaler


@torch.no_grad()
def mlp_predict(
    model: LocalizationMLP,
    scaler: StandardScaler,
    X: np.ndarray,
    le: LabelEncoder,
) -> np.ndarray:
    model.eval()
    X_s = torch.from_numpy(scaler.transform(X).astype(np.float32)).to(DEVICE)
    logits = model(X_s)
    idx = logits.argmax(dim=1).cpu().numpy()
    return le.inverse_transform(idx)


# ---------------------------------------------------------------------------
# Hierarchical localization
# ---------------------------------------------------------------------------

def assign_zones(
    cells: np.ndarray,
    lookup: pd.DataFrame,
    zone_size: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Assign a zone ID to each cell based on coarse spatial binning."""
    coords = lookup.reindex(cells)[["coord_x_m", "coord_y_m"]].to_numpy(dtype=float)
    zx = np.floor(coords[:, 0] / zone_size[0]).astype(int)
    zy = np.floor(coords[:, 1] / zone_size[1]).astype(int)
    # Unique integer zone ID
    zx_min, zy_min = zx.min(), zy.min()
    n_zx = zx.max() - zx_min + 1
    return (zx - zx_min) * (zy.max() - zy_min + 2) + (zy - zy_min)


def hierarchical_predict(
    zone_model: LocalizationMLP,
    zone_scaler: StandardScaler,
    zone_le: LabelEncoder,
    cell_model: LocalizationMLP,
    cell_scaler: StandardScaler,
    cell_le: LabelEncoder,
    X_test: np.ndarray,
    cell_zones: dict,  # cell -> zone_id
) -> np.ndarray:
    """Predict zone first, then restrict cell search to that zone."""
    zone_model.eval()
    cell_model.eval()

    with torch.no_grad():
        X_s = torch.from_numpy(zone_scaler.transform(X_test).astype(np.float32)).to(DEVICE)
        zone_logits = zone_model(X_s)
        pred_zones = zone_le.inverse_transform(zone_logits.argmax(1).cpu().numpy())

        X_s2 = torch.from_numpy(cell_scaler.transform(X_test).astype(np.float32)).to(DEVICE)
        cell_logits = cell_model(X_s2).cpu().numpy()

    preds = []
    for i, zone in enumerate(pred_zones):
        # Mask logits to only cells in predicted zone
        valid_mask = np.array([
            cell_zones.get(c, -1) == zone for c in cell_le.classes_
        ], dtype=bool)
        if not valid_mask.any():
            valid_mask[:] = True  # fall back to all cells
        masked = cell_logits[i].copy()
        masked[~valid_mask] = -1e9
        preds.append(cell_le.classes_[masked.argmax()])

    return np.array(preds)


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

    # Shared label encoder fitted on training cells
    le = LabelEncoder().fit(y_train)

    # ---- KNN baseline ----
    scaler_knn = StandardScaler()
    X_tr_s = scaler_knn.fit_transform(X_train)
    X_te_s = scaler_knn.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_tr_s, y_train)
    results["KNN"] = _metrics(y_test, knn.predict(X_te_s), lookup)
    print(f"    KNN              acc={results['KNN']['cell_acc']:.4f}  err={results['KNN']['mean_error_m']:.3f}m")

    # ---- Plain MLP (CE loss) ----
    mlp, scaler_mlp = train_mlp(
        X_train, y_train, le, lookup,
        use_spatial_loss=False,
        epochs=args.epochs, lr=args.lr, hidden=args.hidden, batch_size=args.batch_size,
    )
    pred_mlp = mlp_predict(mlp, scaler_mlp, X_test, le)
    results["MLP_CE"] = _metrics(y_test, pred_mlp, lookup)
    print(f"    MLP_CE           acc={results['MLP_CE']['cell_acc']:.4f}  err={results['MLP_CE']['mean_error_m']:.3f}m")

    # ---- Spatial-loss MLP ----
    mlp_sp, scaler_sp = train_mlp(
        X_train, y_train, le, lookup,
        use_spatial_loss=True, alpha=args.alpha,
        epochs=args.epochs, lr=args.lr, hidden=args.hidden, batch_size=args.batch_size,
    )
    pred_sp = mlp_predict(mlp_sp, scaler_sp, X_test, le)
    results["MLP_SpatialCE"] = _metrics(y_test, pred_sp, lookup)
    print(f"    MLP_SpatialCE    acc={results['MLP_SpatialCE']['cell_acc']:.4f}  err={results['MLP_SpatialCE']['mean_error_m']:.3f}m")

    # ---- Hierarchical localization ----
    train_cells = y_train
    zone_labels = assign_zones(train_cells, lookup, zone_size=(args.zone_size, args.zone_size))
    test_cells_z  = y_test
    test_zones    = assign_zones(test_cells_z, lookup, zone_size=(args.zone_size, args.zone_size))

    zone_le = LabelEncoder().fit(zone_labels)
    y_zone_enc = zone_le.transform(zone_labels)
    n_zones = len(zone_le.classes_)

    # cell -> zone mapping
    all_cells_z = np.unique(np.concatenate([train_cells, y_test]))
    all_cell_zones_arr = assign_zones(all_cells_z, lookup, zone_size=(args.zone_size, args.zone_size))
    cell_to_zone = {c: int(z) for c, z in zip(all_cells_z, all_cell_zones_arr)}

    # Zone classifier (separate MLP, same architecture but smaller)
    zone_mlp = LocalizationMLP(X_train.shape[1], n_zones, hidden=64).to(DEVICE)
    zone_optimizer = optim.Adam(zone_mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    z_scaler = StandardScaler()
    X_tr_zs = z_scaler.fit_transform(X_train).astype(np.float32)
    X_t_z = torch.from_numpy(X_tr_zs).to(DEVICE)
    y_z_t = torch.from_numpy(y_zone_enc.astype(np.int64)).to(DEVICE)
    zone_scheduler = optim.lr_scheduler.CosineAnnealingLR(zone_optimizer, T_max=args.epochs)
    zone_mlp.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(X_t_z), device=DEVICE)
        for s in range(0, len(X_t_z), args.batch_size):
            idx = perm[s:s + args.batch_size]
            loss = F.cross_entropy(zone_mlp(X_t_z[idx]), y_z_t[idx])
            zone_optimizer.zero_grad(); loss.backward(); zone_optimizer.step()
        zone_scheduler.step()

    pred_hier = hierarchical_predict(
        zone_mlp, z_scaler, zone_le,
        mlp, scaler_mlp, le,
        X_test, cell_to_zone,
    )
    results["Hierarchical"] = _metrics(y_test, pred_hier, lookup)
    print(f"    Hierarchical     acc={results['Hierarchical']['cell_acc']:.4f}  err={results['Hierarchical']['mean_error_m']:.3f}m")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spatial-loss + Hierarchical localization")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight of CE in spatial loss (1=pure CE, 0=pure spatial)")
    parser.add_argument("--zone-size", type=float, default=1.5,
                        help="Zone side length in metres for hierarchical localization")
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
        folds = {}
        for held_out in rooms:
            tr_l = df[df["room"] != held_out]
            te_l = df[df["room"] == held_out]
            folds[held_out] = run_protocol(
                f"LORO_{held_out}", tr_l, te_l, df, include_room=False, args=args
            )
        model_names = list(next(iter(folds.values())).keys())
        avgs = {}
        print("\n  LORO averages:")
        for m in model_names:
            accs = [folds[r][m]["cell_acc"] for r in rooms if m in folds[r]]
            valid = [a for a in accs if not np.isnan(a)]
            avg = float(np.mean(valid)) if valid else float("nan")
            avgs[m] = {"cell_acc_mean": avg}
            print(f"    {m:<20} acc={avg:.4f}")
        all_results["loro"] = {"folds": folds, "averages": avgs}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "spatial_loss_benchmark.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    print("\n=== SUMMARY ===")
    for proto, res in all_results.items():
        if proto == "loro":
            print(f"\n{proto} averages:")
            for m, v in res.get("averages", {}).items():
                print(f"  {m:<22} acc={v['cell_acc_mean']:.4f}")
        else:
            print(f"\n{proto}:")
            for m, v in res.items():
                print(f"  {m:<22} acc={v.get('cell_acc', float('nan')):.4f}  "
                      f"err={v.get('mean_error_m', float('nan')):.3f}m")


if __name__ == "__main__":
    main()
