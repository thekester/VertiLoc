"""Stacked Autoencoder + Multi-label classification for RSSI localization.

Architecture (inspired by Nabati et al., Nature/Sensors 2025):
  1. SAE encoder: RSSI → latent representation (unsupervised pre-training)
  2. Multi-label heads: latent → row class + column class (separate sub-problems)
  3. Cell reconstruction: (predicted_row, predicted_col) → cell ID

Advantages over direct cell classification:
  - Decomposes N_cells classes into N_rows + N_cols (e.g., 75 → 5+15)
  - Row and column may be partially independent signals → better generalization
  - SAE regularization enforces a compact, denoised representation

Tested on:
  - Room-aware (80/20 stratified, with one-hot room)
  - Intra-room E102 (80/20)
  - LORO (generalization under domain shift)
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def parse_cell_labels(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parse 'row_col' strings into (row_array, col_array)."""
    rows, cols = [], []
    for cell in cells:
        parts = str(cell).split("_")
        rows.append(int(parts[0]))
        cols.append(int(parts[1]))
    return np.array(rows), np.array(cols)


# ---------------------------------------------------------------------------
# SAE + Multi-label model
# ---------------------------------------------------------------------------

class SAEMultiLabel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_row_classes: int,
        n_col_classes: int,
        hidden: int = 32,
        bottleneck: int = 16,
    ):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
        )
        # Decoder (for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )
        # Classification heads
        self.row_head = nn.Linear(bottleneck, n_row_classes)
        self.col_head = nn.Linear(bottleneck, n_col_classes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        row_logits = self.row_head(z)
        col_logits = self.col_head(z)
        return z, x_recon, row_logits, col_logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_sae(
    X: np.ndarray,
    y_rows: np.ndarray,
    y_cols: np.ndarray,
    n_row_classes: int,
    n_col_classes: int,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    hidden: int = 32,
    bottleneck: int = 16,
    recon_weight: float = 0.3,
    random_state: int = SEED,
) -> tuple[SAEMultiLabel, StandardScaler]:
    torch.manual_seed(random_state)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X).astype(np.float32)

    model = SAEMultiLabel(
        X_s.shape[1], n_row_classes, n_col_classes,
        hidden=hidden, bottleneck=bottleneck,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_s).to(DEVICE)
    y_row_t = torch.from_numpy(y_rows.astype(np.int64)).to(DEVICE)
    y_col_t = torch.from_numpy(y_cols.astype(np.int64)).to(DEVICE)

    # Shift column labels to start from 0
    col_offset = int(y_cols.min())
    y_col_t = y_col_t - col_offset

    n, batch = len(X_t), min(512, len(X_t))
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            z, x_recon, row_logits, col_logits = model(X_t[idx])
            loss_row = F.cross_entropy(row_logits, y_row_t[idx])
            loss_col = F.cross_entropy(col_logits, y_col_t[idx])
            loss_recon = F.mse_loss(x_recon, X_t[idx])
            loss = loss_row + loss_col + recon_weight * loss_recon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Store col_offset in scaler for later use
    scaler.col_offset_ = col_offset
    return model, scaler


@torch.no_grad()
def sae_predict(
    model: SAEMultiLabel,
    scaler: StandardScaler,
    X_test: np.ndarray,
    valid_cells: np.ndarray,
    *,
    row_weight: float = 1.0,
    col_weight: float = 1.0,
) -> np.ndarray:
    """Predict cells by combining row + col softmax probabilities."""
    model.eval()
    X_s = torch.from_numpy(scaler.transform(X_test).astype(np.float32)).to(DEVICE)
    _, _, row_logits, col_logits = model(X_s)

    row_probs = F.softmax(row_logits * row_weight, dim=1).cpu().numpy()  # (n, n_rows)
    col_probs = F.softmax(col_logits * col_weight, dim=1).cpu().numpy()  # (n, n_cols)
    col_offset = getattr(scaler, "col_offset_", 0)

    preds = []
    for i in range(len(X_test)):
        # For each valid cell, compute P(row) * P(col) and pick argmax
        best_cell, best_score = None, -1.0
        for cell in valid_cells:
            parts = str(cell).split("_")
            r, c = int(parts[0]), int(parts[1])
            c_idx = c - col_offset
            r_prob = float(row_probs[i, r]) if r < row_probs.shape[1] else 0.0
            c_prob = float(col_probs[i, c_idx]) if 0 <= c_idx < col_probs.shape[1] else 0.0
            score = r_prob * c_prob
            if score > best_score:
                best_score, best_cell = score, cell
        preds.append(best_cell)
    return np.array(preds)


@torch.no_grad()
def sae_encode(model: SAEMultiLabel, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_s = torch.from_numpy(scaler.transform(X).astype(np.float32)).to(DEVICE)
    return model.encode(X_s).cpu().numpy()


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
    valid_cells = np.unique(y_train)

    print(f"\n  [{label}]  n_train={len(X_train)}  n_test={len(X_test)}")
    results = {}

    # KNN baseline
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_train, y_train)
    results["KNN"] = _metrics(y_test, knn.predict(X_test), lookup)
    print(f"    KNN            acc={results['KNN']['cell_acc']:.4f}  err={results['KNN']['mean_error_m']:.3f}m")

    # SAE + multi-label
    y_rows, y_cols = parse_cell_labels(y_train)
    n_row_classes = int(y_rows.max()) + 1
    n_col_classes = int(y_cols.max()) - int(y_cols.min()) + 1

    model, scaler = train_sae(
        X_train, y_rows, y_cols, n_row_classes, n_col_classes,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        bottleneck=args.bottleneck,
        recon_weight=args.recon_weight,
        random_state=SEED,
    )
    pred_sae = sae_predict(model, scaler, X_test, valid_cells)
    results["SAE+MultiLabel"] = _metrics(y_test, pred_sae, lookup)
    print(f"    SAE+MultiLabel acc={results['SAE+MultiLabel']['cell_acc']:.4f}  err={results['SAE+MultiLabel']['mean_error_m']:.3f}m")

    # SAE embedding + KNN (use latent space for KNN)
    Z_train = sae_encode(model, scaler, X_train)
    Z_test  = sae_encode(model, scaler, X_test)
    knn_latent = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn_latent.fit(Z_train, y_train)
    results["SAE+KNN"] = _metrics(y_test, knn_latent.predict(Z_test), lookup)
    print(f"    SAE+KNN        acc={results['SAE+KNN']['cell_acc']:.4f}  err={results['SAE+KNN']['mean_error_m']:.3f}m")

    return results


def main():
    parser = argparse.ArgumentParser(description="SAE + Multi-label benchmark")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--bottleneck", type=int, default=16)
    parser.add_argument("--recon-weight", type=float, default=0.3,
                        help="Weight of reconstruction loss vs classification.")
    parser.add_argument("--skip-loro", action="store_true")
    args = parser.parse_args()

    all_results: dict = {}

    # Room-aware
    print("\n=== Room-aware (multi-room, 80/20) ===")
    df = load_multiroom()
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["grid_cell"])
    all_results["room_aware"] = run_protocol("room_aware", tr, te, df, include_room=True, args=args)

    # E102 intra-room
    print("\n=== E102 intra-room (80/20) ===")
    df_e = load_multiroom(["E102"])
    tr_e, te_e = train_test_split(df_e, test_size=0.2, random_state=SEED, stratify=df_e["grid_cell"])
    all_results["e102"] = run_protocol("E102", tr_e, te_e, df_e, include_room=False, args=args)

    # LORO
    if not args.skip_loro:
        print("\n=== LORO ===")
        df = load_multiroom()
        rooms = sorted(df["room"].unique())
        folds, averages = {}, {}
        for held_out in rooms:
            tr = df[df["room"] != held_out]
            te = df[df["room"] == held_out]
            fold = run_protocol(f"LORO_{held_out}", tr, te, df, include_room=False, args=args)
            folds[held_out] = fold
        for m in ["KNN", "SAE+MultiLabel", "SAE+KNN"]:
            accs = [folds[r][m]["cell_acc"] for r in rooms if m in folds[r]]
            valid = [a for a in accs if not np.isnan(a)]
            averages[m] = {"cell_acc_mean": float(np.mean(valid)) if valid else float("nan")}
        print("\n  LORO averages:")
        for m, v in averages.items():
            print(f"    {m:<20} acc={v['cell_acc_mean']:.4f}")
        all_results["loro"] = {"folds": folds, "averages": averages}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "sae_multilabel_benchmark.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    print("\n=== SUMMARY ===")
    for proto, res in all_results.items():
        if proto == "loro":
            print(f"\n{proto} averages:")
            for m, v in res.get("averages", {}).items():
                print(f"  {m:<20} acc={v['cell_acc_mean']:.4f}")
        else:
            print(f"\n{proto}:")
            for m, v in res.items():
                print(f"  {m:<20} acc={v.get('cell_acc', float('nan')):.4f}  err={v.get('mean_error_m', float('nan')):.3f}m")


if __name__ == "__main__":
    main()
