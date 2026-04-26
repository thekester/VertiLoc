"""TabM and KAN benchmark for RSSI indoor localization.

TabM (Gorishniy et al., NeurIPS 2024 / ICLR 2025):
  Ensemble of small MLPs with parameter mixing — strong tabular deep learning
  baseline, often beats XGBoost while using fewer parameters than transformers.

KAN (Liu et al., MIT 2024):
  Kolmogorov-Arnold Networks replace fixed activations with learnable splines.
  Potentially more expressive on low-dimensional structured inputs like RSSI.

Protocols:
  - Room-aware (80/20 stratified, 29 730 samples)
  - Intra-room E102 (80/20)
  - LORO (generalization, 4 folds)

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localization.catalog import BENCHMARK_REPORT_DIR, FEATURE_COLUMNS, filter_room_campaigns
from localization.data import load_measurements

try:
    from kan import KAN as _KAN_impl
    HAS_KAN = True
except ImportError:
    HAS_KAN = False

warnings.filterwarnings("ignore")
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
    p90 = float(np.percentile(np.linalg.norm(pc[mask] - tc[mask], axis=1), 90)) if mask.sum() else float("nan")
    return {"cell_acc": acc, "mean_error_m": err, "p90_error_m": p90}


def build_X(df: pd.DataFrame, include_room: bool = True) -> np.ndarray:
    base = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not include_room:
        return base
    cols = sorted(c for c in df.columns if c.startswith("room_"))
    return np.hstack([base, df[cols].to_numpy(dtype=float)]) if cols else base


# ---------------------------------------------------------------------------
# TabM implementation
# ---------------------------------------------------------------------------

class TabM(nn.Module):
    """TabM: ensemble of k small MLPs sharing input, with per-model parameters.

    Simplified version of Gorishniy et al. (NeurIPS 2024): instead of full
    'MLP-mixer' weight sharing, we use k independent small MLPs whose outputs
    are averaged — this captures the ensemble spirit with low memory cost.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        k: int = 32,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k
        # Each ensemble member: input → hidden → ... → hidden → logit
        self.members = nn.ModuleList([
            self._make_mlp(n_features, n_classes, hidden, n_layers, dropout)
            for _ in range(k)
        ])

    @staticmethod
    def _make_mlp(n_in, n_out, hidden, n_layers, dropout):
        layers: list[nn.Module] = []
        dim = n_in
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden
        layers.append(nn.Linear(dim, n_out))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average logits across ensemble members
        logits = torch.stack([m(x) for m in self.members], dim=0).mean(0)
        return logits


def tabm_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    k: int = 16,
    hidden: int = 64,
    n_layers: int = 2,
    epochs: int = 150,
    lr: float = 1e-3,
    dropout: float = 0.1,
    random_state: int = SEED,
) -> np.ndarray:
    torch.manual_seed(random_state)
    le = LabelEncoder().fit(y_train)
    y_enc = le.transform(y_train)
    scaler = StandardScaler()
    X_tr = torch.from_numpy(scaler.fit_transform(X_train).astype(np.float32)).to(DEVICE)
    X_te = torch.from_numpy(scaler.transform(X_test).astype(np.float32)).to(DEVICE)
    y_t = torch.from_numpy(y_enc).long().to(DEVICE)

    model = TabM(X_tr.shape[1], len(le.classes_), k=k, hidden=hidden,
                 n_layers=n_layers, dropout=dropout).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n, batch = len(X_tr), min(512, len(X_tr))
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            idx = perm[s:s+batch]
            loss = F.cross_entropy(model(X_tr[idx]), y_t[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        pred_enc = model(X_te).argmax(1).cpu().numpy()
    return le.inverse_transform(pred_enc)


# ---------------------------------------------------------------------------
# KAN wrapper
# ---------------------------------------------------------------------------

def kan_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    grid: int = 5,
    k_spline: int = 3,
    hidden_nodes: int = 10,
    epochs: int = 50,
    lr: float = 1e-2,
    random_state: int = SEED,
) -> np.ndarray:
    if not HAS_KAN:
        raise ImportError("pykan not installed. Run: pip install pykan")

    torch.manual_seed(random_state)
    le = LabelEncoder().fit(y_train)
    y_enc = le.transform(y_train)
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X_tr = torch.from_numpy(scaler.fit_transform(X_train).astype(np.float32))
    X_te = torch.from_numpy(scaler.transform(X_test).astype(np.float32))
    y_t = torch.from_numpy(y_enc).long()

    n_features = X_tr.shape[1]
    # KAN architecture: [n_features, hidden_nodes, n_classes]
    model = _KAN_impl(
        width=[n_features, hidden_nodes, n_classes],
        grid=grid,
        k=k_spline,
        seed=random_state,
        device=DEVICE,
    )

    # Use KAN's built-in train method for small datasets
    dataset = {
        "train_input": X_tr,
        "train_label": y_t,
        "test_input": X_te,
        "test_label": torch.zeros(len(X_te), dtype=torch.long),
    }

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n, batch = len(X_tr), min(512, len(X_tr))
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            idx = perm[s:s+batch]
            logits = model(X_tr[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_enc = model(X_te).argmax(1).numpy()
    return le.inverse_transform(pred_enc)


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
    results = {}
    print(f"\n  [{label}]  n_train={len(X_train)}  n_test={len(X_test)}")

    # KNN baseline
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_train, y_train)
    results["KNN"] = _metrics(y_test, knn.predict(X_test), lookup)
    print(f"    KNN   : acc={results['KNN']['cell_acc']:.4f}  err={results['KNN']['mean_error_m']:.3f}m")

    # RF baseline
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    results["RF"] = _metrics(y_test, rf.predict(X_test), lookup)
    print(f"    RF    : acc={results['RF']['cell_acc']:.4f}  err={results['RF']['mean_error_m']:.3f}m")

    # TabM
    try:
        pred_tabm = tabm_fit_predict(
            X_train, y_train, X_test,
            k=args.tabm_k, hidden=args.tabm_hidden,
            epochs=args.tabm_epochs, lr=args.tabm_lr,
        )
        results["TabM"] = _metrics(y_test, pred_tabm, lookup)
        print(f"    TabM  : acc={results['TabM']['cell_acc']:.4f}  err={results['TabM']['mean_error_m']:.3f}m")
    except Exception as exc:
        print(f"    TabM  : FAILED — {exc}")
        results["TabM"] = {"cell_acc": float("nan"), "error": str(exc)}

    # KAN (skip if too many classes — KAN is slow on very wide output layers)
    n_classes = len(np.unique(y_train))
    if HAS_KAN and not args.skip_kan and n_classes <= args.kan_max_classes:
        try:
            pred_kan = kan_fit_predict(
                X_train, y_train, X_test,
                hidden_nodes=args.kan_hidden,
                epochs=args.kan_epochs,
                grid=args.kan_grid,
            )
            results["KAN"] = _metrics(y_test, pred_kan, lookup)
            print(f"    KAN   : acc={results['KAN']['cell_acc']:.4f}  err={results['KAN']['mean_error_m']:.3f}m")
        except Exception as exc:
            print(f"    KAN   : FAILED — {exc}")
            results["KAN"] = {"cell_acc": float("nan"), "error": str(exc)}
    elif n_classes > args.kan_max_classes:
        print(f"    KAN   : skipped ({n_classes} classes > --kan-max-classes={args.kan_max_classes})")

    return results


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

def run_room_aware(args: argparse.Namespace) -> dict:
    print("\n=== Room-aware (multi-room 80/20) ===")
    df = load_multiroom()
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["grid_cell"])
    lookup = _cell_lookup(df)
    return run_protocol("room_aware", build_X(tr), tr["grid_cell"].to_numpy(),
                        build_X(te), te["grid_cell"].to_numpy(), lookup, args)


def run_e102(args: argparse.Namespace) -> dict:
    print("\n=== E102 intra-room (80/20) ===")
    df = load_multiroom(["E102"])
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["grid_cell"])
    lookup = _cell_lookup(df)
    return run_protocol("E102", build_X(tr, include_room=False), tr["grid_cell"].to_numpy(),
                        build_X(te, include_room=False), te["grid_cell"].to_numpy(), lookup, args)


def run_loro(args: argparse.Namespace) -> dict:
    print("\n=== LORO (Leave-One-Room-Out) ===")
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)
    folds: dict[str, dict] = {}
    for held_out in rooms:
        tr = df[df["room"] != held_out]
        te = df[df["room"] == held_out]
        fold = run_protocol(
            f"LORO_{held_out}",
            build_X(tr, include_room=False), tr["grid_cell"].to_numpy(),
            build_X(te, include_room=False), te["grid_cell"].to_numpy(),
            lookup, args,
        )
        folds[held_out] = fold

    # Averages
    model_names = list(next(iter(folds.values())).keys())
    averages: dict[str, dict] = {}
    print("\n  LORO averages:")
    for m in model_names:
        accs = [folds[r][m]["cell_acc"] for r in rooms if m in folds[r]]
        valid = [a for a in accs if not np.isnan(a)]
        avg = float(np.mean(valid)) if valid else float("nan")
        std = float(np.std(valid)) if valid else float("nan")
        averages[m] = {"cell_acc_mean": avg, "cell_acc_std": std}
        print(f"    {m:<12} acc={avg:.4f} ± {std:.4f}")
    return {"folds": folds, "averages": averages}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TabM + KAN + TabPFN benchmark")
    # TabM
    parser.add_argument("--tabm-k", type=int, default=16, help="TabM ensemble size.")
    parser.add_argument("--tabm-hidden", type=int, default=64)
    parser.add_argument("--tabm-epochs", type=int, default=150)
    parser.add_argument("--tabm-lr", type=float, default=1e-3)
    # KAN
    parser.add_argument("--skip-kan", action="store_true")
    parser.add_argument("--kan-hidden", type=int, default=16)
    parser.add_argument("--kan-epochs", type=int, default=50)
    parser.add_argument("--kan-grid", type=int, default=5)
    parser.add_argument("--kan-max-classes", type=int, default=80,
                        help="Skip KAN if n_classes exceeds this (slow for very wide outputs).")
    # Protocols
    parser.add_argument("--skip-loro", action="store_true")
    parser.add_argument("--skip-room-aware", action="store_true")
    parser.add_argument("--skip-e102", action="store_true")
    args = parser.parse_args()

    if not HAS_KAN:
        print("[WARNING] pykan not installed — KAN will be skipped. pip install pykan")

    results: dict = {}
    if not args.skip_room_aware:
        results["room_aware"] = run_room_aware(args)
    if not args.skip_e102:
        results["e102"] = run_e102(args)
    if not args.skip_loro:
        results["loro"] = run_loro(args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "tabm_kan_benchmark.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    print("\n=== SUMMARY ===")
    for proto, res in results.items():
        if proto == "loro":
            print(f"\n{proto} (averages):")
            for m, v in res.get("averages", {}).items():
                print(f"  {m:<12} acc={v['cell_acc_mean']:.4f} ± {v['cell_acc_std']:.4f}")
        else:
            print(f"\n{proto}:")
            for m, v in res.items():
                print(f"  {m:<12} acc={v.get('cell_acc', float('nan')):.4f}  "
                      f"err={v.get('mean_error_m', float('nan')):.3f}m")


if __name__ == "__main__":
    main()
