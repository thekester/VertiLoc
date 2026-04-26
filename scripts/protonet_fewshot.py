"""Prototypical Networks for few-shot room calibration.

Trains a shared embedding on seen rooms (episodic meta-learning), then
evaluates how accuracy grows as K labeled samples per cell are added for
a NEW, never-seen room.

Key result: accuracy vs K curve (K=1,3,5,10,25) showing how much
calibration effort can be saved vs full 25-sample baseline.

Protocol:
  - Meta-train: all rooms except held-out
  - Meta-test : held-out room, K support samples per cell
  - Repeated for all 4 LORO folds

Reference: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

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
    return {"cell_acc": acc, "mean_error_m": err, "n": int(len(y_true))}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ProtoEncoder(nn.Module):
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
        return self.net(x)


# ---------------------------------------------------------------------------
# Episodic training helpers
# ---------------------------------------------------------------------------

def _sample_episode(
    X: np.ndarray,
    y: np.ndarray,
    n_way: int,
    n_support: int,
    n_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a few-shot episode with n_way classes."""
    classes = np.unique(y)
    if len(classes) < n_way:
        n_way = len(classes)
    chosen = rng.choice(classes, size=n_way, replace=False)

    X_sup, y_sup_enc, X_qry, y_qry_enc = [], [], [], []
    for enc, cls in enumerate(chosen):
        idx = np.where(y == cls)[0]
        if len(idx) < n_support + n_query:
            n_q = max(1, len(idx) - n_support)
            n_s = min(n_support, len(idx) - n_q)
        else:
            n_s, n_q = n_support, n_query
        perm = rng.permutation(len(idx))
        sup_idx = idx[perm[:n_s]]
        qry_idx = idx[perm[n_s:n_s + n_q]]
        X_sup.append(X[sup_idx])
        y_sup_enc.extend([enc] * n_s)
        X_qry.append(X[qry_idx])
        y_qry_enc.extend([enc] * n_q)

    return (
        np.vstack(X_sup), np.array(y_sup_enc, dtype=np.int64),
        np.vstack(X_qry), np.array(y_qry_enc, dtype=np.int64),
    )


def proto_loss(
    encoder: ProtoEncoder,
    X_sup: torch.Tensor,
    y_sup: torch.Tensor,
    X_qry: torch.Tensor,
    y_qry: torch.Tensor,
    n_way: int,
) -> torch.Tensor:
    z_sup = encoder(X_sup)
    z_qry = encoder(X_qry)

    # Prototype = mean embedding per class
    prototypes = torch.stack([
        z_sup[y_sup == c].mean(0) for c in range(n_way)
    ])  # (n_way, emb_dim)

    # Negative squared euclidean distances as logits
    dists = torch.cdist(z_qry, prototypes)          # (n_qry, n_way)
    logits = -dists
    return F.cross_entropy(logits, y_qry)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def meta_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_episodes: int = 2000,
    n_way: int = 20,
    n_support: int = 5,
    n_query: int = 10,
    hidden: int = 64,
    emb_dim: int = 32,
    lr: float = 1e-3,
    random_state: int = SEED,
    device: str = "cpu",
) -> tuple[ProtoEncoder, StandardScaler]:
    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train).astype(np.float32)

    encoder = ProtoEncoder(X_s.shape[1], hidden=hidden, emb_dim=emb_dim).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes)

    encoder.train()
    losses = []
    for ep in range(n_episodes):
        X_sup, y_sup, X_qry, y_qry = _sample_episode(
            X_s, y_train, n_way, n_support, n_query, rng
        )
        n_w = len(np.unique(y_sup))
        X_sup_t = torch.from_numpy(X_sup).to(device)
        y_sup_t = torch.from_numpy(y_sup).to(device)
        X_qry_t = torch.from_numpy(X_qry).to(device)
        y_qry_t = torch.from_numpy(y_qry).to(device)

        optimizer.zero_grad()
        loss = proto_loss(encoder, X_sup_t, y_sup_t, X_qry_t, y_qry_t, n_w)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

        if (ep + 1) % 500 == 0:
            print(f"    episode {ep+1}/{n_episodes}  loss={np.mean(losses[-100:]):.4f}")

    return encoder, scaler


# ---------------------------------------------------------------------------
# Few-shot inference
# ---------------------------------------------------------------------------

def protonet_predict_k(
    encoder: ProtoEncoder,
    scaler: StandardScaler,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_test: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Predict test labels using K support samples per class as prototypes."""
    encoder.eval()
    X_sup_s = scaler.transform(X_support).astype(np.float32)
    X_te_s = scaler.transform(X_test).astype(np.float32)

    classes = np.unique(y_support)
    with torch.no_grad():
        z_sup = encoder(torch.from_numpy(X_sup_s).to(device))
        z_te = encoder(torch.from_numpy(X_te_s).to(device))

    prototypes = torch.stack([
        z_sup[torch.from_numpy(y_support == c)].mean(0) for c in classes
    ])  # (n_classes, emb_dim)

    dists = torch.cdist(z_te, prototypes.to(device))
    pred_idx = dists.argmin(dim=1).cpu().numpy()
    return classes[pred_idx]


def raw_centroid_predict_k(
    scaler: StandardScaler,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Nearest target-room centroid in standardized RSSI space."""
    X_sup_s = scaler.transform(X_support).astype(np.float32)
    X_te_s = scaler.transform(X_test).astype(np.float32)

    classes = np.unique(y_support)
    prototypes = np.stack([
        X_sup_s[y_support == c].mean(axis=0) for c in classes
    ])
    dists = np.linalg.norm(X_te_s[:, None, :] - prototypes[None, :, :], axis=2)
    return classes[dists.argmin(axis=1)]


def target_knn_predict_k(
    scaler: StandardScaler,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """KNN trained only on the labeled support samples from the target room."""
    from sklearn.neighbors import KNeighborsClassifier

    X_sup_s = scaler.transform(X_support).astype(np.float32)
    X_te_s = scaler.transform(X_test).astype(np.float32)
    n_neighbors = min(3, len(X_sup_s))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(X_sup_s, y_support)
    return knn.predict(X_te_s)


# ---------------------------------------------------------------------------
# LORO few-shot evaluation
# ---------------------------------------------------------------------------

def run_loro_fewshot(args: argparse.Namespace) -> dict:
    print("\n=== ProtoNet Few-Shot LORO ===")
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = df["grid_cell"].to_numpy()

    k_values = args.k_values
    all_folds: dict[str, dict] = {}

    for fold_idx, held_out in enumerate(rooms):
        print(f"\n  Fold: held_out={held_out}")
        train_mask = df["room"].to_numpy() != held_out
        test_mask  = df["room"].to_numpy() == held_out

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_all[test_mask]
        y_test  = y_all[test_mask]
        df_test = df[test_mask]

        print(f"    Meta-training on {X_train.shape[0]} samples...")
        encoder, scaler = meta_train(
            X_train, y_train,
            n_episodes=args.episodes,
            n_way=args.n_way,
            n_support=args.n_support,
            n_query=args.n_query,
            hidden=args.hidden,
            emb_dim=args.emb_dim,
            lr=args.lr,
            random_state=args.seed + fold_idx * 100,
        )

        # Also KNN baseline (no labels in target)
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn.fit(X_train, y_train)
        fold_results: dict = {
            "KNN_vanilla_LORO": _metrics(y_test, knn.predict(X_test), lookup)
        }

        # Few-shot: for each K, randomly sample K per cell from test set
        cells_in_target = np.unique(y_test)
        for k in k_values:
            proto_accs, proto_errs = [], []
            centroid_accs, centroid_errs = [], []
            target_knn_accs, target_knn_errs = [], []
            for trial in range(args.n_trials):
                # Sample K support examples per cell (stratified)
                sup_idx, qry_idx = [], []
                trial_rng = np.random.default_rng(args.seed + fold_idx * 10000 + trial * 1000)
                for cell in cells_in_target:
                    cell_idx = np.where(y_test == cell)[0]
                    if len(cell_idx) == 0:
                        continue
                    k_eff = min(k, len(cell_idx))
                    perm = trial_rng.permutation(len(cell_idx))
                    sup_idx.extend(cell_idx[perm[:k_eff]])
                    qry_idx.extend(cell_idx[perm[k_eff:]])

                if not qry_idx:
                    continue

                X_sup = X_test[sup_idx]
                y_sup = y_test[sup_idx]
                X_qry = X_test[qry_idx]
                y_qry = y_test[qry_idx]

                pred = protonet_predict_k(encoder, scaler, X_sup, y_sup, X_qry)
                m = _metrics(y_qry, pred, lookup)
                proto_accs.append(m["cell_acc"])
                proto_errs.append(m["mean_error_m"])

                pred_centroid = raw_centroid_predict_k(scaler, X_sup, y_sup, X_qry)
                m_centroid = _metrics(y_qry, pred_centroid, lookup)
                centroid_accs.append(m_centroid["cell_acc"])
                centroid_errs.append(m_centroid["mean_error_m"])

                pred_target_knn = target_knn_predict_k(scaler, X_sup, y_sup, X_qry)
                m_target_knn = _metrics(y_qry, pred_target_knn, lookup)
                target_knn_accs.append(m_target_knn["cell_acc"])
                target_knn_errs.append(m_target_knn["mean_error_m"])

            fold_results[f"ProtoNet_K{k}"] = {
                "cell_acc": float(np.mean(proto_accs)) if proto_accs else float("nan"),
                "cell_acc_std": float(np.std(proto_accs)) if proto_accs else float("nan"),
                "mean_error_m": float(np.mean(proto_errs)) if proto_errs else float("nan"),
                "k": k,
                "n_trials": len(proto_accs),
            }
            fold_results[f"RawCentroid_K{k}"] = {
                "cell_acc": float(np.mean(centroid_accs)) if centroid_accs else float("nan"),
                "cell_acc_std": float(np.std(centroid_accs)) if centroid_accs else float("nan"),
                "mean_error_m": float(np.mean(centroid_errs)) if centroid_errs else float("nan"),
                "k": k,
                "n_trials": len(centroid_accs),
            }
            fold_results[f"TargetKNN_K{k}"] = {
                "cell_acc": float(np.mean(target_knn_accs)) if target_knn_accs else float("nan"),
                "cell_acc_std": float(np.std(target_knn_accs)) if target_knn_accs else float("nan"),
                "mean_error_m": float(np.mean(target_knn_errs)) if target_knn_errs else float("nan"),
                "k": k,
                "n_trials": len(target_knn_accs),
            }
            print(
                f"    K={k:>2}  acc={fold_results[f'ProtoNet_K{k}']['cell_acc']:.4f}"
                f" ± {fold_results[f'ProtoNet_K{k}']['cell_acc_std']:.4f}"
                f"  err={fold_results[f'ProtoNet_K{k}']['mean_error_m']:.3f}m"
                f"  | centroid={fold_results[f'RawCentroid_K{k}']['cell_acc']:.4f}"
                f"  targetKNN={fold_results[f'TargetKNN_K{k}']['cell_acc']:.4f}"
            )

        all_folds[held_out] = fold_results

    # Cross-room averages for each K
    averages: dict[str, dict] = {}
    print("\n  Cross-room averages:")
    for method in ("ProtoNet", "RawCentroid", "TargetKNN"):
        for k in k_values:
            key = f"{method}_K{k}"
            accs = [all_folds[r][key]["cell_acc"] for r in rooms if key in all_folds[r]]
            errs = [all_folds[r][key]["mean_error_m"] for r in rooms if key in all_folds[r]]
            valid_a = [a for a in accs if not np.isnan(a)]
            valid_e = [e for e in errs if not np.isnan(e)]
            avg_a = float(np.mean(valid_a)) if valid_a else float("nan")
            avg_e = float(np.mean(valid_e)) if valid_e else float("nan")
            averages[key] = {"cell_acc_mean": avg_a, "mean_error_m": avg_e}
            print(f"    {key:<18} acc_mean={avg_a:.4f}  err_mean={avg_e:.3f}m")

    # KNN LORO baseline average
    knn_accs = [all_folds[r]["KNN_vanilla_LORO"]["cell_acc"] for r in rooms]
    print(f"    KNN LORO baseline: acc={np.mean(knn_accs):.4f}")

    return {
        "config": vars(args),
        "folds": all_folds,
        "k_values": k_values,
        "averages": averages,
        "knn_loro_baseline_acc_mean": float(np.mean(knn_accs)),
    }


def main():
    parser = argparse.ArgumentParser(description="ProtoNet few-shot LORO localization")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Meta-training episodes per fold.")
    parser.add_argument("--n-way", type=int, default=20,
                        help="Number of classes per episode.")
    parser.add_argument("--n-support", type=int, default=5,
                        help="Support samples per class per episode.")
    parser.add_argument("--n-query", type=int, default=10,
                        help="Query samples per class per episode.")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Random trials per K value to reduce variance.")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10, 25])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name", default="protonet_fewshot.json")
    args = parser.parse_args()

    results = run_loro_fewshot(args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / args.output_name
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
