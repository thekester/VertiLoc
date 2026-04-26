"""Conformal Prediction / Selective Classification for RSSI localization.

Standard classifiers always return a single cell prediction, even when the
RSSI fingerprint is ambiguous. This script applies two complementary ideas:

1. Split Conformal Prediction (SCP)
   Use a calibration set to compute cell-level nonconformity scores
   (1 - softmax[true_class]). At test time, output a *prediction set* — the
   smallest set of cells that contains the true cell with probability ≥ 1-α.
   This gives a principled coverage guarantee without assuming i.i.d. domains.

   Metrics:
     - Marginal coverage (should be ≥ 1-α)
     - Average prediction set size (smaller = more informative)
     - "Empty prediction" rate (0 if threshold is correct)

2. Selective Classification (abstain-or-predict)
   A classifier with a reject option: abstain when max-softmax < threshold τ.
   Trades off coverage (% answered) vs accuracy-when-answered.
   We sweep τ ∈ [0.1, 0.9] and report the coverage-accuracy frontier.

Both methods are model-agnostic and applied on top of a simple MLP or KNN.

Protocols: room-aware (80/20 with calibration split) and E102 intra-room.
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
# Base MLP
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
            nn.Linear(hidden // 2 * 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 512,
    random_state: int = SEED,
) -> tuple[LocalizationMLP, StandardScaler]:
    torch.manual_seed(random_state)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X).astype(np.float32)

    model = LocalizationMLP(X_s.shape[1], n_classes, hidden=hidden).to(DEVICE)
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


@torch.no_grad()
def get_softmax(
    model: LocalizationMLP,
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    model.eval()
    X_s = torch.from_numpy(scaler.transform(X).astype(np.float32)).to(DEVICE)
    return F.softmax(model(X_s), dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Split Conformal Prediction
# ---------------------------------------------------------------------------

def conformal_calibrate(
    softmax_cal: np.ndarray,
    y_cal_enc: np.ndarray,
) -> np.ndarray:
    """Compute nonconformity scores = 1 - softmax[true_class] on calibration set."""
    n = len(y_cal_enc)
    scores = 1.0 - softmax_cal[np.arange(n), y_cal_enc]
    return scores


def conformal_threshold(scores: np.ndarray, alpha: float = 0.1) -> float:
    """Quantile-corrected threshold for (1-alpha) marginal coverage."""
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, min(q_level, 1.0)))


def conformal_predict_sets(
    softmax_test: np.ndarray,
    tau: float,
    le: LabelEncoder,
) -> list[list[str]]:
    """Return prediction sets: cells with score <= tau."""
    sets = []
    for probs in softmax_test:
        scores = 1.0 - probs
        in_set = np.where(scores <= tau)[0]
        sets.append(list(le.inverse_transform(in_set)))
    return sets


def conformal_metrics(
    pred_sets: list[list[str]],
    y_true: np.ndarray,
    lookup: pd.DataFrame,
    alpha: float,
) -> dict:
    coverage = float(np.mean([y in s for y, s in zip(y_true, pred_sets)]))
    set_sizes = [len(s) for s in pred_sets]
    empty_rate = float(np.mean([len(s) == 0 for s in pred_sets]))

    # Spatial coverage: max distance from any set member to true cell
    coverages_m = []
    tc = lookup.reindex(y_true)[["coord_x_m", "coord_y_m"]].to_numpy()
    for i, (s, true) in enumerate(zip(pred_sets, y_true)):
        if not s:
            continue
        pc = lookup.reindex(s)[["coord_x_m", "coord_y_m"]].to_numpy()
        mask = ~np.isnan(pc).any(1)
        if mask.sum() == 0:
            continue
        dists = np.linalg.norm(pc[mask] - tc[i], axis=1)
        coverages_m.append(float(dists.min()))  # best cell in set

    return {
        "target_alpha": alpha,
        "marginal_coverage": coverage,
        "avg_set_size": float(np.mean(set_sizes)),
        "empty_rate": empty_rate,
        "mean_min_dist_m": float(np.mean(coverages_m)) if coverages_m else float("nan"),
    }


# ---------------------------------------------------------------------------
# Selective Classification
# ---------------------------------------------------------------------------

def selective_frontier(
    softmax_test: np.ndarray,
    y_true: np.ndarray,
    le: LabelEncoder,
    lookup: pd.DataFrame,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.99, 30)

    pred_all = le.inverse_transform(softmax_test.argmax(axis=1))
    conf_all = softmax_test.max(axis=1)
    results = []

    for tau in thresholds:
        answered = conf_all >= tau
        coverage_rate = float(answered.mean())
        if answered.sum() == 0:
            continue
        pred_answered = pred_all[answered]
        true_answered = y_true[answered]
        m = _metrics(true_answered, pred_answered, lookup)
        results.append({
            "threshold": float(tau),
            "coverage_rate": coverage_rate,
            "cell_acc_when_answered": m["cell_acc"],
            "mean_error_m_when_answered": m["mean_error_m"],
        })

    return results


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
    X_all_train = build_X(train_df, include_room=include_room)
    X_test      = build_X(test_df,  include_room=include_room)
    y_all_train = train_df["grid_cell"].to_numpy()
    y_test      = test_df["grid_cell"].to_numpy()

    print(f"\n  [{label}]  n_train={len(X_all_train)}")
    results = {}

    le = LabelEncoder().fit(y_all_train)
    y_all_enc = le.transform(y_all_train)

    # Split off a calibration set (20% of training data)
    cal_frac = 0.2
    idx_tr, idx_cal = train_test_split(
        np.arange(len(X_all_train)), test_size=cal_frac,
        random_state=SEED, stratify=y_all_enc,
    )
    X_train, X_cal = X_all_train[idx_tr], X_all_train[idx_cal]
    y_train_enc, y_cal_enc = y_all_enc[idx_tr], y_all_enc[idx_cal]
    y_train_str = y_all_train[idx_tr]

    # Train MLP
    model, scaler = train_mlp(
        X_train, y_train_enc, len(le.classes_),
        epochs=args.epochs, lr=args.lr, hidden=args.hidden,
    )

    # Baseline accuracy
    softmax_test = get_softmax(model, scaler, X_test)
    pred_all = le.inverse_transform(softmax_test.argmax(axis=1))
    results["MLP_baseline"] = _metrics(y_test, pred_all, lookup)
    print(f"    MLP_baseline   acc={results['MLP_baseline']['cell_acc']:.4f}  "
          f"err={results['MLP_baseline']['mean_error_m']:.3f}m")

    # Conformal prediction
    softmax_cal = get_softmax(model, scaler, X_cal)
    ncal_scores = conformal_calibrate(softmax_cal, y_cal_enc)

    conformal_results = {}
    for alpha in args.alphas:
        tau = conformal_threshold(ncal_scores, alpha=alpha)
        pred_sets = conformal_predict_sets(softmax_test, tau, le)
        cm = conformal_metrics(pred_sets, y_test, lookup, alpha)
        conformal_results[f"alpha_{alpha}"] = {**cm, "tau": tau}
        print(f"    CP alpha={alpha:.2f}  coverage={cm['marginal_coverage']:.3f}  "
              f"set_size={cm['avg_set_size']:.1f}  best_dist={cm['mean_min_dist_m']:.3f}m")
    results["conformal"] = conformal_results

    # Selective classification frontier
    frontier = selective_frontier(softmax_test, y_test, le, lookup)
    results["selective_frontier"] = frontier
    # Print a few key points
    for pt in frontier[::6]:
        print(f"    Selective tau={pt['threshold']:.2f}  "
              f"cov={pt['coverage_rate']:.3f}  "
              f"acc={pt['cell_acc_when_answered']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Conformal prediction + selective classification")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20],
                        help="Conformal error levels (1-coverage)")
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
        all_results["loro"] = {"folds": folds}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "conformal_localization.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
