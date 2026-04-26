"""Confidence-weighted Prototypical Network for WiFi Indoor Positioning — v2.

Implements the core innovations from:
  "Few-Shot Learning for WiFi Fingerprinting Indoor Positioning"
  Ma & Shi, Sensors 2023, 23, 8458.

Key difference from v1: the TCN encoder is swapped for an MLP encoder that is
more appropriate for our 5-feature RSSI fingerprints (Signal, Noise, A1, A2, A3
are not a temporal sequence; causal dilated convolutions add no benefit here).
The paper's two main contributions are preserved exactly:
  1. Confidence-based prototype: variance-weighted mean (Eq. 4-5)
  2. Mahalanobis distance classification (Eq. 6)

Three encoders are compared:
  - MLP_plain   : standard MLP, no confidence (= vanilla ProtoNet)
  - MLP_conf    : MLP + confidence scalar  (key paper contribution)
  - TCN_conf    : TCN + confidence (v1 architecture, as reference)

Evaluation: LORO × K ∈ {1,3,5,10,25}, 5 trials each.
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
from torch.nn.utils.parametrizations import weight_norm
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localization.catalog import BENCHMARK_REPORT_DIR, FEATURE_COLUMNS, filter_room_campaigns
from localization.data import load_measurements

REPORT_DIR = BENCHMARK_REPORT_DIR
SEED = 42
EMB_DIM = 64


# ---------------------------------------------------------------------------
# Kalman Filter
# ---------------------------------------------------------------------------

def kalman_filter_rssi(rssi_window: np.ndarray, Q: float = 1e-3, R: float = 0.5) -> np.ndarray:
    """Per-channel Kalman filter; returns estimate at last timestep."""
    n, n_ch = rssi_window.shape
    result = np.empty(n_ch)
    for ch in range(n_ch):
        x = rssi_window[0, ch]
        P = 1.0
        for t in range(1, n):
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x + K * (rssi_window[t, ch] - x)
            P = (1.0 - K) * P_pred
        result[ch] = x
    return result


def simulate_kf_online(
    X: np.ndarray, window_size: int = 10, noise_std: float = 2.0,
    Q: float = 1e-3, R: float = 0.5, rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(SEED)
    out = np.empty_like(X)
    for i, x in enumerate(X):
        noise = rng.normal(0, noise_std, (window_size - 1, X.shape[1]))
        window = np.vstack([x[None] + noise, x[None]])
        out[i] = kalman_filter_rssi(window, Q=Q, R=R)
    return out


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    """Standard MLP encoder (vanilla ProtoNet). No confidence output."""

    def __init__(self, n_features: int, hidden: int = 128, emb_dim: int = EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x), None   # (emb, sigma=None)


class MLPConfEncoder(nn.Module):
    """MLP encoder with confidence scalar (key paper contribution).

    Output: (embedding [batch, EMB_DIM], sigma [batch, 1]) where sigma > 1.
    The inverse 1/sigma is the precision used for prototype weighting.
    """

    def __init__(self, n_features: int, hidden: int = 128, emb_dim: int = EMB_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.emb_head = nn.Linear(hidden, emb_dim)
        self.cov_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        emb = self.emb_head(h)
        # Eq. (3): S = 1 + softplus(S_raw) ensures S > 1
        sigma = 1.0 + F.softplus(self.cov_head(h))
        return emb, sigma


class _ResidualBlock(nn.Module):
    """TCN residual block: dilated causal conv × 2, WeightNorm, ReLU, Dropout."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = pad
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _trim(self, x):
        return x[:, :, :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        out = self.drop1(self.relu(self._trim(self.conv1(x))))
        out = self.drop2(self.relu(self._trim(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNConfEncoder(nn.Module):
    """TCN encoder from the paper (5 blocks, dilations 1-16).

    Input is treated as [batch, 1, n_features] (1 channel, seq_len=n_features).
    The LAST position is taken (has seen full input via causal conv)
    rather than avg-pool, which averages partial-context positions.
    """

    _BLOCKS = [(1, 32, 0.2, 1), (32, 32, 0.2, 2), (32, 64, 0.5, 4),
               (64, 64, 0.5, 8), (64, EMB_DIM + 1, 0.5, 16)]

    def __init__(self, n_features: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            _ResidualBlock(ic, oc, kernel_size=5, dilation=d, dropout=dr)
            for ic, oc, dr, d in self._BLOCKS
        ])

    def forward(self, x: torch.Tensor):
        h = x.unsqueeze(1)              # [B, 1, F]
        for block in self.blocks:
            h = block(h)                # [B, ch, F]
        h = h[:, :, -1]                # Take LAST position [B, ch] — sees full input
        emb = h[:, :EMB_DIM]
        sigma = 1.0 + F.softplus(h[:, EMB_DIM:])
        return emb, sigma


# ---------------------------------------------------------------------------
# Prototypes and distance (Eq. 4-6 from the paper)
# ---------------------------------------------------------------------------

def compute_prototypes(
    emb_sup: torch.Tensor, sigma_sup: torch.Tensor | None,
    y_sup: torch.Tensor, n_way: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Eq. 4-5: confidence-weighted prototypes. Falls back to mean if sigma=None."""
    prototypes, prec_c = [], []
    for c in range(n_way):
        mask = (y_sup == c)
        if mask.sum() == 0:
            prototypes.append(torch.zeros(EMB_DIM, device=emb_sup.device))
            prec_c.append(torch.zeros(1, device=emb_sup.device))
            continue
        x_i = emb_sup[mask]
        if sigma_sup is not None:
            s_i = 1.0 / sigma_sup[mask]              # precision [k, 1]
            num = (s_i * x_i).sum(0)
            denom = s_i.sum()
            prototypes.append(num / denom)
            prec_c.append(denom.squeeze())
        else:
            prototypes.append(x_i.mean(0))
            prec_c.append(None)

    p = torch.stack(prototypes)
    pc = torch.stack(prec_c) if sigma_sup is not None else None
    return p, pc


def classify(
    emb_qry: torch.Tensor,
    prototypes: torch.Tensor,
    prec_c: torch.Tensor | None,
) -> torch.Tensor:
    """Eq. 6-7: nearest prototype by Mahalanobis (or Euclidean if no prec_c)."""
    dists = torch.cdist(emb_qry, prototypes)      # [Q, C]
    if prec_c is not None:
        dists = dists * prec_c.sqrt().unsqueeze(0)
    return -dists   # logits for cross-entropy


# ---------------------------------------------------------------------------
# Episodic loss
# ---------------------------------------------------------------------------

def episode_loss(
    encoder: nn.Module,
    X_sup: torch.Tensor, y_sup: torch.Tensor,
    X_qry: torch.Tensor, y_qry: torch.Tensor,
    n_way: int,
) -> torch.Tensor:
    emb_sup, sigma_sup = encoder(X_sup)
    emb_qry, _ = encoder(X_qry)
    prototypes, prec_c = compute_prototypes(emb_sup, sigma_sup, y_sup, n_way)
    logits = classify(emb_qry, prototypes, prec_c)
    return F.cross_entropy(logits, y_qry)


# ---------------------------------------------------------------------------
# Episode sampler
# ---------------------------------------------------------------------------

def _sample_episode(X, y, n_way, n_support, n_query, rng):
    classes = np.unique(y)
    if len(classes) < n_way:
        n_way = len(classes)
    chosen = rng.choice(classes, size=n_way, replace=False)
    X_s, y_s, X_q, y_q = [], [], [], []
    for enc, cls in enumerate(chosen):
        idx = np.where(y == cls)[0]
        n_q = max(1, min(n_query, len(idx) - n_support))
        n_s = min(n_support, len(idx) - n_q)
        perm = rng.permutation(len(idx))
        X_s.append(X[idx[perm[:n_s]]]); y_s.extend([enc]*n_s)
        X_q.append(X[idx[perm[n_s:n_s+n_q]]]); y_q.extend([enc]*n_q)
    return (np.vstack(X_s), np.array(y_s, dtype=np.int64),
            np.vstack(X_q), np.array(y_q, dtype=np.int64))


# ---------------------------------------------------------------------------
# Meta-training
# ---------------------------------------------------------------------------

def meta_train(X_train, y_train, *, encoder_cls, n_episodes=3000, n_way=20,
               n_support=10, n_query=10, lr=1e-3, random_state=SEED, device="cpu"):
    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train).astype(np.float32)

    n_features = X_s.shape[1]
    encoder = encoder_cls(n_features).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes)

    encoder.train()
    losses = []
    for ep in range(n_episodes):
        Xs, ys, Xq, yq = _sample_episode(X_s, y_train, n_way, n_support, n_query, rng)
        n_w = len(np.unique(ys))
        loss = episode_loss(
            encoder,
            torch.from_numpy(Xs).to(device), torch.from_numpy(ys).to(device),
            torch.from_numpy(Xq).to(device), torch.from_numpy(yq).to(device),
            n_w,
        )
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        losses.append(float(loss.item()))
        if (ep + 1) % 1000 == 0:
            print(f"      ep {ep+1}/{n_episodes}  loss={np.mean(losses[-200:]):.4f}")

    return encoder, scaler


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(encoder, scaler, X_sup, y_sup, X_test, device="cpu"):
    encoder.eval()
    X_sup_s = torch.from_numpy(scaler.transform(X_sup).astype(np.float32)).to(device)
    X_te_s  = torch.from_numpy(scaler.transform(X_test).astype(np.float32)).to(device)
    classes = np.unique(y_sup)
    y_enc = torch.from_numpy(np.searchsorted(classes, y_sup)).to(device)
    with torch.no_grad():
        emb_s, sig_s = encoder(X_sup_s)
        emb_q, _ = encoder(X_te_s)
        proto, prec = compute_prototypes(emb_s, sig_s, y_enc, len(classes))
        pred_idx = classify(emb_q, proto, prec).argmax(dim=1).cpu().numpy()
    return classes[pred_idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _cell_lookup(df):
    return df.groupby("grid_cell")[["coord_x_m", "coord_y_m"]].first()

def _metrics(y_true, y_pred, lookup):
    acc = float((y_true == y_pred).mean())
    pc = lookup.reindex(y_pred)[["coord_x_m","coord_y_m"]].to_numpy()
    tc = lookup.reindex(y_true)[["coord_x_m","coord_y_m"]].to_numpy()
    mask = ~(np.isnan(pc).any(1)|np.isnan(tc).any(1))
    err = float(np.linalg.norm(pc[mask]-tc[mask],axis=1).mean()) if mask.sum() else float("nan")
    return {"cell_acc": acc, "mean_error_m": err}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_multiroom():
    campaigns = filter_room_campaigns()
    frames = []
    for room, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df = load_measurements([spec]); df["room"] = room; frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# LORO evaluation
# ---------------------------------------------------------------------------

ENCODER_CLASSES = {
    "MLP_plain": MLPEncoder,
    "MLP_conf":  MLPConfEncoder,
    "TCN_conf":  TCNConfEncoder,
}


def run_loro(args):
    print(f"\n=== Confidence-weighted ProtoNet v2 ({', '.join(args.encoders)}) ===")
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = df["grid_cell"].to_numpy()
    k_values = args.k_values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    all_folds: dict[str, dict] = {r: {} for r in rooms}
    averages: dict[str, dict] = {}

    for enc_name in args.encoders:
        enc_cls = ENCODER_CLASSES[enc_name]
        print(f"\n  [{enc_name}]")

        for fold_idx, held_out in enumerate(rooms):
            print(f"    Fold: held_out={held_out}")
            train_mask = df["room"].to_numpy() != held_out
            test_mask  = df["room"].to_numpy() == held_out
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

            encoder, scaler = meta_train(
                X_train, y_train,
                encoder_cls=enc_cls,
                n_episodes=args.episodes,
                n_way=args.n_way,
                n_support=args.n_support,
                n_query=args.n_query,
                lr=args.lr,
                random_state=args.seed + fold_idx * 100,
                device=device,
            )

            rng_kf = np.random.default_rng(args.seed + fold_idx)
            X_test_kf = simulate_kf_online(
                X_test, window_size=args.kf_window,
                noise_std=args.kf_noise_std, Q=args.kf_Q, R=args.kf_R,
                rng=rng_kf,
            )

            cells = np.unique(y_test)
            for k in k_values:
                accs, errs, accs_kf, errs_kf = [], [], [], []
                for trial in range(args.n_trials):
                    rng_t = np.random.default_rng(args.seed + fold_idx*10_000 + trial*1_000)
                    sup_idx, qry_idx = [], []
                    for cell in cells:
                        ci = np.where(y_test == cell)[0]
                        ke = min(k, len(ci))
                        p = rng_t.permutation(len(ci))
                        sup_idx.extend(ci[p[:ke]]); qry_idx.extend(ci[p[ke:]])
                    if not qry_idx:
                        continue
                    Xs, ys = X_test[sup_idx], y_test[sup_idx]
                    Xq, yq = X_test[qry_idx], y_test[qry_idx]

                    m  = _metrics(yq, predict(encoder, scaler, Xs, ys, Xq, device), lookup)
                    mk = _metrics(yq, predict(encoder, scaler, Xs, ys, X_test_kf[qry_idx], device), lookup)
                    accs.append(m["cell_acc"]); errs.append(m["mean_error_m"])
                    accs_kf.append(mk["cell_acc"]); errs_kf.append(mk["mean_error_m"])

                def agg(v): return {"mean": float(np.mean(v)) if v else float("nan"),
                                    "std":  float(np.std(v))  if v else float("nan")}

                key     = f"{enc_name}_K{k}"
                key_kf  = f"{enc_name}_KF_K{k}"
                all_folds[held_out][key]    = {**agg(accs),    "err": agg(errs)}
                all_folds[held_out][key_kf] = {**agg(accs_kf), "err": agg(errs_kf)}

                print(f"      K={k:>2}  acc={np.mean(accs):.4f}±{np.std(accs):.4f}"
                      f"  err={np.mean(errs):.3f}m"
                      f"  | +KF acc={np.mean(accs_kf):.4f}"
                      f"  err={np.mean(errs_kf):.3f}m")

    # Cross-room averages
    print("\n  ── Cross-room averages ──")
    for enc_name in args.encoders:
        for suffix in ("", "_KF"):
            for k in k_values:
                key = f"{enc_name}{suffix}_K{k}"
                accs = [all_folds[r][key]["mean"] for r in rooms if key in all_folds[r]]
                errs = [all_folds[r][key]["err"]["mean"] for r in rooms if key in all_folds[r]]
                va = [a for a in accs if not np.isnan(a)]
                ve = [e for e in errs if not np.isnan(e)]
                avg_a = float(np.mean(va)) if va else float("nan")
                avg_e = float(np.mean(ve)) if ve else float("nan")
                averages[key] = {"cell_acc_mean": avg_a, "mean_error_m": avg_e}
                print(f"    {key:<25} acc={avg_a:.4f}  err={avg_e:.3f}m")

    return {"config": vars(args), "folds": all_folds, "k_values": k_values, "averages": averages}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+",
                        choices=list(ENCODER_CLASSES), default=["MLP_plain", "MLP_conf"])
    parser.add_argument("--episodes",   type=int,   default=3000)
    parser.add_argument("--n-way",      type=int,   default=20)
    parser.add_argument("--n-support",  type=int,   default=10)
    parser.add_argument("--n-query",    type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--n-trials",   type=int,   default=5)
    parser.add_argument("--k-values",   nargs="+",  type=int, default=[1, 3, 5, 10, 25])
    parser.add_argument("--kf-window",     type=int,   default=10)
    parser.add_argument("--kf-noise-std",  type=float, default=2.0)
    parser.add_argument("--kf-Q",          type=float, default=1e-3)
    parser.add_argument("--kf-R",          type=float, default=0.5)
    parser.add_argument("--seed",          type=int,   default=SEED)
    parser.add_argument("--output-name",   default="tcn_protonet_wifi_v2.json")
    args = parser.parse_args()

    results = run_loro(args)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / args.output_name
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
