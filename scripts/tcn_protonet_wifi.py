"""TCN-based Prototypical Network for WiFi Fingerprinting Indoor Positioning.

Implements the method from:
  "Few-Shot Learning for WiFi Fingerprinting Indoor Positioning"
  Ma & Shi, Sensors 2023, 23, 8458.

Key components:
  1. TCN Encoder (5 dilated causal residual blocks) outputting embedding + covariance scalar
  2. Confidence-based prototype: variance-weighted mean of support embeddings
  3. Mahalanobis distance for classification: d_c = sqrt(s_c) * ||q - p_c||
  4. Kalman Filter for online RSSI denoising (simulation over noisy test windows)
  5. LORO (Leave-One-Room-Out) evaluation

TCN block structure (Table 1 from the paper):
  Block 1: kernel=5, ch=32, dropout=0.2, dilation=1
  Block 2: kernel=5, ch=32, dropout=0.2, dilation=2
  Block 3: kernel=5, ch=64, dropout=0.5, dilation=4
  Block 4: kernel=5, ch=64, dropout=0.5, dilation=8
  Block 5: kernel=5, ch=65, dropout=0.5, dilation=16
           (65 = 64-d embedding + 1-d covariance scalar)
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
# Kalman Filter for RSSI denoising
# ---------------------------------------------------------------------------

class KalmanFilter1D:
    """Scalar Kalman filter for a single RSSI channel.

    State model: x_{t} = x_{t-1} + w_t   (w_t ~ N(0, Q))
    Measurement: z_t   = x_t     + v_t   (v_t ~ N(0, R))
    """

    def __init__(self, Q: float = 1e-3, R: float = 0.1):
        self.Q = Q  # process noise variance
        self.R = R  # measurement noise variance

    def filter(self, measurements: np.ndarray) -> np.ndarray:
        n = len(measurements)
        x = measurements[0]
        P = 1.0
        filtered = np.empty(n)
        filtered[0] = x
        for t in range(1, n):
            # Predict
            P_pred = P + self.Q
            # Update
            K = P_pred / (P_pred + self.R)
            x = x + K * (measurements[t] - x)
            P = (1.0 - K) * P_pred
            filtered[t] = x
        return filtered


def kalman_filter_rssi(
    rssi_window: np.ndarray,
    Q: float = 1e-3,
    R: float = 0.1,
) -> np.ndarray:
    """Apply per-channel Kalman filter to an RSSI time window.

    Args:
        rssi_window: array of shape [T, n_features]
        Q: process noise variance
        R: measurement noise variance

    Returns:
        estimated RSSI at last time step, shape [n_features]
    """
    kf = KalmanFilter1D(Q=Q, R=R)
    result = np.empty(rssi_window.shape[1])
    for ch in range(rssi_window.shape[1]):
        filtered = kf.filter(rssi_window[:, ch])
        result[ch] = filtered[-1]
    return result


def simulate_kf_online(
    X_test: np.ndarray,
    window_size: int = 10,
    noise_std: float = 3.0,
    Q: float = 1e-3,
    R: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate KF online denoising over the test set.

    For each test sample, creates a noisy time window of T measurements by
    adding Gaussian noise to the true RSSI, then uses the KF to estimate the
    current RSSI. This mimics the paper's online stage.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    X_denoised = np.empty_like(X_test)
    for i, x in enumerate(X_test):
        # Build a fake time window: [x + noise] * (T-1) + [x]
        window = x[None, :] + rng.normal(0, noise_std, (window_size - 1, X_test.shape[1]))
        window = np.vstack([window, x[None, :]])  # [T, n_features]
        X_denoised[i] = kalman_filter_rssi(window, Q=Q, R=R)
    return X_denoised


# ---------------------------------------------------------------------------
# TCN residual block
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """Residual block with two dilated causal conv layers.

    Each conv layer is followed by WeightNorm, ReLU, Dropout as per Fig. 2.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.padding, dilation=dilation)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=self.padding, dilation=dilation)
        )
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        """Remove right padding to enforce causality."""
        if self.padding > 0:
            return x[:, :, : -self.padding]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu(self._trim(self.conv1(x))))
        out = self.drop2(self.relu(self._trim(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ---------------------------------------------------------------------------
# TCN encoder
# ---------------------------------------------------------------------------

class TCNEncoder(nn.Module):
    """5-block TCN encoder outputting (embedding, precision_scalar) per sample.

    Architecture follows Table 1 from the paper. The last block outputs
    EMB_DIM + 1 channels; the final channel is the raw covariance scalar
    S_raw, converted via S = 1 + softplus(S_raw) to ensure S > 1.

    The inverse variance (precision) s = 1/S is used for prototype weighting.
    """

    # (in_ch, out_ch, dropout, dilation)
    _BLOCK_CFGS = [
        (1, 32, 0.2, 1),
        (32, 32, 0.2, 2),
        (32, 64, 0.5, 4),
        (64, 64, 0.5, 8),
        (64, EMB_DIM + 1, 0.5, 16),
    ]

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.blocks = nn.ModuleList([
            _ResidualBlock(in_ch, out_ch, kernel_size=5, dilation=dil, dropout=drop)
            for in_ch, out_ch, drop, dil in self._BLOCK_CFGS
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, n_features]
        Returns:
            emb: [batch, EMB_DIM]
            sigma: [batch, 1]  precision scalar, always > 1
        """
        h = x.unsqueeze(1)  # [batch, 1, n_features]
        for block in self.blocks:
            h = block(h)
        h = self.pool(h).squeeze(-1)   # [batch, EMB_DIM+1]

        emb = h[:, :EMB_DIM]           # [batch, EMB_DIM]
        s_raw = h[:, EMB_DIM:]         # [batch, 1]
        # Eq. (3): S = 1 + softplus(S_raw), ensures S > 1
        sigma = 1.0 + F.softplus(s_raw)
        return emb, sigma


# ---------------------------------------------------------------------------
# Prototype computation and distance (Eq. 4-6)
# ---------------------------------------------------------------------------

def compute_prototypes(
    emb_sup: torch.Tensor,
    sigma_sup: torch.Tensor,
    y_sup: torch.Tensor,
    n_way: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute confidence-weighted prototypes (Eq. 4-5).

    Args:
        emb_sup:   [n_support, EMB_DIM]
        sigma_sup: [n_support, 1]  variance scalars (S > 1)
        y_sup:     [n_support]     class labels 0..n_way-1
        n_way:     number of classes

    Returns:
        prototypes: [n_way, EMB_DIM]  variance-weighted class centroids
        prec_c:     [n_way]           class precision scalars s_c = sum(1/S_i)
    """
    inv_sigma = 1.0 / sigma_sup  # precision: [n_support, 1]
    prototypes = []
    prec_c = []
    for c in range(n_way):
        mask = (y_sup == c)
        if mask.sum() == 0:
            prototypes.append(torch.zeros(EMB_DIM, device=emb_sup.device))
            prec_c.append(torch.zeros(1, device=emb_sup.device))
            continue
        s_i = inv_sigma[mask]            # [k, 1]
        x_i = emb_sup[mask]             # [k, EMB_DIM]
        # Eq. (4): p_c = sum(s_i * x_i) / sum(s_i)
        num = (s_i * x_i).sum(0)        # [EMB_DIM]
        denom = s_i.sum()               # scalar
        prototypes.append(num / denom)
        # Eq. (5): s_c = sum(s_i)
        prec_c.append(denom.squeeze())

    return torch.stack(prototypes), torch.stack(prec_c)


def mahalanobis_distances(
    emb_qry: torch.Tensor,
    prototypes: torch.Tensor,
    prec_c: torch.Tensor,
) -> torch.Tensor:
    """Compute scaled Mahalanobis distances (Eq. 6).

    d_c(i) = sqrt(s_c) * ||q - p_c||

    Args:
        emb_qry:    [n_query, EMB_DIM]
        prototypes: [n_way, EMB_DIM]
        prec_c:     [n_way]

    Returns:
        distances: [n_query, n_way]
    """
    # Euclidean distance: [n_query, n_way]
    euc = torch.cdist(emb_qry, prototypes)
    # Scale by sqrt(s_c): [1, n_way]
    scale = prec_c.sqrt().unsqueeze(0)
    return euc * scale


# ---------------------------------------------------------------------------
# Episodic loss (Eq. 8)
# ---------------------------------------------------------------------------

def episode_loss(
    encoder: TCNEncoder,
    X_sup: torch.Tensor,
    y_sup: torch.Tensor,
    X_qry: torch.Tensor,
    y_qry: torch.Tensor,
    n_way: int,
) -> torch.Tensor:
    emb_sup, sigma_sup = encoder(X_sup)
    emb_qry, _ = encoder(X_qry)

    prototypes, prec_c = compute_prototypes(emb_sup, sigma_sup, y_sup, n_way)

    # Negative distances as logits for cross-entropy (Eq. 8)
    dists = mahalanobis_distances(emb_qry, prototypes, prec_c)
    logits = -dists
    return F.cross_entropy(logits, y_qry)


# ---------------------------------------------------------------------------
# Episode sampler
# ---------------------------------------------------------------------------

def _sample_episode(
    X: np.ndarray,
    y: np.ndarray,
    n_way: int,
    n_support: int,
    n_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(y)
    if len(classes) < n_way:
        n_way = len(classes)
    chosen = rng.choice(classes, size=n_way, replace=False)

    X_sup, y_sup_enc, X_qry, y_qry_enc = [], [], [], []
    for enc, cls in enumerate(chosen):
        idx = np.where(y == cls)[0]
        n_q = max(1, min(n_query, len(idx) - n_support))
        n_s = min(n_support, len(idx) - n_q)
        perm = rng.permutation(len(idx))
        X_sup.append(X[idx[perm[:n_s]]])
        y_sup_enc.extend([enc] * n_s)
        X_qry.append(X[idx[perm[n_s: n_s + n_q]]])
        y_qry_enc.extend([enc] * n_q)

    return (
        np.vstack(X_sup), np.array(y_sup_enc, dtype=np.int64),
        np.vstack(X_qry), np.array(y_qry_enc, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Meta-training
# ---------------------------------------------------------------------------

def meta_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_episodes: int = 2000,
    n_way: int = 20,
    n_support: int = 5,
    n_query: int = 10,
    lr: float = 1e-3,
    random_state: int = SEED,
    device: str = "cpu",
) -> tuple[TCNEncoder, StandardScaler]:
    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train).astype(np.float32)

    encoder = TCNEncoder(n_features=X_s.shape[1]).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes)

    encoder.train()
    losses: list[float] = []
    for ep in range(n_episodes):
        X_sup, y_sup, X_qry, y_qry = _sample_episode(
            X_s, y_train, n_way, n_support, n_query, rng
        )
        n_w = int(len(np.unique(y_sup)))
        X_sup_t = torch.from_numpy(X_sup).to(device)
        y_sup_t = torch.from_numpy(y_sup).to(device)
        X_qry_t = torch.from_numpy(X_qry).to(device)
        y_qry_t = torch.from_numpy(y_qry).to(device)

        optimizer.zero_grad()
        loss = episode_loss(encoder, X_sup_t, y_sup_t, X_qry_t, y_qry_t, n_w)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

        if (ep + 1) % 500 == 0:
            print(f"    ep {ep+1}/{n_episodes}  loss={np.mean(losses[-100:]):.4f}")

    return encoder, scaler


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def tcn_predict(
    encoder: TCNEncoder,
    scaler: StandardScaler,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_test: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Predict test labels using confidence-weighted prototypes."""
    encoder.eval()
    X_sup_s = torch.from_numpy(scaler.transform(X_support).astype(np.float32)).to(device)
    X_te_s = torch.from_numpy(scaler.transform(X_test).astype(np.float32)).to(device)

    classes = np.unique(y_support)
    y_sup_enc = np.searchsorted(classes, y_support)
    y_sup_t = torch.from_numpy(y_sup_enc).to(device)

    with torch.no_grad():
        emb_sup, sigma_sup = encoder(X_sup_s)
        emb_qry, _ = encoder(X_te_s)
        prototypes, prec_c = compute_prototypes(emb_sup, sigma_sup, y_sup_t, len(classes))
        dists = mahalanobis_distances(emb_qry, prototypes, prec_c)
        pred_idx = dists.argmin(dim=1).cpu().numpy()

    return classes[pred_idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _cell_lookup(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("grid_cell")[["coord_x_m", "coord_y_m"]].first()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, lookup: pd.DataFrame) -> dict:
    acc = float((y_true == y_pred).mean())
    pred_c = lookup.reindex(y_pred)[["coord_x_m", "coord_y_m"]].to_numpy()
    true_c = lookup.reindex(y_true)[["coord_x_m", "coord_y_m"]].to_numpy()
    mask = ~(np.isnan(pred_c).any(1) | np.isnan(true_c).any(1))
    err = (
        float(np.linalg.norm(pred_c[mask] - true_c[mask], axis=1).mean())
        if mask.sum() else float("nan")
    )
    return {"cell_acc": acc, "mean_error_m": err, "n": int(len(y_true))}


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# LORO evaluation
# ---------------------------------------------------------------------------

def run_loro(args: argparse.Namespace) -> dict:
    print("\n=== TCN Prototypical Network (Few-Shot WiFi Positioning) ===")
    df = load_multiroom()
    rooms = sorted(df["room"].unique())
    lookup = _cell_lookup(df)
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = df["grid_cell"].to_numpy()

    k_values = args.k_values
    all_folds: dict[str, dict] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    for fold_idx, held_out in enumerate(rooms):
        print(f"\n  Fold: held_out={held_out}")
        train_mask = df["room"].to_numpy() != held_out
        test_mask = df["room"].to_numpy() == held_out

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        print(f"    Meta-training on {X_train.shape[0]} samples "
              f"({len(np.unique(y_train))} cells)...")
        encoder, scaler = meta_train(
            X_train, y_train,
            n_episodes=args.episodes,
            n_way=args.n_way,
            n_support=args.n_support,
            n_query=args.n_query,
            lr=args.lr,
            random_state=args.seed + fold_idx * 100,
            device=device,
        )

        # KF-denoised test set (simulated online stage)
        rng_kf = np.random.default_rng(args.seed + fold_idx)
        X_test_kf = simulate_kf_online(
            X_test,
            window_size=args.kf_window,
            noise_std=args.kf_noise_std,
            Q=args.kf_Q,
            R=args.kf_R,
            rng=rng_kf,
        )

        cells_in_target = np.unique(y_test)
        fold_results: dict = {}

        for k in k_values:
            tcn_accs, tcn_errs = [], []
            tcn_kf_accs, tcn_kf_errs = [], []
            for trial in range(args.n_trials):
                trial_rng = np.random.default_rng(
                    args.seed + fold_idx * 10_000 + trial * 1_000
                )
                sup_idx, qry_idx = [], []
                for cell in cells_in_target:
                    cell_idx = np.where(y_test == cell)[0]
                    k_eff = min(k, len(cell_idx))
                    perm = trial_rng.permutation(len(cell_idx))
                    sup_idx.extend(cell_idx[perm[:k_eff]])
                    qry_idx.extend(cell_idx[perm[k_eff:]])

                if not qry_idx:
                    continue

                X_sup, y_sup = X_test[sup_idx], y_test[sup_idx]
                X_qry, y_qry = X_test[qry_idx], y_test[qry_idx]
                X_qry_kf = X_test_kf[qry_idx]

                # TCN PN without KF
                pred = tcn_predict(encoder, scaler, X_sup, y_sup, X_qry, device)
                m = _metrics(y_qry, pred, lookup)
                tcn_accs.append(m["cell_acc"])
                tcn_errs.append(m["mean_error_m"])

                # TCN PN with KF online denoising
                pred_kf = tcn_predict(encoder, scaler, X_sup, y_sup, X_qry_kf, device)
                m_kf = _metrics(y_qry, pred_kf, lookup)
                tcn_kf_accs.append(m_kf["cell_acc"])
                tcn_kf_errs.append(m_kf["mean_error_m"])

            def _agg(vals: list[float]) -> dict:
                return {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std": float(np.std(vals)) if vals else float("nan"),
                    "n_trials": len(vals),
                }

            fold_results[f"TCN_PN_K{k}"] = {**_agg(tcn_accs), "err": _agg(tcn_errs)}
            fold_results[f"TCN_PN_KF_K{k}"] = {**_agg(tcn_kf_accs), "err": _agg(tcn_kf_errs)}

            print(
                f"    K={k:>2}  acc={np.mean(tcn_accs):.4f}±{np.std(tcn_accs):.4f}"
                f"  err={np.mean(tcn_errs):.3f}m"
                f"  | +KF acc={np.mean(tcn_kf_accs):.4f}"
                f"  err={np.mean(tcn_kf_errs):.3f}m"
            )

        all_folds[held_out] = fold_results

    # Cross-room averages
    print("\n  Cross-room averages:")
    averages: dict[str, dict] = {}
    for method in ("TCN_PN", "TCN_PN_KF"):
        for k in k_values:
            key = f"{method}_K{k}"
            accs = [all_folds[r][key]["mean"] for r in rooms if key in all_folds[r]]
            errs = [all_folds[r][key]["err"]["mean"] for r in rooms if key in all_folds[r]]
            valid_a = [a for a in accs if not np.isnan(a)]
            valid_e = [e for e in errs if not np.isnan(e)]
            avg_a = float(np.mean(valid_a)) if valid_a else float("nan")
            avg_e = float(np.mean(valid_e)) if valid_e else float("nan")
            averages[key] = {"cell_acc_mean": avg_a, "mean_error_m": avg_e, "k": k}
            print(f"    {key:<20} acc={avg_a:.4f}  err={avg_e:.3f}m")

    return {
        "config": vars(args),
        "folds": all_folds,
        "k_values": k_values,
        "averages": averages,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TCN Prototypical Network for WiFi Indoor Positioning (Sensors 2023)"
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--n-way", type=int, default=20)
    parser.add_argument("--n-support", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10, 25])
    parser.add_argument("--kf-window", type=int, default=10,
                        help="Time window length T for Kalman filter (paper uses 10).")
    parser.add_argument("--kf-noise-std", type=float, default=3.0,
                        help="Simulated online RSSI noise std (dBm).")
    parser.add_argument("--kf-Q", type=float, default=1e-3,
                        help="Kalman filter process noise variance.")
    parser.add_argument("--kf-R", type=float, default=1.0,
                        help="Kalman filter measurement noise variance.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name", default="tcn_protonet_wifi.json")
    args = parser.parse_args()

    results = run_loro(args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / args.output_name
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
