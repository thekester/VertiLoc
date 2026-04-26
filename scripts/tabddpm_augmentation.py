"""TabDDPM-style conditional diffusion for RSSI data augmentation.

Implements a conditional Denoising Diffusion Probabilistic Model (DDPM)
for tabular RSSI data. Conditioned on (cell_id, campaign_id) to generate
synthetic fingerprints for under-represented or held-out conditions.

Tested on the LOCO protocol:
  - Train DDPM on all campaigns except one
  - Generate synthetic samples for the held-out campaign's conditions
  - Train KNN/RF on real + synthetic, test on real held-out

Reference: Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models",
           ICML 2023
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localization.catalog import BENCHMARK_REPORT_DIR, FEATURE_COLUMNS, filter_room_campaigns
from localization.data import load_measurements

REPORT_DIR = BENCHMARK_REPORT_DIR
SEED = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_room(room: str) -> pd.DataFrame:
    campaigns = filter_room_campaigns(room_filter=[room])
    frames = []
    for r, specs in campaigns.items():
        for spec in specs:
            if spec.path.exists():
                df = load_measurements([spec])
                df["room"] = r
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


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


# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------

def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_bar = torch.cos(((steps / T + s) / (1 + s)) * (torch.pi / 2)) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


class DDPMSchedule:
    def __init__(self, T: int = 100):
        self.T = T
        betas = _cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register = {
            "betas": betas,
            "alphas": alphas,
            "alphas_bar": alphas_bar,
            "sqrt_alphas_bar": alphas_bar.sqrt(),
            "sqrt_one_minus_alphas_bar": (1 - alphas_bar).sqrt(),
        }

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.register[key]

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t from x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self["sqrt_alphas_bar"][t].view(-1, 1)
        sqrt_1ab = self["sqrt_one_minus_alphas_bar"][t].view(-1, 1)
        return sqrt_ab * x0 + sqrt_1ab * noise, noise


# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------

class ConditionalDenoiser(nn.Module):
    """Simple MLP denoiser conditioned on timestep + class label embedding."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int = 256,
        n_layers: int = 4,
        emb_dim: int = 32,
    ):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.label_emb = nn.Embedding(n_classes, emb_dim)

        layers: list[nn.Module] = []
        in_dim = n_features + 2 * emb_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_features))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_emb(t.float().unsqueeze(1) / 100.0)
        l_emb = self.label_emb(label)
        h = torch.cat([x, t_emb, l_emb], dim=1)
        return self.net(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ddpm(
    X: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    *,
    T: int = 100,
    epochs: int = 300,
    lr: float = 2e-4,
    hidden: int = 256,
    batch_size: int = 256,
    random_state: int = SEED,
    device: str = "cpu",
) -> tuple[ConditionalDenoiser, DDPMSchedule]:
    torch.manual_seed(random_state)
    schedule = DDPMSchedule(T=T)
    model = ConditionalDenoiser(X.shape[1], n_classes, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_enc.astype(np.int64)).to(device)
    n = len(X_t)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x0 = X_t[idx]
            labels = y_t[idx]
            t = torch.randint(0, T, (len(x0),), device=device)
            xt, noise = schedule.q_sample(x0, t)
            pred_noise = model(xt, t, labels)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss={epoch_loss/n_batches:.5f}")

    return model, schedule


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample(
    model: ConditionalDenoiser,
    schedule: DDPMSchedule,
    n_samples: int,
    label: int,
    n_features: int,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Generate n_samples for a given class label using DDPM reverse process."""
    model.eval()
    T = schedule.T
    betas = schedule["betas"].to(device)
    alphas = schedule["alphas"].to(device)
    alphas_bar = schedule["alphas_bar"].to(device)

    x = torch.randn(n_samples, n_features, device=device)
    label_t = torch.full((n_samples,), label, dtype=torch.long, device=device)

    for t_idx in reversed(range(T)):
        t_tensor = torch.full((n_samples,), t_idx, dtype=torch.long, device=device)
        pred_noise = model(x, t_tensor, label_t)

        alpha_t = alphas[t_idx]
        alpha_bar_t = alphas_bar[t_idx]
        beta_t = betas[t_idx]

        # DDPM reverse step
        coef = beta_t / (1 - alpha_bar_t).sqrt()
        x_prev = (x - coef * pred_noise) / alpha_t.sqrt()

        if t_idx > 0:
            noise = torch.randn_like(x)
            x_prev = x_prev + beta_t.sqrt() * noise
        x = x_prev

    return x.cpu().numpy()


# ---------------------------------------------------------------------------
# LOCO augmentation benchmark
# ---------------------------------------------------------------------------

def run_loco_augmentation(room: str, args: argparse.Namespace) -> dict:
    print(f"\n=== TabDDPM LOCO Augmentation — {room} ===")
    df = load_room(room)
    campaigns = sorted(df["campaign"].unique())

    if len(campaigns) < 2:
        print(f"  Only {len(campaigns)} campaign(s) in {room}, skipping LOCO.")
        return {}

    lookup = _cell_lookup(df)
    all_results: dict[str, dict] = {}
    device = "cpu"

    for held_out_camp in campaigns:
        train_df = df[df["campaign"] != held_out_camp].copy()
        test_df  = df[df["campaign"] == held_out_camp].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_df["grid_cell"].to_numpy()
        X_test  = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test  = test_df["grid_cell"].to_numpy()

        # Encode cell labels
        le = LabelEncoder().fit(y_train)
        y_enc = le.transform(y_train)
        n_classes = len(le.classes_)

        scaler = StandardScaler().fit(X_train)
        X_tr_s = scaler.transform(X_train).astype(np.float32)
        X_te_s = scaler.transform(X_test).astype(np.float32)

        camp_short = held_out_camp.split("/")[-1]
        print(f"\n  Held-out campaign: {camp_short}  (train={len(X_tr_s)}, test={len(X_te_s)})")

        # Baseline without augmentation
        knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn.fit(X_tr_s, y_train)
        baseline = _metrics(y_test, knn.predict(X_te_s), lookup)
        print(f"    KNN baseline (no aug):  acc={baseline['cell_acc']:.4f}")

        # Train DDPM
        print(f"    Training DDPM ({args.ddpm_epochs} epochs, T={args.T})...")
        ddpm, schedule = train_ddpm(
            X_tr_s, y_enc, n_classes,
            T=args.T,
            epochs=args.ddpm_epochs,
            lr=args.ddpm_lr,
            hidden=args.ddpm_hidden,
            batch_size=args.batch_size,
            random_state=SEED,
            device=device,
        )

        # Generate synthetic samples for each cell present in test
        test_cells_in_train = [c for c in np.unique(y_test) if c in le.classes_]
        X_syn_list, y_syn_list = [], []
        for cell in test_cells_in_train:
            cell_enc = le.transform([cell])[0]
            syn = ddpm_sample(ddpm, schedule, args.n_synth, cell_enc,
                              X_tr_s.shape[1], device=device)
            X_syn_list.append(syn)
            y_syn_list.extend([cell] * args.n_synth)

        X_syn = np.vstack(X_syn_list).astype(np.float32)
        y_syn = np.array(y_syn_list)

        # Augmented training
        X_aug = np.vstack([X_tr_s, X_syn])
        y_aug = np.concatenate([y_train, y_syn])

        knn_aug = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn_aug.fit(X_aug, y_aug)
        aug_result = _metrics(y_test, knn_aug.predict(X_te_s), lookup)
        delta = aug_result["cell_acc"] - baseline["cell_acc"]
        sign = "+" if delta >= 0 else ""
        print(f"    KNN + TabDDPM aug:       acc={aug_result['cell_acc']:.4f}  Δ={sign}{delta:.4f}")

        rf_aug = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
        rf_aug.fit(X_aug, y_aug)
        rf_aug_result = _metrics(y_test, rf_aug.predict(X_te_s), lookup)
        delta_rf = rf_aug_result["cell_acc"] - baseline["cell_acc"]
        sign_rf = "+" if delta_rf >= 0 else ""
        print(f"    RF  + TabDDPM aug:       acc={rf_aug_result['cell_acc']:.4f}  Δ={sign_rf}{delta_rf:.4f}")

        all_results[camp_short] = {
            "baseline_knn": baseline,
            "augmented_knn": aug_result,
            "augmented_rf": rf_aug_result,
            "n_synthetic": int(len(X_syn)),
            "n_train_original": int(len(X_tr_s)),
        }

    # Summary
    if all_results:
        base_accs  = [v["baseline_knn"]["cell_acc"] for v in all_results.values()]
        aug_accs   = [v["augmented_knn"]["cell_acc"] for v in all_results.values()]
        print(f"\n  Average LOCO: baseline={np.mean(base_accs):.4f}  "
              f"augmented={np.mean(aug_accs):.4f}  "
              f"Δ={np.mean(aug_accs)-np.mean(base_accs):+.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="TabDDPM augmentation for RSSI LOCO")
    parser.add_argument("--rooms", nargs="+", default=["E102", "B121"],
                        help="Rooms to run LOCO augmentation on.")
    parser.add_argument("--T", type=int, default=100,
                        help="Diffusion timesteps.")
    parser.add_argument("--ddpm-epochs", type=int, default=300,
                        help="DDPM training epochs.")
    parser.add_argument("--ddpm-lr", type=float, default=2e-4)
    parser.add_argument("--ddpm-hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-synth", type=int, default=25,
                        help="Synthetic samples per cell per held-out campaign.")
    args = parser.parse_args()

    all_results = {}
    for room in args.rooms:
        all_results[room] = run_loco_augmentation(room, args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "tabddpm_augmentation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
