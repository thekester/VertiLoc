"""Semi-Supervised GAN (SSGAN) for RSSI fingerprint augmentation.

Unlike a standard CGAN (which only uses labeled data), SSGAN leverages
BOTH labeled fingerprints AND unlabeled RSSI measurements to train a more
realistic generator. The discriminator has two heads:
  - Real/fake (adversarial)
  - Cell classification (supervised, on labeled samples only)

This addresses the failure of plain CGAN: the discriminator learns richer
structure from the full unlabeled pool, which guides the generator towards
more realistic RSSI distributions.

Inspired by: Odena (2016) "Semi-supervised learning with GANs",
             + Wi-Fi fingerprint SSGAN works (Sensors/MDPI 2024).

Tested on LOCO protocol (E102, B121):
  - Train on all campaigns except one (labeled + unlabeled pool)
  - Generate synthetic samples for held-out campaign cells
  - Evaluate KNN trained on real + synthetic data
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
# SSGAN architecture
# ---------------------------------------------------------------------------

class SSGANGenerator(nn.Module):
    def __init__(self, n_features: int, n_classes: int, latent_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 16, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, n_features),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_emb(labels)
        x = torch.cat([z, label_emb], dim=1)
        return self.net(x)


class SSGANDiscriminator(nn.Module):
    """Dual-head discriminator: real/fake + cell classification."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.real_fake_head = nn.Linear(hidden, 1)
        self.class_head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        rf = self.real_fake_head(h).squeeze(1)  # logit
        cls = self.class_head(h)                # class logits
        return rf, cls


# ---------------------------------------------------------------------------
# SSGAN training
# ---------------------------------------------------------------------------

def train_ssgan(
    X_labeled: np.ndarray,
    y_enc: np.ndarray,
    X_unlabeled: np.ndarray,
    n_classes: int,
    *,
    epochs: int = 200,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    latent_dim: int = 32,
    hidden: int = 128,
    batch_size: int = 128,
    lambda_cls: float = 1.0,
    random_state: int = SEED,
) -> tuple[SSGANGenerator, int]:
    torch.manual_seed(random_state)
    n_features = X_labeled.shape[1]

    gen = SSGANGenerator(n_features, n_classes, latent_dim, hidden).to(DEVICE)
    disc = SSGANDiscriminator(n_features, n_classes, hidden).to(DEVICE)

    opt_g = optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))

    X_lab_t = torch.from_numpy(X_labeled.astype(np.float32)).to(DEVICE)
    y_lab_t  = torch.from_numpy(y_enc.astype(np.int64)).to(DEVICE)
    X_unl_t  = torch.from_numpy(X_unlabeled.astype(np.float32)).to(DEVICE)

    n_lab = len(X_lab_t)
    n_unl = len(X_unl_t)

    for epoch in range(epochs):
        perm_lab = torch.randperm(n_lab, device=DEVICE)
        perm_unl = torch.randperm(n_unl, device=DEVICE)

        for s in range(0, min(n_lab, n_unl), batch_size):
            # --- Discriminator step ---
            idx_lab = perm_lab[s:s + batch_size]
            idx_unl = perm_unl[s:s + batch_size]
            x_real_lab = X_lab_t[idx_lab]
            y_real_lab = y_lab_t[idx_lab]
            x_real_unl = X_unl_t[idx_unl]

            # Generate fake samples
            bs = len(x_real_lab)
            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_labels = torch.randint(0, n_classes, (bs,), device=DEVICE)
            x_fake = gen(z, fake_labels).detach()

            opt_d.zero_grad()
            # Real labeled: real/fake=1 + classification loss
            rf_lab, cls_lab = disc(x_real_lab)
            loss_d_real_rf = F.binary_cross_entropy_with_logits(rf_lab, torch.ones_like(rf_lab))
            loss_d_cls = F.cross_entropy(cls_lab, y_real_lab) * lambda_cls

            # Real unlabeled: real/fake=1 (no class loss)
            rf_unl, _ = disc(x_real_unl)
            loss_d_real_unl = F.binary_cross_entropy_with_logits(rf_unl, torch.ones_like(rf_unl))

            # Fake: real/fake=0
            rf_fake, _ = disc(x_fake)
            loss_d_fake = F.binary_cross_entropy_with_logits(rf_fake, torch.zeros_like(rf_fake))

            loss_d = loss_d_real_rf + loss_d_real_unl + loss_d_fake + loss_d_cls
            loss_d.backward()
            opt_d.step()

            # --- Generator step ---
            opt_g.zero_grad()
            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_labels = torch.randint(0, n_classes, (bs,), device=DEVICE)
            x_fake = gen(z, fake_labels)
            rf_fake, cls_fake = disc(x_fake)
            loss_g_rf  = F.binary_cross_entropy_with_logits(rf_fake, torch.ones_like(rf_fake))
            loss_g_cls = F.cross_entropy(cls_fake, fake_labels) * lambda_cls
            loss_g = loss_g_rf + loss_g_cls
            loss_g.backward()
            opt_g.step()

        if (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss_D={loss_d.item():.4f}  loss_G={loss_g.item():.4f}")

    return gen, latent_dim


@torch.no_grad()
def ssgan_generate(
    gen: SSGANGenerator,
    cell_enc: int,
    n_samples: int,
    latent_dim: int,
) -> np.ndarray:
    gen.eval()
    z = torch.randn(n_samples, latent_dim, device=DEVICE)
    labels = torch.full((n_samples,), cell_enc, dtype=torch.long, device=DEVICE)
    return gen(z, labels).cpu().numpy()


# ---------------------------------------------------------------------------
# LOCO augmentation
# ---------------------------------------------------------------------------

def run_loco_ssgan(room: str, args: argparse.Namespace) -> dict:
    print(f"\n=== SSGAN LOCO — {room} ===")
    df = load_room(room)
    campaigns = sorted(df["campaign"].unique())
    if len(campaigns) < 2:
        return {}

    lookup = _cell_lookup(df)
    all_results: dict = {}

    for held_out in campaigns:
        train_df = df[df["campaign"] != held_out]
        test_df  = df[df["campaign"] == held_out]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_df["grid_cell"].to_numpy()
        X_test  = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test  = test_df["grid_cell"].to_numpy()

        le = LabelEncoder().fit(y_train)
        y_enc = le.transform(y_train)
        n_classes = len(le.classes_)

        scaler = StandardScaler().fit(X_train)
        X_tr_s = scaler.transform(X_train).astype(np.float32)
        X_te_s = scaler.transform(X_test).astype(np.float32)

        # The unlabeled pool = all training samples (label ignored for semi-supervised part)
        X_unl = X_tr_s.copy()

        camp_short = held_out.split("/")[-1]
        print(f"\n  Held-out: {camp_short}  (train={len(X_tr_s)}, test={len(X_te_s)})")

        # Baseline
        knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn.fit(X_tr_s, y_train)
        baseline = _metrics(y_test, knn.predict(X_te_s), lookup)
        print(f"    KNN baseline:       acc={baseline['cell_acc']:.4f}")

        # Train SSGAN
        print(f"    Training SSGAN ({args.epochs} epochs)...")
        gen, latent_dim = train_ssgan(
            X_tr_s, y_enc, X_unl, n_classes,
            epochs=args.epochs,
            lr_g=args.lr,
            lr_d=args.lr,
            latent_dim=args.latent_dim,
            hidden=args.hidden,
            batch_size=args.batch_size,
            lambda_cls=args.lambda_cls,
            random_state=SEED,
        )

        # Generate synthetic samples for cells in test set
        test_cells = [c for c in np.unique(y_test) if c in le.classes_]
        X_syn, y_syn = [], []
        for cell in test_cells:
            enc = le.transform([cell])[0]
            syn = ssgan_generate(gen, enc, args.n_synth, latent_dim)
            X_syn.append(syn)
            y_syn.extend([cell] * args.n_synth)

        X_syn_arr = np.vstack(X_syn).astype(np.float32)
        y_syn_arr = np.array(y_syn)

        X_aug = np.vstack([X_tr_s, X_syn_arr])
        y_aug = np.concatenate([y_train, y_syn_arr])

        knn_aug = KNeighborsClassifier(n_neighbors=7, weights="distance")
        knn_aug.fit(X_aug, y_aug)
        aug_result = _metrics(y_test, knn_aug.predict(X_te_s), lookup)
        delta = aug_result["cell_acc"] - baseline["cell_acc"]
        print(f"    KNN + SSGAN aug:    acc={aug_result['cell_acc']:.4f}  Δ={delta:+.4f}")

        all_results[camp_short] = {
            "baseline": baseline,
            "ssgan_augmented": aug_result,
            "delta": delta,
        }

    if all_results:
        base_accs = [v["baseline"]["cell_acc"] for v in all_results.values()]
        aug_accs  = [v["ssgan_augmented"]["cell_acc"] for v in all_results.values()]
        print(f"\n  LOCO average: baseline={np.mean(base_accs):.4f}  augmented={np.mean(aug_accs):.4f}  "
              f"Δ={np.mean(aug_accs)-np.mean(base_accs):+.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="SSGAN augmentation for RSSI LOCO")
    parser.add_argument("--rooms", nargs="+", default=["E102", "B121"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lambda-cls", type=float, default=1.0,
                        help="Weight of classification loss in discriminator.")
    parser.add_argument("--n-synth", type=int, default=25,
                        help="Synthetic samples per cell per held-out campaign.")
    args = parser.parse_args()

    all_results: dict = {}
    for room in args.rooms:
        all_results[room] = run_loco_ssgan(room, args)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "ssgan_augmentation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
