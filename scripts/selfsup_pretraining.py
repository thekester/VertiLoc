"""Self-supervised pretraining for RSSI localization.

Three SSL objectives tested on the RSSI encoder, then fine-tuned for cell
classification:

1. Masking autoencoder (MAE-style)
   Randomly mask p% of features → 0, reconstruct originals.
   Forces the encoder to learn cross-feature correlations.

2. Denoising autoencoder (DAE)
   Add Gaussian noise N(0, sigma) to inputs, reconstruct clean version.
   Closely related to BERT masking but continuous.

3. Contrastive pretraining (SimCLR-style)
   Two augmented views of the same sample (independent noise) → pull
   together, push others apart. NT-Xent loss.

Pipeline:
  Phase 1 — pretrain encoder on ALL available unlabeled data (all rooms, all
             campaigns). No cell labels used.
  Phase 2 — fine-tune a classification head on labeled training data.
  Evaluate on standard room-aware / E102 / LORO protocols.

Reference: BERT (Devlin 2019), SimCLR (Chen 2020), MAE (He 2022)
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
# Shared encoder backbone
# ---------------------------------------------------------------------------

class RSSIEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, emb_dim: int = 64):
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


class ClassificationHead(nn.Module):
    def __init__(self, emb_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def mask_features(x: torch.Tensor, mask_ratio: float = 0.4) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(x) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def add_noise(x: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
    return x + sigma * torch.randn_like(x)


# ---------------------------------------------------------------------------
# SSL pretraining: MAE / DAE
# ---------------------------------------------------------------------------

def pretrain_mae_dae(
    X_unlabeled: np.ndarray,
    n_features: int,
    *,
    method: str = "mae",
    epochs: int = 100,
    lr: float = 1e-3,
    hidden: int = 128,
    emb_dim: int = 64,
    batch_size: int = 512,
    mask_ratio: float = 0.4,
    noise_sigma: float = 0.3,
    random_state: int = SEED,
) -> RSSIEncoder:
    torch.manual_seed(random_state)

    encoder = RSSIEncoder(n_features, hidden=hidden, emb_dim=emb_dim).to(DEVICE)
    decoder = nn.Sequential(
        nn.Linear(emb_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_features),
    ).to(DEVICE)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_unlabeled.astype(np.float32)).to(DEVICE)
    n = len(X_t)

    encoder.train()
    decoder.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            x_clean = X_t[idx]
            if method == "mae":
                x_in, _ = mask_features(x_clean, mask_ratio=mask_ratio)
            else:  # dae
                x_in = add_noise(x_clean, sigma=noise_sigma)
            z = encoder(x_in)
            x_recon = decoder(z)
            loss = F.mse_loss(x_recon, x_clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return encoder


# ---------------------------------------------------------------------------
# SSL pretraining: SimCLR NT-Xent
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2n, d)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature  # (2n, 2n)

    mask = ~torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~mask, -1e9)

    labels = torch.cat([
        torch.arange(n, 2 * n, device=z.device),
        torch.arange(0, n, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


def pretrain_simclr(
    X_unlabeled: np.ndarray,
    n_features: int,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    hidden: int = 128,
    emb_dim: int = 64,
    proj_dim: int = 32,
    batch_size: int = 256,
    noise_sigma: float = 0.2,
    temperature: float = 0.1,
    random_state: int = SEED,
) -> RSSIEncoder:
    torch.manual_seed(random_state)

    encoder = RSSIEncoder(n_features, hidden=hidden, emb_dim=emb_dim).to(DEVICE)
    projector = nn.Sequential(
        nn.Linear(emb_dim, emb_dim),
        nn.ReLU(),
        nn.Linear(emb_dim, proj_dim),
    ).to(DEVICE)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_unlabeled.astype(np.float32)).to(DEVICE)
    n = len(X_t)

    encoder.train()
    projector.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            x = X_t[idx]
            x1 = add_noise(x, sigma=noise_sigma)
            x2 = add_noise(x, sigma=noise_sigma)
            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return encoder


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(
    encoder: RSSIEncoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    le: LabelEncoder,
    *,
    freeze_encoder: bool = False,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    random_state: int = SEED,
) -> tuple[RSSIEncoder, ClassificationHead]:
    torch.manual_seed(random_state)
    y_enc = le.transform(y_train)
    n_classes = len(le.classes_)

    emb_dim = encoder.net[-1].out_features
    head = ClassificationHead(emb_dim, n_classes).to(DEVICE)

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = head.parameters()
    else:
        params = list(encoder.parameters()) + list(head.parameters())

    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    y_t = torch.from_numpy(y_enc.astype(np.int64)).to(DEVICE)
    n = len(X_t)

    encoder.train()
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            z = encoder(X_t[idx])
            logits = head(z)
            loss = F.cross_entropy(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = True

    return encoder, head


@torch.no_grad()
def ssl_predict(
    encoder: RSSIEncoder,
    head: ClassificationHead,
    X_test: np.ndarray,
    le: LabelEncoder,
) -> np.ndarray:
    encoder.eval()
    head.eval()
    X_t = torch.from_numpy(X_test.astype(np.float32)).to(DEVICE)
    z = encoder(X_t)
    logits = head(z)
    return le.inverse_transform(logits.argmax(1).cpu().numpy())


@torch.no_grad()
def ssl_encode(encoder: RSSIEncoder, X: np.ndarray) -> np.ndarray:
    encoder.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
    return encoder(X_t).cpu().numpy()


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

def run_protocol(
    label: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    df_full: pd.DataFrame,
    X_unlabeled: np.ndarray,
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

    le = LabelEncoder().fit(y_train)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train).astype(np.float32)
    X_te_s = scaler.transform(X_test).astype(np.float32)
    X_unl_s = scaler.transform(X_unlabeled).astype(np.float32)
    n_features = X_tr_s.shape[1]

    # ---- KNN baseline ----
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_tr_s, y_train)
    results["KNN"] = _metrics(y_test, knn.predict(X_te_s), lookup)
    print(f"    KNN          acc={results['KNN']['cell_acc']:.4f}  err={results['KNN']['mean_error_m']:.3f}m")

    # ---- No pretraining (random encoder, finetune only) ----
    enc_scratch = RSSIEncoder(n_features, hidden=args.hidden, emb_dim=args.emb_dim).to(DEVICE)
    enc_s, head_s = finetune(
        enc_scratch, X_tr_s, y_train, le,
        freeze_encoder=False, epochs=args.finetune_epochs, lr=args.lr,
    )
    results["MLP_scratch"] = _metrics(y_test, ssl_predict(enc_s, head_s, X_te_s, le), lookup)
    print(f"    MLP_scratch  acc={results['MLP_scratch']['cell_acc']:.4f}  err={results['MLP_scratch']['mean_error_m']:.3f}m")

    # ---- SSL methods ----
    for method in args.methods:
        try:
            print(f"    Pretraining {method} ({args.pretrain_epochs} epochs)...")
            if method in ("mae", "dae"):
                enc_pre = pretrain_mae_dae(
                    X_unl_s, n_features,
                    method=method,
                    epochs=args.pretrain_epochs, lr=args.lr,
                    hidden=args.hidden, emb_dim=args.emb_dim,
                    batch_size=args.batch_size,
                )
            elif method == "simclr":
                enc_pre = pretrain_simclr(
                    X_unl_s, n_features,
                    epochs=args.pretrain_epochs, lr=args.lr,
                    hidden=args.hidden, emb_dim=args.emb_dim,
                    batch_size=args.batch_size,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Fine-tune with full encoder update
            enc_ft, head_ft = finetune(
                enc_pre, X_tr_s, y_train, le,
                freeze_encoder=False,
                epochs=args.finetune_epochs, lr=args.lr * 0.1,
            )
            pred_ft = ssl_predict(enc_ft, head_ft, X_te_s, le)
            results[f"{method}_finetune"] = _metrics(y_test, pred_ft, lookup)
            r = results[f"{method}_finetune"]
            print(f"    {method}_finetune acc={r['cell_acc']:.4f}  err={r['mean_error_m']:.3f}m")

            # KNN in pretrained embedding space (frozen encoder)
            Z_tr = ssl_encode(enc_ft, X_tr_s)
            Z_te = ssl_encode(enc_ft, X_te_s)
            knn_emb = KNeighborsClassifier(n_neighbors=7, weights="distance")
            knn_emb.fit(Z_tr, y_train)
            results[f"{method}_knn"] = _metrics(y_test, knn_emb.predict(Z_te), lookup)
            r2 = results[f"{method}_knn"]
            print(f"    {method}_knn      acc={r2['cell_acc']:.4f}  err={r2['mean_error_m']:.3f}m")

        except Exception as exc:
            print(f"    {method}: FAILED — {exc}")
            results[f"{method}_finetune"] = {"cell_acc": float("nan"), "error": str(exc)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for RSSI localization")
    parser.add_argument("--methods", nargs="+", default=["mae", "dae", "simclr"],
                        choices=["mae", "dae", "simclr"])
    parser.add_argument("--pretrain-epochs", type=int, default=150,
                        help="Epochs for SSL pretraining phase")
    parser.add_argument("--finetune-epochs", type=int, default=150,
                        help="Epochs for supervised fine-tuning phase")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-loro", action="store_true", default=True)
    args = parser.parse_args()

    # All unlabeled data (all rooms) used for pretraining
    df_all = load_multiroom()
    X_unlabeled_full = df_all[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    all_results: dict = {}

    print("\n=== Room-aware (multi-room, 80/20) ===")
    df = load_multiroom()
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["grid_cell"])
    X_unl = build_X(tr, include_room=True).astype(np.float32)
    all_results["room_aware"] = run_protocol(
        "room_aware", tr, te, df, X_unl, include_room=True, args=args
    )

    print("\n=== E102 intra-room (80/20) ===")
    df_e = load_multiroom(["E102"])
    tr_e, te_e = train_test_split(df_e, test_size=0.2, random_state=SEED, stratify=df_e["grid_cell"])
    X_unl_e = build_X(tr_e, include_room=False).astype(np.float32)
    all_results["e102"] = run_protocol(
        "E102", tr_e, te_e, df_e, X_unl_e, include_room=False, args=args
    )

    if not args.skip_loro:
        print("\n=== LORO ===")
        df = load_multiroom()
        rooms = sorted(df["room"].unique())
        folds, avgs = {}, {}
        for held_out in rooms:
            tr_l = df[df["room"] != held_out]
            te_l = df[df["room"] == held_out]
            X_unl_l = build_X(tr_l, include_room=False).astype(np.float32)
            folds[held_out] = run_protocol(
                f"LORO_{held_out}", tr_l, te_l, df, X_unl_l, include_room=False, args=args
            )
        model_names = list(next(iter(folds.values())).keys())
        print("\n  LORO averages:")
        for m in model_names:
            accs = [folds[r][m]["cell_acc"] for r in rooms if m in folds[r]]
            valid = [a for a in accs if not np.isnan(a)]
            avg = float(np.mean(valid)) if valid else float("nan")
            avgs[m] = {"cell_acc_mean": avg}
            print(f"    {m:<22} acc={avg:.4f}")
        all_results["loro"] = {"folds": folds, "averages": avgs}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "selfsup_pretraining.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    print("\n=== SUMMARY ===")
    for proto, res in all_results.items():
        if proto == "loro":
            print(f"\n{proto} averages:")
            for m, v in res.get("averages", {}).items():
                print(f"  {m:<25} acc={v['cell_acc_mean']:.4f}")
        else:
            print(f"\n{proto}:")
            for m, v in res.items():
                print(f"  {m:<25} acc={v.get('cell_acc', float('nan')):.4f}  "
                      f"err={v.get('mean_error_m', float('nan')):.3f}m")


if __name__ == "__main__":
    main()
