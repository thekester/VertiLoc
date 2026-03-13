"""Explore generative radio map estimation with cGAN and CycleGAN baselines.

This script targets the practical question:
can synthetic RSSI fingerprints help when the target campaign has very few labels?

Two generative modes are provided:
- cGAN: class-conditional generator trained on target low-data samples.
- cyclegan: unpaired domain transfer from source campaigns to target campaign.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "PyTorch is required for this script. Install it with: pip install torch"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
REPORT_DIR = ROOT / "reports" / "benchmarks"


@dataclass(frozen=True)
class RunData:
    X_train_low: np.ndarray
    y_train_low: np.ndarray
    X_train_full: np.ndarray
    y_train_full: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    target_lookup: pd.DataFrame
    X_source: np.ndarray | None
    y_source: np.ndarray | None
    scaler: StandardScaler


class CondGenerator(nn.Module):
    def __init__(self, noise_dim: int, num_classes: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, y_onehot], dim=1))


class CondDiscriminator(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y_onehot], dim=1))


class MlpGenerator(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MlpDiscriminator(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_campaign_entry(entry: str) -> CampaignSpec:
    if ":" in entry:
        folder, distance = entry.split(":", 1)
        return CampaignSpec(Path(folder), float(distance))
    return CampaignSpec(Path(entry))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_fraction(
    df: pd.DataFrame,
    *,
    label_col: str,
    fraction: float,
    min_per_class: int,
    seed: int,
) -> pd.DataFrame:
    if fraction >= 1.0:
        return df.copy()

    sampled = []
    for _, group in df.groupby(label_col, sort=False):
        target_n = max(min_per_class, int(round(len(group) * fraction)))
        target_n = min(target_n, len(group))
        sampled.append(group.sample(n=target_n, random_state=seed))
    return pd.concat(sampled, ignore_index=True)


def compute_mean_error_m(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lookup: pd.DataFrame,
) -> float:
    coords = lookup.set_index("grid_cell")[["coord_x_m", "coord_y_m"]]
    true_xy = coords.loc[y_true].to_numpy(dtype=float)
    pred_xy = coords.loc[y_pred].to_numpy(dtype=float)
    return float(np.linalg.norm(true_xy - pred_xy, axis=1).mean())


def evaluate_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lookup: pd.DataFrame,
    *,
    k_neighbors: int,
) -> dict[str, float]:
    clf = KNeighborsClassifier(
        n_neighbors=k_neighbors,
        weights="distance",
        metric="euclidean",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "cell_accuracy": float(accuracy_score(y_test, y_pred)),
        "mean_error_m": compute_mean_error_m(y_test, y_pred, lookup),
    }


def load_and_split_data(
    *,
    target_spec: CampaignSpec,
    source_specs: list[CampaignSpec],
    test_size: float,
    low_data_ratio: float,
    seed: int,
) -> RunData:
    target_df = load_measurements([target_spec])
    source_df = load_measurements(source_specs) if source_specs else None

    missing = [col for col in FEATURE_COLUMNS if col not in target_df.columns]
    if missing:
        raise ValueError(f"Target campaign missing features: {missing}")
    if source_df is not None:
        missing_src = [col for col in FEATURE_COLUMNS if col not in source_df.columns]
        if missing_src:
            raise ValueError(f"Source campaigns missing features: {missing_src}")

    target_df = target_df.copy()
    target_df["grid_cell"] = target_df["grid_cell"].astype(str)

    target_train, target_test = train_test_split(
        target_df,
        test_size=test_size,
        random_state=seed,
        stratify=target_df["grid_cell"],
    )
    target_low = stratified_fraction(
        target_train,
        label_col="grid_cell",
        fraction=low_data_ratio,
        min_per_class=1,
        seed=seed,
    )

    scaler = StandardScaler()
    X_train_low = scaler.fit_transform(target_low[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    X_train_full = scaler.transform(target_train[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    X_test = scaler.transform(target_test[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    y_train_low = target_low["grid_cell"].to_numpy()
    y_train_full = target_train["grid_cell"].to_numpy()
    y_test = target_test["grid_cell"].to_numpy()
    target_lookup = (
        target_df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .reset_index(drop=True)
    )

    X_source = None
    y_source = None
    if source_df is not None:
        source_df = source_df.copy()
        source_df["grid_cell"] = source_df["grid_cell"].astype(str)
        keep_cells = set(target_lookup["grid_cell"])
        source_df = source_df[source_df["grid_cell"].isin(keep_cells)].reset_index(drop=True)
        if not source_df.empty:
            X_source = scaler.transform(source_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            y_source = source_df["grid_cell"].to_numpy()

    return RunData(
        X_train_low=X_train_low,
        y_train_low=y_train_low,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        target_lookup=target_lookup,
        X_source=X_source,
        y_source=y_source,
        scaler=scaler,
    )


def train_cgan_and_generate(
    run_data: RunData,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_dim: int,
    synthetic_per_class: int,
    lambda_center: float,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    classes = np.unique(run_data.y_train_low)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_idx = np.asarray([class_to_idx[c] for c in run_data.y_train_low], dtype=np.int64)

    X = torch.from_numpy(run_data.X_train_low.astype(np.float32)).to(device)
    y = torch.from_numpy(y_idx).to(device)

    num_classes = len(classes)
    in_dim = X.shape[1]
    G = CondGenerator(noise_dim=noise_dim, num_classes=num_classes, out_dim=in_dim).to(device)
    D = CondDiscriminator(in_dim=in_dim, num_classes=num_classes).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    class_means = []
    for idx in range(num_classes):
        mask = y == idx
        class_means.append(X[mask].mean(dim=0))
    class_means_t = torch.stack(class_means, dim=0).detach()

    n_samples = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb = X[batch_idx]
            yb = y[batch_idx]
            yb_onehot = torch.nn.functional.one_hot(yb, num_classes=num_classes).float()

            valid = torch.ones((xb.size(0), 1), device=device)
            fake = torch.zeros((xb.size(0), 1), device=device)

            z = torch.randn((xb.size(0), noise_dim), device=device)
            x_fake = G(z, yb_onehot).detach()

            opt_d.zero_grad()
            d_real = D(xb, yb_onehot)
            d_fake = D(x_fake, yb_onehot)
            d_loss = 0.5 * (bce(d_real, valid) + bce(d_fake, fake))
            d_loss.backward()
            opt_d.step()

            z = torch.randn((xb.size(0), noise_dim), device=device)
            x_gen = G(z, yb_onehot)
            opt_g.zero_grad()
            g_adv = bce(D(x_gen, yb_onehot), valid)
            centers = class_means_t[yb]
            g_center = l1(x_gen, centers)
            g_loss = g_adv + lambda_center * g_center
            g_loss.backward()
            opt_g.step()

    X_syn = []
    y_syn = []
    G.eval()
    with torch.no_grad():
        for label in classes:
            idx = class_to_idx[label]
            y_cond = torch.full((synthetic_per_class,), idx, device=device, dtype=torch.long)
            y_onehot = torch.nn.functional.one_hot(y_cond, num_classes=num_classes).float()
            z = torch.randn((synthetic_per_class, noise_dim), device=device)
            xg = G(z, y_onehot).cpu().numpy()
            X_syn.append(xg)
            y_syn.extend([label] * synthetic_per_class)

    return np.vstack(X_syn).astype(np.float32), np.asarray(y_syn)


def train_cycle_gan_and_translate(
    run_data: RunData,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_cycle: float,
    lambda_identity: float,
    synthetic_per_class: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if run_data.X_source is None or run_data.y_source is None:
        raise ValueError("CycleGAN mode requires --source campaigns.")

    set_seed(seed)
    XA = torch.from_numpy(run_data.X_source.astype(np.float32)).to(device)
    yA = run_data.y_source
    XB = torch.from_numpy(run_data.X_train_low.astype(np.float32)).to(device)

    dim = XA.shape[1]
    G_AB = MlpGenerator(dim).to(device)
    G_BA = MlpGenerator(dim).to(device)
    D_A = MlpDiscriminator(dim).to(device)
    D_B = MlpDiscriminator(dim).to(device)

    opt_g = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
    )
    opt_da = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_db = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    n_steps = max(1, int(max(len(XA), len(XB)) / max(1, batch_size)))
    for _ in range(epochs):
        for _ in range(n_steps):
            idx_a = torch.randint(0, len(XA), (batch_size,), device=device)
            idx_b = torch.randint(0, len(XB), (batch_size,), device=device)
            real_a = XA[idx_a]
            real_b = XB[idx_b]

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            fake_b = G_AB(real_a).detach()
            fake_a = G_BA(real_b).detach()

            opt_da.zero_grad()
            d_a_loss = 0.5 * (bce(D_A(real_a), valid) + bce(D_A(fake_a), fake))
            d_a_loss.backward()
            opt_da.step()

            opt_db.zero_grad()
            d_b_loss = 0.5 * (bce(D_B(real_b), valid) + bce(D_B(fake_b), fake))
            d_b_loss.backward()
            opt_db.step()

            opt_g.zero_grad()
            fake_b = G_AB(real_a)
            fake_a = G_BA(real_b)

            adv_loss = bce(D_B(fake_b), valid) + bce(D_A(fake_a), valid)
            cycle_loss = l1(G_BA(fake_b), real_a) + l1(G_AB(fake_a), real_b)
            id_loss = l1(G_BA(real_a), real_a) + l1(G_AB(real_b), real_b)
            g_loss = adv_loss + lambda_cycle * cycle_loss + lambda_identity * id_loss
            g_loss.backward()
            opt_g.step()

    G_AB.eval()
    with torch.no_grad():
        translated = G_AB(XA).cpu().numpy()

    target_classes = set(np.unique(run_data.y_train_low))
    X_syn = []
    y_syn = []
    rng = np.random.default_rng(seed)
    for label in sorted(target_classes):
        mask = yA == label
        if not np.any(mask):
            continue
        candidates = translated[mask]
        if len(candidates) <= synthetic_per_class:
            chosen = candidates
        else:
            take = rng.choice(len(candidates), size=synthetic_per_class, replace=False)
            chosen = candidates[take]
        X_syn.append(chosen)
        y_syn.extend([label] * len(chosen))

    if not X_syn:
        raise RuntimeError("CycleGAN produced no synthetic samples for shared grid cells.")

    return np.vstack(X_syn).astype(np.float32), np.asarray(y_syn)


def run_experiment(args: argparse.Namespace) -> dict:
    device = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    target_spec = parse_campaign_entry(args.target)
    source_specs = [parse_campaign_entry(entry) for entry in args.source]

    run_data = load_and_split_data(
        target_spec=target_spec,
        source_specs=source_specs,
        test_size=args.test_size,
        low_data_ratio=args.low_data_ratio,
        seed=args.seed,
    )

    baseline = evaluate_knn(
        run_data.X_train_low,
        run_data.y_train_low,
        run_data.X_test,
        run_data.y_test,
        run_data.target_lookup,
        k_neighbors=args.k_neighbors,
    )
    full_upper = evaluate_knn(
        run_data.X_train_full,
        run_data.y_train_full,
        run_data.X_test,
        run_data.y_test,
        run_data.target_lookup,
        k_neighbors=args.k_neighbors,
    )

    if args.mode == "cgan":
        X_syn, y_syn = train_cgan_and_generate(
            run_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            noise_dim=args.noise_dim,
            synthetic_per_class=args.synthetic_per_class,
            lambda_center=args.lambda_center,
            device=device,
            seed=args.seed,
        )
    else:
        X_syn, y_syn = train_cycle_gan_and_translate(
            run_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_cycle=args.lambda_cycle,
            lambda_identity=args.lambda_identity,
            synthetic_per_class=args.synthetic_per_class,
            device=device,
            seed=args.seed,
        )

    X_aug = np.vstack([run_data.X_train_low, X_syn])
    y_aug = np.concatenate([run_data.y_train_low, y_syn])
    augmented = evaluate_knn(
        X_aug,
        y_aug,
        run_data.X_test,
        run_data.y_test,
        run_data.target_lookup,
        k_neighbors=args.k_neighbors,
    )

    result = {
        "mode": args.mode,
        "target": args.target,
        "sources": args.source,
        "device": device,
        "seed": args.seed,
        "params": {
            "low_data_ratio": args.low_data_ratio,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "k_neighbors": args.k_neighbors,
            "synthetic_per_class": args.synthetic_per_class,
        },
        "counts": {
            "train_low": int(len(run_data.X_train_low)),
            "train_full": int(len(run_data.X_train_full)),
            "test": int(len(run_data.X_test)),
            "synthetic": int(len(X_syn)),
        },
        "metrics": {
            "baseline_low_data": baseline,
            "augmented_with_generative": augmented,
            "upper_bound_full_train": full_upper,
        },
    }
    return result


def save_results(result: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target_name = Path(result["target"].split(":", 1)[0]).name
    ratio_tag = str(result["params"]["low_data_ratio"]).replace(".", "p")
    source_tag = "nosrc"
    if result["sources"]:
        source_names = [Path(src.split(":", 1)[0]).name for src in result["sources"]]
        source_tag = "src-" + "-".join(source_names)
    filename = (
        f"generative_{result['mode']}_{target_name}_{source_tag}"
        f"_r{ratio_tag}_e{result['params']['epochs']}_s{result['params']['synthetic_per_class']}"
        f"_seed{result['seed']}.json"
    )
    out_path = REPORT_DIR / filename
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    rows = []
    for metric_name, values in result["metrics"].items():
        rows.append(
            {
                "metric_set": metric_name,
                "cell_accuracy": values["cell_accuracy"],
                "mean_error_m": values["mean_error_m"],
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generative radio map estimation benchmark (cGAN / CycleGAN)."
    )
    parser.add_argument("--mode", choices=["cgan", "cyclegan"], required=True)
    parser.add_argument(
        "--target",
        required=True,
        help="Target campaign path, optionally with distance override: path:distance",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source campaign path for transfer (repeatable). Required for cyclegan.",
    )
    parser.add_argument("--low-data-ratio", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--synthetic-per-class", type=int, default=40)
    parser.add_argument("--noise-dim", type=int, default=16)
    parser.add_argument("--lambda-center", type=float, default=0.15)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "cyclegan" and not args.source:
        parser.error("--source is required when --mode cyclegan.")

    torch.set_num_threads(max(1, int(args.torch_threads)))
    set_seed(args.seed)
    result = run_experiment(args)
    out_path = save_results(result)

    baseline = result["metrics"]["baseline_low_data"]
    augmented = result["metrics"]["augmented_with_generative"]
    upper = result["metrics"]["upper_bound_full_train"]

    print("=== Generative Radio Map Experiment ===")
    print(f"mode={result['mode']} target={result['target']} seed={result['seed']}")
    print(
        f"counts: low={result['counts']['train_low']} synthetic={result['counts']['synthetic']} "
        f"test={result['counts']['test']}"
    )
    print(
        "baseline_low_data: "
        f"acc={baseline['cell_accuracy']:.4f} err={baseline['mean_error_m']:.4f}m"
    )
    print(
        "augmented_with_generative: "
        f"acc={augmented['cell_accuracy']:.4f} err={augmented['mean_error_m']:.4f}m"
    )
    print(
        "upper_bound_full_train: "
        f"acc={upper['cell_accuracy']:.4f} err={upper['mean_error_m']:.4f}m"
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
