"""Train a PyTorch NN encoder + local KNN decoder for RSSI localization.

This script mirrors the sklearn NN+L-KNN approach already used in the project,
but replaces the encoder training with PyTorch.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - runtime guard for missing torch
    raise ImportError(
        "PyTorch is required for this script. Install it with: pip install torch"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]


class MlpEncoderClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return self.classifier(emb)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def parse_campaign_args(entries: list[str]) -> list[CampaignSpec]:
    if not entries:
        return [
            CampaignSpec(Path("data/D005/ddeuxmetres"), 2.0),
            CampaignSpec(Path("data/D005/dquatremetres"), 4.0),
        ]

    specs: list[CampaignSpec] = []
    for entry in entries:
        if ":" in entry:
            folder, distance = entry.split(":", 1)
            specs.append(CampaignSpec(Path(folder), float(distance)))
        else:
            specs.append(CampaignSpec(Path(entry)))
    return specs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_mean_error_m(y_true_cells: np.ndarray, y_pred_cells: np.ndarray, df) -> float:
    cell_lookup = (
        df[["grid_cell", "coord_x_m", "coord_y_m"]]
        .drop_duplicates("grid_cell")
        .set_index("grid_cell")
    )
    true_coords = cell_lookup.loc[y_true_cells][["coord_x_m", "coord_y_m"]].to_numpy()
    pred_coords = cell_lookup.loc[y_pred_cells][["coord_x_m", "coord_y_m"]].to_numpy()
    return float(np.linalg.norm(true_coords - pred_coords, axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a PyTorch NN encoder + local KNN decoder on RSSI data."
    )
    parser.add_argument(
        "--campaign",
        action="append",
        default=[],
        help="Campaign folder, optionally with explicit distance: folder:distance",
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    campaigns = parse_campaign_args(args.campaign)
    df = load_measurements(campaigns)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["grid_cell"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train).astype(np.int64)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train_enc),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = MlpEncoderClassifier(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=args.hidden_dims,
        num_classes=len(label_encoder.classes_),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            epoch_loss = running_loss / len(train_ds)
            print(f"[epoch {epoch:4d}/{args.epochs}] train_loss={epoch_loss:.5f}")

    model.eval()
    with torch.no_grad():
        train_embeddings = model.encode(torch.from_numpy(X_train_scaled).to(device)).cpu().numpy()
        test_embeddings = model.encode(torch.from_numpy(X_test_scaled).to(device)).cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=args.k_neighbors, weights="distance", metric="euclidean")
    knn.fit(train_embeddings, y_train)

    y_pred = knn.predict(test_embeddings)
    acc = accuracy_score(y_test, y_pred)
    mean_error_m = compute_mean_error_m(y_test, y_pred, df)

    print("\n=== PyTorch NN + L-KNN Results ===")
    print(f"Samples: train={len(X_train)} | test={len(X_test)}")
    print(f"Device: {device}")
    print(f"Cell accuracy: {acc:.4f}")
    print(f"Mean localization error (m): {mean_error_m:.6f}")

    top_k = min(3, args.k_neighbors)
    distances, indices = knn.kneighbors(test_embeddings[:1], n_neighbors=top_k)
    neighbor_cells = np.asarray(y_train)[indices[0]]
    print("\nExample (first test sample):")
    print(f"True cell: {y_test[0]} | Pred cell: {y_pred[0]}")
    for rank, (cell, dist) in enumerate(zip(neighbor_cells, distances[0]), start=1):
        print(f"  #{rank}: neighbor_cell={cell} | embedding_distance={dist:.4f}")


if __name__ == "__main__":
    main()
