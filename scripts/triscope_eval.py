"""TriScope evaluation: one model, three tasks (cell, distance, room) from RSSI only.

Runs 50 random draws of samples and reports aggregated accuracies.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import CampaignSpec, load_measurements  # noqa: E402
from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer  # noqa: E402

# Campaign folders grouped by room, excluding circular trajectories.
ROOM_CAMPAIGNS = {
    "D005": [
        ROOT / "data" / "D005" / "ddeuxmetres",
        ROOT / "data" / "D005" / "dquatremetres",
    ],
    "E101": [
        ROOT / "data" / "E101" / "dtroismetres",
        ROOT / "data" / "E101" / "dcinqmetres",
    ],
}
FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]


def load_cross_room() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for room, folders in ROOM_CAMPAIGNS.items():
        for folder in folders:
            if folder.exists():
                df_room = load_measurements([CampaignSpec(folder)])
                df_room["room"] = room
                df_room["campaign"] = f"{room}/{folder.name}"
                frames.append(df_room)
            else:
                missing.append(folder)
    if missing:
        raise FileNotFoundError(f"Missing campaign folders: {missing}")
    if not frames:
        raise RuntimeError("No measurement data found for any room.")
    df = pd.concat(frames, ignore_index=True)
    return df


def build_features(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].reset_index(drop=True).to_numpy()


def fit_triscope(df: pd.DataFrame, random_state: int = 64):
    cfg = EmbeddingKnnConfig(
        hidden_layer_sizes=(64, 32),
        k_neighbors=5,
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=1200,
        random_state=random_state,
    )
    localizer = EmbeddingKnnLocalizer(config=cfg)
    localizer.fit(build_features(df), df["grid_cell"])
    emb = localizer.transform(build_features(df))

    dist_head = LogisticRegression(max_iter=800, solver="lbfgs")
    dist_head.fit(emb, df["router_distance_m"])

    room_head = LogisticRegression(max_iter=800, solver="lbfgs")
    room_head.fit(emb, df["room"])

    return localizer, dist_head, room_head


def evaluate_triscope(df: pd.DataFrame, iterations: int = 50, sample_size: int = 5, seed: int = 7):
    rng = np.random.default_rng(seed)
    localizer, dist_head, room_head = fit_triscope(df)

    cell_hits = []
    dist_hits = []
    room_hits = []
    for _ in range(iterations):
        idx = rng.choice(len(df), size=sample_size, replace=False)
        batch = df.iloc[idx]
        X = build_features(batch)
        cell_pred = localizer.predict(X)
        emb = localizer.transform(X)
        dist_pred = dist_head.predict(emb)
        room_pred = room_head.predict(emb)

        cell_hits.append((cell_pred == batch["grid_cell"].to_numpy()).mean())
        dist_hits.append((dist_pred == batch["router_distance_m"].to_numpy()).mean())
        room_hits.append((room_pred == batch["room"].to_numpy()).mean())

    return {
        "cell_acc_mean": float(np.mean(cell_hits)),
        "cell_acc_std": float(np.std(cell_hits)),
        "router_dist_acc_mean": float(np.mean(dist_hits)),
        "router_dist_acc_std": float(np.std(dist_hits)),
        "room_acc_mean": float(np.mean(room_hits)),
        "room_acc_std": float(np.std(room_hits)),
        "iterations": iterations,
        "sample_size": sample_size,
    }


def main():
    df = load_cross_room()
    print(
        f"Loaded {len(df)} samples | rooms={sorted(df['room'].unique())} | "
        f"cells={df['grid_cell'].nunique()} | distances={sorted(df['router_distance_m'].unique())}"
    )
    metrics = evaluate_triscope(df, iterations=50, sample_size=5, seed=7)
    print("TriScope (50 random draws):")
    print(metrics)
    out_path = ROOT / "reports" / "benchmarks" / "triscope_runs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
