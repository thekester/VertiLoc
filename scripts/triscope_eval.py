"""TriScope evaluation: one model, three tasks (cell, distance, room) from RSSI only.

Runs 50 random draws of samples and reports aggregated accuracies.
"""

from __future__ import annotations

from pathlib import Path
import time
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

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
        hidden_layer_sizes=(48, 24),
        k_neighbors=5,
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=400,
        random_state=random_state,
    )
    localizer = EmbeddingKnnLocalizer(config=cfg)
    t0 = time.time()
    localizer.fit(build_features(df), df["grid_cell"])
    emb = localizer.transform(build_features(df))
    print(f"TriScope embedding fit in {time.time() - t0:.2f}s", flush=True)

    t1 = time.time()
    dist_head = LogisticRegression(max_iter=800, solver="lbfgs")
    dist_head.fit(emb, df["router_distance_m"])
    print(f"TriScope distance head fit in {time.time() - t1:.2f}s", flush=True)

    t2 = time.time()
    room_head = LogisticRegression(max_iter=800, solver="lbfgs")
    room_head.fit(emb, df["room"])
    print(f"TriScope room head fit in {time.time() - t2:.2f}s", flush=True)

    return localizer, dist_head, room_head


def evaluate_triscope(df: pd.DataFrame, iterations: int = 50, sample_size: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    localizer, dist_head, room_head = fit_triscope(df)
    print("TriScope fit done. Running random draws...", flush=True)

    cell_hits = []
    dist_hits = []
    room_hits = []
    draw_logs = []
    for i in tqdm(range(iterations), desc="TriScope random draws"):
        idx = rng.choice(len(df), size=sample_size, replace=False)
        batch = df.iloc[idx]
        X = build_features(batch)
        cell_pred = localizer.predict(X)
        emb = localizer.transform(X)
        dist_pred = dist_head.predict(emb)
        room_pred = room_head.predict(emb)

        cell_acc = (cell_pred == batch["grid_cell"].to_numpy()).mean()
        dist_acc = (dist_pred == batch["router_distance_m"].to_numpy()).mean()
        room_acc = (room_pred == batch["room"].to_numpy()).mean()

        cell_hits.append(cell_acc)
        dist_hits.append(dist_acc)
        room_hits.append(room_acc)
        draw_logs.append({"draw": i + 1, "cell_acc": cell_acc, "router_dist_acc": dist_acc, "room_acc": room_acc})

    return {
        "cell_acc_mean": float(np.mean(cell_hits)),
        "cell_acc_std": float(np.std(cell_hits)),
        "router_dist_acc_mean": float(np.mean(dist_hits)),
        "router_dist_acc_std": float(np.std(dist_hits)),
        "room_acc_mean": float(np.mean(room_hits)),
        "room_acc_std": float(np.std(room_hits)),
        "iterations": iterations,
        "sample_size": sample_size,
        "draw_logs": draw_logs,
    }


def eval_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 99):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    loc, dist_head, room_head = fit_triscope(train_df, random_state=seed)

    X_test = build_features(test_df)
    cell_pred = loc.predict(X_test)
    emb_test = loc.transform(X_test)
    dist_pred = dist_head.predict(emb_test)
    room_pred = room_head.predict(emb_test)

    return {
        "cell_acc": float(accuracy_score(test_df["grid_cell"], cell_pred)),
        "router_dist_acc": float(accuracy_score(test_df["router_distance_m"], dist_pred)),
        "room_acc": float(accuracy_score(test_df["room"], room_pred)),
        "test_size": test_size,
    }


def main():
    df = load_cross_room()
    print(
        f"Loaded {len(df)} samples | rooms={sorted(df['room'].unique())} | "
        f"cells={df['grid_cell'].nunique()} | distances={sorted(df['router_distance_m'].unique())}"
    )
    print("Fitting TriScope (embedding + distance/room heads)...", flush=True)
    metrics = evaluate_triscope(df, iterations=50, sample_size=20, seed=7)
    print("TriScope (50 random draws):")
    summary_line = (
        f"cell_acc={metrics['cell_acc_mean']:.3f}±{metrics['cell_acc_std']:.3f} | "
        f"dist_acc={metrics['router_dist_acc_mean']:.3f}±{metrics['router_dist_acc_std']:.3f} | "
        f"room_acc={metrics['room_acc_mean']:.3f}±{metrics['room_acc_std']:.3f} "
        f"(sample_size={metrics['sample_size']}, draws={metrics['iterations']})"
    )
    print(summary_line)
    # Show a few draw samples for quick “CheckeurFou” style sanity.
    preview_logs = metrics.get("draw_logs", [])[:5]
    if preview_logs:
        print("First 5 draws (cell | dist | room acc):")
        for log in preview_logs:
            print(
                f"  draw {log['draw']:2d}: "
                f"{log['cell_acc']:.2f} | {log['router_dist_acc']:.2f} | {log['room_acc']:.2f}"
            )
    out_path = ROOT / "reports" / "benchmarks" / "triscope_runs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {out_path}")

    split_metrics = eval_split(df, test_size=0.2, seed=99)
    print("TriScope (fixed 80/20 split):")
    print(
        f"cell_acc={split_metrics['cell_acc']:.3f} | "
        f"dist_acc={split_metrics['router_dist_acc']:.3f} | "
        f"room_acc={split_metrics['room_acc']:.3f} (test_size={split_metrics['test_size']})"
    )
    split_path = ROOT / "reports" / "benchmarks" / "triscope_split.json"
    split_path.write_text(json.dumps(split_metrics, indent=2))
    print(f"Saved to {split_path}")


if __name__ == "__main__":
    main()
