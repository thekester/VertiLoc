"""
Notebook-inspired smoke test.

This script mirrors the steps demonstrated in `notebooks/explain_pipeline.ipynb`:
1. Load and merge all measurement campaigns.
2. Train the NN encoder + L-KNN model on a stratified split.
3. Evaluate accuracy to ensure the pipeline still works.
4. Run `explain()` on a sample to confirm neighbor retrieval.

The goal is not to achieve the best accuracy but to guarantee that the code path used
in the notebook remains functional during CI runs.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.data import (
    CampaignSpec,
    load_measurements,
)
from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]


def main() -> None:
    campaigns = [
        CampaignSpec(Path("data/D005/ddeuxmetres"), 2.0),
        CampaignSpec(Path("data/D005/dquatremetres"), 4.0),
        CampaignSpec(Path("data/E101/dtroismetres"), 3.0),
        CampaignSpec(Path("data/E101/dcinqmetres"), 5.0),
    ]
    df = load_measurements(campaigns)
    print(f"[CI notebook smoke] loaded {len(df)} samples across {df['grid_cell'].nunique()} cells.")

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["grid_cell"].to_numpy()
    distances = df["router_distance_m"].to_numpy()

    X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
        X,
        y,
        distances,
        test_size=0.25,
        random_state=123,
        stratify=y,
    )

    # Use the default max_iter from the library to avoid convergence warnings in CI.
    config = EmbeddingKnnConfig(hidden_layer_sizes=(64, 32), k_neighbors=5)
    localizer = EmbeddingKnnLocalizer(config=config).fit(X_train, y_train)
    y_pred = localizer.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[CI notebook smoke] accuracy on held-out set: {accuracy:.3f}")
    if accuracy < 0.8:
        raise AssertionError("Expected accuracy >= 0.8 for the smoke test.")

    # Optional head: predict router distance from embeddings (no distance used as input).
    train_emb = localizer.train_embeddings_
    test_emb = localizer.transform(X_test)
    dist_clf = LogisticRegression(max_iter=2000)
    dist_clf.fit(train_emb, d_train)
    dist_pred = dist_clf.predict(test_emb)
    dist_acc = accuracy_score(d_test, dist_pred)
    # Baseline without logistic head: guess distance from the modal distance of the predicted cell.
    mode_map = (
        pd.DataFrame({"cell": y_train, "distance": d_train})
        .groupby("cell")["distance"]
        .agg(lambda s: s.mode().iloc[0])
    )
    dist_pred_baseline = np.array([mode_map.get(cell, mode_map.mode().iloc[0]) for cell in y_pred])
    dist_acc_baseline = accuracy_score(d_test, dist_pred_baseline)
    print(f"[CI notebook smoke] router-distance accuracy (LogReg on embeddings): {dist_acc:.3f}")
    print(f"[CI notebook smoke] router-distance accuracy (cell-mode baseline):   {dist_acc_baseline:.3f}")

    # Grab a single sample to replicate the notebook explainability step.
    sample_features = X_test[:1]
    sample_pred = localizer.predict(sample_features)[0]
    sample_proba = localizer.predict_proba(sample_features).max()
    neighbor_dist, neighbor_cells = localizer.explain(sample_features, top_k=3)
    print(f"[CI notebook smoke] sample prediction: {sample_pred} (confidence={sample_proba:.3f})")
    print("[CI notebook smoke] closest neighbors (distance = latent embedding distance, not physical):")
    for rank, (cell, dist) in enumerate(zip(neighbor_cells[0], neighbor_dist[0]), start=1):
        print(f"  #{rank}: cell={cell} | embedding_distance={dist:.4f}")


if __name__ == "__main__":
    main()
