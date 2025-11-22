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

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from localization.data import (
    CampaignSpec,
    load_measurements,
)
from localization.embedding_knn import EmbeddingKnnConfig, EmbeddingKnnLocalizer

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]


def main() -> None:
    campaigns = [
        CampaignSpec(Path("ddeuxmetres"), 2.0),
        CampaignSpec(Path("dquatremetres"), 4.0),
    ]
    df = load_measurements(campaigns)
    print(f"[CI notebook smoke] loaded {len(df)} samples across {df['grid_cell'].nunique()} cells.")

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["grid_cell"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=123,
        stratify=y,
    )

    config = EmbeddingKnnConfig(hidden_layer_sizes=(64, 32), max_iter=200, k_neighbors=5)
    localizer = EmbeddingKnnLocalizer(config=config).fit(X_train, y_train)
    y_pred = localizer.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[CI notebook smoke] accuracy on held-out set: {accuracy:.3f}")
    if accuracy < 0.8:
        raise AssertionError("Expected accuracy >= 0.8 for the smoke test.")

    # Grab a single sample to replicate the notebook explainability step.
    sample_features = X_test[:1]
    sample_pred = localizer.predict(sample_features)[0]
    sample_proba = localizer.predict_proba(sample_features).max()
    neighbor_dist, neighbor_cells = localizer.explain(sample_features, top_k=3)
    print(f"[CI notebook smoke] sample prediction: {sample_pred} (confidence={sample_proba:.3f})")
    print("[CI notebook smoke] closest neighbors:")
    for rank, (cell, dist) in enumerate(zip(neighbor_cells[0], neighbor_dist[0]), start=1):
        print(f"  #{rank}: cell={cell} | embedding_dist={dist:.4f}")


if __name__ == "__main__":
    main()
