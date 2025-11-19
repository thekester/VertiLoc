from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(x, 0)
    if activation == "tanh":
        return np.tanh(x)
    if activation == "logistic":
        return 1.0 / (1.0 + np.exp(-x))
    if activation == "identity":
        return x
    raise ValueError(f"Unsupported activation `{activation}`.")


@dataclass
class EmbeddingKnnConfig:
    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    max_iter: int = 800
    learning_rate_init: float = 1e-3
    alpha: float = 1e-4
    k_neighbors: int = 5
    weights: str = "distance"
    random_state: int = 42


@dataclass
class EmbeddingKnnLocalizer:
    """Neural network encoder + k-NN decoder for RSSI-based localization."""

    config: EmbeddingKnnConfig = field(default_factory=EmbeddingKnnConfig)
    scaler_: StandardScaler | None = field(init=False, default=None)
    encoder_: MLPClassifier | None = field(init=False, default=None)
    knn_: KNeighborsClassifier | None = field(init=False, default=None)
    train_labels_: np.ndarray | None = field(init=False, default=None)
    train_embeddings_: np.ndarray | None = field(init=False, default=None)

    def fit(self, X, y):
        X_arr = self._ensure_2d_array(X)
        # RSSI scales differ per campaign, therefore we always re-fit a scaler.
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        cfg = self.config
        self.encoder_ = MLPClassifier(
            hidden_layer_sizes=cfg.hidden_layer_sizes,
            activation=cfg.activation,
            max_iter=cfg.max_iter,
            learning_rate_init=cfg.learning_rate_init,
            alpha=cfg.alpha,
            solver="adam",
            random_state=cfg.random_state,
        )
        self.encoder_.fit(X_scaled, y)

        train_embeddings = self._compute_embeddings(X_scaled)
        self.train_embeddings_ = train_embeddings
        self.train_labels_ = np.asarray(y)
        self.knn_ = KNeighborsClassifier(
            n_neighbors=cfg.k_neighbors,
            weights=cfg.weights,
            metric="euclidean",
        )
        self.knn_.fit(train_embeddings, y)
        return self

    def predict(self, X):
        # Prediction = encode + run through the k-NN decoder.
        embeddings = self.transform(X)
        return self.knn_.predict(embeddings)

    def predict_proba(self, X):
        embeddings = self.transform(X)
        return self.knn_.predict_proba(embeddings)

    def kneighbors(self, X, n_neighbors: int | None = None):
        embeddings = self.transform(X)
        return self.knn_.kneighbors(embeddings, n_neighbors=n_neighbors)

    def transform(self, X):
        X_arr = self._ensure_2d_array(X)
        if self.scaler_ is None or self.encoder_ is None:
            raise RuntimeError("The model must be fitted before calling transform.")
        X_scaled = self.scaler_.transform(X_arr)
        return self._compute_embeddings(X_scaled)

    def _compute_embeddings(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.encoder_ is None:
            raise RuntimeError("Encoder is not trained yet.")

        activation = X_scaled
        # Skip the output layer and only keep hidden activations as the embedding.
        for coef, intercept in zip(self.encoder_.coefs_[:-1], self.encoder_.intercepts_[:-1]):
            activation = activation @ coef + intercept
            activation = _apply_activation(activation, self.encoder_.activation)
        return activation

    def explain(self, X, top_k: int = 3):
        """Return distances and labels for the closest training embeddings."""
        if self.knn_ is None or self.train_labels_ is None:
            raise RuntimeError("Model must be fitted before explanations can be generated.")
        # Reuse the fitted k-NN to fetch both distances and labels of the top_k neighbors.
        distances, indices = self.kneighbors(X, n_neighbors=top_k)
        neighbor_labels = self.train_labels_[indices]
        return distances, neighbor_labels

    @staticmethod
    def _ensure_2d_array(X) -> np.ndarray:
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError("Feature matrix must be 2-D.")
            return X
        return np.asarray(X, dtype=float)
