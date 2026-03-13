"""Neural-network encoder + local KNN decoder used for RSSI localization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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
    max_iter: int = 1500
    learning_rate_init: float = 1e-3
    alpha: float = 1e-4
    k_neighbors: int = 5
    weights: str = "distance"
    random_state: int = 42
    # Early stopping is left disabled by default because sklearn's MLP uses
    # string labels and triggers dtype issues when scoring the validation set.
    early_stopping: bool = False
    n_iter_no_change: int = 25
    tol: float = 1e-4
    validation_fraction: float = 0.1


@dataclass
class EmbeddingKnnLocalizer:
    """Neural network encoder + k-NN decoder for RSSI-based localization."""

    config: EmbeddingKnnConfig = field(default_factory=EmbeddingKnnConfig)
    scaler_: StandardScaler | None = field(init=False, default=None)
    encoder_: MLPClassifier | None = field(init=False, default=None)
    knn_: KNeighborsClassifier | None = field(init=False, default=None)
    train_labels_: np.ndarray | None = field(init=False, default=None)
    train_embeddings_: np.ndarray | None = field(init=False, default=None)
    train_scaled_: np.ndarray | None = field(init=False, default=None)
    distance_clf_: LogisticRegression | None = field(init=False, default=None)
    distance_classes_: np.ndarray | None = field(init=False, default=None)
    ood_energy_threshold_: float | None = field(init=False, default=None)
    ood_distance_threshold_: float | None = field(init=False, default=None)
    ood_energy_percentile_: float | None = field(init=False, default=None)
    ood_distance_percentile_: float | None = field(init=False, default=None)
    ood_temperature_: float = field(init=False, default=1.0)

    def fit(self, X, y):
        """Train scaler, MLP encoder, and k-NN classifier on RSSI data."""
        X_arr = self._ensure_2d_array(X)
        # RSSI scales differ per campaign, therefore we always re-fit a scaler so
        # the MLP operates on standardized inputs irrespective of router distance.
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)
        self.train_scaled_ = X_scaled

        cfg = self.config
        self.encoder_ = MLPClassifier(
            hidden_layer_sizes=cfg.hidden_layer_sizes,
            activation=cfg.activation,
            max_iter=cfg.max_iter,
            learning_rate_init=cfg.learning_rate_init,
            alpha=cfg.alpha,
            solver="adam",
            random_state=cfg.random_state,
            early_stopping=cfg.early_stopping,
            n_iter_no_change=cfg.n_iter_no_change,
            tol=cfg.tol,
            validation_fraction=cfg.validation_fraction,
        )
        self.encoder_.fit(X_scaled, y)

        # Compute embeddings for the training set once and store them: they are
        # reused by KNN and the explain() helper to expose nearest neighbors.
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
        """Predict the grid_cell label for each RSSI vector in X."""
        # Prediction = encode + run through the k-NN decoder.
        embeddings = self.transform(X)
        return self.knn_.predict(embeddings)

    def predict_proba(self, X):
        """Return class probabilities computed by the k-NN classifier."""
        embeddings = self.transform(X)
        return self.knn_.predict_proba(embeddings)

    def kneighbors(self, X, n_neighbors: int | None = None):
        """Expose k-NN distances/indices for downstream explainability tools."""
        embeddings = self.transform(X)
        return self.knn_.kneighbors(embeddings, n_neighbors=n_neighbors)

    def transform(self, X):
        """Encode raw RSSI vectors into the learned embedding space."""
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

    def _compute_logits(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.encoder_ is None:
            raise RuntimeError("Encoder is not trained yet.")
        activation = X_scaled
        for coef, intercept in zip(self.encoder_.coefs_[:-1], self.encoder_.intercepts_[:-1]):
            activation = activation @ coef + intercept
            activation = _apply_activation(activation, self.encoder_.activation)
        return activation @ self.encoder_.coefs_[-1] + self.encoder_.intercepts_[-1]

    @staticmethod
    def _logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
        vmax = np.max(values, axis=axis, keepdims=True)
        stable = values - vmax
        return np.log(np.sum(np.exp(stable), axis=axis)) + np.squeeze(vmax, axis=axis)

    def energy_score(self, X, temperature: float = 1.0) -> np.ndarray:
        """Return softmax energy score (higher => more likely OOD)."""
        X_arr = self._ensure_2d_array(X)
        if self.scaler_ is None:
            raise RuntimeError("The model must be fitted before calling energy_score.")
        X_scaled = self.scaler_.transform(X_arr)
        return self._energy_from_scaled(X_scaled, temperature=temperature)

    def _energy_from_scaled(self, X_scaled: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        logits = self._compute_logits(X_scaled)
        t = max(float(temperature), 1e-6)
        return -t * self._logsumexp(logits / t, axis=1)

    def embedding_distance_score(self, X) -> np.ndarray:
        """Distance to the nearest training embedding."""
        if self.knn_ is None:
            raise RuntimeError("The model must be fitted before calling embedding_distance_score.")
        embeddings = self.transform(X)
        distances, _ = self.knn_.kneighbors(embeddings, n_neighbors=1)
        return distances[:, 0]

    def calibrate_ood(
        self,
        *,
        energy_percentile: float = 95.0,
        distance_percentile: float = 95.0,
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """Calibrate OOD thresholds from the training distribution."""
        if self.train_scaled_ is None or self.train_embeddings_ is None or self.knn_ is None:
            raise RuntimeError("Fit the model before calibrating OOD thresholds.")

        energy_train = self._energy_from_scaled(self.train_scaled_, temperature=temperature)
        # Use the second neighbor to avoid the trivial self-distance = 0 on train points.
        n_neighbors = 2 if len(self.train_embeddings_) > 1 else 1
        dist_train_all, _ = self.knn_.kneighbors(self.train_embeddings_, n_neighbors=n_neighbors)
        if n_neighbors == 2:
            dist_train = dist_train_all[:, 1]
        else:
            dist_train = dist_train_all[:, 0]

        self.ood_energy_threshold_ = float(np.percentile(energy_train, energy_percentile))
        self.ood_distance_threshold_ = float(np.percentile(dist_train, distance_percentile))
        self.ood_energy_percentile_ = float(energy_percentile)
        self.ood_distance_percentile_ = float(distance_percentile)
        self.ood_temperature_ = float(temperature)
        return {
            "ood_energy_threshold": self.ood_energy_threshold_,
            "ood_distance_threshold": self.ood_distance_threshold_,
            "ood_energy_percentile": self.ood_energy_percentile_,
            "ood_distance_percentile": self.ood_distance_percentile_,
            "ood_temperature": self.ood_temperature_,
        }

    def ood_scores(self, X) -> dict[str, np.ndarray]:
        """Return both OOD signals for each sample."""
        return {
            "ood_energy": self.energy_score(X, temperature=self.ood_temperature_),
            "ood_embedding_distance": self.embedding_distance_score(X),
        }

    def is_ood(self, X, *, scores: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Return a boolean mask: True means sample considered unknown/OOD."""
        if self.ood_energy_threshold_ is None or self.ood_distance_threshold_ is None:
            raise RuntimeError("OOD thresholds are missing. Call calibrate_ood() first.")
        score_dict = self.ood_scores(X) if scores is None else scores
        return (score_dict["ood_energy"] > self.ood_energy_threshold_) | (
            score_dict["ood_embedding_distance"] > self.ood_distance_threshold_
        )

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
        # Pandas DataFrame / list-like inputs end up here.
        return np.asarray(X, dtype=float)
