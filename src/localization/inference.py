"""Inference wrapper around the trained VertiLoc localizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from .board_geometry import BoardGeometry, add_board_geometry, add_board_zones
from .embedding_knn import EmbeddingKnnLocalizer

FEATURE_COLUMNS = ["Signal", "Noise", "signal_A1", "signal_A2", "signal_A3"]
DEFAULT_RUN_NAME = "vertiloc-beacon-v1"


def _parse_grid_cell(cell: str) -> tuple[int, int]:
    left, right = str(cell).split("_", 1)
    return int(left), int(right)


def _coordinate_mode_for_room(room: str) -> str:
    if room in {"B121", "D005", "E101"}:
        return "fit_to_data"
    return "absolute"


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Unable to read {path}: {exc}")
        return None


@dataclass
class VertiLocInferenceModel:
    """Serializable inference facade for RSSI -> location predictions."""

    localizer: EmbeddingKnnLocalizer
    run_name: str = DEFAULT_RUN_NAME
    latest_metrics: dict[str, object] | None = None

    @classmethod
    def load(
        cls,
        model_path: Path,
        *,
        metrics_path: Path | None = None,
        run_name: str | None = None,
    ) -> "VertiLocInferenceModel":
        try:
            localizer: EmbeddingKnnLocalizer = joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the serialized model. This usually means the joblib artifact "
                "was created with incompatible numpy/scikit-learn versions. "
                "Retrain it in the current environment with:\n"
                "PYTHONPATH=src python -m localization.pipeline "
                "--campaign data/D005/ddeuxmetres:2 --campaign data/D005/dquatremetres:4"
            ) from exc

        resolved_run_name = run_name or getattr(localizer, "run_name_", None) or DEFAULT_RUN_NAME
        latest_metrics = _load_json(metrics_path) if metrics_path is not None else None
        return cls(localizer=localizer, run_name=resolved_run_name, latest_metrics=latest_metrics)

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        *,
        top_k: int = 3,
        room: str = "auto",
        router_height_m: float = 0.75,
        include_neighbors: bool = False,
    ) -> pd.DataFrame:
        feature_df = df[FEATURE_COLUMNS].astype(float).copy()
        X = feature_df.to_numpy(dtype=float)
        y_pred = self.localizer.predict(X)
        y_proba = self.localizer.predict_proba(X)
        confidences = y_proba.max(axis=1)
        coordinate_mode = _coordinate_mode_for_room(room) if room != "auto" else "absolute"

        has_ood = (
            getattr(self.localizer, "ood_energy_threshold_", None) is not None
            and getattr(self.localizer, "ood_distance_threshold_", None) is not None
        )
        if has_ood:
            ood_scores = self.localizer.ood_scores(X)
            ood_unknown = self.localizer.is_ood(X, scores=ood_scores)
        else:
            ood_scores = {
                "ood_energy": np.full(len(df), np.nan, dtype=float),
                "ood_embedding_distance": np.full(len(df), np.nan, dtype=float),
            }
            ood_unknown = np.zeros(len(df), dtype=bool)

        neighbor_distances = None
        neighbor_labels = None
        if include_neighbors or len(df) == 1:
            neighbor_distances, neighbor_labels = self.localizer.explain(X, top_k=top_k)

        rows: list[dict[str, object]] = []
        for idx, pred_cell in enumerate(y_pred):
            router_distance_m, distance_confidence = self._infer_router_distance(X[idx : idx + 1])
            row: dict[str, object] = {
                "sample_id": int(idx),
                "run_name": self.run_name,
                **{col: float(feature_df.iloc[idx][col]) for col in FEATURE_COLUMNS},
                "pred_cell": str(pred_cell),
                "confidence": float(confidences[idx]),
                "ood_energy": float(ood_scores["ood_energy"][idx]),
                "ood_embedding_distance": float(ood_scores["ood_embedding_distance"][idx]),
                "ood_is_unknown": bool(ood_unknown[idx]),
                "pred_router_distance_m": router_distance_m,
                "distance_confidence": distance_confidence,
                "room_projection_mode": coordinate_mode,
            }
            if router_distance_m is not None:
                row.update(
                    self._project_board(
                        str(pred_cell),
                        router_distance_m=float(router_distance_m),
                        router_height_m=router_height_m,
                        coordinate_mode=coordinate_mode,
                    )
                )
            if include_neighbors and neighbor_distances is not None and neighbor_labels is not None:
                for rank in range(neighbor_labels.shape[1]):
                    row[f"neighbor_{rank+1}_cell"] = str(neighbor_labels[idx, rank])
                    row[f"neighbor_{rank+1}_embedding_distance"] = float(neighbor_distances[idx, rank])
            rows.append(row)
        return pd.DataFrame(rows)

    def _infer_router_distance(
        self,
        sample: np.ndarray,
    ) -> tuple[float | None, float | None]:
        dist_clf = getattr(self.localizer, "distance_clf_", None)
        if dist_clf is None:
            return None, None
        try:
            dist_proba = dist_clf.predict_proba(self.localizer.transform(sample))[0]
            dist_classes = getattr(self.localizer, "distance_classes_", dist_clf.classes_)
            best_idx = int(dist_proba.argmax())
            return float(dist_classes[best_idx]), float(dist_proba[best_idx])
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Could not infer router distance: {exc}")
            return None, None

    @staticmethod
    def _project_board(
        pred_cell: str,
        *,
        router_distance_m: float,
        router_height_m: float,
        coordinate_mode: str,
    ) -> dict[str, object]:
        grid_x, grid_y = _parse_grid_cell(pred_cell)
        geometry = BoardGeometry()
        df = pd.DataFrame(
            [{"grid_x": grid_x, "grid_y": grid_y, "router_distance_m": float(router_distance_m)}]
        )
        projected = add_board_geometry(
            df,
            geometry=geometry,
            router_height_m=float(router_height_m),
            grid_x_top_is_zero=True,
            coordinate_mode=coordinate_mode,
        )
        projected = add_board_zones(
            projected,
            geometry=geometry,
            n_cols=3,
            n_rows=3,
            use_clamped_coordinates=True,
            label_language="en",
        ).iloc[0]
        return {
            "board_x_m": float(projected["board_x_m"]),
            "board_z_m": float(projected["board_z_m"]),
            "board_x_clamped_m": float(projected["board_x_clamped_m"]),
            "board_z_clamped_m": float(projected["board_z_clamped_m"]),
            "zone_id": str(projected["zone_id"]),
            "zone_row": int(projected["zone_row"]),
            "zone_col": int(projected["zone_col"]),
            "within_board": bool(projected["is_within_board"]),
            "projection_distance_m": float(projected["board_projection_distance_m"]),
            "router_esp_3d_raw_m": float(projected["router_esp_3d_m"]),
            "router_esp_3d_clamped_m": float(projected["router_esp_3d_clamped_m"]),
        }
