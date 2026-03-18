"""Orientation helpers for circular ESP32 pose labels."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

CANONICAL_ORIENTATION_DEGREES: dict[str, float] = {
    "back": 0.0,
    "back_right": 45.0,
    "right": 90.0,
    "front_right": 135.0,
    "front": 180.0,
    "front_left": 225.0,
    "left": 270.0,
    "back_left": 315.0,
}


def normalize_orientation_label(label: str) -> str:
    """Normalize free-form orientation labels to the canonical snake_case form."""
    cleaned = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    if cleaned not in CANONICAL_ORIENTATION_DEGREES:
        raise KeyError(f"Unknown orientation label: {label!r}")
    return cleaned


def orientation_label_to_degrees(label: str) -> float:
    """Return the canonical angle in degrees for an orientation label."""
    return float(CANONICAL_ORIENTATION_DEGREES[normalize_orientation_label(label)])


def angle_to_unit_circle(angle_deg: float) -> tuple[float, float]:
    """Encode an angle on the unit circle."""
    radians = math.radians(float(angle_deg))
    return math.cos(radians), math.sin(radians)


def unit_circle_to_angle(cos_value: float, sin_value: float) -> float:
    """Decode a raw unit-circle prediction to [0, 360) degrees."""
    angle = math.degrees(math.atan2(float(sin_value), float(cos_value)))
    return float(angle % 360.0)


def circular_distance_degrees(angle_a: float, angle_b: float) -> float:
    """Return the shortest signed-free angular distance in degrees."""
    delta = abs(float(angle_a) - float(angle_b)) % 360.0
    return float(min(delta, 360.0 - delta))


def nearest_orientation_label(angle_deg: float, labels: Iterable[str] | None = None) -> str:
    """Snap an angle to the closest canonical label."""
    available = [
        normalize_orientation_label(label)
        for label in (labels if labels is not None else CANONICAL_ORIENTATION_DEGREES.keys())
    ]
    best_label = min(
        available,
        key=lambda label: circular_distance_degrees(angle_deg, CANONICAL_ORIENTATION_DEGREES[label]),
    )
    return best_label


def angles_to_unit_circle(angles_deg: Iterable[float]) -> np.ndarray:
    """Vectorized conversion from degrees to [cos, sin]."""
    angles = np.asarray(list(angles_deg), dtype=float)
    radians = np.deg2rad(angles)
    return np.column_stack([np.cos(radians), np.sin(radians)])

