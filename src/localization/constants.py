"""Shared constants for VertiLoc localization code."""

from __future__ import annotations

DEFAULT_RUN_NAME = "vertiloc-beacon-v1"

RSSI_FEATURE_COLUMNS: tuple[str, ...] = (
    "Signal",
    "Noise",
    "signal_A1",
    "signal_A2",
    "signal_A3",
)
