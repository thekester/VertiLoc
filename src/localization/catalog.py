"""Dataset catalog shared by benchmark and inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .constants import RSSI_FEATURE_COLUMNS
from .data import CampaignSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"
BENCHMARK_REPORT_DIR = REPORT_DIR / "benchmarks"

# Keep script compatibility: several experiments expect a mutable list-like
# FEATURE_COLUMNS symbol, while the source of truth remains the tuple above.
FEATURE_COLUMNS = list(RSSI_FEATURE_COLUMNS)

_E102_ROOT = PROJECT_ROOT / "data" / "E102"

# Campaign folders grouped by room. E102 folders need explicit distances: names
# like "exp1" would otherwise be inferred as 1 m instead of the actual 4 m
# acquisition distance.
ROOM_CAMPAIGNS: dict[str, list[CampaignSpec]] = {
    "B121": [
        CampaignSpec(
            PROJECT_ROOT / "data" / "B121" / "dtroismetres" / "routeurcentretableau",
            router_distance_m=3.0,
        ),
        CampaignSpec(
            PROJECT_ROOT / "data" / "B121" / "dtroismetres" / "routeurgauche",
            router_distance_m=3.0,
        ),
        CampaignSpec(
            PROJECT_ROOT / "data" / "B121" / "dtroismetres" / "routeuradroitedutableau",
            router_distance_m=3.0,
        ),
    ],
    "D005": [
        CampaignSpec(PROJECT_ROOT / "data" / "D005" / "ddeuxmetres"),
        CampaignSpec(PROJECT_ROOT / "data" / "D005" / "dquatremetres"),
    ],
    "E101": [
        CampaignSpec(PROJECT_ROOT / "data" / "E101" / "dtroismetres"),
        CampaignSpec(PROJECT_ROOT / "data" / "E101" / "dcinqmetres"),
        CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "back", router_distance_m=3.0),
        CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "front", router_distance_m=3.0),
        CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "left", router_distance_m=3.0),
        CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "right", router_distance_m=3.0),
    ],
    "E102": [
        CampaignSpec(_E102_ROOT / "exp1", router_distance_m=4.0),
        CampaignSpec(_E102_ROOT / "exp2", router_distance_m=4.0),
        CampaignSpec(_E102_ROOT / "exp3", router_distance_m=4.0),
        CampaignSpec(_E102_ROOT / "exp4", router_distance_m=4.0),
        CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0),
        CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0),
    ],
}

OrientationCampaign = tuple[str, CampaignSpec, str]

ORIENTATION_CAMPAIGNS: dict[str, list[OrientationCampaign]] = {
    "E101": [
        ("E101/back", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "back", router_distance_m=3.0), "back"),
        ("E101/right", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "right", router_distance_m=3.0), "right"),
        ("E101/front", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "front", router_distance_m=3.0), "front"),
        ("E101/left", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "left", router_distance_m=3.0), "left"),
    ],
    "E102": [
        ("E102/exp1_back_right", CampaignSpec(_E102_ROOT / "exp1", router_distance_m=4.0), "back_right"),
        ("E102/exp2_front_right", CampaignSpec(_E102_ROOT / "exp2", router_distance_m=4.0), "front_right"),
        ("E102/exp3_front_left", CampaignSpec(_E102_ROOT / "exp3", router_distance_m=4.0), "front_left"),
        ("E102/exp4_back_left", CampaignSpec(_E102_ROOT / "exp4", router_distance_m=4.0), "back_left"),
        ("E102/exp5_ground", CampaignSpec(_E102_ROOT / "elevation" / "exp5", router_distance_m=4.0), "front"),
        ("E102/exp6_1m50", CampaignSpec(_E102_ROOT / "elevation" / "exp6", router_distance_m=4.0), "front"),
    ],
}


def normalize_room_name(room: str) -> str:
    """Return the canonical uppercase room identifier."""
    return str(room).strip().upper()


def room_campaigns() -> dict[str, list[CampaignSpec]]:
    """Return a shallow copy of the known room campaign catalog."""
    return {room: list(specs) for room, specs in ROOM_CAMPAIGNS.items()}


def distance_matches(distance: float, targets: Iterable[float]) -> bool:
    """Return whether a distance matches any target with a tiny float tolerance."""
    return any(abs(float(distance) - float(target)) < 1e-6 for target in targets)


def filter_room_campaigns(
    room_filter: Iterable[str] | None = None,
    distance_filter: Iterable[float] | None = None,
) -> dict[str, list[CampaignSpec]]:
    """Filter the catalog by room names and/or acquisition distances."""
    normalized_rooms = None
    if room_filter:
        normalized_rooms = {
            normalize_room_name(room)
            for room in room_filter
            if str(room).strip()
        }
        if not normalized_rooms:
            normalized_rooms = None

    distance_set = set(distance_filter) if distance_filter else None
    filtered: dict[str, list[CampaignSpec]] = {}
    for room, specs in ROOM_CAMPAIGNS.items():
        if normalized_rooms and normalize_room_name(room) not in normalized_rooms:
            continue
        selected: list[CampaignSpec] = []
        for spec in specs:
            if distance_set:
                try:
                    distance = spec.resolved_distance()
                except ValueError:
                    continue
                if not distance_matches(distance, distance_set):
                    continue
            selected.append(spec)
        if selected:
            filtered[room] = selected
    return filtered
