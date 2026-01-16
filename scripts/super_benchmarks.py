#!/usr/bin/env python3
"""Interactive helper to run benchmark_models with room/distance selection."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_models as bm
from localization.data import infer_router_distance


def _split_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[,\s]+", text.strip()) if token]


def _format_distances(distances: list[float]) -> str:
    return ", ".join(f"{dist:g}" for dist in distances)


def _collect_available() -> dict[str, list[float]]:
    rooms: dict[str, list[float]] = {}
    for room, folders in bm.ROOM_CAMPAIGNS.items():
        distances = set()
        for folder in folders:
            if not folder.exists():
                continue
            inferred = infer_router_distance(folder.name)
            if inferred is not None:
                distances.add(float(inferred))
        if distances:
            rooms[room] = sorted(distances)
    return rooms


def _resolve_rooms(
    raw_rooms: list[str] | None,
    available_rooms: list[str],
    *,
    interactive: bool,
) -> tuple[list[str], bool]:
    if not available_rooms:
        raise ValueError("No rooms available to select.")
    if raw_rooms:
        tokens = []
        for entry in raw_rooms:
            tokens.extend(_split_tokens(entry))
        if not tokens:
            tokens = available_rooms
        return _match_rooms(tokens, available_rooms), False
    if not interactive:
        raise ValueError("Missing --room and interactive mode disabled.")
    return _prompt_rooms(available_rooms)


def _match_rooms(tokens: list[str], available_rooms: list[str]) -> list[str]:
    lookup = {room.lower(): room for room in available_rooms}
    resolved: list[str] = []
    missing: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        if key in ("all", "a", "*"):
            return available_rooms
        if key in lookup:
            resolved.append(lookup[key])
        else:
            missing.append(token)
    if missing:
        raise ValueError(f"Unknown rooms: {', '.join(missing)}")
    return sorted(set(resolved), key=available_rooms.index)


def _prompt_rooms(available_rooms: list[str]) -> tuple[list[str], bool]:
    print("Available rooms:")
    for room in available_rooms:
        print(f"- {room}")
    while True:
        raw = input("Rooms to include [all]: ").strip()
        if not raw or raw.lower() in ("all", "a", "*"):
            return available_rooms, True
        tokens = _split_tokens(raw)
        try:
            return _match_rooms(tokens, available_rooms), False
        except ValueError as exc:
            print(f"{exc}. Try again.")


def _resolve_distances(
    raw_distances: list[str] | None,
    available_distances: list[float],
    *,
    interactive: bool,
) -> tuple[list[float] | None, bool]:
    if not available_distances:
        raise ValueError("No distances available to select.")
    if raw_distances:
        tokens = []
        for entry in raw_distances:
            tokens.extend(_split_tokens(entry))
        if any(token.lower() in ("all", "a", "*") for token in tokens):
            return None, True
        values = _parse_distance_tokens(tokens)
        matched = _match_distances(values, available_distances)
        return matched, False
    if not interactive:
        raise ValueError("Missing --distances and interactive mode disabled.")
    return _prompt_distances(available_distances)


def _parse_distance_tokens(tokens: list[str]) -> list[float]:
    values: list[float] = []
    for token in tokens:
        for part in _split_tokens(token):
            values.append(float(part.replace(",", ".")))
    return values


def _match_distances(values: list[float], available: list[float]) -> list[float] | None:
    if not values:
        return None
    matched: list[float] = []
    for value in values:
        found = None
        for candidate in available:
            if abs(candidate - value) < 1e-6:
                found = candidate
                break
        if found is None:
            raise ValueError(f"Unknown distance: {value:g}")
        matched.append(found)
    return sorted(set(matched))


def _prompt_distances(available_distances: list[float]) -> tuple[list[float] | None, bool]:
    print(f"Available distances (m): {_format_distances(available_distances)}")
    while True:
        raw = input("Distances to include [all]: ").strip()
        if not raw or raw.lower() in ("all", "a", "*"):
            return None, True
        try:
            values = _parse_distance_tokens([raw])
            matched = _match_distances(values, available_distances)
            return matched, False
        except ValueError as exc:
            print(f"{exc}. Try again.")


def _resolve_holdout(
    raw_holdout: str | None,
    *,
    rooms_count: int,
    distances_count: int,
    interactive: bool,
) -> str:
    if raw_holdout:
        return raw_holdout
    if not interactive:
        return "auto"
    default = "auto"
    if rooms_count > 1:
        default = "room"
    elif distances_count > 1:
        default = "distance"
    print("Holdout options: auto, room, distance, none")
    raw = input(f"Holdout strategy [{default}]: ").strip().lower()
    if not raw:
        return default
    if raw in ("auto", "room", "distance", "none"):
        return raw
    print("Unknown holdout option, using auto.")
    return "auto"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive benchmark runner.")
    parser.add_argument(
        "--room",
        action="append",
        help="Room filter (repeatable). Example: --room D005 --room E101",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        help="Distance filter in meters. Example: --distances 2 4",
    )
    parser.add_argument(
        "--holdout",
        choices=["auto", "room", "distance", "none"],
        help="Holdout strategy to avoid near-duplicate leakage.",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional/unstable models (Mahalanobis, GPC, CatBoost, LightGBM, XGBoost).",
    )
    parser.add_argument(
        "--run-gpc",
        action="store_true",
        help="Run the Gaussian Process Classifier stages (very slow on large datasets).",
    )
    parser.add_argument(
        "--gpc-max-samples",
        type=int,
        help="Optional cap on training samples for GPC to keep runtime reasonable.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all optional models, including GPC.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail if required inputs are missing instead of prompting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rooms_map = _collect_available()
    if not rooms_map:
        raise RuntimeError("No campaign folders found under data/.")
    available_rooms = sorted(rooms_map.keys())

    rooms, rooms_all = _resolve_rooms(
        args.room,
        available_rooms,
        interactive=not args.non_interactive,
    )
    available_distances = sorted({d for room in rooms for d in rooms_map[room]})
    distances, _distances_all = _resolve_distances(
        args.distances,
        available_distances,
        interactive=not args.non_interactive,
    )
    holdout = _resolve_holdout(
        args.holdout,
        rooms_count=len(rooms),
        distances_count=len(available_distances if distances is None else distances),
        interactive=not args.non_interactive,
    )
    if args.all_models:
        skip_optional = False
        run_gpc = True
    elif args.skip_optional:
        skip_optional = True
        run_gpc = False
    elif args.non_interactive:
        skip_optional = False
        run_gpc = args.run_gpc
    else:
        raw = input("Mode all (run all optional models including GPC)? [no]: ").strip().lower()
        if raw in ("y", "yes", "o", "oui"):
            skip_optional = False
            run_gpc = True
        else:
            raw = input("Include optional models (Mahalanobis, GPC, CatBoost, LightGBM, XGBoost)? [no]: ").strip().lower()
            skip_optional = raw not in ("y", "yes", "o", "oui")
            if skip_optional:
                run_gpc = False
            elif args.run_gpc:
                run_gpc = True
            else:
                raw = input("Include GPC (very slow on large datasets)? [no]: ").strip().lower()
                run_gpc = raw in ("y", "yes", "o", "oui")
    gpc_max_samples = args.gpc_max_samples
    if run_gpc and not args.non_interactive:
        raw = input("GPC max samples [1400]: ").strip()
        if raw:
            try:
                gpc_max_samples = int(raw)
            except ValueError:
                print("Invalid integer, using default (1400).")
                gpc_max_samples = 1400
        elif gpc_max_samples is None:
            gpc_max_samples = 1400
    if run_gpc and gpc_max_samples is None:
        gpc_max_samples = 1400

    print("\nSelections:")
    print(f"- rooms: {', '.join(rooms)}")
    if distances is None:
        print(f"- distances: all ({_format_distances(available_distances)})")
    else:
        print(f"- distances: {_format_distances(distances)}")
    print(f"- holdout: {holdout}")
    print(f"- optional models: {'skipped' if skip_optional else 'included'}")
    print(f"- GPC: {'enabled' if run_gpc else 'disabled'}")
    if run_gpc and gpc_max_samples:
        print(f"- GPC max samples: {gpc_max_samples}")

    cli_args: list[str] = []
    if not rooms_all:
        for room in rooms:
            cli_args += ["--room", room]
    if distances is not None:
        cli_args += ["--distances"] + [f"{dist:g}" for dist in distances]
    if holdout:
        cli_args += ["--holdout", holdout]
    if skip_optional:
        cli_args += ["--skip-optional"]
    if run_gpc:
        cli_args += ["--run-gpc"]
    if gpc_max_samples:
        cli_args += ["--gpc-max-samples", str(gpc_max_samples)]
    if args.all_models:
        cli_args += ["--all-models"]

    print("\nLaunching benchmarks...")
    bm.main(cli_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
