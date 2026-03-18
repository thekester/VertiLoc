#!/usr/bin/env python3
"""Project arbitrary predicted coordinates onto the physical board rectangle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.board_geometry import BoardGeometry, clamp_board_coordinates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-x-m", type=float, required=True, help="Predicted horizontal coordinate on the board axis.")
    parser.add_argument("--board-z-m", type=float, required=True, help="Predicted vertical coordinate / height on the board axis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry = BoardGeometry()
    x_clamped, z_clamped = clamp_board_coordinates(
        [args.board_x_m],
        [args.board_z_m],
        geometry=geometry,
    )
    result = {
        "input": {
            "board_x_m": float(args.board_x_m),
            "board_z_m": float(args.board_z_m),
        },
        "projected": {
            "board_x_m": float(x_clamped[0]),
            "board_z_m": float(z_clamped[0]),
        },
        "board_limits": {
            "x_min_m": 0.0,
            "x_max_m": geometry.board_width_m,
            "z_min_m": geometry.board_bottom_height_m,
            "z_max_m": geometry.board_top_height_m,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
