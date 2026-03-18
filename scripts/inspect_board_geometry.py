#!/usr/bin/env python3
"""Generate board-relative physical coordinates for the measurement campaigns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from localization.board_geometry import BoardGeometry, add_board_geometry  # noqa: E402
from localization.data import CampaignSpec, load_measurements  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "benchmarks"

ROOM_CAMPAIGNS: dict[str, list[tuple[str, CampaignSpec]]] = {
    "B121": [
        (
            "B121/routeurcentretableau",
            CampaignSpec(PROJECT_ROOT / "data" / "B121" / "dtroismetres" / "routeurcentretableau", router_distance_m=3.0),
        ),
        (
            "B121/routeurgauche",
            CampaignSpec(PROJECT_ROOT / "data" / "B121" / "dtroismetres" / "routeurgauche", router_distance_m=3.0),
        ),
        (
            "B121/routeuradroitedutableau",
            CampaignSpec(PROJECT_ROOT / "data" / "B121" / "dtroismetres" / "routeuradroitedutableau", router_distance_m=3.0),
        ),
    ],
    "D005": [
        ("D005/ddeuxmetres", CampaignSpec(PROJECT_ROOT / "data" / "D005" / "ddeuxmetres", router_distance_m=2.0)),
        ("D005/dquatremetres", CampaignSpec(PROJECT_ROOT / "data" / "D005" / "dquatremetres", router_distance_m=4.0)),
    ],
    "E101": [
        ("E101/dtroismetres", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "dtroismetres", router_distance_m=3.0)),
        ("E101/dcinqmetres", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "dcinqmetres", router_distance_m=5.0)),
        ("E101/back", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "back", router_distance_m=3.0)),
        ("E101/front", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "front", router_distance_m=3.0)),
        ("E101/left", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "left", router_distance_m=3.0)),
        ("E101/right", CampaignSpec(PROJECT_ROOT / "data" / "E101" / "circulaire" / "right", router_distance_m=3.0)),
    ],
    "E102": [
        ("E102/exp1", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp1", router_distance_m=4.0)),
        ("E102/exp2", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp2", router_distance_m=4.0)),
        ("E102/exp3", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp3", router_distance_m=4.0)),
        ("E102/exp4", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "exp4", router_distance_m=4.0)),
        ("E102/exp5_ground", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "elevation" / "exp5", router_distance_m=4.0)),
        ("E102/exp6_1m50", CampaignSpec(PROJECT_ROOT / "data" / "E102" / "elevation" / "exp6", router_distance_m=4.0)),
    ],
}

ROOM_ROUTER_HEIGHTS = {
    "B121": 0.75,
    "D005": 0.75,
    "E101": 0.75,
    "E102": 0.75,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms", default="B121,D005,E101,E102", help="Comma-separated rooms among B121,D005,E101,E102.")
    parser.add_argument(
        "--output-prefix",
        default="board_geometry_projection",
        help="Prefix used for output artifacts in reports/benchmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rooms = [token.strip().upper() for token in str(args.rooms).split(",") if token.strip()]
    invalid = sorted(set(rooms) - set(ROOM_CAMPAIGNS))
    if invalid:
        raise ValueError(f"Unknown rooms: {invalid}")

    geometry = BoardGeometry()
    frames: list[pd.DataFrame] = []
    for room in rooms:
        router_height_m = ROOM_ROUTER_HEIGHTS[room]
        for campaign_name, spec in ROOM_CAMPAIGNS[room]:
            df = load_measurements([spec])
            df["room"] = room
            df["campaign_full"] = campaign_name
            df = add_board_geometry(
                df,
                geometry=geometry,
                router_height_m=router_height_m,
                grid_x_top_is_zero=True,
            )
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    cell_df = (
        combined[
            [
                "room",
                "campaign_full",
                "grid_cell",
                "grid_x",
                "grid_y",
                "router_distance_m",
                "board_x_m",
                "board_z_m",
                "board_x_clamped_m",
                "board_z_clamped_m",
                "is_within_board_width",
                "is_within_board_height",
                "is_within_board",
                "board_projection_dx_m",
                "board_projection_dz_m",
                "board_projection_distance_m",
                "esp_height_m",
                "router_height_m",
                "router_vertical_delta_m",
                "board_center_offset_x_m",
                "router_esp_3d_m",
                "router_vertical_delta_clamped_m",
                "router_esp_3d_clamped_m",
            ]
        ]
        .drop_duplicates(["room", "campaign_full", "grid_cell"])
        .sort_values(["room", "campaign_full", "grid_x", "grid_y"])
        .reset_index(drop=True)
    )

    summary = {
        "rooms": rooms,
        "geometry": {
            "board_width_m": geometry.board_width_m,
            "board_bottom_height_m": geometry.board_bottom_height_m,
            "board_top_height_m": geometry.board_top_height_m,
            "board_height_m": geometry.board_height_m,
            "cell_width_m": geometry.cell_width_m,
            "cell_height_m": geometry.cell_height_m,
            "table_height_m": geometry.table_height_m,
        },
        "router_heights_m": {room: ROOM_ROUTER_HEIGHTS[room] for room in rooms},
        "board_z_range_m": {
            "min": float(cell_df["board_z_m"].min()),
            "max": float(cell_df["board_z_m"].max()),
        },
        "within_board_ratio": float(cell_df["is_within_board"].mean()),
        "projection_distance_m": {
            "mean": float(cell_df["board_projection_distance_m"].mean()),
            "max": float(cell_df["board_projection_distance_m"].max()),
        },
        "router_esp_3d_range_m": {
            "min": float(cell_df["router_esp_3d_m"].min()),
            "max": float(cell_df["router_esp_3d_m"].max()),
        },
        "router_esp_3d_clamped_range_m": {
            "min": float(cell_df["router_esp_3d_clamped_m"].min()),
            "max": float(cell_df["router_esp_3d_clamped_m"].max()),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix).strip() or "board_geometry_projection"
    csv_path = REPORT_DIR / f"{prefix}.csv"
    json_path = REPORT_DIR / f"{prefix}.json"
    cell_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))

    print(cell_df.head(12).to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
