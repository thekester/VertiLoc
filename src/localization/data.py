from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Physical grid definition (meters). The notebook uses the same constants so we keep
# everything in sync at module level.
DEFAULT_CELL_WIDTH_M = 0.25
DEFAULT_CELL_HEIGHT_M = 0.30

_FRENCH_NUMBER_MAP = {
    "un": 1.0,
    "deux": 2.0,
    "trois": 3.0,
    "quatre": 4.0,
    "cinq": 5.0,
}


@dataclass(frozen=True)
class CampaignSpec:
    """Definition of a measurement campaign folder."""

    path: Path
    router_distance_m: Optional[float] = None

    def resolved_distance(self) -> float:
        """Return the router distance, inferring it from the folder name if needed."""
        if self.router_distance_m is not None:
            return float(self.router_distance_m)

        inferred = infer_router_distance(self.path.name)
        if inferred is None:
            raise ValueError(
                f"Unable to infer router distance from folder `{self.path}`. "
                "Pass the value explicitly using the `folder:distance` syntax."
            )
        return inferred


def infer_router_distance(folder_name: str) -> Optional[float]:
    """Infer the router distance (meters) from a folder name."""
    lower = folder_name.lower()
    match = re.search(r"(\d+(?:[.,]\d+)?)", lower)
    if match:
        return float(match.group(1).replace(",", "."))

    for word, value in _FRENCH_NUMBER_MAP.items():
        if word in lower:
            return value
    return None


def _grid_to_physical(
    grid_x: int,
    grid_y: int,
    *,
    cell_width_m: float,
    cell_height_m: float,
) -> tuple[float, float]:
    """Return the physical coordinate (meters) of the cell center."""
    x_m = grid_y * cell_width_m + cell_width_m / 2.0
    y_m = grid_x * cell_height_m + cell_height_m / 2.0
    return x_m, y_m


def load_measurements(
    specs: Iterable[CampaignSpec],
    *,
    cell_width_m: float = DEFAULT_CELL_WIDTH_M,
    cell_height_m: float = DEFAULT_CELL_HEIGHT_M,
) -> pd.DataFrame:
    """Load all CSV measurements for the provided campaigns into a dataframe."""

    records = []
    for spec in specs:
        # Each campaign folder is tied to a router distance (either specified or inferred).
        distance = spec.resolved_distance()
        path = spec.path
        if not path.exists():
            raise FileNotFoundError(f"Measurement folder `{path}` not found.")

        for csv_path in sorted(path.glob("*.csv")):
            if csv_path.suffix.lower() != ".csv":
                continue

            # Filenames follow the pattern "<grid_x>_<grid_y>.csv".
            grid_part = csv_path.stem.split("_")
            if len(grid_part) != 2:
                raise ValueError(f"Unexpected filename format for {csv_path}")
            grid_x, grid_y = map(int, grid_part)

            df = pd.read_csv(csv_path)
            # Add spatial metadata so downstream consumers do not have to guess it again.
            df["grid_x"] = grid_x
            df["grid_y"] = grid_y
            df["grid_cell"] = f"{grid_x}_{grid_y}"
            df["router_distance_m"] = distance
            df["campaign"] = path.name
            coord_x_m, coord_y_m = _grid_to_physical(
                grid_x, grid_y, cell_width_m=cell_width_m, cell_height_m=cell_height_m
            )
            df["coord_x_m"] = coord_x_m
            df["coord_y_m"] = coord_y_m
            records.append(df)

    if not records:
        raise RuntimeError("No CSV files found for the provided campaigns.")

    return pd.concat(records, ignore_index=True)


def default_campaign_specs() -> list[CampaignSpec]:
    """Generate CampaignSpec entries for common folder names."""
    candidate_dirs = [
        Path("ddeuxmetres"),
        Path("dquatremetres"),
    ]
    specs: list[CampaignSpec] = []
    for directory in candidate_dirs:
        if directory.exists():
            specs.append(CampaignSpec(directory))
    if not specs:
        raise FileNotFoundError(
            "No measurement folders found. "
            "Pass the desired folders explicitly using the CLI."
        )
    return specs
