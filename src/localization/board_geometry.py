"""Geometry helpers for projecting grid cells onto the physical board."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BoardGeometry:
    """Physical dimensions of the board and router support."""

    board_width_m: float = 3.95
    board_bottom_height_m: float = 1.18
    board_top_height_m: float = 2.20
    cell_width_m: float = 0.25
    cell_height_m: float = 0.30
    table_height_m: float = 0.75

    @property
    def board_height_m(self) -> float:
        return float(self.board_top_height_m - self.board_bottom_height_m)


def clamp_board_coordinates(
    board_x_m: np.ndarray | pd.Series,
    board_z_m: np.ndarray | pd.Series,
    *,
    geometry: BoardGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Project arbitrary board coordinates onto the physical board rectangle."""
    x = np.asarray(board_x_m, dtype=float)
    z = np.asarray(board_z_m, dtype=float)
    x_clamped = np.clip(x, 0.0, geometry.board_width_m)
    z_clamped = np.clip(z, geometry.board_bottom_height_m, geometry.board_top_height_m)
    return x_clamped, z_clamped


def add_board_geometry(
    df: pd.DataFrame,
    *,
    geometry: BoardGeometry,
    router_height_m: float,
    grid_x_top_is_zero: bool = True,
    coordinate_mode: str = "absolute",
) -> pd.DataFrame:
    """Annotate a dataframe with raw and clamped board-relative coordinates.

    The raw coordinates preserve the experimental grid even when it extends
    outside the physical board. The clamped coordinates project the point onto
    the board rectangle, which is useful when downstream predictions should
    remain physically on the board.
    """

    required = {"grid_x", "grid_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    grid_x = out["grid_x"].to_numpy(dtype=float)
    grid_y = out["grid_y"].to_numpy(dtype=float)

    if coordinate_mode == "absolute":
        board_x_m = grid_y * geometry.cell_width_m + geometry.cell_width_m / 2.0
        if grid_x_top_is_zero:
            board_z_m = geometry.board_top_height_m - (grid_x * geometry.cell_height_m + geometry.cell_height_m / 2.0)
        else:
            board_z_m = geometry.board_bottom_height_m + (grid_x * geometry.cell_height_m + geometry.cell_height_m / 2.0)
    elif coordinate_mode == "fit_to_data":
        unique_y = np.sort(np.unique(grid_y))
        unique_x = np.sort(np.unique(grid_x))
        if len(unique_y) == 0 or len(unique_x) == 0:
            raise ValueError("Cannot fit board coordinates on an empty grid.")
        x_step = geometry.board_width_m / float(len(unique_y))
        z_step = geometry.board_height_m / float(len(unique_x))
        y_rank = np.searchsorted(unique_y, grid_y)
        x_rank = np.searchsorted(unique_x, grid_x)
        board_x_m = y_rank * x_step + x_step / 2.0
        if grid_x_top_is_zero:
            board_z_m = geometry.board_top_height_m - (x_rank * z_step + z_step / 2.0)
        else:
            board_z_m = geometry.board_bottom_height_m + (x_rank * z_step + z_step / 2.0)
    else:
        raise ValueError(f"Unsupported coordinate_mode: {coordinate_mode}")

    out["board_x_m"] = board_x_m
    out["board_z_m"] = board_z_m
    board_x_clamped_m, board_z_clamped_m = clamp_board_coordinates(
        out["board_x_m"],
        out["board_z_m"],
        geometry=geometry,
    )
    out["board_x_clamped_m"] = board_x_clamped_m
    out["board_z_clamped_m"] = board_z_clamped_m
    out["is_within_board_width"] = (out["board_x_m"] >= 0.0) & (out["board_x_m"] <= geometry.board_width_m)
    out["is_within_board_height"] = (
        (out["board_z_m"] >= geometry.board_bottom_height_m) & (out["board_z_m"] <= geometry.board_top_height_m)
    )
    out["is_within_board"] = out["is_within_board_width"] & out["is_within_board_height"]
    out["board_projection_dx_m"] = out["board_x_clamped_m"] - out["board_x_m"]
    out["board_projection_dz_m"] = out["board_z_clamped_m"] - out["board_z_m"]
    out["board_projection_distance_m"] = np.sqrt(
        np.square(out["board_projection_dx_m"].to_numpy(dtype=float))
        + np.square(out["board_projection_dz_m"].to_numpy(dtype=float))
    )
    out["esp_height_m"] = board_z_m
    out["router_height_m"] = float(router_height_m)
    out["router_vertical_delta_m"] = out["esp_height_m"] - out["router_height_m"]
    out["board_center_offset_x_m"] = out["board_x_m"] - (geometry.board_width_m / 2.0)
    out["board_center_offset_x_clamped_m"] = out["board_x_clamped_m"] - (geometry.board_width_m / 2.0)
    out["router_esp_3d_m"] = np.sqrt(
        np.square(out["router_distance_m"].to_numpy(dtype=float))
        + np.square(out["router_vertical_delta_m"].to_numpy(dtype=float))
        + np.square(out["board_center_offset_x_m"].to_numpy(dtype=float))
    )
    out["router_vertical_delta_clamped_m"] = out["board_z_clamped_m"] - out["router_height_m"]
    out["router_esp_3d_clamped_m"] = np.sqrt(
        np.square(out["router_distance_m"].to_numpy(dtype=float))
        + np.square(out["router_vertical_delta_clamped_m"].to_numpy(dtype=float))
        + np.square(out["board_center_offset_x_clamped_m"].to_numpy(dtype=float))
    )
    return out


def add_board_zones(
    df: pd.DataFrame,
    *,
    geometry: BoardGeometry,
    n_cols: int = 3,
    n_rows: int = 3,
    use_clamped_coordinates: bool = True,
    label_language: str = "en",
) -> pd.DataFrame:
    """Annotate board-relative macro-zones such as top_left or middle_center."""
    if n_cols < 1 or n_rows < 1:
        raise ValueError("n_cols and n_rows must be >= 1.")

    out = df.copy()
    x_col = "board_x_clamped_m" if use_clamped_coordinates else "board_x_m"
    z_col = "board_z_clamped_m" if use_clamped_coordinates else "board_z_m"
    required = {x_col, z_col}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns for zoning: {sorted(missing)}")

    x = out[x_col].to_numpy(dtype=float)
    z = out[z_col].to_numpy(dtype=float)

    col_edges = np.linspace(0.0, geometry.board_width_m, num=n_cols + 1)
    row_edges = np.linspace(geometry.board_bottom_height_m, geometry.board_top_height_m, num=n_rows + 1)
    col_idx = np.clip(np.digitize(x, col_edges[1:-1], right=False), 0, n_cols - 1)
    row_from_bottom = np.clip(np.digitize(z, row_edges[1:-1], right=False), 0, n_rows - 1)
    row_idx = (n_rows - 1) - row_from_bottom

    if label_language == "fr":
        col_names_map = {
            1: ["centre"],
            2: ["gauche", "droite"],
            3: ["gauche", "centre", "droite"],
        }
        row_names_map = {
            1: ["milieu"],
            2: ["haut", "bas"],
            3: ["haut", "milieu", "bas"],
        }
    elif label_language == "en":
        col_names_map = {
            1: ["center"],
            2: ["left", "right"],
            3: ["left", "center", "right"],
        }
        row_names_map = {
            1: ["middle"],
            2: ["top", "bottom"],
            3: ["top", "middle", "bottom"],
        }
    else:
        raise ValueError(f"Unsupported label_language: {label_language}")
    col_names = col_names_map.get(n_cols, [f"col{i}" for i in range(n_cols)])
    row_names = row_names_map.get(n_rows, [f"row{i}" for i in range(n_rows)])

    zone_labels = np.asarray(
        [f"{row_names[r]}_{col_names[c]}" for r, c in zip(row_idx, col_idx, strict=False)],
        dtype=object,
    )
    out["zone_row"] = row_idx
    out["zone_col"] = col_idx
    out["zone_id"] = zone_labels
    out["zone_rows_total"] = int(n_rows)
    out["zone_cols_total"] = int(n_cols)
    return out
