from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from localization.board_geometry import BoardGeometry, add_board_geometry, add_board_zones, clamp_board_coordinates


class GeometryTests(unittest.TestCase):
    def test_clamp_board_coordinates(self) -> None:
        geometry = BoardGeometry(board_width_m=2.0, board_bottom_height_m=1.0, board_top_height_m=2.0)

        x, z = clamp_board_coordinates(np.array([-0.5, 0.5, 3.0]), np.array([0.2, 1.5, 3.0]), geometry=geometry)

        np.testing.assert_allclose(x, [0.0, 0.5, 2.0])
        np.testing.assert_allclose(z, [1.0, 1.5, 2.0])

    def test_add_board_geometry_absolute_mode(self) -> None:
        geometry = BoardGeometry(board_width_m=2.0, board_bottom_height_m=1.0, board_top_height_m=2.2)
        df = pd.DataFrame([{"grid_x": 0, "grid_y": 1, "router_distance_m": 4.0}])

        out = add_board_geometry(df, geometry=geometry, router_height_m=0.75)
        row = out.iloc[0]

        self.assertAlmostEqual(row["board_x_m"], 0.375)
        self.assertAlmostEqual(row["board_z_m"], 2.05)
        self.assertTrue(bool(row["is_within_board"]))
        self.assertAlmostEqual(row["board_projection_distance_m"], 0.0)

    def test_add_board_zones_uses_clamped_coordinates(self) -> None:
        geometry = BoardGeometry(board_width_m=3.0, board_bottom_height_m=1.0, board_top_height_m=2.0)
        df = pd.DataFrame([{"board_x_clamped_m": 0.1, "board_z_clamped_m": 1.9}])

        out = add_board_zones(df, geometry=geometry, n_cols=3, n_rows=3, label_language="en")

        self.assertEqual(out.iloc[0]["zone_id"], "top_left")
        self.assertEqual(out.iloc[0]["zone_row"], 0)
        self.assertEqual(out.iloc[0]["zone_col"], 0)


if __name__ == "__main__":
    unittest.main()
