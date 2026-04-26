from __future__ import annotations

import unittest

import numpy as np

from localization.orientation import (
    angle_to_unit_circle,
    angles_to_unit_circle,
    circular_distance_degrees,
    nearest_orientation_label,
    normalize_orientation_label,
    unit_circle_to_angle,
)


class OrientationTests(unittest.TestCase):
    def test_normalize_orientation_label(self) -> None:
        self.assertEqual(normalize_orientation_label("Front Left"), "front_left")
        self.assertEqual(normalize_orientation_label("back-right"), "back_right")
        with self.assertRaises(KeyError):
            normalize_orientation_label("unknown")

    def test_unit_circle_round_trip(self) -> None:
        cos_value, sin_value = angle_to_unit_circle(225.0)

        self.assertAlmostEqual(unit_circle_to_angle(cos_value, sin_value), 225.0)

    def test_circular_distance_wraps_around_zero(self) -> None:
        self.assertEqual(circular_distance_degrees(350.0, 10.0), 20.0)

    def test_nearest_orientation_label(self) -> None:
        self.assertEqual(nearest_orientation_label(88.0), "right")

    def test_angles_to_unit_circle_vectorized(self) -> None:
        values = angles_to_unit_circle([0.0, 90.0])

        self.assertEqual(values.shape, (2, 2))
        np.testing.assert_allclose(values[0], [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(values[1], [0.0, 1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
