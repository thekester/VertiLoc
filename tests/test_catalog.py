from __future__ import annotations

import unittest

from localization.catalog import (
    FEATURE_COLUMNS,
    ORIENTATION_CAMPAIGNS,
    ROOM_CAMPAIGNS,
    filter_room_campaigns,
    normalize_room_name,
)
from localization.constants import RSSI_FEATURE_COLUMNS


class CatalogTests(unittest.TestCase):
    def test_feature_columns_match_shared_constant(self) -> None:
        self.assertEqual(FEATURE_COLUMNS, list(RSSI_FEATURE_COLUMNS))

    def test_e102_campaign_distances_are_explicit(self) -> None:
        distances = [spec.resolved_distance() for spec in ROOM_CAMPAIGNS["E102"]]
        self.assertEqual(distances, [4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

    def test_orientation_campaign_catalog_contains_360_sets(self) -> None:
        self.assertEqual(
            [label for _, _, label in ORIENTATION_CAMPAIGNS["E101"]],
            ["back", "right", "front", "left"],
        )
        self.assertEqual(
            [spec.resolved_distance() for _, spec, _ in ORIENTATION_CAMPAIGNS["E102"][:4]],
            [4.0, 4.0, 4.0, 4.0],
        )

    def test_filter_room_campaigns_by_room_and_distance(self) -> None:
        filtered = filter_room_campaigns(room_filter=["d005"], distance_filter=[2.0])

        self.assertEqual(list(filtered), ["D005"])
        self.assertEqual(len(filtered["D005"]), 1)
        self.assertEqual(filtered["D005"][0].resolved_distance(), 2.0)

    def test_normalize_room_name(self) -> None:
        self.assertEqual(normalize_room_name(" e102 "), "E102")


if __name__ == "__main__":
    unittest.main()
