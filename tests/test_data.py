from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from localization.data import CampaignSpec, infer_router_distance, load_measurements


class DataTests(unittest.TestCase):
    def test_infer_router_distance_from_digits_and_french_words(self) -> None:
        self.assertEqual(infer_router_distance("ddeuxmetres"), 2.0)
        self.assertEqual(infer_router_distance("dcinqmetres"), 5.0)
        self.assertEqual(infer_router_distance("distance_3_5m"), 3.0)
        self.assertIsNone(infer_router_distance("exp_front"))

    def test_load_measurements_adds_labels_and_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "ddeuxmetres"
            folder.mkdir()
            pd.DataFrame(
                [
                    {
                        "Signal": -40,
                        "Noise": -95,
                        "signal_A1": -41,
                        "signal_A2": -42,
                        "signal_A3": -43,
                    }
                ]
            ).to_csv(folder / "1_2.csv", index=False)

            df = load_measurements([CampaignSpec(folder)])

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["grid_x"], 1)
        self.assertEqual(row["grid_y"], 2)
        self.assertEqual(row["grid_cell"], "1_2")
        self.assertEqual(row["router_distance_m"], 2.0)
        self.assertAlmostEqual(row["coord_x_m"], 0.625)
        self.assertAlmostEqual(row["coord_y_m"], 0.45)

    def test_load_measurements_rejects_unexpected_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "ddeuxmetres"
            folder.mkdir()
            (folder / "bad.csv").write_text("Signal,Noise,signal_A1,signal_A2,signal_A3\n-40,-95,-41,-42,-43\n")
            with self.assertRaises(ValueError):
                load_measurements([CampaignSpec(folder)])


if __name__ == "__main__":
    unittest.main()
