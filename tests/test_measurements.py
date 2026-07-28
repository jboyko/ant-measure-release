import sys
import tempfile
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from measurements import build_measurements, normalize_unit


class MeasurementTests(unittest.TestCase):
    def test_unit_normalisation(self):
        self.assertEqual(normalize_unit(1, "mm"), 1)
        self.assertEqual(normalize_unit(1000, "um"), 1)
        self.assertEqual(normalize_unit(0.1, "cm"), 1)

    def test_dorsal_traits(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            (labels / "sample.txt").write_text(
                "0 0.5 0.5 1 1 0.1 0.5 2 0.3 0.5 2 0.7 0.5 2 0.9 0.5 2\n"
            )
            dimensions = pd.DataFrame([{"image_name": "sample.jpg", "width": 100, "height": 50}])
            calibration = pd.DataFrame([
                {"image": "sample.jpg", "final_value": 1, "final_unit": "mm", "pixel_width": 20}
            ])
            output = build_measurements(labels, dimensions, calibration, "dorsal")
            self.assertEqual(len(output), 1)
            self.assertAlmostEqual(output.loc[0, "pronotum_width_px"], 40)
            self.assertAlmostEqual(output.loc[0, "body_length_px"], 80)
            self.assertAlmostEqual(output.loc[0, "pronotum_width_mm"], 2)
            self.assertAlmostEqual(output.loc[0, "body_length_mm"], 4)


if __name__ == "__main__":
    unittest.main()
