import unittest
import numpy as np
import pandas as pd
from main import CalibrationRun, fit_calibration, error_metrics, load_data, analyse, DATA_PATH

class CalibrationTests(unittest.TestCase):
    def test_valid_record(self):
        self.assertTrue(CalibrationRun("A", 10, 11).is_valid())

    def test_negative_record_invalid(self):
        self.assertFalse(CalibrationRun("B", 10, -1).is_valid())

    def test_linear_fit(self):
        slope, intercept = fit_calibration([10, 20, 30], [12, 22, 32])
        self.assertAlmostEqual(slope, 1.0, places=6)
        self.assertAlmostEqual(intercept, -2.0, places=6)

    def test_error_metrics(self):
        metrics = error_metrics([10, 20], [12, 18])
        self.assertAlmostEqual(metrics["mae"], 2.0)

    def test_dataset_has_train_and_test(self):
        df = load_data(DATA_PATH)
        self.assertIn("train", set(df["split"]))
        self.assertIn("test", set(df["split"]))

    def test_calibration_reduces_test_mae(self):
        _, summary = analyse(load_data(DATA_PATH))
        self.assertLess(summary["held_out_corrected_error"]["mae"],
                        summary["held_out_raw_error"]["mae"])

if __name__ == "__main__":
    unittest.main()
