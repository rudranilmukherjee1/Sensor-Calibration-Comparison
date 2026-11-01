"""Sensor calibration comparison project."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "calibration_data.csv"
PLOT_DIR = ROOT / "plots"
OUTPUT_DIR = ROOT / "outputs"

class CalibrationRun:
    """One sensor observation paired with a known reference value."""
    def __init__(self, run_id, reference_value, raw_reading, split="train"):
        self.run_id = str(run_id)
        self.reference_value = float(reference_value)
        self.raw_reading = float(raw_reading)
        self.split = str(split)

    def is_valid(self):
        values = [self.reference_value, self.raw_reading]
        return all(np.isfinite(values)) and self.reference_value >= 0 and self.raw_reading >= 0

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = {"run_id", "reference_value", "raw_reading", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Dataset contains missing required values.")
    if not set(df["split"]).issubset({"train", "test"}):
        raise ValueError("split must contain only 'train' or 'test'.")
    if (df[["reference_value", "raw_reading"]] < 0).any().any():
        raise ValueError("Reference and raw readings must be non-negative.")
    return df

def fit_calibration(reference, raw):
    """Fit reference = slope * raw + intercept using training pairs."""
    reference = np.asarray(reference, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if len(reference) < 2 or len(reference) != len(raw):
        raise ValueError("Need at least two paired readings of equal length.")
    if not np.isfinite(reference).all() or not np.isfinite(raw).all():
        raise ValueError("Calibration inputs must be finite.")
    slope, intercept = np.polyfit(raw, reference, 1)
    return float(slope), float(intercept)

def error_metrics(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    error = estimate - reference
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_error": float(np.mean(error)),
    }

def analyse(df):
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    if len(train) < 2 or len(test) < 1:
        raise ValueError("Need at least two training rows and one held-out test row.")
    slope, intercept = fit_calibration(train["reference_value"], train["raw_reading"])
    result = df.copy()
    result["corrected_reading"] = slope * result["raw_reading"] + intercept
    test_result = result[result["split"] == "test"]
    raw_metrics = error_metrics(test_result["reference_value"], test_result["raw_reading"])
    corrected_metrics = error_metrics(test_result["reference_value"], test_result["corrected_reading"])
    return result, {
        "calibration_equation": f"corrected = {slope:.5f} * raw + ({intercept:.5f})",
        "slope": slope,
        "intercept": intercept,
        "training_samples": int(len(train)),
        "held_out_samples": int(len(test)),
        "held_out_raw_error": raw_metrics,
        "held_out_corrected_error": corrected_metrics,
        "mae_improvement_percent": float(
            100 * (raw_metrics["mae"] - corrected_metrics["mae"]) / raw_metrics["mae"]
        ) if raw_metrics["mae"] else 0.0,
    }

def make_plots(result):
    PLOT_DIR.mkdir(exist_ok=True)
    test = result[result["split"] == "test"]
    # Plot 1: raw and corrected against the known reference.
    plt.figure(figsize=(8, 5))
    plt.scatter(test["reference_value"], test["raw_reading"], label="Raw sensor reading", marker="o")
    plt.scatter(test["reference_value"], test["corrected_reading"], label="Corrected reading", marker="x")
    lo = min(test["reference_value"].min(), test["raw_reading"].min(), test["corrected_reading"].min())
    hi = max(test["reference_value"].max(), test["raw_reading"].max(), test["corrected_reading"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="Ideal: estimate = reference")
    plt.xlabel("Known reference value (units)")
    plt.ylabel("Sensor estimate (units)")
    plt.title("Held-out sensor readings before and after calibration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "calibration_scatter.png", dpi=160)
    plt.close()

    # Plot 2: absolute errors before and after calibration.
    raw_error = np.abs(test["raw_reading"] - test["reference_value"])
    corrected_error = np.abs(test["corrected_reading"] - test["reference_value"])
    plt.figure(figsize=(8, 5))
    plt.hist(raw_error, bins=8, alpha=0.65, label="Raw absolute error")
    plt.hist(corrected_error, bins=8, alpha=0.65, label="Corrected absolute error")
    plt.xlabel("Absolute error (units)")
    plt.ylabel("Number of test readings")
    plt.title("Held-out absolute-error distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "error_histogram.png", dpi=160)
    plt.close()

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    result, summary = analyse(df)
    result.to_csv(OUTPUT_DIR / "calibrated_readings.csv", index=False)
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    result.groupby("split")[["reference_value", "raw_reading", "corrected_reading"]].mean().to_csv(
        OUTPUT_DIR / "split_summary.csv"
    )
    make_plots(result)
    print("SENSOR CALIBRATION ANALYSIS")
    print(summary["calibration_equation"])
    print(f"Held-out raw MAE:       {summary['held_out_raw_error']['mae']:.3f}")
    print(f"Held-out corrected MAE: {summary['held_out_corrected_error']['mae']:.3f}")
    print(f"MAE improvement:        {summary['mae_improvement_percent']:.1f}%")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Plots saved in:   {PLOT_DIR}")

if __name__ == "__main__":
    main()
