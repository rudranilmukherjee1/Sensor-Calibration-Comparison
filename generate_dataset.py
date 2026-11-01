"""Regenerate the shareable synthetic calibration dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)
reference = np.tile(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float), 4)
rng.shuffle(reference)
raw = 1.08 * reference + 4.5 + rng.normal(0, 2.2, size=len(reference))
df = pd.DataFrame({
    "run_id": [f"R{i+1:03d}" for i in range(len(reference))],
    "reference_value": reference,
    "raw_reading": np.round(raw, 3),
    "split": ["train" if i % 4 != 0 else "test" for i in range(len(reference))]
})
df.to_csv(ROOT / "data" / "calibration_data.csv", index=False)
print(f"Generated {len(df)} rows at data/calibration_data.csv")
