# Sensor Calibration Comparison

## Project overview
This Python project compares raw sensor readings with known reference values and checks whether a linear calibration improves accuracy. It uses a small reproducible synthetic dataset, an object-oriented record class, NumPy, Pandas, Matplotlib, JSON export, and unit tests.

**Important:** The included dataset is synthetic for demonstration and testing; it is not a real laboratory measurement.

## Requirements
- Python 3.9+
- numpy
- pandas
- matplotlib

Install dependencies:
```bash
python -m pip install numpy pandas matplotlib
```

## Run
From this folder:
```bash
python main.py
```
Outputs are written to `outputs/`; labelled plots are written to `plots/`.

Regenerate the dataset:
```bash
python generate_dataset.py
```

Run tests:
```bash
python -m unittest discover -s tests -v
```

## Dataset columns
- `run_id`: unique observation identifier
- `reference_value`: known reference value
- `raw_reading`: uncorrected sensor measurement
- `split`: `train` for fitting the calibration line or `test` for held-out evaluation

## Method
1. Validate required columns, missing values, non-negative readings and split labels.
2. Fit `reference = slope * raw + intercept` using only training rows.
3. Evaluate raw and corrected readings on held-out test rows.
4. Report MAE, RMSE and mean error, and save CSV/JSON outputs.
5. Plot the held-out readings and absolute-error distributions.

## Files
- `main.py`: analysis, class, metrics, plots and exports
- `generate_dataset.py`: reproducible dataset generator (seed 42)
- `data/calibration_data.csv`: shareable dataset
- `tests/test_calibration.py`: unit tests
- `plots/`: two labelled plots
- `outputs/`: sample generated outputs
- `analysis_report.md`: brief results and limitations
- `contributions.md`: team member names 

## Interpretation
A calibration fit is not automatically trustworthy just because training error is low. This project keeps a held-out test split to check whether correction generalises. The dataset is small and synthetic, so results demonstrate the method rather than validate a physical sensor.
