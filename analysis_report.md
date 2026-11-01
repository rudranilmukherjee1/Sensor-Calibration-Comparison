# Brief Analysis Report: Sensor Calibration Comparison

## Objective
Estimate a linear correction for raw sensor readings using paired known reference values, then evaluate the correction on observations excluded from calibration fitting.

## Dataset and method
The shareable dataset contains 44 synthetic paired observations across reference values from 0 to 100 units. A fixed random seed makes the generator reproducible. Three quarters of rows are assigned to training and one quarter to held-out testing. The model fits `reference = slope × raw + intercept` using training rows only.

## Results
The exact equation and test metrics are generated in `outputs/summary.json` whenever `python main.py` is run. The two plots show (1) raw and corrected readings relative to the ideal line, and (2) the distribution of absolute errors before and after correction. A lower held-out MAE/RMSE after correction supports the claim that the linear adjustment improved predictions for this synthetic sample.

## Evaluation and limitations
- The synthetic sensor was generated with an offset, scale error and random noise, so it is useful for checking the analysis pipeline.
- Only a small set of reference levels and 44 observations are included; results may vary with noise and split.
- Linear correction assumes a roughly linear relationship. It may fail for nonlinear drift, saturation, temperature dependence or sensor aging.
- A few missing or unrecorded runs could bias summaries. The loader rejects missing required fields rather than silently imputing them.
- Real deployment requires collecting repeated physical sensor readings against a traceable reference instrument, then validating on later readings not used to fit the correction.

## Conclusion
This project demonstrates a reproducible calibration workflow and tests whether correction improves held-out error. It does not claim that the model is ready for a real sensor until real experimental data are collected.
