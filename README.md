# Solar Flare Activity Forecasting

An end-to-end, reproducible multi-output regression baseline for estimating the C-, M-, and X-class flare counts associated with a solar active region over the next 24 hours.

## Scope and limitations

This uses UCI's 1,389-row Solar Flare dataset, which contains active-region descriptors and three next-24-hour flare-count targets. The source has no timestamps, so the evaluation is a random holdout rather than a time-based forecast test. It is a research/learning benchmark and must not be used for operational space-weather warnings.

## Run

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python main.py --output-dir artifacts
```

The script downloads the dataset through `ucimlrepo`, reports per-target MAE, median absolute error, and R², and writes holdout predictions, metrics, transformed-feature importances, and a fitted model under `artifacts/`.

## Data and attribution

Solar Flare, UCI Machine Learning Repository, DOI [10.24432/C5530G](https://doi.org/10.24432/C5530G), licensed CC BY 4.0. Original source: https://archive.ics.uci.edu/dataset/89/solar+flare

## Method

- Explicitly separates the three count targets from the 10 active-region predictors.
- Median-imputes numeric predictors and most-frequent imputes plus one-hot encodes categorical predictors inside a scikit-learn pipeline.
- Fits a multi-output random forest on a held-out split (seed 42).
- Reports per-target errors; scores are descriptive for this small historical dataset and do not establish operational skill.

## Reproducibility

The split seed and model configuration are in `main.py`. The script records the dataset row count, target names, and holdout metrics in `metrics.json`.

