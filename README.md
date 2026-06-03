# Bike Rental Demand Forecasting

This project compares classical, machine learning and foundation-model approaches for hourly bike rental demand forecasting. The target variable is `cnt`, the total number of bike rentals in a given hour.

The project is inspired by the Chronos paper, which argues that pretrained time-series models can simplify forecasting pipelines and produce strong zero-shot forecasts on unseen forecasting tasks. We test this idea on one hourly bike rental dataset by comparing Chronos 2 against models trained specifically on this dataset.

## Project Goal

The main goal is to answer:

> Can a zero-shot Chronos 2 foundation model perform competitively against classical baselines and a supervised LightGBM model on hourly bike rental demand forecasting?

To answer this, we build and compare:

- Naive forecast
- Seasonal naive forecast
- AutoARIMA
- Recursive LightGBM
- Zero-shot univariate Chronos 2

The final comparison is based on test-set forecasts saved in a common format:

```text
actual,prediction
```

## Dataset

The project uses the hourly Bike Sharing dataset stored in:

```text
data/raw/hour.csv
```

The chronological splits are:

| Split | Rows | Role |
|---|---:|---|
| Train | 12,047 | Model fitting, validation scaling, feature construction |
| Validation | 3,442 | Model selection and hyperparameter tuning where applicable |
| Test | 1,722 | Final model evaluation only |

The processed split files are:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

## Methodology

### Baseline Models

Notebook: `notebooks/02_baseline_models.ipynb`

The baseline notebook evaluates simple and classical time-series models:

- Naive: repeats the last observed value.
- Seasonal naive: repeats the last observed 24-hour pattern.
- AutoARIMA: selected with `pmdarima.auto_arima`.

Validation is performed with 24-hour expanding-window forecasts. Final test forecasts are saved separately for each baseline model.

### LightGBM

Notebook: `notebooks/03_lightgbm_forecasting.ipynb`

LightGBM is used as a supervised machine learning model with engineered time-series features, including lag and rolling features. Forecasting is recursive: predictions are fed back into the feature pipeline when future lag values are needed.

LightGBM is not the main model from the Chronos paper, but it is a strong practical benchmark and extends the comparison beyond purely classical methods.

### Chronos 2

Notebook: `notebooks/04_fm_chronos-2.ipynb`

Chronos 2 is used through AutoGluon TimeSeries in zero-shot mode. The final project version uses the univariate setup only:

```text
target = cnt
```

Chronos 2 is not fine-tuned on this dataset. The `.fit()` call in AutoGluon is used to infer metadata, store the predictor state and prepare the forecasting interface. The final Chronos 2 setup receives the available pre-test history and is evaluated on the test set.

Because Chronos 2 is computationally heavier than the other models, this notebook is best run on a GPU-enabled environment such as Databricks.

### Final Model Comparison

Notebook: `notebooks/05_final_model_comparison.ipynb`

The final notebook reloads all saved test forecast CSV files and recalculates the metrics with one shared evaluation function. This makes the final comparison reproducible and prevents differences caused by notebook-specific metric implementations.

The final comparison uses:

- MAE
- RMSE
- MASE

MASE is calculated with a seasonal period of 24 hours. The numerator is the model error on the test set, and the denominator is the 24-hour seasonal naive error calculated on the training set. This is consistent with the baseline, LightGBM and Chronos notebooks.

## Final Results

Final test-set comparison:

| Rank | Model | MAE | RMSE | MASE |
|---:|---|---:|---:|---:|
| 1 | Chronos 2 | 43.7287 | 75.2708 | 0.7487 |
| 2 | LightGBM | 68.5865 | 105.1321 | 1.1743 |
| 3 | Seasonal Naive | 75.1297 | 125.2061 | 1.2864 |
| 4 | AutoARIMA | 186.6068 | 255.3413 | 3.1951 |
| 5 | Naive | 191.1144 | 260.7123 | 3.2723 |

The final table is saved in:

```text
outputs/metrics/final_model_comparison.csv
```

The final MASE plot is saved in:

```text
outputs/figures/final_model_comparison_mase.png
```

## Interpretation

Chronos 2 achieved the best test-set performance in this project. It outperformed:

- the classical baselines,
- AutoARIMA,
- and the supervised recursive LightGBM model.

This result is consistent with the practical claim made in the Chronos paper: pretrained time-series models can be a viable way to simplify forecasting workflows while still producing competitive or superior zero-shot forecasts.

However, this project should not be interpreted as a full replication of the Chronos paper benchmark. The paper evaluates Chronos on 42 datasets and compares performance across a broad benchmark. Our project evaluates one hourly bike rental dataset and uses Chronos 2, a newer model version. Therefore, our conclusion is narrower:

> On this dataset, zero-shot univariate Chronos 2 provides the strongest final test performance and supports the idea that pretrained time-series foundation models can simplify forecasting pipelines.

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── hour.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── notebooks/
│   ├── 01_eda_feature_engineering.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_lightgbm_forecasting.ipynb
│   ├── 04_fm_chronos-2.ipynb
│   └── 05_final_model_comparison.ipynb
├── outputs/
│   ├── forecasts/
│   ├── metrics/
│   └── figures/
├── requirements.txt
└── README.md
```

## How To Run The Project

### 1. Clone The Repository

```bash
git clone https://github.com/Kat-Pelka/RR_project.git
cd RR_project
```

### 2. Create A Python Environment

Python 3.10-3.13 is recommended. Chronos 2 was run in a Databricks GPU environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you run Chronos 2 on Databricks, install the Chronos dependencies in the notebook environment and restart Python:

```python
%pip install -U "autogluon.timeseries[chronos]" "chronos-forecasting>=2.0"
dbutils.library.restartPython()
```

### 3. Run The Notebooks In Order

Run the notebooks in this order:

```text
01_eda_feature_engineering.ipynb
02_baseline_models.ipynb
03_lightgbm_forecasting.ipynb
04_fm_chronos-2.ipynb
05_final_model_comparison.ipynb
```

The recommended workflow is:

1. Run notebook 01 to create processed splits and engineered features.
2. Run notebook 02 to generate baseline forecasts and metrics.
3. Run notebook 03 to tune and evaluate LightGBM.
4. Run notebook 04 on GPU/Databricks to run Chronos 2.
5. Run notebook 05 to regenerate the final model comparison table.

### 4. Expected Outputs

Final forecast files:

```text
outputs/forecasts/naive_test_forecasts.csv
outputs/forecasts/seasonal_naive_test_forecasts.csv
outputs/forecasts/auto_arima_test_forecasts.csv
outputs/forecasts/lightgbm_test_recursive_forecasts.csv
outputs/forecasts/chronos2_test_forecasts.csv
```

Final comparison files:

```text
outputs/metrics/final_model_comparison.csv
outputs/figures/final_model_comparison_mase.png
```

## Reproducibility Notes

- All models are evaluated on the same chronological test split.
- Final forecast files use the same two-column structure: `actual`, `prediction`.
- Final metrics are recalculated in notebook 05 from the saved forecast files.
- MASE uses the same seasonal period, `m = 24`, for all models.
- The final comparison checks that all forecast files contain the same `actual` values.
- Random seed is set to `42` where applicable.
- Model artifacts in `outputs/models/` are not required to reproduce the final comparison if forecast CSV files are available.

## Team Responsibilities

- Iza: EDA, train/validation/test split, Naive, Seasonal Naive, AutoARIMA
- Kasia: repository structure, feature engineering, recursive LightGBM
- Pawel: Chronos 2, final comparison, reproducibility, README

## References

- Chronos paper: [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- Amazon Science summary: [Chronos: Learning the language of time series](https://www.amazon.science/publications/chronos-learning-the-language-of-time-series)
- AutoGluon TimeSeries Chronos documentation: [Forecasting with Chronos-2](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)
- Chronos 2 model card: [autogluon/chronos-2](https://huggingface.co/autogluon/chronos-2)
