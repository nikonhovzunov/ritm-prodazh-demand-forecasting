# Ritm Prodazh - Demand Forecasting

GitHub-ready version of my demand forecasting solution for the **Ritm Prodazh** hackathon project on All Cups.

The original work was done in notebooks. This repository keeps the strongest confirmed CatBoost autoregressive pipeline and wraps it into a cleaner Python project with reusable modules, a command-line submission script, and no raw data committed to Git.

## Project Card

| Field | Value |
| --- | --- |
| Project | Ritm Prodazh |
| Platform | All Cups |
| Task URL | https://cups.online/ru/tasks/3268 |
| Result | 16th place out of 125 |
| Domain | Retail demand forecasting |
| Object | `nm_id` item-date rows |
| Forecast horizon | 14 days |
| Target | `qty` |
| Metric used in experiments | Weighted MAE |
| Best confirmed CV run | `CB_LAGS_XL_PLUS_QTYROLL` |
| Best confirmed CV | `wMAE 2.761114 +/- 0.445312` |
| Main model | CatBoostRegressor |
| Main signal | lag/rolling demand features + price/promo/stock features |

## Task

Forecast daily item demand for the next 14 days.

Known data:

- `train.parquet`: `nm_id`, `dt`, `qty`, `price`, `is_promo`, `prev_leftovers`
- `test.parquet`: `nm_id`, `dt`, `price`, `is_promo`, `prev_leftovers`

The submission format is:

```text
nm_id,dt,qty
...
```

The data is sparse: around 87% of training rows have zero demand.

## Solution

The final confirmed pipeline is a CatBoost autoregressive demand forecast.

Feature groups:

- calendar features: day of week, month, ISO week, day of month, weekend flag;
- price and promo features: `price`, `log_price`, `is_promo`, promo interactions;
- stock features: `prev_leftovers`, small-stock flags and log leftovers;
- target lags: `1..14`, `21`, `28`, `35`, `42`;
- rolling demand features: mean/std/non-zero share over 7, 14 and 28 days.

Forecasting strategy:

1. predict demand for the first test date;
2. append predictions to the known history;
3. recompute lag and rolling features for the next date;
4. repeat until the full 14-day horizon is predicted.

Post-processing:

- clip predictions below zero;
- cap predictions by `prev_leftovers`;
- round final `qty` to integer.

Validation setup from the competition notebooks:

- temporal folds;
- 14-day holdout;
- honest autoregressive rollout metric for the multi-step forecast.

## Repository Structure

```text
ritm-prodazh-demand-forecasting/
  README.md
  requirements.txt
  data/
    README.md
    raw/
  scripts/
    make_submission.py
    train.py
  src/
    ritm_prodazh/
      config.py
      data.py
      metrics.py
      model.py
      pipeline.py
      postprocessing.py
      rollout.py
      submission.py
      features/
        base.py
        time_series.py
  models/
  submissions/
  docs/
    project_card.md
    resume_ru.md
```

## Data

Raw data is not included.

Place the files into `data/raw/`:

```text
data/raw/
  train.parquet
  test.parquet
  sample_submission.csv
```

`sample_submission.csv` is optional, but recommended because it preserves the expected row order.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Generate Submission

PowerShell:

```powershell
python scripts/make_submission.py `
  --data-dir data/raw `
  --submission-path submissions/catboost_rollout_submission.csv `
  --model-path models/catboost_rollout.cbm `
  --task-type GPU
```

CPU fallback:

```powershell
python scripts/make_submission.py `
  --data-dir data/raw `
  --submission-path submissions/catboost_rollout_submission.csv `
  --model-path models/catboost_rollout.cbm `
  --task-type CPU
```

To skip saving the model, add `--no-save-model`.

## Reproducibility Notes

The implementation follows the strongest confirmed notebook experiment:

```text
04_cb_features_rollout.ipynb
```

The best confirmed row in `experiments_log_V2.csv`:

```text
CB_LAGS_XL_PLUS_QTYROLL
rollout_pp_mean = 2.761114
rollout_pp_std  = 0.445312
```

## Resume

Russian resume wording is available in:

```text
docs/resume_ru.md
```
