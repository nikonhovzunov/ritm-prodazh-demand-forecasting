# Project Card

| Field | Value |
| --- | --- |
| Project | Ritm Prodazh |
| Platform | All Cups |
| Result | 16th place out of 125 |
| Domain | Retail demand forecasting |
| Object | `nm_id` item-date rows |
| Forecast horizon | 14 days |
| Target | `qty` |
| Metric used in experiments | Weighted MAE |
| Best confirmed CV run | `CB_LAGS_XL_PLUS_QTYROLL` |
| Best confirmed CV | `wMAE 2.761114 +/- 0.445312` |
| Main model | CatBoostRegressor |
| Key techniques | lag features, rolling demand features, autoregressive rollout |

## Short Summary

Forecasting sparse 14-day item demand for retail products. The final confirmed pipeline uses CatBoostRegressor with price/promo/stock features, target lags, rolling target statistics and day-by-day autoregressive inference.

## Reproducible Entry Point

```powershell
python scripts/make_submission.py `
  --data-dir data/raw `
  --submission-path submissions/catboost_rollout_submission.csv `
  --model-path models/catboost_rollout.cbm `
  --task-type GPU
```
