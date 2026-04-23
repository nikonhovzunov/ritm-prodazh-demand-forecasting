# Resume Alignment

This file records the exact claims from the current resume PDF and where they are represented in the project.

## Resume Claim

**Ритмы продаж 2026** — 16 место из 125  
All Cups | прогнозирование 14-дневного спроса

- Построил CatBoost-модель для 14-дневного прогноза спроса по истории продаж, ценам, промо, остаткам и лагово-роллинговым признакам.
- Реализовал полный авторегрессионный rollout и валидировал решение как многошаговый прогноз на временных фолдах и отложенной выборке.

## Project Coverage

- Rank/platform: `README.md`, `docs/project_card.md`, `docs/resume_ru.md`.
- CatBoost model: `src/ritm_prodazh/model.py`, `src/ritm_prodazh/pipeline.py`.
- History, price, promo, stock features: `src/ritm_prodazh/features/base.py`.
- Lag and rolling features: `src/ritm_prodazh/features/time_series.py`.
- Autoregressive rollout: `src/ritm_prodazh/rollout.py`.
- Temporal validation facts: `README.md`, `docs/resume_ru.md`; original notebook evidence is `04_cb_features_rollout.ipynb`.
