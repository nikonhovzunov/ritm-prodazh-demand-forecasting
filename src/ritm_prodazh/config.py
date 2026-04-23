"""Project configuration."""

from __future__ import annotations

from dataclasses import dataclass

DATE_COL = "dt"
ID_COL = "nm_id"
TARGET_COL = "qty"

VAL_DAYS = 14
QTY_WINDOWS = (7, 14, 28)

LAGS_XL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 28, 35, 42]

BASE_FEATURES = [
    ID_COL,
    "price",
    "is_promo",
    "prev_leftovers",
    "dow",
    "month",
    "weekofyear",
    "dayofmonth",
    "is_weekend",
    "log_price",
    "stock_small_5",
    "stock_small_10",
    "promo_x_log_price",
    "promo_x_left",
]

ROLLING_FEATURES = [
    f"{stat}_{window}"
    for window in QTY_WINDOWS
    for stat in ("qty_mean", "qty_std", "qty_nz_share")
]

CAT_FEATURES = [ID_COL]

BEST_MODEL_FEATURES = BASE_FEATURES + ROLLING_FEATURES + [f"lag_{lag}" for lag in LAGS_XL]

# Best logged CV run in experiments_log_V2.csv:
# CB_LAGS_XL_PLUS_QTYROLL, rollout_pp_mean=2.761114, rollout_pp_std=0.445312.
BEST_CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "iterations": 4775,
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 3,
    "random_strength": 1.0,
    "random_seed": 42,
    "allow_writing_files": False,
}


@dataclass(frozen=True)
class ProjectFacts:
    name: str = "Ritm Prodazh"
    platform: str = "All Cups"
    result: str = "16th place out of 125"
    task: str = "14-day item-level demand forecasting"
    target: str = TARGET_COL
    best_logged_cv_model: str = "CB_LAGS_XL_PLUS_QTYROLL"
    best_logged_cv_wmae_mean: float = 2.761114
    best_logged_cv_wmae_std: float = 0.445312
