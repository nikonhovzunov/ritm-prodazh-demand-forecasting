"""CatBoost model helpers."""

from __future__ import annotations

from catboost import CatBoostRegressor, Pool

from ritm_prodazh.config import BEST_CATBOOST_PARAMS, CAT_FEATURES


def make_catboost_params(task_type: str = "GPU", iterations: int | None = None) -> dict:
    params = dict(BEST_CATBOOST_PARAMS)
    params["task_type"] = task_type
    if task_type.upper() == "GPU":
        params["devices"] = "0"
    if iterations is not None:
        params["iterations"] = iterations
    return params


def make_pool(X, y=None, weights=None) -> Pool:
    return Pool(X, y, cat_features=CAT_FEATURES, weight=weights)


def train_catboost_regressor(train_pool: Pool, params: dict) -> CatBoostRegressor:
    model = CatBoostRegressor(**params)
    model.fit(train_pool, verbose=200)
    return model
