"""End-to-end pipeline for the CatBoost autoregressive solution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ritm_prodazh.config import (
    BEST_MODEL_FEATURES,
    LAGS_XL,
    QTY_WINDOWS,
    TARGET_COL,
)
from ritm_prodazh.data import load_data
from ritm_prodazh.features.base import build_base_features
from ritm_prodazh.features.time_series import (
    add_lags_to_target,
    add_rolling_qty_features,
    fill_lag_columns,
)
from ritm_prodazh.metrics import make_weights
from ritm_prodazh.model import make_catboost_params, make_pool, train_catboost_regressor
from ritm_prodazh.rollout import autoregressive_rollout
from ritm_prodazh.submission import make_submission


@dataclass(frozen=True)
class PipelineResult:
    submission_path: Path
    model_path: Path | None
    train_rows: int
    test_rows: int
    feature_count: int


def build_training_matrix(train_df):
    train_features = build_base_features(train_df)
    train_features = add_rolling_qty_features(train_features, windows=QTY_WINDOWS)
    train_features = add_lags_to_target(train_features, train_features, LAGS_XL)
    train_features = fill_lag_columns(train_features, LAGS_XL)

    X_train = train_features[BEST_MODEL_FEATURES]
    y_train = train_features[TARGET_COL].astype(np.float32).values
    weights = make_weights(y_train)
    return train_features, X_train, y_train, weights


def run_submission_pipeline(
    data_dir: str | Path,
    submission_path: str | Path,
    model_path: str | Path | None = None,
    task_type: str = "GPU",
    iterations: int | None = None,
) -> PipelineResult:
    data = load_data(data_dir)

    train_features, X_train, y_train, weights = build_training_matrix(data.train)
    train_pool = make_pool(X_train, y_train, weights)

    params = make_catboost_params(task_type=task_type, iterations=iterations)
    model = train_catboost_regressor(train_pool, params)

    test_features = build_base_features(data.test)
    predictions = autoregressive_rollout(
        train_history=train_features,
        forecast_df=test_features,
        lags=LAGS_XL,
        model=model,
        model_features=BEST_MODEL_FEATURES,
        qty_windows=QTY_WINDOWS,
        clip_by_leftovers=True,
        round_int=True,
    )

    submission_path = Path(submission_path)
    make_submission(
        predictions,
        output_path=submission_path,
        sample_submission=data.sample_submission,
    )

    model_path = Path(model_path) if model_path is not None else None
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))

    return PipelineResult(
        submission_path=submission_path,
        model_path=model_path,
        train_rows=len(data.train),
        test_rows=len(data.test),
        feature_count=len(BEST_MODEL_FEATURES),
    )
