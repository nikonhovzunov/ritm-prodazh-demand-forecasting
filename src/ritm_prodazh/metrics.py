"""Metrics and sample weights."""

from __future__ import annotations

import numpy as np


def make_weights(
    y: np.ndarray,
    w_pos: float = 7.0,
    w_zero: float = 1.0,
) -> np.ndarray:
    y = np.asarray(y)
    return np.where(y > 0, w_pos, w_zero).astype(np.float32)


def weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w_pos: float = 7.0,
    w_zero: float = 1.0,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = make_weights(y_true, w_pos=w_pos, w_zero=w_zero)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))
