"""Prediction post-processing."""

from __future__ import annotations

import numpy as np


def postprocess_predictions(
    predictions: np.ndarray,
    leftovers: np.ndarray,
    round_int: bool = True,
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float32)
    predictions = np.clip(predictions, 0, None)
    predictions = np.minimum(predictions, np.asarray(leftovers, dtype=np.float32))
    if round_int:
        predictions = np.rint(predictions)
    return predictions.astype(np.float32)
