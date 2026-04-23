"""Base calendar, price, promo and stock features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ritm_prodazh.config import DATE_COL, ID_COL


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build exogenous features available for both train and test rows."""
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])

    out["dow"] = out[DATE_COL].dt.dayofweek.astype(np.int16)
    out["month"] = out[DATE_COL].dt.month.astype(np.int16)
    out["weekofyear"] = out[DATE_COL].dt.isocalendar().week.astype(np.int16)
    out["dayofmonth"] = out[DATE_COL].dt.day.astype(np.int16)
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)

    out["log_price"] = np.log1p(out["price"].astype(np.float32))
    out["log_left"] = np.log1p(out["prev_leftovers"].astype(np.float32))

    out["stock_small_5"] = (out["prev_leftovers"] <= 5).astype(np.int8)
    out["stock_small_10"] = (out["prev_leftovers"] <= 10).astype(np.int8)

    out["promo_x_log_price"] = (
        out["is_promo"].astype(np.float32) * out["log_price"]
    ).astype(np.float32)
    out["promo_x_left"] = (
        out["is_promo"].astype(np.float32) * out["log_left"]
    ).astype(np.float32)

    out[ID_COL] = out[ID_COL].astype(str)
    return out
