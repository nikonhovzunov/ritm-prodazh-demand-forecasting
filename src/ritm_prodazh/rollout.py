"""Autoregressive multi-day forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ritm_prodazh.config import CAT_FEATURES, DATE_COL, ID_COL, TARGET_COL
from ritm_prodazh.features.time_series import add_qty_rolling_from_history_for_day
from ritm_prodazh.postprocessing import postprocess_predictions


def autoregressive_rollout(
    train_history: pd.DataFrame,
    forecast_df: pd.DataFrame,
    lags: list[int],
    model,
    model_features: list[str],
    qty_windows: tuple[int, ...] = (7, 14, 28),
    clip_by_leftovers: bool = True,
    round_int: bool = True,
) -> pd.DataFrame:
    """Forecast day by day and feed predictions back into history."""
    history = train_history[[ID_COL, DATE_COL, TARGET_COL]].copy()
    history = history.rename(columns={TARGET_COL: "qty_known"})

    rolling_prefixes = ("qty_mean_", "qty_std_", "qty_nz_share_")
    base_feature_cols = [
        col
        for col in model_features
        if not str(col).startswith("lag_")
        and not str(col).startswith(rolling_prefixes)
    ]
    day_cols = list(dict.fromkeys([ID_COL, DATE_COL, "prev_leftovers"] + base_feature_cols))

    missing = [col for col in [ID_COL, DATE_COL, "prev_leftovers"] if col not in forecast_df]
    if missing:
        raise KeyError(f"forecast_df is missing required columns: {missing}")

    predictions = []

    for current_date in np.sort(forecast_df[DATE_COL].unique()):
        day = forecast_df.loc[forecast_df[DATE_COL] == current_date, day_cols].copy()

        for lag in lags:
            lookup_date = current_date - pd.Timedelta(days=lag)
            lag_values = history.loc[
                history[DATE_COL] == lookup_date,
                [ID_COL, "qty_known"],
            ].rename(columns={"qty_known": f"lag_{lag}"})
            day = day.merge(lag_values, on=ID_COL, how="left")

        for lag in lags:
            day[f"lag_{lag}"] = day[f"lag_{lag}"].fillna(0).astype(np.float32)

        day = add_qty_rolling_from_history_for_day(
            day,
            history,
            pd.Timestamp(current_date),
            windows=qty_windows,
        )

        raw_pred = model.predict(day[model_features]).astype(np.float32)
        if clip_by_leftovers:
            pred = postprocess_predictions(
                raw_pred,
                day["prev_leftovers"].values,
                round_int=round_int,
            )
        else:
            pred = np.clip(raw_pred, 0, None)
            if round_int:
                pred = np.rint(pred)

        day_out = day[[ID_COL, DATE_COL]].copy()
        day_out[TARGET_COL] = pred.astype(np.float32)
        predictions.append(day_out)

        history = pd.concat(
            [history, day_out.rename(columns={TARGET_COL: "qty_known"})],
            ignore_index=True,
        )

    return pd.concat(predictions, ignore_index=True)
