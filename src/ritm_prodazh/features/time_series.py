"""Lag and rolling target features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ritm_prodazh.config import BASE_FEATURES, DATE_COL, ID_COL, TARGET_COL


def make_model_features(
    lags: list[int],
    extra: list[str] | None = None,
) -> list[str]:
    extra = [] if extra is None else list(extra)
    return BASE_FEATURES + extra + [f"lag_{lag}" for lag in lags]


def add_lags_to_target(
    df_target: pd.DataFrame,
    history_df: pd.DataFrame,
    lags: list[int],
) -> pd.DataFrame:
    """Add lag columns to target rows using only the supplied history."""
    out = df_target.copy()
    base = history_df[[ID_COL, DATE_COL, TARGET_COL]].copy()

    for lag in lags:
        shifted = base.copy()
        shifted[DATE_COL] = shifted[DATE_COL] + pd.Timedelta(days=lag)
        shifted = shifted.rename(columns={TARGET_COL: f"lag_{lag}"})
        out = out.merge(
            shifted[[ID_COL, DATE_COL, f"lag_{lag}"]],
            on=[ID_COL, DATE_COL],
            how="left",
        )

    return out


def fill_lag_columns(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[f"lag_{lag}"].fillna(0).astype(np.float32)
    return out


def add_rolling_qty_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    """Teacher-forced rolling target statistics for training pools."""
    out = df.copy()
    out["_row"] = np.arange(len(out))
    out = out.sort_values([ID_COL, DATE_COL])

    qty_shift = out.groupby(ID_COL)[TARGET_COL].shift(1)

    for window in windows:
        roll = qty_shift.groupby(out[ID_COL]).rolling(window, min_periods=1)
        out[f"qty_mean_{window}"] = (
            roll.mean().reset_index(level=0, drop=True).fillna(0).astype(np.float32)
        )
        out[f"qty_std_{window}"] = (
            roll.std(ddof=0).reset_index(level=0, drop=True).fillna(0).astype(np.float32)
        )

        non_zero = qty_shift.gt(0).astype(np.float32)
        out[f"qty_nz_share_{window}"] = (
            non_zero.groupby(out[ID_COL])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0)
            .astype(np.float32)
        )

    return out.sort_values("_row").drop(columns="_row")


def add_qty_rolling_from_history_for_day(
    day: pd.DataFrame,
    history: pd.DataFrame,
    current_date: pd.Timestamp,
    windows: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    """Add honest rolling target features for one forecast date."""
    out = day.copy()

    for window in windows:
        left = current_date - pd.Timedelta(days=window)
        hist_window = history[
            (history[DATE_COL] < current_date) & (history[DATE_COL] >= left)
        ]

        if hist_window.empty:
            out[f"qty_mean_{window}"] = 0.0
            out[f"qty_std_{window}"] = 0.0
            out[f"qty_nz_share_{window}"] = 0.0
            continue

        grouped = hist_window.groupby(ID_COL)["qty_known"]
        stats = pd.concat(
            [
                grouped.mean().rename(f"qty_mean_{window}"),
                grouped.std(ddof=0).fillna(0.0).rename(f"qty_std_{window}"),
                grouped.apply(lambda s: (s.values > 0).mean()).rename(
                    f"qty_nz_share_{window}"
                ),
            ],
            axis=1,
        ).reset_index()

        out = out.merge(stats, on=ID_COL, how="left")

        out[f"qty_mean_{window}"] = out[f"qty_mean_{window}"].fillna(0).astype(np.float32)
        out[f"qty_std_{window}"] = out[f"qty_std_{window}"].fillna(0).astype(np.float32)
        out[f"qty_nz_share_{window}"] = (
            out[f"qty_nz_share_{window}"].fillna(0).astype(np.float32)
        )

    return out
