"""Submission helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ritm_prodazh.config import DATE_COL, ID_COL, TARGET_COL


def make_submission(
    predictions: pd.DataFrame,
    output_path: str | Path,
    sample_submission: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submit = predictions[[ID_COL, DATE_COL, TARGET_COL]].copy()
    submit[DATE_COL] = pd.to_datetime(submit[DATE_COL])

    if sample_submission is not None:
        sample = sample_submission[[ID_COL, DATE_COL]].copy()
        sample[DATE_COL] = pd.to_datetime(sample[DATE_COL])
        submit = sample.merge(submit, on=[ID_COL, DATE_COL], how="left")
        submit[TARGET_COL] = submit[TARGET_COL].fillna(0)

    submit[TARGET_COL] = submit[TARGET_COL].round().astype(int)
    submit[DATE_COL] = submit[DATE_COL].dt.strftime("%Y-%m-%d")
    submit.to_csv(output_path, index=False)
    return submit
