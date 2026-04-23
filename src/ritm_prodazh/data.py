"""Data loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ritm_prodazh.config import DATE_COL


@dataclass(frozen=True)
class DemandData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame | None = None


REQUIRED_FILES = {
    "train": "train.parquet",
    "test": "test.parquet",
}

OPTIONAL_FILES = {
    "sample_submission": "sample_submission.csv",
}


def validate_data_dir(data_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES.values() if not (data_dir / name).exists()]
    if missing:
        formatted = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            f"Missing required data files in {data_dir}:\n{formatted}\n\n"
            "Expected files: train.parquet, test.parquet. "
            "Optional file: sample_submission.csv."
        )


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    return out


def load_data(data_dir: str | Path) -> DemandData:
    data_dir = Path(data_dir)
    validate_data_dir(data_dir)

    train = _normalize_dates(pd.read_parquet(data_dir / REQUIRED_FILES["train"]))
    test = _normalize_dates(pd.read_parquet(data_dir / REQUIRED_FILES["test"]))

    sample_path = data_dir / OPTIONAL_FILES["sample_submission"]
    sample_submission = pd.read_csv(sample_path) if sample_path.exists() else None
    if sample_submission is not None:
        sample_submission = _normalize_dates(sample_submission)

    return DemandData(train=train, test=test, sample_submission=sample_submission)
