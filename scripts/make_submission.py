"""Train CatBoost and generate a Ritm Prodazh submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory with train.parquet, test.parquet and optional sample_submission.csv.",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=PROJECT_ROOT / "submissions" / "catboost_rollout_submission.csv",
        help="Where to write the submission CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "catboost_rollout.cbm",
        help="Where to save the trained CatBoost model.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Do not save the trained CatBoost model.",
    )
    parser.add_argument(
        "--task-type",
        choices=["GPU", "CPU"],
        default="GPU",
        help="CatBoost task type. GPU matches the original notebook.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override CatBoost iterations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ritm_prodazh.pipeline import run_submission_pipeline

    model_path = None if args.no_save_model else args.model_path
    result = run_submission_pipeline(
        data_dir=args.data_dir,
        submission_path=args.submission_path,
        model_path=model_path,
        task_type=args.task_type,
        iterations=args.iterations,
    )

    print(f"Submission saved to: {result.submission_path}")
    print(f"Train rows:          {result.train_rows}")
    print(f"Test rows:           {result.test_rows}")
    print(f"Feature count:       {result.feature_count}")
    if result.model_path is not None:
        print(f"Model saved to:      {result.model_path}")


if __name__ == "__main__":
    main()
