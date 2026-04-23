# Release Check

Date: 2026-04-23

## Status

Ready for GitHub publication.

## Checks Passed

- Repository initialized with Git.
- `.gitignore` excludes raw data, submissions, models, virtual environments and caches.
- `.gitattributes` added for stable text line endings.
- No raw competition data is stored in the repository.
- No model binaries or generated submissions are stored in the repository.
- No large files over 1 MB are stored in the repository.
- No local machine-specific absolute paths are present in project files.
- No mojibake / broken Cyrillic markers were found in project text files.
- Python syntax check passed with `compileall`.
- CLI help works for `scripts/make_submission.py` and `scripts/train.py`.
- Fresh virtual environment installation from `requirements.txt` completed successfully.
- Heavy pipeline imports passed after installing requirements:
  - `ritm_prodazh.pipeline`
  - `ritm_prodazh.model`
- End-to-end smoke test passed on synthetic data:
  - generated parquet train/test files;
  - trained CatBoost on CPU with one iteration;
  - ran autoregressive rollout;
  - wrote a valid submission CSV.
- Resume claims are mapped in `docs/resume_alignment.md`.

## Not Run

- Full training on the original competition dataset was not run because raw data is intentionally not included in this repository.

## Main Entry Point

```powershell
python scripts/make_submission.py `
  --data-dir data/raw `
  --submission-path submissions/catboost_rollout_submission.csv `
  --model-path models/catboost_rollout.cbm `
  --task-type GPU
```
