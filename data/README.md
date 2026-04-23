# Data

Competition data is not stored in this repository.

Place the raw files here:

```text
data/raw/
  train.parquet
  test.parquet
  sample_submission.csv  # optional, used to preserve submission order
```

Expected columns:

- `train.parquet`: `nm_id`, `dt`, `qty`, `price`, `is_promo`, `prev_leftovers`
- `test.parquet`: `nm_id`, `dt`, `price`, `is_promo`, `prev_leftovers`
- `sample_submission.csv`: `nm_id`, `dt`, `qty`
