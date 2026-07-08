# Data Setup

## Included in Git (demo / dashboard)

| File | Purpose |
|------|---------|
| `raw_mobile_reviews_sample.csv` | Subset of mobile review records |
| `cleaned_mobile_reviews_sample.csv` | Cleaned sample for segmentation demos |
| `product_features.csv` | Product specification features |
| `segmented_products.csv` | Pre-computed segment assignments |

## Full dataset (local only)

| File | Purpose |
|------|---------|
| `raw_mobile_reviews.csv` | Full review export for KMeans training |

**How to obtain:** Place your full review CSV at `data/raw_mobile_reviews.csv`.

**Resolution order:** `src/data_ingestion.py` uses the full file when present; otherwise `raw_mobile_reviews_sample.csv`.

**Ingest:** `python -m src.data_ingestion` (requires PostgreSQL; see `src/db_config.py`).
