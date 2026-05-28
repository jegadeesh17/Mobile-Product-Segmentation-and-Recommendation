import pandas as pd
from db_config import get_engine
from pathlib import Path

# Build path to data folder relative to this script
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
csv_path = DATA_DIR / 'Mobile Reviews Sentiment null.csv'

# Load raw CSV
df = pd.read_csv(csv_path)

# Clean date column formatting before loading to Postgres
df['review_date'] = pd.to_datetime(df['review_date']).dt.date

# Connect and write to PostgreSQL
engine = get_engine()
df.to_sql('raw_mobile_reviews', engine, if_exists='replace', index=False)
print("Data ingestion complete!")