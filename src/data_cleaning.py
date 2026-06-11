import pandas as pd
import numpy as np
from db_config import get_engine

print("Running Data Cleaning...")

# 1. Fetch data from PostgreSQL
engine = get_engine()
df = pd.read_sql('SELECT * FROM raw_mobile_reviews', engine)

# 2. Drop rows with missing critical information (price and rating)
if 'rating' in df.columns and 'price_usd' in df.columns:
    df = df.dropna(subset=['rating', 'price_usd'])

# 4. Impute Missing Sentiment conditionally based on Rating
def impute_sentiment(row):
    if pd.isna(row['sentiment']):
        if row['rating'] >= 4.0: return 'Positive'
        elif row['rating'] == 3.0: return 'Neutral'
        else: return 'Negative'
    return row['sentiment']

df['sentiment'] = df.apply(impute_sentiment, axis=1)

# 5. Drop duplicate reviews if any exist
df.drop_duplicates(subset=['review_id'], inplace=True)

# 6. Save clean data back to PostgreSQL
df.to_sql('cleaned_mobile_reviews', engine, if_exists='replace', index=False)
print(f"Data cleaning complete! Cleaned reviews saved to 'cleaned_mobile_reviews' table ({len(df)} rows).")