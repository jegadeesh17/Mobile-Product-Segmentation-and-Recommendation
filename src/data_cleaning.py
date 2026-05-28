import pandas as pd
import numpy as np
from sqlalchemy import create_engine

print("Running Data Cleaning...")

# 1. Fetch data from PostgreSQL
engine = create_engine('postgresql://postgres:jaundice@localhost:5432/product_segmentation')
df = pd.read_sql('SELECT * FROM raw_mobile_reviews', engine)

# 2. Fix missing price_usd using model/brand medians (fall back to global median if still missing)
df['price_usd'] = df.groupby(['brand', 'model'])['price_usd'].transform(lambda x: x.fillna(x.median()))
df['price_usd'] = df['price_usd'].fillna(df['price_usd'].median())

# 3. Fix missing ratings using model averages (fall back to global mean if still missing)
df['rating'] = df.groupby('model')['rating'].transform(lambda x: x.fillna(x.mean()))
df['rating'] = df['rating'].fillna(df['rating'].mean())

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