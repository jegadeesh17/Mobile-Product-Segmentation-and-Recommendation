import pandas as pd
from sqlalchemy import create_engine

print("Running Feature Engineering...")

# 1. Fetch cleaned data from PostgreSQL
engine = create_engine('postgresql://postgres:jaundice@localhost:5432/product_segmentation')
df = pd.read_sql('SELECT * FROM cleaned_mobile_reviews', engine)

# 2. Create a comprehensive composite specifications score
df['specs_average'] = df[['battery_life_rating', 'camera_rating', 'performance_rating', 'design_rating', 'display_rating']].mean(axis=1)

# 3. Group by brand and model to find product-level attributes
product_df = df.groupby(['brand', 'model']).agg(
    avg_price=('price_usd', 'mean'),
    avg_rating=('rating', 'mean'),
    avg_specs_score=('specs_average', 'mean'),
    total_reviews=('review_id', 'count'),
    positive_sentiment_ratio=('sentiment', lambda x: (x == 'Positive').sum() / len(x))
).reset_index()

# 4. Drop models with insufficient data if necessary (minimum 5 reviews)
product_df = product_df[product_df['total_reviews'] >= 5].reset_index(drop=True)

# 5. Save the aggregated product features back to PostgreSQL
product_df.to_sql('product_features', engine, if_exists='replace', index=False)

print(f"Feature engineering complete! Extracted features for {len(product_df)} unique phone models and saved to 'product_features' table.")