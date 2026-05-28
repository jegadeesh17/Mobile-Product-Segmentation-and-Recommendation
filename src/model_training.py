import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

print("Running Model Training...")

# 1. Fetch product features from PostgreSQL
engine = create_engine('postgresql://postgres:jaundice@localhost:5432/product_segmentation')
product_df = pd.read_sql('SELECT * FROM product_features', engine)

# 2. Select features for segmentation
features = ['avg_price', 'avg_rating', 'avg_specs_score', 'positive_sentiment_ratio']
X = product_df[features]

# 3. Create two scalers: one for recommendations (4 features) and one for quality (3 features)
scaler = StandardScaler()
scaler.fit(X)  # Fit on all 4 features (for compatibility with recommendation similarity)

# Quality scaler (3 features: rating, specs, sentiment ratio)
quality_features = ['avg_rating', 'avg_specs_score', 'positive_sentiment_ratio']
quality_scaler = StandardScaler()
X_quality_scaled = quality_scaler.fit_transform(product_df[quality_features])

# 4. Fit 2-cluster KMeans for quality segmentation (High vs Low Quality)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
product_df['quality_cluster'] = kmeans.fit_predict(X_quality_scaled)

# Identify which cluster index is High Quality vs Low Quality
cluster_0_mean = product_df[product_df['quality_cluster'] == 0]['avg_rating'].mean()
cluster_1_mean = product_df[product_df['quality_cluster'] == 1]['avg_rating'].mean()
high_quality_cluster = 0 if cluster_0_mean > cluster_1_mean else 1

product_df['quality_label'] = product_df['quality_cluster'].apply(
    lambda x: 'High Quality' if x == high_quality_cluster else 'Low Quality'
)

# 5. Classify price tiers
def get_price_tier(price):
    if price >= 1000:
        return 'Premium'
    elif price >= 500:
        return 'Mid-Range'
    else:
        return 'Budget'

product_df['price_tier'] = product_df['avg_price'].apply(get_price_tier)

# Apply Hybrid Logic for final Business Persona Mapping
def get_hybrid_cluster_name(row):
    if row['quality_label'] == 'Low Quality':
        return 'Underperformers'
    else:
        if row['price_tier'] == 'Premium':
            return 'Premium Flagships'
        elif row['price_tier'] == 'Mid-Range':
            return 'Mid-Range Value'
        else:
            return 'Budget Workhorses'

product_df['cluster_name'] = product_df.apply(get_hybrid_cluster_name, axis=1)

# Maintain numerical 'cluster' mapping for reverse compatibility:
# 0: Budget Workhorses, 1: Underperformers, 2: Premium Flagships, 3: Mid-Range Value
CLUSTER_INT_MAPPING = {
    'Budget Workhorses': 0,
    'Underperformers': 1,
    'Premium Flagships': 2,
    'Mid-Range Value': 3
}
product_df['cluster'] = product_df['cluster_name'].map(CLUSTER_INT_MAPPING)

# Clean up helper columns before saving to keep database clean
product_df = product_df.drop(columns=['quality_cluster'])

# Show cluster profiles with business names
cluster_profiles = product_df.groupby('cluster_name')[features].mean()
print("Cluster Profiles (Mean values of features per cluster):")
print(cluster_profiles)

# 6. Save models/artifacts to disk using pickle
# Get the project root folder to save models and CSV
PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(PROJECT_ROOT / 'kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# 7. Save clustered product data back to PostgreSQL and a CSV in the data folder
product_df.to_sql('segmented_products', engine, if_exists='replace', index=False)
product_df.to_csv(PROJECT_ROOT / 'data' / 'segmented_products.csv', index=False)

print("Model training complete! Scaler and KMeans models saved. Segmented products saved to database and CSV.")