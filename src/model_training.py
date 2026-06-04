import pandas as pd
import pickle
from db_config import get_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path

print("Running Model Training...")

# 1. Fetch product features from PostgreSQL (fallback to CSV)
try:
    engine = get_engine()
    product_df = pd.read_sql('SELECT * FROM product_features', engine)
except Exception as e:
    print(f"Database connection failed: {e}. Falling back to CSV.")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    product_df = pd.read_csv(PROJECT_ROOT / 'data' / 'segmented_products.csv')
# 2. Select features for segmentation
features = ['avg_price', 'avg_rating', 'avg_specs_score', 'positive_sentiment_ratio']
X = product_df[features]

# 3. Standardize all 4 features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find the optimal number of clusters using Silhouette Score and Elbow Method
print("Evaluating optimal number of clusters (Elbow & Silhouette)...")
best_k = 4
best_score = -1

wcss = []
silhouette_scores = []
k_range = range(3, 11)

for k in k_range:
    temp_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    temp_labels = temp_kmeans.fit_predict(X_scaled)
    
    # WCSS (Inertia) for Elbow
    wcss.append(temp_kmeans.inertia_)
    
    # Silhouette
    score = silhouette_score(X_scaled, temp_labels)
    silhouette_scores.append(score)
    print(f"  k={k}, WCSS: {temp_kmeans.inertia_:.2f}, Silhouette: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"Optimal number of clusters chosen: {best_k} (Silhouette Score: {best_score:.4f})")

# Generate Plot
PROJECT_ROOT = Path(__file__).resolve().parent.parent
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('WCSS (Elbow Method)', color=color)
ax1.plot(k_range, wcss, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_range, silhouette_scores, marker='s', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Highlight best K
ax2.axvline(x=best_k, color='green', linestyle='--', label=f'Best k={best_k}')
fig.tight_layout()
plt.title('Optimal K Analysis: Elbow Method vs Silhouette Score')
plt.savefig(PROJECT_ROOT / 'reports' / 'figures' / 'optimal_k_analysis.png', bbox_inches='tight')
plt.close()

# 5. Fit Final KMeans with optimal K
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
product_df['cluster'] = kmeans.fit_predict(X_scaled)

# 6. Dynamically Name Clusters based on Centroids
# We will compute the mean price and specs for each cluster to label them
cluster_profiles = product_df.groupby('cluster')[features].mean().reset_index()

def assign_dynamic_name(row):
    # Basic heuristic based on percentiles of the cluster means
    # For a truly dynamic naming, we rank the clusters by price and specs
    return f"Segment_{int(row['cluster'])}"

# Let's do a relative ranking of clusters
cluster_profiles['price_rank'] = cluster_profiles['avg_price'].rank()
cluster_profiles['spec_rank'] = cluster_profiles['avg_specs_score'].rank()

cluster_names = {}
for _, row in cluster_profiles.iterrows():
    c = int(row['cluster'])
    pr = row['price_rank']
    sr = row['spec_rank']
    
    # Simple heuristic to name the segments
    if pr >= best_k * 0.75:
        name = "Premium Flagships"
    elif pr <= best_k * 0.35 and sr <= best_k * 0.5:
        name = "Budget Workhorses"
    elif pr <= best_k * 0.5 and sr > best_k * 0.5:
        name = "High-Value Mid-Range"
    else:
        name = "Standard Mid-Range"
        
    # Ensure unique names if collisions occur
    if name in cluster_names.values():
        name = f"{name} (Tier {c})"
        
    cluster_names[c] = name

product_df['cluster_name'] = product_df['cluster'].map(cluster_names)

# Show cluster profiles with business names
final_profiles = product_df.groupby('cluster_name')[features].mean()
print("\nCluster Profiles (Mean values of features per cluster):")
print(final_profiles)
    
# 6. Save models/artifacts to disk using pickle
# Get the project root folder to save models and CSV
PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / 'models' / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(PROJECT_ROOT / 'models' / 'kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# 7. Save clustered product data back to PostgreSQL and a CSV in the data folder
try:
    product_df.to_sql('segmented_products', engine, if_exists='replace', index=False)
except Exception as e:
    print(f"Warning: Could not save to database ({e}).")

product_df.to_csv(PROJECT_ROOT / 'data' / 'segmented_products.csv', index=False)

print("Model training complete! Scaler and KMeans models saved. Segmented products saved to database and CSV.")