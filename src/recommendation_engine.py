import pandas as pd
import pickle
from db_config import get_engine
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Get the project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Fetch segmented products from PostgreSQL (fallback to CSV if DB is unavailable)
try:
    engine = get_engine()
    product_df = pd.read_sql('SELECT * FROM segmented_products', engine)
except Exception:
    # Fallback to local CSV
    product_df = pd.read_csv(PROJECT_ROOT / 'data' / 'segmented_products.csv')

# 2. Load the trained scaler to scale features
with open(PROJECT_ROOT / 'models' / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 3. Scale features and compute the pairwise similarity matrix
features = ['avg_price', 'avg_rating', 'avg_specs_score', 'positive_sentiment_ratio']
X = product_df[features]
X_scaled = scaler.transform(X)

similarity_matrix = cosine_similarity(X_scaled)

def get_recommendations(model_name, top_n=5):
    # Find index of target phone model
    if model_name not in product_df['model'].values:
        return None
    
    target_row = product_df[product_df['model'] == model_name]
    idx = target_row.index[0]
    target_cluster = target_row['cluster_name'].values[0]
    
    # Get similarity scores for all phones with this phone
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Filter candidates:
    # 1. Exclude the selected phone itself.
    # 2. Prioritize models in the same cluster.
    filtered_sim_scores = []
    for other_idx, score in sim_scores:
        if other_idx != idx:
            other_row = product_df.loc[other_idx]
            if other_row['cluster_name'] == target_cluster:
                filtered_sim_scores.append((other_idx, score))
    
    # Fallback: if we don't have enough recommendations in the same cluster,
    # pull other highly similar devices from other clusters.
    if len(filtered_sim_scores) < top_n:
        additional_scores = []
        for other_idx, score in sim_scores:
            if other_idx != idx and (other_idx, score) not in filtered_sim_scores:
                additional_scores.append((other_idx, score))
        
        # Sort additional recommendations by similarity score and append
        additional_scores = sorted(additional_scores, key=lambda x: x[1], reverse=True)
        filtered_sim_scores.extend(additional_scores[:(top_n - len(filtered_sim_scores))])
    
    # Sort by similarity score in descending order
    filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Extract phone details
    recommended_indices = [item[0] for item in filtered_sim_scores]
    
    if not recommended_indices:
        return pd.DataFrame(columns=['brand', 'model', 'avg_price', 'avg_rating', 'cluster_name'])
        
    res_df = product_df.iloc[recommended_indices][['brand', 'model', 'avg_price', 'avg_rating', 'cluster_name']].copy()
    
    # Sort by similarity score inherently handled by the recommendation ordering
    return res_df

import numpy as np

def get_custom_recommendations(price, rating, specs, sentiment, top_n=5):
    """
    Get recommendations based on custom user-defined values rather than an existing phone.
    """
    # Create input feature array (must match the order of `features` list)
    # features = ['avg_price', 'avg_rating', 'avg_specs_score', 'positive_sentiment_ratio']
    input_features = np.array([[price, rating, specs, sentiment]])
    input_scaled = scaler.transform(input_features)
    
    # Calculate cosine similarity with all existing scaled products
    sim_scores = cosine_similarity(input_scaled, X_scaled)[0]
    
    # Enumerate and sort by similarity score in descending order
    sim_scores_list = list(enumerate(sim_scores))
    sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Extract phone details
    recommended_indices = [item[0] for item in sim_scores_list]
    
    if not recommended_indices:
        return pd.DataFrame(columns=['brand', 'model', 'avg_price', 'avg_rating', 'cluster_name'])
        
    res_df = product_df.iloc[recommended_indices][['brand', 'model', 'avg_price', 'avg_rating', 'cluster_name']].copy()
    
    return res_df


def get_brand_price_recommendations(brand, target_price, top_n=5):
    """
    Get recommendations based on brand and price filters.
    Optimizes for the highest quality specs, rating, and sentiment matching the target price.
    """
    # Filter by brand first
    if brand != 'All':
        filtered_df = product_df[product_df['brand'] == brand].copy()
    else:
        filtered_df = product_df.copy()
        
    if filtered_df.empty:
        return filtered_df
        
    # We want the highest rating, spec score, and sentiment for the given price
    max_rating = product_df['avg_rating'].max()
    max_specs = product_df['avg_specs_score'].max()
    max_sentiment = product_df['positive_sentiment_ratio'].max()
    
    import numpy as np
    target_features = np.array([[target_price, max_rating, max_specs, max_sentiment]])
    target_scaled = scaler.transform(target_features)
    
    # Scale features of the filtered products
    X_filtered = filtered_df[features]
    X_filtered_scaled = scaler.transform(X_filtered)
    
    # Calculate similarities
    sim_scores = cosine_similarity(target_scaled, X_filtered_scaled)[0]
    
    # Sort and get top_n
    filtered_df['similarity'] = sim_scores
    res_df = filtered_df.sort_values(by='similarity', ascending=False).head(top_n)
    
    return res_df[['brand', 'model', 'avg_price', 'avg_rating', 'cluster_name']]