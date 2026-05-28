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
with open(PROJECT_ROOT / 'scaler.pkl', 'rb') as f:
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
    target_price_tier = target_row['price_tier'].values[0]
    
    # Get similarity scores for all phones with this phone
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Filter candidates:
    # 1. Exclude the selected phone itself.
    # 2. Prioritize High Quality models in the same price tier.
    filtered_sim_scores = []
    for other_idx, score in sim_scores:
        if other_idx != idx:
            other_row = product_df.loc[other_idx]
            if other_row['quality_label'] == 'High Quality' and other_row['price_tier'] == target_price_tier:
                filtered_sim_scores.append((other_idx, score))
    
    # Fallback: if we don't have enough recommendations in the same price tier,
    # pull other High Quality devices from adjacent/other price tiers.
    if len(filtered_sim_scores) < top_n:
        additional_scores = []
        for other_idx, score in sim_scores:
            if other_idx != idx and (other_idx, score) not in filtered_sim_scores:
                other_row = product_df.loc[other_idx]
                if other_row['quality_label'] == 'High Quality':
                    additional_scores.append((other_idx, score))
        
        # Sort additional recommendations by similarity score and append
        additional_scores = sorted(additional_scores, key=lambda x: x[1], reverse=True)
        filtered_sim_scores.extend(additional_scores[:(top_n - len(filtered_sim_scores))])
    
    # Sort by similarity score in descending order
    filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Extract phone details
    recommended_indices = [item[0] for item in filtered_sim_scores]
    
    if not recommended_indices:
        return pd.DataFrame(columns=['brand', 'model', 'avg_price', 'avg_rating', 'price_tier', 'cluster_name'])
        
    res_df = product_df.iloc[recommended_indices][['brand', 'model', 'avg_price', 'avg_rating', 'price_tier', 'cluster_name']].copy()
    
    # Custom sort order: Premium first, then Mid-Range, then Budget
    price_tier_order = {'Premium': 0, 'Mid-Range': 1, 'Budget': 2}
    res_df['price_tier_sort'] = res_df['price_tier'].map(price_tier_order)
    
    # Sort by price tier (ascending order of sort index) and then user rating (descending)
    res_df = res_df.sort_values(by=['price_tier_sort', 'avg_rating'], ascending=[True, False])
    res_df = res_df.drop(columns=['price_tier_sort'])
    
    return res_df