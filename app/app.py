import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import sys
from pathlib import Path

# Configure import paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.append(str(PROJECT_ROOT / 'src'))

from recommendation_engine import get_recommendations, get_custom_recommendations

st.set_page_config(layout="wide", page_title="Telecom Device Intelligence App")
st.title("📱 Mobile Product Segmentation & Recommendation Engine")

# Load your precomputed aggregated product_df (fallback to local CSV if DB is down)
try:
    from db_config import get_engine
    engine = get_engine()
    product_df = pd.read_sql('SELECT * FROM segmented_products', engine)
except Exception:
    product_df = pd.read_csv(PROJECT_ROOT / 'data' / 'segmented_products.csv')

menu = st.sidebar.radio(
    "Navigate Application", 
    ["Market Segmentation", "Segment Explorer", "Similar Device Alternatives"]
)

if menu == "Market Segmentation":
    st.subheader("📊 Strategic Product Clusters")
    # Interactive visualization using Plotly
    fig = px.scatter(product_df, x="avg_price", y="avg_rating", color="cluster_name",
                     hover_data=["brand", "model"], title="Price vs Rating Segmented by ML Clusters")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### Cluster Insights Table")
    # Compute mean and format it elegantly using st.column_config
    cluster_stats = product_df.groupby('cluster_name').mean(numeric_only=True).reset_index()
    
    st.dataframe(
        cluster_stats,
        column_config={
            "cluster_name": st.column_config.TextColumn(
                "Segment Name", 
                width="medium"
            ),
            "avg_price": st.column_config.NumberColumn(
                "Average Price",
                help="Average price of phones in USD",
                format="$ %.2f",
            ),
            "avg_rating": st.column_config.NumberColumn(
                "Average Rating",
                help="Average user rating (out of 5.0)",
                format="%.2f ⭐",
            ),
            "avg_specs_score": st.column_config.NumberColumn(
                "Avg Specs Score",
                help="Average specification score (out of 3.0)",
                format="%.2f",
            ),
            "total_reviews": st.column_config.NumberColumn(
                "Total Reviews",
                help="Average number of reviews",
                format="%d",
            ),
            "positive_sentiment_ratio": st.column_config.NumberColumn(
                "Positive Sentiment",
                help="Ratio of positive reviews",
                format="%.2f",
            ),
            "cluster": None, # Hide the internal cluster ID column
        },
        hide_index=True,
        use_container_width=True
    )

elif menu == "Segment Explorer":
    st.subheader("🔍 ML-Driven Segment Explorer")
    st.write("Explore the automatically generated ML segments to find top devices.")
    
    available_clusters = product_df['cluster_name'].unique()
    cluster_choice = st.selectbox(
        "Select an ML Segment:", 
        available_clusters
    )
    
    st.info(f"💡 Viewing devices classified by K-Means into: **{cluster_choice}**")
        
    # Filter for the chosen cluster
    explorer_df = product_df[product_df['cluster_name'] == cluster_choice].copy()
    
    # Format and present results
    if not explorer_df.empty:
        # Sort by rating and specifications score to show best first
        explorer_df = explorer_df.sort_values(by=['avg_rating', 'avg_specs_score'], ascending=False)
        
        # Format metrics for clean display
        display_df = explorer_df[['brand', 'model', 'avg_price', 'avg_rating', 'avg_specs_score', 'positive_sentiment_ratio', 'cluster_name']].copy()
        display_df['avg_price'] = display_df['avg_price'].map(lambda x: f"${x:,.2f}")
        display_df['avg_rating'] = display_df['avg_rating'].map(lambda x: f"{x:.2f} / 5.0")
        display_df['avg_specs_score'] = display_df['avg_specs_score'].map(lambda x: f"{x:.2f} / 3.0")
        display_df['positive_sentiment_ratio'] = display_df['positive_sentiment_ratio'].map(lambda x: f"{x*100:.1f}%")
        
        display_df.columns = ['Brand', 'Model', 'Average Price', 'User Rating', 'Specifications Score', 'Positive Sentiment %', 'Segment']
        
        st.write(f"### Top Performing Devices in the **{cluster_choice}** segment:")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning(f"No devices found in the {cluster_choice} segment.")

elif menu == "Similar Device Alternatives":
    st.subheader("🎯 Intelligent Device Alternatives")
    st.write("Find high-performing alternatives within your preferred budget tier or by matching a specific model.")
    
    tab1, tab2 = st.tabs(["By Phone Model", "By Custom Features"])
    
    with tab1:
        st.write("Select a phone model to find high-performing alternatives.")
        
        selected_phone = st.selectbox("Select a Smartphone Model:", product_df['model'].unique())
        
        # Show characteristics of the selected phone
        selected_info = product_df[product_df['model'] == selected_phone].iloc[0]
        st.markdown(
            f"**Selected Model details:** Price: `${selected_info['avg_price']:,.2f}` | "
            f"Rating: `{selected_info['avg_rating']:.2f}` | "
            f"Specs: `{selected_info['avg_specs_score']:.2f}` | "
            f"Segment: **{selected_info['cluster_name']}**"
        )
        
        if st.button("Generate Tailored Recommendations"):
            recommendations = get_recommendations(selected_phone)
            if recommendations is not None and not recommendations.empty:
                st.write("### Recommended Alternatives:")
                
                # Format display df
                display_recs = recommendations.copy()
                display_recs['avg_price'] = display_recs['avg_price'].map(lambda x: f"${x:,.2f}")
                display_recs['avg_rating'] = display_recs['avg_rating'].map(lambda x: f"{x:.2f} / 5.0")
                display_recs.columns = ['Brand', 'Model', 'Average Price', 'User Rating', 'Segment']
                
                st.dataframe(display_recs, use_container_width=True)
            else:
                st.error("No recommendations could be found for this device.")

    with tab2:
        st.write("Define your ideal phone specs to find the closest matches.")
        
        col1, col2 = st.columns(2)
        with col1:
            user_price = st.slider("Target Price ($)", 
                                   min_value=float(product_df['avg_price'].min()), 
                                   max_value=float(product_df['avg_price'].max()), 
                                   value=float(product_df['avg_price'].mean()))
            user_rating = st.slider("Target Rating (out of 5)", 
                                    min_value=float(product_df['avg_rating'].min()), 
                                    max_value=float(product_df['avg_rating'].max()), 
                                    value=float(product_df['avg_rating'].mean()))
        with col2:
            user_specs = st.slider("Target Specs Score (out of 3)", 
                                   min_value=float(product_df['avg_specs_score'].min()), 
                                   max_value=float(product_df['avg_specs_score'].max()), 
                                   value=float(product_df['avg_specs_score'].mean()))
            user_sentiment = st.slider("Target Positive Sentiment (%)", 
                                       min_value=float(product_df['positive_sentiment_ratio'].min()), 
                                       max_value=float(product_df['positive_sentiment_ratio'].max()), 
                                       value=float(product_df['positive_sentiment_ratio'].mean()))
            
        if st.button("Generate Custom Recommendations"):
            recommendations = get_custom_recommendations(user_price, user_rating, user_specs, user_sentiment)
            if recommendations is not None and not recommendations.empty:
                st.write("### Recommended Alternatives:")
                
                # Format display df
                display_recs = recommendations.copy()
                display_recs['avg_price'] = display_recs['avg_price'].map(lambda x: f"${x:,.2f}")
                display_recs['avg_rating'] = display_recs['avg_rating'].map(lambda x: f"{x:.2f} / 5.0")
                display_recs.columns = ['Brand', 'Model', 'Average Price', 'User Rating', 'Segment']
                
                st.dataframe(display_recs, use_container_width=True)
            else:
                st.error("No recommendations could be found.")