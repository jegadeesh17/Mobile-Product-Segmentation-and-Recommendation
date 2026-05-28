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

from recommendation_engine import get_recommendations

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
    ["Market Segmentation", "Price-Tier Budget Explorer", "Similar Device Alternatives"]
)

if menu == "Market Segmentation":
    st.subheader("📊 Strategic Product Clusters")
    # Interactive visualization using Plotly
    fig = px.scatter(product_df, x="avg_price", y="avg_rating", color="cluster_name",
                     hover_data=["brand", "model"], title="Price vs Rating Segmented by Clusters",
                     color_discrete_map={
                         'Budget Workhorses': '#1f77b4',
                         'Underperformers': '#d62728',
                         'Premium Flagships': '#2ca02c',
                         'Mid-Range Value': '#ff7f0e'
                     })
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### Cluster Insights Table")
    st.dataframe(product_df.groupby('cluster_name').mean(numeric_only=True))

elif menu == "Price-Tier Budget Explorer":
    st.subheader("🔍 Price-Tier Budget Explorer")
    st.write("Find the best-performing mobile devices grouped by pricing brackets.")
    
    budget_choice = st.selectbox(
        "Select your Budget Tier:", 
        ["Premium", "Mid-Range", "Budget"],
        index=1
    )
    
    # Display Price Tier Mark / Reference
    if budget_choice == "Premium":
        st.info("💡 **Premium Range Mark**: $1,000 and above")
    elif budget_choice == "Mid-Range":
        st.info("💡 **Mid-Range Range Mark**: $500 to $1,000")
    else:
        st.info("💡 **Budget Range Mark**: Under $500")
        
    # Filter for High Quality phones in the chosen tier
    explorer_df = product_df[
        (product_df['price_tier'] == budget_choice) & 
        (product_df['quality_label'] == 'High Quality')
    ].copy()
    
    # Format and present results
    if not explorer_df.empty:
        # Sort by rating and specifications score for quality sorting
        explorer_df = explorer_df.sort_values(by=['avg_rating', 'avg_specs_score'], ascending=False)
        
        # Format metrics for clean display
        display_df = explorer_df[['brand', 'model', 'avg_price', 'avg_rating', 'avg_specs_score', 'positive_sentiment_ratio', 'cluster_name']].copy()
        display_df['avg_price'] = display_df['avg_price'].map(lambda x: f"${x:,.2f}")
        display_df['avg_rating'] = display_df['avg_rating'].map(lambda x: f"{x:.2f} / 5.0")
        display_df['avg_specs_score'] = display_df['avg_specs_score'].map(lambda x: f"{x:.2f} / 3.0")
        display_df['positive_sentiment_ratio'] = display_df['positive_sentiment_ratio'].map(lambda x: f"{x*100:.1f}%")
        
        display_df.columns = ['Brand', 'Model', 'Average Price', 'User Rating', 'Specifications Score', 'Positive Sentiment %', 'Segment']
        
        st.write(f"### Top Performing Devices in the **{budget_choice}** segment:")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning(f"No high-performing devices found in the {budget_choice} segment.")

elif menu == "Similar Device Alternatives":
    st.subheader("🎯 Intelligent Device Alternatives")
    st.write("Select a phone model to find high-performing alternatives within the same budget tier.")
    
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
            display_recs = display_recs.drop(columns=['price_tier'])
            display_recs['avg_price'] = display_recs['avg_price'].map(lambda x: f"${x:,.2f}")
            display_recs['avg_rating'] = display_recs['avg_rating'].map(lambda x: f"{x:.2f} / 5.0")
            display_recs.columns = ['Brand', 'Model', 'Average Price', 'User Rating', 'Segment']
            
            st.dataframe(display_recs, use_container_width=True)
        else:
            st.error("No high-performing recommendations could be found for this device.")