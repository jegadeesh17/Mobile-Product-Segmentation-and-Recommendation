# 🎯 Mobile Product Segmentation & Recommendation System

An end-to-end data science and unsupervised machine learning pipeline designed to clean raw smartphone customer reviews, group devices into strategic market clusters, and recommend similar high-performing devices based on product specifications and consumer sentiment.

## 📖 Project Overview
The smartphone industry generates high volumes of reviews, ratings, prices, and specifications. Analyzing market trends or recommending devices using raw text or raw averages directly is difficult due to missing values, unorganized specifications, and clustering instability when dealing with multi-scale features like price vs. ratings.

This project implements a complete PostgreSQL and Python pipeline to impute missing data, calculate aggregated quality features, segment devices using K-Means clustering, and power a recommendation engine based on Cosine Similarity.

## 🏗️ Architecture & Pipeline

1. **Data Ingestion & Cleaning**: A Python pipeline that loads raw CSV review records into PostgreSQL. Missing prices and ratings are resolved via group-level median/mean transformations, and a rule-based imputation system resolves missing review sentiments.
2. **Feature Engineering**: Calculates a composite specifications score by averaging individual hardware ratings and aggregates raw reviews into distinct product-level profiles representing pricing, overall ratings, and positive sentiment ratios.
3. **Hybrid K-Means Clustering**: Standardizes quality features (rating, sentiment, specs) to train a K-Means model that classifies smartphones into 'High Quality' and 'Low Quality' tiers. A hybrid logic block then maps devices into 4 distinct commercial personas: Budget Workhorses, Underperformers, Premium Flagships, and Mid-Range Value.
4. **Cosine Similarity Recommendation Engine**: A recommendation pipeline utilizing Cosine Similarity on scaled product profiles. It incorporates custom filtering to suggest high-quality alternative devices within the same price bracket.
5. **Streamlit UI**: An interactive web app featuring Plotly market segmentation scatter plots, a price-tier budget explorer, and a live similarity recommendation engine.

## 🚀 How to Run

1. **Verify Database Configuration**: Ensure PostgreSQL contains the `product_segmentation` database.
2. **Execute Ingestion & ML Pipeline**:
   ```bash
   python src/data_ingestion.py
   python src/data_cleaning.py
   python src/feature_engineering.py
   python src/model_training.py
   ```
3. **Run the Streamlit Dashboard**:
   ```bash
   python -m streamlit run app/app.py
   ```
