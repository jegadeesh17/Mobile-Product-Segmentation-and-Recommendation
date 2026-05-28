# Mobile Product Segmentation & Recommendation System

---

### **Project Overview**

The smartphone market is highly competitive and diverse, with hundreds of models catering to different user needs. This project builds an end-to-end Data Science pipeline designed to ingest smartphone customer reviews, clean and engineer features, segment products using K-Means clustering, and recommend similar devices based on specifications and consumer sentiment. 

The platform connects to a PostgreSQL database for reliable storage, runs machine learning pipelines for analysis, and serves results via an interactive Streamlit Dashboard containing market segmentation plots and a product recommendation engine.

---

### **Key Features**

* **PostgreSQL Ingestion Pipeline:** Automated loader to transfer raw CSV review data to a relational database.
* **Intelligent Data Cleaning:** Handles missing specification columns, drops duplicates, and imputes ratings.
* **Review Sentiment Aggregation:** Automatically calculates consumer sentiment ratios from raw reviews.
* **Market Segmentation (K-Means):** Groups devices into 4 distinct commercial categories.
* **Specs-Based Recommendation Engine:** Uses Cosine Similarity to find similar smartphone products.
* **Interactive Streamlit Web App:** Deployed dashboard for market segmentation visualization and recommendation search.

---

### **Dataset**

* **Source:** Mobile Reviews Sentiment Dataset
* **Coverage:** Detailed smartphone specifications and customer reviews
* **Data Type:** Relational text reviews and tabular specifications

#### **Key Features Analyzed**

* Device price and specifications (RAM, ROM, Battery, Camera, Screen Size)
* Customer ratings (1-5 stars)
* Review sentiment polarities (Positive, Neutral, Negative)
* Brand and model names

---

### **Project Structure**

```bash
Mobile-Product-Segmentation/
│
├── data/
│   ├── Mobile Reviews Sentiment null.csv     # Raw customer reviews dataset
│   └── segmented_products.csv                # Output containing product aggregated clusters
│
├── src/
│   ├── app.py                                # Streamlit dashboard application
│   ├── data_cleaning.py                      # Data cleaning and imputation script
│   ├── data_ingestion.py                     # Raw CSV to PostgreSQL database loader
│   ├── eda.py                                # Exploratory Data Analysis & visual correlation heatmap
│   ├── feature_engineering.py                # Review aggregation and custom metric creation
│   ├── model_training.py                     # K-Means clustering and persona profiles exporter
│   └── recommendation_engine.py              # Cosine similarity recommendation logic
│
├── specifications_correlation.png            # Correlation matrix plot
├── scaler.pkl                                # Standard Scaler model
├── kmeans.pkl                                # K-Means model
└── README.md
```

---

### **How It Works**

### **1. Data Ingestion & Cleaning**

* **`data_ingestion.py`**: Reads raw review records, formats date-times, and writes them to a PostgreSQL instance.
* **`data_cleaning.py`**: Fetches database records, handles missing specs, imputes price ranges, and removes invalid or duplicated entries.

---

### **2. Feature Engineering & Sentiment Aggregation**

* Aggregates fine-grained user reviews into macro product profiles.
* Generates rating statistics, specification scores, and positive sentiment ratios (`positive_reviews / total_reviews`).

---

### **3. K-Means Market Segmentation**

The model groups products into 4 clear commercial profiles based on price, performance, and feedback metrics:

| Segment | Profile Name | Description |
| --- | --- | --- |
| **Cluster 0** | Budget Workhorses | Low-price devices with moderate specs and high review volumes. |
| **Cluster 1** | Underperformers | Mid-to-high price points but low customer ratings and negative sentiment. |
| **Cluster 2** | Premium Flagships | High-price tier with stellar specifications and very high positive sentiment ratios. |
| **Cluster 3** | Mid-Range Value | Balanced price-to-performance ratio, offering decent specs at a mid-tier price. |

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)
```

---

### **4. Recommendation Engine**

Uses Cosine Similarity across price, specs, and rating scores to return the top 5 most similar devices for any selected phone model.

---

### **Database Configuration**

The system is configured to interface with a local **PostgreSQL** database:

* **Database Name:** `product_segmentation`
* **Username:** `postgres`
* **Password:** `jaundice`
* **Port:** `5432`

---

### **Interactive Application Deployment**

The project features a multi-tab **Streamlit Dashboard** allowing product managers and buyers to examine segment scatter plots and run product recommendations.

#### **To Launch the Platform Locally:**
```powershell
python -m streamlit run ".\Mobile Product Segmentation and Recommendation System\src\app.py"
```

---

### **Technology Stack**

| Category             | Tools                                         |
| -------------------- | --------------------------------------------- |
| Programming          | Python                                        |
| Data Processing      | Pandas, NumPy, SQL (PostgreSQL)               |
| Machine Learning     | Scikit-learn (K-Means, Cosine Similarity)     |
| Database Connection  | SQLAlchemy, Psycopg2                          |
| Visualization        | Matplotlib, Seaborn, Plotly                   |
| Web Framework        | Streamlit                                     |

---

### **Getting Started**

### **1. Setup Database**

Create a PostgreSQL database named `product_segmentation` and run your database server on port `5432` with username `postgres`.

---

### **2. Install Dependencies**

```bash
pip install pandas numpy scikit-learn psycopg2 sqlalchemy matplotlib seaborn streamlit plotly python-dotenv
```

---

### **3. Configure Environment Variables**

Create a `.env` file in the root of the project folder:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=product_segmentation
DB_USER=postgres
DB_PASSWORD=your_postgres_password
```

---

### **3. Execute Data & ML Pipeline**

Run the following scripts in order to load the data, clean it, build features, and train the model:

```bash
# Ingest raw CSV data
python src/data_ingestion.py

# Clean and impute data
python src/data_cleaning.py

# Run EDA
python src/eda.py

# Feature engineering
python src/feature_engineering.py

# Train K-Means clustering model
python src/model_training.py
```

---

### **Example Use Case**

E-commerce managers and retail analysts can use this system to:

1. Identify underperforming inventory units (Cluster 1) to apply discounts.
2. Recommend alternative devices to customers based on their specifications preference.
3. Classify newly introduced phone models into appropriate market tiers.

---

### **Future Improvements**

* Real-time scrape of web reviews for dynamic model updates.
* NLP-based aspect sentiment analysis (e.g. tracking specific issues with battery vs camera).
* Collaborative filtering incorporation for user-profile recommendations.

---

### **Contributors**

* **Jegadeesh D** — Database administration, data engineering, feature extraction, K-Means clustering, and dashboard development

---

### **License**

MIT License
