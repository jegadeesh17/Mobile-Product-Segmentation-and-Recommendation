# Mobile Product Segmentation & Recommendation System

---

### **Project Overview**

The smartphone industry generates high volumes of reviews, ratings, prices, and specifications across thousands of device models. Analyzing market positioning or recommending devices from raw data is difficult due to missing values, unstructured specifications, and clustering instability when features span vastly different scales.

This project builds an end-to-end data science and unsupervised machine learning platform to clean raw smartphone customer reviews, segment devices into strategic market clusters, and power a cosine similarity-based recommendation engine — all served through an interactive Streamlit dashboard.

---

### **Key Features**

* **Automated Data Cleaning:** Group-level median/mean imputation for missing prices and ratings with rule-based sentiment resolution.
* **Composite Specification Scoring:** Aggregates individual hardware ratings into a unified product quality metric.
* **Hybrid K-Means Clustering:** Classifies smartphones into 4 commercial personas — Budget Workhorses, Underperformers, Premium Flagships, and Mid-Range Value.
* **Cosine Similarity Recommendation Engine:** Recommends high-quality alternative devices within the same price bracket.
* **Market Segmentation Visualization:** Plotly scatter plots showing cluster distributions across quality and price dimensions.
* **Price-Tier Budget Explorer:** Interactive budget filter to explore devices within a specified price range.
* **Live Recommendation Dashboard:** Real-time similarity-based device suggestions through the Streamlit UI.
* **PostgreSQL Backend:** Persistent storage for cleaned product profiles, enabling fast dashboard queries.

---

### **Dataset**

* **Source:** Global Mobile Reviews Dataset
* **In repo:** `raw_mobile_reviews_sample.csv`, `cleaned_mobile_reviews_sample.csv`, plus feature/segment CSVs
* **Full data:** Place `raw_mobile_reviews.csv` in `data/` — see [data/DATA_SETUP.md](data/DATA_SETUP.md)
* **Coverage:** Multi-brand smartphone products with customer reviews and specifications
* **Format:** Raw CSV review records ingested into PostgreSQL

#### **Key Features**

* Device model name and brand
* Customer ratings and review sentiment
* Selling price and specifications
* Individual hardware component ratings (camera, battery, display, performance)
* Review count per device

---

### **Project Structure**

```bash
MobileProductSegmentation/
│
├── app/                          # Streamlit application files
│   └── app.py                    # Main Streamlit dashboard
├── data/                         # Project datasets
├── docs/                         # Documentation and visualizations
├── models/                       # Saved trained models
├── notebooks/                    # Jupyter notebooks (Source of Truth)
├── src/                          # Core Python logic and scripts
├── requirements.txt              # Python dependencies
└── README.md
```

---

### **How It Works**

### **1. Data Cleaning & Imputation**

* Loads raw CSV review records into PostgreSQL
* Resolves missing prices and ratings using group-level statistical imputation
* Applies rule-based logic to fill missing review sentiments

| Step                   | Operation                                       |
| ---------------------- | ----------------------------------------------- |
| Median Price Imputation| Fills missing prices by brand-tier group        |
| Mean Rating Imputation | Fills missing ratings by device category        |
| Sentiment Resolution   | Rule-based fill for missing positive/negative   |
| Duplicate Removal      | Drops repeated device-review pairs              |

---

### **2. Feature Engineering**

Creates a composite product-level profile for each device:

| Feature               | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| `spec_score`          | Average of all hardware component ratings          |
| `positive_ratio`      | Proportion of positive reviews per device          |
| `price_normalized`    | Scaled price for clustering stability              |
| `rating_aggregated`   | Product-level mean rating across all reviews       |

---

### **3. Hybrid K-Means Clustering**

Uses K-Means on standardized quality features to group devices, then applies a hybrid logic block to assign commercial personas:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['rating', 'positive_ratio', 'spec_score']])

kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
```

| Persona              | Description                                     |
| -------------------- | ----------------------------------------------- |
| Premium Flagships    | High quality, high price                        |
| Mid-Range Value      | Good quality, moderate price                    |
| Budget Workhorses    | Adequate quality, low price                     |
| Underperformers      | Low quality relative to price                   |

---

### **4. Cosine Similarity Recommendation Engine**

Recommends similar devices based on scaled product profiles, filtering within the same price bracket:

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(X_scaled)
```

---

### **Model Performance**

| Metric                     | Result                                    |
| -------------------------- | ----------------------------------------- |
| Clustering Quality (Inertia)| Optimized via Elbow Method               |
| Recommendation Relevance   | High-quality devices within price bracket |
| Dashboard Query Speed      | Fast (PostgreSQL-backed)                  |

---

### **Interactive Application Deployment**

The project features an interactive **Streamlit Web Application** structured into four main modules:

* **Market Clusters Overview:** Interactive Plotly scatter plots and statistical summaries visualizing the K-Means cluster segmentation.
* **Device Segment Explorer:** A tool to browse and drill down into top-performing models within each ML-generated cluster.
* **Similar Device Finder:** A recommendation tool to select a brand and model and find the top 5 most similar device alternatives using cosine similarity.
* **Custom Recommendation Engine:** An intuitive matching engine where selecting a target brand (or "All") and a target price slider recommends the best quality options within the budget.

#### **To Launch the Platform Locally:**
```powershell
streamlit run app/app.py
```

---

### **Technology Stack**

| Category             | Tools                          |
| -------------------- | ------------------------------ |
| Programming          | Python                         |
| Data Processing      | Pandas, NumPy                  |
| Database             | PostgreSQL, SQLAlchemy         |
| Machine Learning     | Scikit-learn (KMeans, Cosine)  |
| Visualization        | Plotly                         |
| Notebook Environment | Jupyter Notebook               |
| Web Framework        | Streamlit                      |

---

### **Getting Started**

### **1. Clone Repository**

```bash
git clone https://github.com/jegadeesh17/Mobile-Product-Segmentation-and-Recommendation-System.git

cd MobileProductSegmentation
```

---

### **2. Configure Database**

Ensure PostgreSQL is running with the `product_segmentation` database. Update `.env` with your credentials:

```env
DB_HOST=localhost
DB_NAME=product_segmentation
DB_USER=your_user
DB_PASSWORD=your_password
```

---

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### **4. Run Ingestion & Training**

```bash
python src/data_ingestion.py
python src/data_cleaning.py
python src/feature_engineering.py
python src/model_training.py
```

---

### **5. Launch Dashboard**

```bash
python -m streamlit run app/app.py
```

---

### **Example Use Case**

A consumer electronics retailer or e-commerce platform can use this system to:

1. Segment their catalog into market tiers for strategic pricing
2. Identify underperforming devices receiving poor value perception
3. Recommend similar alternatives to customers based on their viewed product
4. Analyze which price tiers contain the highest-rated devices

---

### **Future Improvements**

* Real-time product review scraping and live clustering updates
* Deep learning-based sentiment analysis for richer feature signals
* Price elasticity modeling for revenue optimization
* A/B testing framework for recommendation engine variants

---

### **Contributors**

* **Jegadeesh D** — Data ingestion, cleaning, feature engineering, K-Means clustering, cosine similarity recommendation, and Streamlit dashboard development

---

### **License**

MIT License
