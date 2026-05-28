import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

print("Running Exploratory Data Analysis (EDA)...")

# 1. Fetch cleaned data from PostgreSQL
engine = create_engine('postgresql://postgres:jaundice@localhost:5432/product_segmentation')
df = pd.read_sql('SELECT * FROM cleaned_mobile_reviews', engine)

# 2. Correlation Heatmap for Specification Features
spec_cols = ['price_usd', 'rating', 'battery_life_rating', 'camera_rating', 'performance_rating', 'design_rating', 'display_rating']
plt.figure(figsize=(10, 6))
sns.heatmap(df[spec_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Specifications and Price')
plt.savefig('specifications_correlation.png', bbox_inches='tight')
plt.close()

print("EDA complete! Saved correlation heatmap to 'specifications_correlation.png'.")