# Customer Segmentation (RFM + KMeans)

*Summary:* Segment customers with RFM analysis and KMeans clustering, then provide targeted marketing recommendations per segment.

*Dataset:* Superstore / Online Retail  
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

*Files*
- notebook.ipynb
- code.py
- dataset_link.txt
- insights.txt
- interview_questions.md

code.py
# Customer Segmentation + LLM Recommendations
# ==========================

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# --------------------------
# 1️⃣ Load Dataset
# --------------------------
df = pd.read_csv("your_dataset.csv")  # Replace with your CSV
print("Data shape:", df.shape)
print(df.head())

# --------------------------
# 2️⃣ Data Preprocessing
# --------------------------
# Example columns: 'CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice'

# Handle missing values
df.dropna(subset=['CustomerID', 'InvoiceDate'], inplace=True)
df['Quantity'].fillna(0, inplace=True)
df['UnitPrice'].fillna(df['UnitPrice'].median(), inplace=True)

# Ensure InvoiceDate is datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# --------------------------
# 3️⃣ Feature Engineering (RFM)
# --------------------------
today_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (today_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',                                   # Frequency
    'Quantity': lambda x: (x * df.loc[x.index, 'UnitPrice']).sum()  # Monetary
}).reset_index()

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Quantity': 'Monetary'
}, inplace=True)

# Log transform Monetary (optional, reduces skew)
rfm['Monetary'] = np.log1p(rfm['Monetary'])

# --------------------------
# 4️⃣ Feature Scaling
# --------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

# --------------------------
# 5️⃣ K-Means Clustering
# --------------------------
# Decide clusters using elbow method (optional)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# --------------------------
# 6️⃣ Cluster Analysis
# --------------------------
cluster_summary = rfm.groupby('Cluster').mean().round(1)
print("\nCluster Summary:\n", cluster_summary)

# Visualize clusters (2D)
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm, palette='Set2')
plt.title("Customer Segmentation Clusters")
plt.show()

# --------------------------
# 7️⃣ AI / LLM Integration
# --------------------------
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API Key

def get_ai_recommendation(prompt_text):
    """
    Returns AI recommendation for a given customer segment
    """
    response = openai.Completion.create(
        engine="text-davinci-003",  # or "gpt-4o"
        prompt=prompt_text,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Generate recommendations per cluster
for cluster_id in rfm['Cluster'].unique():
    cluster_data = cluster_summary.loc[cluster_id].to_dict()
    prompt = f"Customer Segment {cluster_id} has features: {cluster_data}. Suggest marketing strategies and engagement ideas."
    print(f"\n--- Cluster {cluster_id} AI Recommendations ---")
    print(get_ai_recommendation(prompt))


    dataset_link.txt
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

insights.txt
- Typical segments: high-LTV, loyal frequent buyers, occasional buyers, at-risk customers.
- Action: Upsell to high-LTV, re-engage at-risk with offers.

interview_questions.md
Q1: How choose number of clusters?  
A: Use elbow method and silhouette score; choose the number that balances interpretability and cohesion.

Q2: How to validate clusters?  
A: Check segment behavior, LTV, conversion rates, and align with business
