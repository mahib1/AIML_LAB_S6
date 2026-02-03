import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Create img directory for output
if not os.path.exists('img'):
    os.makedirs('img')

# Load the MGNREGA dataset
df = pd.read_csv('data/MGNREGA_dataset_AtAGlance.csv')

# Selecting features for clustering
features = [
    'average_days_employment_per_household', 
    'total_expenditure', 
    'percentage_expenditure_on_agriculture_and_allied_works'
]
X = df[features].dropna() # Ensure no missing values

# Standardize the features (K-means is distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

maxClu = 20
# 1. Elbow Method to find optimal K
wcss = []
for i in range(1, maxClu, 2):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, maxClu, 2), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.savefig('img/elbow_method.png')
plt.close()

# 2. Applying K-Means (Optimal K chosen as 4 based on results)
k = 10
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add results back to dataframe
df_clustered = df.loc[X.index].copy()
df_clustered['Cluster'] = clusters

# 3. Visualization: Scatter plot of Employment vs Expenditure
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_clustered, 
    x='average_days_employment_per_household', 
    y='total_expenditure', 
    hue='Cluster', 
    palette='viridis', 
    s=100, 
    alpha=0.7
)
plt.title('Districts Clustered by MGNREGA Performance')
plt.xlabel('Avg Days Employment / HH')
plt.ylabel('Total Expenditure (Lakhs)')
plt.savefig('img/cluster_visualization.png')
plt.close()

# 4. Pairplot for detailed distribution analysis
sns.pairplot(df_clustered[features + ['Cluster']], hue='Cluster', palette='viridis')
plt.savefig('img/pairplot_clusters.png')
plt.close()

# Save the resulting data
df_clustered.to_csv('mgnrega_clustered_districts.csv', index=False)
