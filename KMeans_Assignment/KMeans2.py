import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Changed output directory to img2 as requested
output_dir = 'img2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the MGNREGA dataset
df = pd.read_csv('data/MGNREGA_dataset_AtAGlance.csv')

# Selecting features for clustering
features = [
    'average_days_employment_per_household', 
    'total_expenditure', 
    'percentage_expenditure_on_agriculture_and_allied_works'
]
X = df[features].dropna()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Finding Optimal K based on < 2% WCSS change ---
wcss = []
max_clusters = 20

# Initial WCSS for K=1
kmeans = KMeans(n_clusters=1, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled)
wcss.append(kmeans.inertia_)

optimal_k = 1
for k in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    current_wcss = kmeans.inertia_
    
    # Calculate percentage change
    percent_change = (wcss[-1] - current_wcss) / wcss[-1]
    wcss.append(current_wcss)
    
    if percent_change < 0.02:
        optimal_k = k - 1
        break
    else:
        optimal_k = k

# 1. Elbow Method Visualization
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(wcss) + 1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method (Stopped at <2% Change)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.savefig(f'{output_dir}/elbow_method.png') # Saved to img2
plt.close()

# 2. Applying K-Means with Optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = df.loc[X.index].copy()
df_clustered['Cluster'] = clusters

# 3. Visualization: Scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_clustered, 
    x='average_days_employment_per_household', 
    y='total_expenditure', 
    hue='Cluster', 
    palette='viridis'
)
plt.title(f'Districts Clustered (K={optimal_k})')
plt.savefig(f'{output_dir}/cluster_visualization.png') # Saved to img2
plt.close()

# 4. Pairplot
sns.pairplot(df_clustered[features + ['Cluster']], hue='Cluster', palette='viridis')
plt.savefig(f'{output_dir}/pairplot_clusters.png') # Saved to img2
plt.close()

df_clustered.to_csv('mgnrega_clustered_districts.csv', index=False)
