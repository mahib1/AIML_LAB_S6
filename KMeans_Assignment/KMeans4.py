import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import os
import time

# Ensure output directory exists
output_dir = 'img4'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the MGNREGA dataset
df = pd.read_csv('data/MGNREGA_dataset_AtAGlance.csv')

# Selecting features
features = [
    'average_days_employment_per_household', 
    'total_expenditure', 
    'percentage_expenditure_on_agriculture_and_allied_works'
]
X = df[features].dropna()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def find_optimal_k_with_separation(X_data, init_method, max_clusters=100):
    dbi_scores = []
    wcss_scores = []
    start_time = time.time()
    
    # K=1 doesn't have inter-cluster distance, so we start from K=2
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, init=init_method, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_data)
        
        # Intra-cluster (WCSS)
        wcss_scores.append(kmeans.inertia_)
        
        # Inter-cluster + Intra-cluster (DBI)
        # Lower DBI means better separation and tighter clusters
        dbi = davies_bouldin_score(X_data, labels)
        dbi_scores.append(dbi)
            
    end_time = time.time()
    
    # Optimal K is the point where DBI is minimized
    optimal_k = range(2, max_clusters + 1)[np.argmin(dbi_scores)]
    
    return optimal_k, dbi_scores, wcss_scores, end_time - start_time

# Run comparison
methods = ['k-means++', 'random']
results = {}

for method in methods:
    opt_k, dbi_list, wcss_list, elapsed = find_optimal_k_with_separation(X_scaled, method)
    results[method] = {
        'optimal_k': opt_k,
        'dbi': dbi_list,
        'wcss': wcss_list,
        'time': elapsed
    }

# --- Visualization: DBI Score (Quality Metric) ---
plt.figure(figsize=(12, 6))
for method in methods:
    plt.plot(range(2, len(results[method]['dbi']) + 2), 
             results[method]['dbi'], 
             marker='s', linestyle='-', label=f'{method} (Best K: {results[method]["optimal_k"]})')

plt.title('Davies-Bouldin Index (Lower = Better Separation)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('DBI Score')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/dbi_comparison.png')
plt.close()

# --- Visualization: Time Comparison ---
plt.figure(figsize=(8, 6))
plt.bar(results.keys(), [results[m]['time'] for m in methods], color=['teal', 'coral'])
plt.title('Time Taken to Search (K=2 to 20)')
plt.ylabel('Seconds')
plt.savefig(f'{output_dir}/time_comparison_dbi.png')
plt.close()

# Summary Output
summary_df = pd.DataFrame({
    'Method': methods,
    'Optimal K (min DBI)': [results[m]['optimal_k'] for m in methods],
    'Total Search Time (s)': [results[m]['time'] for m in methods],
    'Best DBI Score': [min(results[m]['dbi']) for m in methods]
})

print(summary_df)

# Final Clustering using k-means++ result
final_k = results['k-means++']['optimal_k']
kmeans_final = KMeans(n_clusters=final_k, init='k-means++', random_state=42, n_init=10)
df_clustered = df.loc[X.index].copy()
df_clustered['Cluster'] = kmeans_final.fit_predict(X_scaled)
df_clustered.to_csv('mgnrega_clustered_with_separation.csv', index=False)
