import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import time

# Ensure output directory exists
output_dir = 'img3'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the MGNREGA dataset
# Assuming the file is in 'data/MGNREGA_dataset_AtAGlance.csv' as per previous context
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

def find_optimal_k(X_data, init_method, max_clusters=100, threshold=0.02):
    wcss = []
    start_time = time.time()
    
    # K=1
    kmeans = KMeans(n_clusters=1, init=init_method, random_state=42, n_init=10)
    kmeans.fit(X_data)
    wcss.append(kmeans.inertia_)
    
    optimal_k = 1
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, init=init_method, random_state=42, n_init=10)
        kmeans.fit(X_data)
        current_wcss = kmeans.inertia_
        
        percent_change = (wcss[-1] - current_wcss) / wcss[-1]
        wcss.append(current_wcss)
        
        if percent_change < threshold:
            optimal_k = k - 1
            break
        else:
            optimal_k = k
            
    end_time = time.time()
    return optimal_k, wcss, end_time - start_time

# Run comparison
methods = ['k-means++', 'random']
results = {}

for method in methods:
    opt_k, wcss_list, elapsed = find_optimal_k(X_scaled, method)
    results[method] = {
        'optimal_k': opt_k,
        'wcss': wcss_list,
        'time': elapsed
    }

# --- Visualization 1: Elbow Curves Comparison ---
plt.figure(figsize=(12, 6))
for method in methods:
    plt.plot(range(1, len(results[method]['wcss']) + 1), 
             results[method]['wcss'], 
             marker='o', linestyle='--', label=f'{method} (Time: {results[method]["time"]:.3f}s)')

plt.title('Elbow Method Comparison: k-means++ vs Random')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/elbow_comparison.png')
plt.close()

# --- Visualization 2: Time Taken Comparison ---
plt.figure(figsize=(8, 6))
plt.bar(results.keys(), [results[m]['time'] for m in methods], color=['blue', 'orange'])
plt.title('Time Taken to Find Optimal K')
plt.ylabel('Time (seconds)')
plt.savefig(f'{output_dir}/time_comparison.png')
plt.close()

# Summary table for user
summary_df = pd.DataFrame({
    'Method': methods,
    'Optimal K Found': [results[m]['optimal_k'] for m in methods],
    'Time Taken (s)': [results[m]['time'] for m in methods],
    'Final WCSS': [results[m]['wcss'][-1] for m in methods]
})

print(summary_df)

# Clustering data with k-means++ optimal k for the final CSV
final_k = results['k-means++']['optimal_k']
kmeans_final = KMeans(n_clusters=final_k, init='k-means++', random_state=42, n_init=10)
df_clustered = df.loc[X.index].copy()
df_clustered['Cluster'] = kmeans_final.fit_predict(X_scaled)
df_clustered.to_csv('mgnrega_clustered_comparison.csv', index=False)
