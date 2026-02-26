import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA

# Create directory for comparison plots
output_dir = 'img_comp'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. LOAD DATA
df = pd.read_csv('data/number_of_children_enrolled_in_anganwadis_2025_10.csv')

# 2. PREPROCESSING
num_features = ['Tot_Children_0to6M', 'Tot_Children_7Mto3Y']
cat_feature = 'D_Name'
target_reg = 'Tot_Children_3Yto6Y'

df_clean = df.dropna(subset=num_features + [cat_feature, target_reg]).copy()
le = LabelEncoder()
df_clean['D_Name_Encoded'] = le.fit_transform(df_clean[cat_feature])

total_enroll = df_clean[num_features + [target_reg]].sum(axis=1)
df_clean['Enrollment_Class'] = (total_enroll > total_enroll.median()).astype(int)

X = df_clean[num_features + ['D_Name_Encoded']]
y_clf = df_clean['Enrollment_Class']
y_reg = df_clean[target_reg]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA APPLICATION ---
# Reducing from 3 features to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

def evaluate_knn(X_data, y_classification, y_regression, label_suffix):
    ks = range(1, 31)
    accs, mses = [], []
    
    start_time = time.time()
    for k in ks:
        # Classification
        knn_c = KNeighborsClassifier(n_neighbors=k)
        knn_c.fit(X_data, y_classification)
        accs.append(accuracy_score(y_classification, knn_c.predict(X_data))) # Using full for demo, or split
        
        # Regression
        knn_r = KNeighborsRegressor(n_neighbors=k)
        knn_r.fit(X_data, y_regression)
        mses.append(mean_squared_error(y_regression, knn_r.predict(X_data)))
    
    end_time = time.time()
    return accs, mses, (end_time - start_time)

# Split data for better validation
X_train, X_test, y_train_c, y_test_c = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y_clf, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
X_train_r_pca, X_test_r_pca, _, _ = train_test_split(X_pca, y_reg, test_size=0.2, random_state=42)

# --- RUN EXPERIMENTS ---
ks = range(1, 31)

# Original Data
acc_orig, mse_orig, time_orig = evaluate_knn(X_test, y_test_c, y_test_r, "Original")
# PCA Data
acc_pca, mse_pca, time_pca = evaluate_knn(X_test_pca, y_test_c, y_test_r, "PCA")

# --- COMPARATIVE PLOTTING ---

# Plot 1: Accuracy Comparison
plt.figure(figsize=(10, 6))
plt.plot(ks, acc_orig, label=f'Original (Time: {time_orig:.4f}s)', color='blue')
plt.plot(ks, acc_pca, label=f'PCA (Time: {time_pca:.4f}s)', color='cyan', linestyle='--')
plt.title('Classification Accuracy: Original vs PCA')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/accuracy_comparison.png')

# Plot 2: MSE Comparison
plt.figure(figsize=(10, 6))
plt.plot(ks, mse_orig, label=f'Original MSE', color='red')
plt.plot(ks, mse_pca, label=f'PCA MSE', color='orange', linestyle='--')
plt.title('Regression MSE: Original vs PCA')
plt.xlabel('K')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/mse_comparison.png')

print("--- PCA Impact Analysis ---")
print(f"Original Execution Time: {time_orig:.6f} seconds")
print(f"PCA Execution Time:      {time_pca:.6f} seconds")
print(f"Time Saved:              {((time_orig - time_pca)/time_orig)*100:.2f}%")
