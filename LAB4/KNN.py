import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import os

# Create directory for plots
if not os.path.exists('img'):
    os.makedirs('img')

# 1. LOAD DATA
# Update the path to match your local 'data' folder
df = pd.read_csv('data/number_of_children_enrolled_in_anganwadis_2025_10.csv')

# 2. PREPROCESSING (Numerical + Categorical Features)
num_features = ['Tot_Children_0to6M', 'Tot_Children_7Mto3Y']
cat_feature = 'D_Name'
target_reg = 'Tot_Children_3Yto6Y' # Numerical target for regression

df_clean = df.dropna(subset=num_features + [cat_feature, target_reg]).copy()

# ENCODING CATEGORICAL DATA: 
le = LabelEncoder()
df_clean['D_Name_Encoded'] = le.fit_transform(df_clean[cat_feature])

# Create a target for CLASSIFICATION: 
total_enroll = df_clean[num_features + [target_reg]].sum(axis=1)
df_clean['Enrollment_Class'] = (total_enroll > total_enroll.median()).astype(int)

# --- FEATURE SCALING ---
X = df_clean[num_features + ['D_Name_Encoded']]
y_clf = df_clean['Enrollment_Class']
y_reg = df_clean[target_reg]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split for both tasks
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

# --- ANALYSIS: VARYING K FOR BOTH TASKS ---
ks = range(1, 31)
accuracies = []
mses = []

for k in ks:
    # 1. Classification
    knn_c = KNeighborsClassifier(n_neighbors=k)
    knn_c.fit(X_train_c, y_train_c)
    accuracies.append(accuracy_score(y_test_c, knn_c.predict(X_test_c)))
    
    # 2. Regression
    knn_r = KNeighborsRegressor(n_neighbors=k)
    knn_r.fit(X_train_r, y_train_r)
    y_pred_r = knn_r.predict(X_test_r)
    mses.append(mean_squared_error(y_test_r, y_pred_r))

# --- PLOTTING ---

# Plot 1: Classification Accuracy
plt.figure(figsize=(10, 6))
plt.plot(ks, accuracies, marker='o', color='blue', linestyle='-')
plt.title('Impact of K on Classification Accuracy')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('img/classification_k_impact.png')
plt.close()

# Plot 2: Regression MSE (The "Elbow" for Regression)
plt.figure(figsize=(10, 6))
plt.plot(ks, mses, marker='o', color='red', linestyle='-')
plt.title('Impact of K on Regression MSE')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.savefig('img/regression_k_impact.png')
plt.close()

# FINAL METRICS
print("--- KNN Assignment Metrics ---")
print(f"Max Classification Accuracy: {max(accuracies):.4f} at K={ks[np.argmax(accuracies)]}")
print(f"Min Regression MSE: {min(mses):.4f} at K={ks[np.argmin(mses)]}")
