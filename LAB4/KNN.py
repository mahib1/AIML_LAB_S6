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
df = pd.read_csv('data/number_of_children_enrolled_in_anganwadis_2025_10.csv')

# 2. PREPROCESSING (Numerical + Categorical Features)
# We pick numerical enrollment counts and the Categorical 'District Name'
num_features = ['Tot_Children_0to6M', 'Tot_Children_7Mto3Y']
cat_feature = 'D_Name'
target_reg = 'Tot_Children_3Yto6Y' # Numerical target for regression

df_clean = df.dropna(subset=num_features + [cat_feature, target_reg]).copy()

# ENCODING CATEGORICAL DATA: 
# KNN needs numbers to calculate distance, so we encode the District names
le = LabelEncoder()
df_clean['D_Name_Encoded'] = le.fit_transform(df_clean[cat_feature])

# Create a target for CLASSIFICATION: 
# Is total enrollment 'High' (1) or 'Low' (0)?
total_enroll = df_clean[num_features + [target_reg]].sum(axis=1)
df_clean['Enrollment_Class'] = (total_enroll > total_enroll.median()).astype(int)

# --- TASK 1: KNN CLASSIFICATION & VARYING K ---
X = df_clean[num_features + ['D_Name_Encoded']]
y_clf = df_clean['Enrollment_Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

ks = range(1, 31)
accuracies = []

for k in ks:
    knn_c = KNeighborsClassifier(n_neighbors=k)
    knn_c.fit(X_train_c, y_train_c)
    accuracies.append(accuracy_score(y_test_c, knn_c.predict(X_test_c)))

# SAVE PLOT: Impact of K
plt.figure(figsize=(10, 6))
plt.plot(ks, accuracies, marker='o', color='blue')
plt.title('Impact of K on Classification Accuracy')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('img/classification_k_impact.png')

# --- TASK 2: KNN REGRESSION ---
y_reg = df_clean[target_reg]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

knn_r = KNeighborsRegressor(n_neighbors=5) # Standard K=5 for regression
knn_r.fit(X_train_r, y_train_r)
y_pred_r = knn_r.predict(X_test_r)

# EVALUATION
print("--- KNN Assignment Metrics ---")
print(f"Classification Max Accuracy: {max(accuracies):.4f}")
print(f"Regression Mean Squared Error (MSE): {mean_squared_error(y_test_r, y_pred_r):.2f}")
print(f"Regression R2 Score: {r2_score(y_test_r, y_pred_r):.2f}")
