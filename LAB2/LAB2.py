import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Data Cleaning and Merging
# ---------------------------------------------------------
print("Loading and cleaning datasets...")
mgnrega = pd.read_csv('MGNREGA_dataset_AtAGlance.csv')
anganwadi = pd.read_csv('number_of_children_enrolled_in_anganwadis_2025_10.csv')

# Aggregate Anganwadi data (Summing children count per district)
ang_agg = anganwadi.groupby('D_Name').agg({
    'Tot_Children_0to6M': 'sum', 
    'Tot_Children_7Mto3Y': 'sum', 
    'Tot_Children_3Yto6Y': 'sum'
}).reset_index()

# Harmonize District Names for merging
mgnrega['district_clean'] = mgnrega['district_name'].str.upper().str.strip()
ang_agg['district_clean'] = ang_agg['D_Name'].str.upper().str.strip()

# Merge Datasets
df = pd.merge(mgnrega, ang_agg, on='district_clean', how='inner')

# ---------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------
# Target: total_expenditure
# Artificial Regressor: Interaction between workers and wage rate
df['worker_wage_interaction'] = (df['total_no_of_active_workers'] * df['average_wage_rate_per_day_per_person']) / 100000

features = [
    'total_no_of_active_workers', 
    'average_wage_rate_per_day_per_person', 
    'Tot_Children_0to6M', 
    'Tot_Children_7Mto3Y', 
    'Tot_Children_3Yto6Y',
    'worker_wage_interaction'
]
target = 'total_expenditure'

# Drop rows with missing values
df_final = df[features + [target]].dropna()

# ---------------------------------------------------------
# 3. Exploratory Analysis: Correlation Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df_final.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
print("Heatmap saved as 'correlation_heatmap.png'.")

# ---------------------------------------------------------
# 4. Data Preparation and Scaling
# ---------------------------------------------------------
X = df_final[features]
y = df_final[target]

# Standardize features (Mean=0, Std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 5. Hyperparameter Tuning using Scikit-Learn (GridSearchCV)
# ---------------------------------------------------------
# Range of Lambda (Alpha) values to test
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# A. Tune Lasso
lasso_cv = GridSearchCV(Lasso(max_iter=10000), param_grid={'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train, y_train)
best_lasso_alpha = lasso_cv.best_params_['alpha']

# B. Tune Ridge
ridge_cv = GridSearchCV(Ridge(), param_grid={'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)
best_ridge_alpha = ridge_cv.best_params_['alpha']

# C. Tune Elastic Net
en_params = {
    'alpha': alphas,
    'l1_ratio': [0.2, 0.5, 0.8] # Ratio of L1 vs L2 penalty
}
en_cv = GridSearchCV(ElasticNet(max_iter=10000), param_grid=en_params, cv=5, scoring='neg_mean_squared_error')
en_cv.fit(X_train, y_train)
best_en_params = en_cv.best_params_

print(f"\nBest Lasso Alpha: {best_lasso_alpha}")
print(f"Best Ridge Alpha: {best_ridge_alpha}")
print(f"Best Elastic Net Params: {best_en_params}")

# ---------------------------------------------------------
# 6. Final Model Training & Evaluation
# ---------------------------------------------------------
models = {
    "OLS": LinearRegression(),
    "Tuned Lasso": Lasso(alpha=best_lasso_alpha),
    "Tuned Ridge": Ridge(alpha=best_ridge_alpha),
    "Tuned ElasticNet": ElasticNet(**best_en_params)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate Adjusted R2
    n = len(y_test)
    p = X_train.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    results.append({
        "Model": name,
        "MSE": mse,
        "Adj R2": adj_r2,
        "Coefficients": np.concatenate([[model.intercept_], model.coef_]) if name != "OLS" else np.concatenate([[model.intercept_], model.coef_])
    })

# ---------------------------------------------------------
# 7. Final Report & Plots
# ---------------------------------------------------------
perf_df = pd.DataFrame(results).drop(columns="Coefficients")
print("\n--- Model Performance Comparison ---")
print(perf_df)

# Plot Tuning Results (Lasso Example)
plt.figure(figsize=(10, 6))
cv_results = pd.DataFrame(lasso_cv.cv_results_)
plt.plot(cv_results['param_alpha'], -cv_results['mean_test_score'], marker='o')
plt.xscale('log')
plt.xlabel('Alpha (Lambda)')
plt.ylabel('Mean Squared Error (CV)')
plt.title('Lasso Hyperparameter Tuning Performance')
plt.grid(True)
plt.savefig('tuning_curve.png')

# Coefficient Comparison
coeff_df = pd.DataFrame({
    "Feature": ["Intercept"] + features,
    "OLS": results[0]["Coefficients"],
    "Lasso": results[1]["Coefficients"],
    "Ridge": results[2]["Coefficients"],
    "ElasticNet": results[3]["Coefficients"]
})

print("\n--- Final Coefficients ---")
print(coeff_df)

# Save results
perf_df.to_csv('tuned_performance.csv', index=False)
coeff_df.to_csv('tuned_coefficients.csv', index=False)
print("\nResults saved to CSV. All plots generated.")
