import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# ---------------------------------------------------------
# 1. Data Processing & Feature Engineering
# ---------------------------------------------------------
print("Loading and merging data...")
mgnrega = pd.read_csv('MGNREGA_dataset_AtAGlance.csv')
anganwadi = pd.read_csv('number_of_children_enrolled_in_anganwadis_2025_10.csv')

def save_scatter_plots(df, target_col, output_dir='./img'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    regressors = df.select_dtypes(include=['number']).columns.tolist()
    if target_col in regressors:
        regressors.remove(target_col)

    print(f"Generating {len(regressors)} scatter plots...")
    for col in regressors:
        plt.figure(figsize=(8, 6))
        plt.scatter(df[col], df[target_col], alpha=0.5, color='royalblue', edgecolors='k')
        plt.title(f'{target_col} vs {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel(target_col, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        file_path = os.path.join(output_dir, f'scatter_{col}.png')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

# Aggregation & Merging
ang_agg = anganwadi.groupby('D_Name').agg({
    'Tot_Children_0to6M': 'sum', 'Tot_Children_7Mto3Y': 'sum', 'Tot_Children_3Yto6Y': 'sum'
}).reset_index()

mgnrega['dist_clean'] = mgnrega['district_name'].str.upper().str.strip()
ang_agg['dist_clean'] = ang_agg['D_Name'].str.upper().str.strip()
df = pd.merge(mgnrega, ang_agg, on='dist_clean', how='inner')

# Interaction Term
df['worker_wage_interaction'] = (df['total_no_of_active_workers'] * df['average_wage_rate_per_day_per_person']) / 100000

features = ['total_no_of_active_workers', 'average_wage_rate_per_day_per_person', 
            'Tot_Children_0to6M', 'Tot_Children_7Mto3Y', 'Tot_Children_3Yto6Y', 'worker_wage_interaction']
target = 'total_expenditure'
df_final = df[features + [target]].dropna()

# RUN SCATTER PLOTS ON RAW DATA
save_scatter_plots(df_final, target)

# ---------------------------------------------------------
# 2. Relationship Analysis Plots
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df_final.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_heatmap.png')

plt.figure(figsize=(10, 8))
sns.heatmap(df_final.cov(), annot=True, cmap='viridis', fmt=".1e")
plt.title('Covariance Matrix Heatmap')
plt.savefig('covariance_heatmap.png')

# ---------------------------------------------------------
# 3. Outlier Detection and Removal via Residuals
# ---------------------------------------------------------
print("Detecting and removing outliers using residuals...")
X_outlier_check = df_final[features]
y_outlier_check = df_final[target]

# Scale for residual check
temp_scaler = StandardScaler()
X_temp_scaled = temp_scaler.fit_transform(X_outlier_check)

# Fit a baseline OLS to find residuals
temp_model = LinearRegression()
temp_model.fit(X_temp_scaled, y_outlier_check)
y_pred_temp = temp_model.predict(X_temp_scaled)
residuals = y_outlier_check - y_pred_temp

# Standardize residuals
std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

# Plot Residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_temp, std_residuals, alpha=0.5, color='orange', edgecolors='k')
plt.axhline(y=0, color='red', linestyle='--')
plt.axhline(y=3, color='black', linestyle=':')
plt.axhline(y=-3, color='black', linestyle=':')
plt.title('Residuals vs Fitted Values (Outlier Detection)')
plt.xlabel('Predicted Value')
plt.ylabel('Standardized Residuals')
plt.savefig('residuals_plot.png')
plt.close()

# Identify and Filter Outliers (Threshold |Z| > 3)
outlier_mask = np.abs(std_residuals) <= 3
df_cleaned = df_final[outlier_mask].copy()
print(f"Removed {len(df_final) - len(df_cleaned)} outliers.")

# ---------------------------------------------------------
# 4. Normalization & Hyperparameter Tuning (On Cleaned Data)
# ---------------------------------------------------------
X = df_cleaned[features]
y = df_cleaned[target]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# Tuning Lasso
lasso_gs = GridSearchCV(Lasso(max_iter=10000), {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
lasso_gs.fit(X_train, y_train)
plt.figure(figsize=(8, 6))
plt.plot(alphas, -lasso_gs.cv_results_['mean_test_score'], marker='o')
plt.xscale('log'); plt.title('Lasso: Error vs Lambda'); plt.grid(True)
plt.savefig('lasso_tuning.png')

# Tuning Ridge
ridge_gs = GridSearchCV(Ridge(), {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
ridge_gs.fit(X_train, y_train)
plt.figure(figsize=(8, 6))
plt.plot(alphas, -ridge_gs.cv_results_['mean_test_score'], marker='o', color='green')
plt.xscale('log'); plt.title('Ridge: Error vs Lambda'); plt.grid(True)
plt.savefig('ridge_tuning.png')

# Tuning Elastic Net
en_gs = GridSearchCV(ElasticNet(max_iter=10000), {'alpha': alphas, 'l1_ratio': [0.2, 0.5, 0.8]}, cv=5, scoring='neg_mean_squared_error')
en_gs.fit(X_train, y_train)
en_res = pd.DataFrame(en_gs.cv_results_)
plt.figure(figsize=(10, 6))
for r in [0.2, 0.5, 0.8]:
    sub = en_res[en_res['param_l1_ratio'] == r]
    plt.plot(sub['param_alpha'], -sub['mean_test_score'], marker='o', label=f'l1_ratio={r}')
plt.xscale('log'); plt.title('Elastic Net: Tuning Curves'); plt.legend(); plt.grid(True)
plt.savefig('elastic_net_tuning.png')

# ---------------------------------------------------------
# 5. Model Comparison & Final Evaluation
# ---------------------------------------------------------
models = {
    "OLS": LinearRegression(),
    "Tuned Lasso": Lasso(alpha=lasso_gs.best_params_['alpha']),
    "Tuned Ridge": Ridge(alpha=ridge_gs.best_params_['alpha']),
    "Tuned ElasticNet": ElasticNet(**en_gs.best_params_)
}

comparison = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1)
    comparison.append({"Model": name, "MSE": mse, "R2": r2, "Adj R2": adj_r2})

results_df = pd.DataFrame(comparison)

# MSE Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='MSE', data=results_df, palette='magma')
plt.title('Model Comparison: MSE (Cleaned Data)')
plt.savefig('model_comparison_mse.png')

# R2 and Adj R2 Plot
plt.figure(figsize=(10, 6))
melted = results_df.melt(id_vars='Model', value_vars=['R2', 'Adj R2'], var_name='Metric', value_name='Value')
sns.barplot(x='Model', y='Value', hue='Metric', data=melted, palette='viridis')
plt.title('Model Comparison: R2 and Adjusted R2 (Cleaned Data)')
plt.savefig('model_comparison_r2.png')

print("Process complete. All charts saved, including residual outlier detection.")
