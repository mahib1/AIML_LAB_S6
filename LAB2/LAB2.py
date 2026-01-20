import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Data Processing & Feature Engineering
# ---------------------------------------------------------
print("Loading and merging data...")
mgnrega = pd.read_csv('MGNREGA_dataset_AtAGlance.csv')
anganwadi = pd.read_csv('number_of_children_enrolled_in_anganwadis_2025_10.csv')

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

# ---------------------------------------------------------
# 2. Relationship Analysis Plots
# ---------------------------------------------------------
# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_final.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_heatmap.png')

# Covariance Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_final.corr(), annot=True, cmap='viridis', fmt=".1e")
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_heatmap.png')

# ---------------------------------------------------------
# 3. Preparation & Hyperparameter Tuning
# ---------------------------------------------------------
X = df_final[features]
y = df_final[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# Tuning Lasso
lasso_gs = GridSearchCV(Lasso(), {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
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
# 4. Model Comparison & Final Evaluation
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
plt.title('Model Comparison: MSE')
plt.savefig('model_comparison_mse.png')

# R2 and Adj R2 Plot
plt.figure(figsize=(10, 6))
melted = results_df.melt(id_vars='Model', value_vars=['R2', 'Adj R2'], var_name='Metric', value_name='Value')
sns.barplot(x='Model', y='Value', hue='Metric', data=melted, palette='viridis')
plt.title('Model Comparison: R2 and Adjusted R2')
plt.savefig('model_comparison_r2.png')

print("Process complete. All charts saved.")
