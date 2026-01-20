import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Gradient Descent Implementation (Elastic Net Support)
# ---------------------------------------------------------
def gradient_descent(X, y, lr=0.01, iters=5000, l1=0, l2=0):
    """
    Performs Gradient Descent with Elastic Net regularization.
    l1=0, l2=0 => OLS (Ordinary Least Squares)
    l1>0, l2=0 => Lasso Regression
    l1=0, l2>0 => Ridge Regression
    l1>0, l2>0 => Elastic Net Regression
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    
    for i in range(iters):
        predictions = X.dot(theta)
        errors = predictions - y
        
        # Standard MSE Gradient
        gradient = (1/m) * X.T.dot(errors)
        
        # Add Regularization Gradients (applied to all except intercept theta[0])
        if l1 > 0:
            gradient[1:] += l1 * np.sign(theta[1:])
        if l2 > 0:
            gradient[1:] += 2 * l2 * theta[1:]
            
        theta = theta - lr * gradient
        
        # Calculate Cost for Monitoring
        mse = (1/(2*m)) * np.sum(np.square(errors))
        penalty = l1 * np.sum(np.abs(theta[1:])) + l2 * np.sum(np.square(theta[1:]))
        cost_history.append(mse + penalty)
        
    return theta, cost_history

def calculate_metrics(y_true, y_pred, n_samples, n_features):
    """Calculates Mean Squared Error and Adjusted R-squared."""
    mse = np.mean((y_true - y_pred)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    # Adjusted R2 Formula
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1) if (n_samples - n_features - 1) > 0 else r2
    return mse, r2, adj_r2

# ---------------------------------------------------------
# 2. Data Cleaning and Merging
# ---------------------------------------------------------
print("Loading and cleaning datasets...")
# Load datasets (Ensure these files are in your working directory)
try:
    mgnrega = pd.read_csv('MGNREGA_dataset_AtAGlance.csv')
    anganwadi = pd.read_csv('number_of_children_enrolled_in_anganwadis_2025_10.csv')
except FileNotFoundError:
    print("Error: CSV files not found. Please ensure the filenames match.")
    exit()

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
composite_df = pd.merge(mgnrega, ang_agg, on='district_clean', how='inner')

# Feature Selection
target = 'total_expenditure'
features = [
    'total_no_of_active_workers', 
    'average_wage_rate_per_day_per_person', 
    'Tot_Children_0to6M', 
    'Tot_Children_7Mto3Y', 
    'Tot_Children_3Yto6Y'
]

# Create Artificial Correlated Regressor
# We create an interaction term (Workers * Wage) which we expect to be highly correlated with expenditure
composite_df['worker_wage_interaction'] = (composite_df['total_no_of_active_workers'] * composite_df['average_wage_rate_per_day_per_person']) / 100000
features.append('worker_wage_interaction')

# Remove any missing values
df_final = composite_df[features + [target]].dropna()

# ---------------------------------------------------------
# 3. Correlation Matrix and Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df_final.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap saved as 'correlation_heatmap.png'.")

# ---------------------------------------------------------
# 4. Data Preparation
# ---------------------------------------------------------
X = df_final[features].values
y = df_final[target].values.reshape(-1, 1)

# Feature Scaling (Standardization is crucial for regularization and GD convergence)
X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std
X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled]) # Add Bias/Intercept term

# Train-Test Split (80:20)
def train_test_split(X, y, ratio=0.8):
    m = len(X)
    indices = np.random.permutation(m)
    train_size = int(m * ratio)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 0.8)

# ---------------------------------------------------------
# 5. Regression Experiments
# ---------------------------------------------------------
# Define hyper-parameters
lr = 0.005
iters = 5000

# Dictionary of models to run: { "Name": {"l1": penalty, "l2": penalty} }
experiments = {
    "OLS (No Reg)": {"l1": 0, "l2": 0},
    "Lasso (L1=100)": {"l1": 100, "l2": 0},
    "Ridge (L2=100)": {"l1": 0, "l2": 100},
    "Elastic Net": {"l1": 50, "l2": 50}
}

final_results = []
plt.figure(figsize=(10, 6))

for name, params in experiments.items():
    print(f"Executing {name}...")
    theta, cost_history = gradient_descent(X_train, y_train, lr=lr, iters=iters, **params)
    
    # Predict and evaluate on Test set
    y_pred = X_test.dot(theta)
    mse, r2, adj_r2 = calculate_metrics(y_test, y_pred, len(y_test), len(features))
    
    final_results.append({
        "Model": name,
        "MSE": mse,
        "Adj R2": adj_r2,
        "Coeffs": theta.flatten()
    })
    
    # Plot Convergence Curve
    plt.plot(cost_history, label=name)

# Format and save convergence plot
plt.yscale('log')
plt.title('Model Convergence Comparison')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE + Regularization)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('convergence_comparison.png')

# ---------------------------------------------------------
# 6. Final Report and Analysis
# ---------------------------------------------------------
perf_report = pd.DataFrame(final_results).drop(columns="Coeffs")
print("\n--- Model Performance Comparison ---")
print(perf_report)

# Coefficient Comparison Table
coeff_table = pd.DataFrame({
    "Feature": ["Intercept"] + features,
    "OLS": final_results[0]["Coeffs"],
    "Lasso": final_results[1]["Coeffs"],
    "Ridge": final_results[2]["Coeffs"],
    "ElasticNet": final_results[3]["Coeffs"]
})

print("\n--- Feature Coefficients ---")
print(coeff_table)

# Save results for external use
perf_report.to_csv('metrics_summary.csv', index=False)
coeff_table.to_csv('coefficients_summary.csv', index=False)
print("\nSuccess: Performance and coefficients saved as CSV files.")
