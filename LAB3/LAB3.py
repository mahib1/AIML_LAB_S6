import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# 1. Load the MGNREGA dataset
df = pd.read_csv('data/MGNREGA_dataset_AtAGlance.csv')

# 2. Define the Target: High Employment Classification
# We use the median of 'average_days_employment_per_household' as the threshold for binary classification
median_employment = df['average_days_employment_per_household'].mean()
df['is_high_employment'] = (df['average_days_employment_per_household'] > median_employment).astype(int)

# 3. Select Features for prediction
features = [
    'sc_worker_percentage_against_acive_workers',
    'st_worker_percentage_against_active_workers',
    'total_no_of_active_workers',
    'approved_labour_budget',
    'women_persondays_percentage_of_total',
    'percentage_expenditure_on_agriculture_and_allied_works',
    'total_expenditure'
]

X = df[features].fillna(df[features].mean()) # Handling missing values with mean imputation
y = df['is_high_employment']

# 4. Split the data into Training and Testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale features
# Logistic Regression requires scaling because it uses regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# 8. Generate the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
p = precision_score(y_test, y_pred)
r = recall_score(y_test, y_pred)

# 9. Plot and save the Confusion Matrix as a PNG file
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: High Employment Prediction')
plt.xlabel('Predicted Label (1=High, 0=Low)')
plt.ylabel('Actual Label (1=High, 0=Low)')
plt.tight_layout()

# Save the file
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")

print(f'recall = {r}, precision = {p}')
