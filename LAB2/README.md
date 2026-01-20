# MGNREGA & Anganwadi Expenditure Analysis

## Project Summary

This project performs a comprehensive regression analysis on a composite dataset formed by merging **MGNREGA** (Mahatma Gandhi National Rural Employment Guarantee Act) district-level expenditure data with **Anganwadi** (early childhood care) enrollment data.

The goal is to analyze the relationship between social welfare employment spending and child demographics across various districts. The project implements a custom data pipeline and utilizes **Scikit-Learn** to perform automated hyperparameter tuning via **Grid Search with Cross-Validation**. We compare four different regression techniques:

1. **Ordinary Least Squares (OLS):** Standard linear regression without penalties.
2. **Lasso Regression (L1):** Performs feature selection by zeroing out redundant coefficients.
3. **Ridge Regression (L2):** Performs coefficient shrinkage to manage multicollinearity.
4. **Elastic Net Regression:** A hybrid approach combining L1 and L2 penalties.

---

## Regression Problem Statement

**Objective:** Predict the **Total Expenditure** of MGNREGA at the district level.

**Hypothesis:** Total expenditure is primarily driven by the interaction between the active workforce and their wage rates, but may also be influenced by the demographic pressure of young children (represented by Anganwadi enrollment) in the region.

**Variables:**

* **Output ():** `total_expenditure`
* **Regressors ():**
* `total_no_of_active_workers`
* `average_wage_rate_per_day_per_person`
* `Tot_Children_0to6M`
* `Tot_Children_7Mto3Y`
* `Tot_Children_3Yto6Y`
* **Artificial Regressor:** `worker_wage_interaction` (Calculated as ) to capture the primary financial driver.



---

## AI Thought Partner

This project was developed in collaboration with **Gemini (Advanced/Paid Tier)**, utilizing its Python data science capabilities, tool execution for file processing, and mathematical reasoning.

---

## Prompt History

The following sequence of prompts was used to generate the analysis and the final script:

1. **Initial Assignment:** "Create a composite dataset by using two or more datasets from AIKosh / data.gov.in. Identify an output and multiple regressors. State your regression problem nicely... Clean the dataset. Analyse the dataset to see correlation between regressors. If they are not related, create an artificial regressors which is correlated with another parameter. Next perform GD on data with 80:20 train test division. See MSE and Adj R2. Experiment with different ratios and learning rates. Add L1 and L2 regularisation and see impact on performance as well as regression coefficients."
2. **Visualization & Complexity:** "Add correlation matrix plots and also ridge regression and elastic net regression to the script."
3. **Consolidation:** "Do everything from plotting, grouping, cleaning, merging datasets, regression with ols, lasso, ridge, elastic net in a single script... give me the python script for it."
4. **Mathematical Clarification:** "Explain the math behind the L1 and L2 gradient updates."
5. **Optimization:** "I want this hyperparam tuning in my code, include it in the script (u can use scikit learn no probs)."

---

## How to Run

1. Ensure `MGNREGA_dataset_AtAGlance.csv` and `number_of_children_enrolled_in_anganwadis_2025_10.csv` are in the same directory as the script.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Execute the script: `python analysis_script.py`
4. **Outputs:**
* `correlation_heatmap.png`: Visualizes relationships and multicollinearity.
* `tuning_curve.png`: Shows how model error changes with different Lambda values.
* `tuned_performance.csv` & `tuned_coefficients.csv`: Final model metrics and weights.
