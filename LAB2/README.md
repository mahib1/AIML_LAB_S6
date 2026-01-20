# MGNREGA & Anganwadi Expenditure Analysis

## Project Summary

This project performs a comprehensive regression analysis on a composite dataset formed by merging **MGNREGA** (Mahatma Gandhi National Rural Employment Guarantee Act) district-level expenditure data with **Anganwadi** (early childhood care) enrollment data.

The goal is to predict **Total MGNREGA Expenditure** based on workforce metrics, child demographics, and an engineered interaction term. The project implements a custom **Gradient Descent** engine from scratch to compare four different regression techniques:

1. **Ordinary Least Squares (OLS):** Standard linear regression without penalties.
2. **Lasso Regression (L1):** Performs feature selection by zeroing out redundant coefficients.
3. **Ridge Regression (L2):** Performs coefficient shrinkage to manage multicollinearity.
4. **Elastic Net Regression:** A hybrid approach combining L1 and L2 penalties.

The analysis includes data cleaning, district-level aggregation, feature standardization, correlation heatmaps, and convergence plots to visualize the optimization process.

---

## AI Thought Partner

This project was developed in collaboration with **Gemini (Advanced/Paid Tier)**, utilizing its Python data science capabilities and mathematical reasoning.

---

## Prompt History

The following sequence of prompts was used to generate the analysis and the final script:

1. **Initial Assignment:** "Create a composite dataset by using two or more datasets from AIKosh / data.gov.in. Identify an output and multiple regressors. State your regression problem nicely... Clean the dataset. Analyse the dataset to see correlation between regressors. If they are not related, create an artificial regressors which is correlated with another parameter. Next perform GD on data with 80:20 train test division. See MSE and Adj R2. Experiment with different ratios and learning rates. Add L1 and L2 regularisation and see impact on performance as well as regression coefficients. Especially, which regressors became zero or were reduced."
2. **Visualization & Complexity:** "Add correlation matrix plots and also ridge regression and elastic net regression to the script."
3. **Consolidation:** "Do everything from plotting, grouping, cleaning, merging datasets, regression with ols, lasso, ridge, elastic net in a single script, don't give me the script results, give me the python script for it and I will run it on my machine."
4. **Mathematical Clarification:** "Explain the code to me line by line" and "Explain the math behind the L1 and L2 gradient updates."

---

## How to Run

1. Ensure `MGNREGA_dataset_AtAGlance.csv` and `number_of_children_enrolled_in_anganwadis_2025_10.csv` are in the same directory as the script.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn`
3. Execute the script: `python analysis_script.py`
4. View the generated `correlation_heatmap.png` and `convergence_comparison.png` for results.
