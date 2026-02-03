# Combined Socio-Economic Analysis: KNN Classification & Regression

## What is this project?

This project advances our analysis of India's rural welfare programs by applying **Supervised Machine Learning** to the **Anganwadi** (Child Care) and **MGNREGA** (Employment) datasets. While our previous work focused on grouping districts (clustering), this assignment implements **K-Nearest Neighbors (KNN)** to solve two distinct mathematical problems: **Classification** and **Regression**.

---

## The KNN Dual-Approach

Imagine you are a policy maker trying to understand a new district. You look at its "nearest neighbors"—districts that have similar demographics—to make two types of predictions:

1. **The Category (Classification):** Is this district likely to be a "High Enrollment" or "Low Enrollment" zone?
2. **The Number (Regression):** Based on the number of infants, exactly how many preschool-aged children should we expect to see enrolled?

### Multi-Type Feature Integration

To make these predictions accurate, the model processes a mix of data types:

* **Numerical Data:** Raw enrollment counts for infants (0–6 months) and toddlers (7 months – 3 years).
* **Categorical Data:** The **District Name**. Because KNN relies on distance math, we use **Label Encoding** to transform these names into unique numerical IDs so the computer can calculate "geographical" similarity.

---

## The Science: Understanding K-Impact

The most important setting in this algorithm is **K**—the number of neighbors the computer looks at before making a decision.

### 1. Classification & Accuracy

In classification, we assign a district to a group based on a majority vote of its neighbors. We tested the model with  values ranging from 1 to 30 to find the "Sweet Spot."

* **Low K (e.g., K=1):** The model is a "perfectionist" and follows the training data too closely, often failing on new data (overfitting).
* **High K (e.g., K=20):** The model becomes a "generalist," smoothing out differences but potentially missing unique local trends (underfitting).

### 2. Regression & Precision

In regression, instead of a "vote," the computer takes the **average** of the neighbors' values to predict a specific number. We evaluate this using the **Mean Squared Error (MSE)** and **R2 Score**, which tells us how much of the district's enrollment variance our model can actually explain.

---

## The Math: Euclidean Distance in Action

For both tasks, the computer treats every district as a coordinate in a multi-dimensional graph. It calculates the similarity between two districts,  and , using the **Euclidean Distance** formula:

Before this calculation, we apply **Standard Scaling** to ensure that features with larger numbers (like preschool counts) don't accidentally overpower features with smaller numbers (like infant counts).

---

## How to Read the Results

### 1. `img/classification_k_impact.png`

This plot shows the "Stability" of our model. You will notice the accuracy peaks and then plateaus. The  value at the highest peak represents the most reliable setting for categorizing new districts.

### 2. Command Line Output

* **Optimal K:** The neighbor count that produced the highest accuracy.
* **Overall Accuracy:** The percentage of districts correctly categorized.
* **Regression MSE:** The average "error" in our preschool enrollment predictions—lower is better!

### 3. `img/combined_analysis_viz.png`

This scatter plot visualizes the "Decision Boundaries" discovered by the model. It shows how the MGNREGA performance clusters align with the Anganwadi child demographics.

---

Would you like me to add a section to this README explaining how to **deploy** this model to predict data for a brand new district?
