# Anganwadi Enrollment Analysis: KNN & PCA Dimensionality Reduction

## Project Overview

This project implements **K-Nearest Neighbors (KNN)** to perform both **Classification** and **Regression** on child enrollment data. This version introduces **Principal Component Analysis (PCA)** to evaluate the trade-offs between model complexity, computational efficiency, and predictive performance.

By comparing the original high-dimensional feature set against a reduced PCA-transformed set, we identify how dimensionality reduction impacts the algorithm's ability to categorize enrollment levels and estimate future child counts.

---

## PCA Dimensionality Reduction

In many real-world datasets, features are often correlated. **Principal Component Analysis (PCA)** allows us to "compress" the data by transforming it into a new set of variables (Principal Components) that capture the maximum variance.

* **Goal:** Reduce the feature space from 3 dimensions (Infant count, Toddler count, District ID) down to 2 principal components.
* **Performance Trade-off:** Reducing dimensions typically speeds up distance calculations but may discard subtle signals necessary for accuracy.

---

## Comparative Performance Analysis

The following analysis compares the **Original Dataset** against the **PCA-Reduced Dataset** across a range of  values (1 to 30).

### 1. Classification: Accuracy vs. Efficiency

We track the model's ability to correctly classify centers into "High" or "Low" enrollment tiers.

* **Accuracy (Blue vs. Cyan):** In this dataset, the original features (Solid Blue) consistently outperform the PCA-transformed features (Dashed Cyan). The accuracy gap suggests that the discarded dimension contained non-redundant information critical for classification.
* **Execution Time:** * **Original:** ~1.2878s
* **PCA:** ~0.8139s
* **Impact:** PCA achieved a significant reduction in computation time, demonstrating its utility for large-scale deployments where speed is prioritized over perfect precision.



### 2. Regression: Error Analysis (MSE)

We measure the Mean Squared Error (MSE) when estimating the number of children in the 3Y to 6Y age group.

* **Error Trends:** Both models show increasing MSE as  increases, but the **PCA MSE (Dashed Orange)** is consistently higher than the **Original MSE (Solid Red)**.
* **Analysis:** The "elbow" or optimal  for regression is found at lower values (K=1 or 2). Beyond that, the model becomes too "blurry," and the loss of information from PCA further compounds the prediction error.

---

## Methodology Summary


### The Math: Euclidean Distance & PCA

$$

d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}

$$

The distance between two points in the PCA-transformed space is still calculated using the Euclidean formula, but in a lower-dimensional  space:

### Key Takeaways

1. **Dimensionality vs. Signal:** For this specific dataset, the 3D feature set is "lean" enough that PCA reduction to 2D causes a noticeable drop in accuracy/increase in MSE.
2. **Computational Gain:** PCA reduced execution time by approximately **36%**. This confirms that for massive datasets, PCA is a vital tool for making KNN (an  algorithm) computationally feasible.
3. **Optimal K:** The best performance for both original and PCA models is generally found at low  values, indicating that enrollment patterns are highly localized in the feature space.

---

## How to Access Results

The comparative visualizations are stored in the following directory:

* `img_comp/accuracy_comparison.png`: Comparison of classification accuracy and runtime.
* `img_comp/mse_comparison.png`: Comparison of regression error (MSE) across feature sets.
