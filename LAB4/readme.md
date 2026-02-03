Anganwadi Enrollment Analysis: KNN Classification & Regression
Project Overview
This project implements K-Nearest Neighbors (KNN), a versatile Machine Learning algorithm, to perform both Classification and Regression on child enrollment data from Anganwadi centers. The goal is to understand demographic patterns and build a predictive model that can categorize enrollment levels and estimate future child counts.

By testing a range of K values (from 1 to 30), this analysis identifies the "sweet spot" where the model is most accurate without being overly sensitive to noise.

The Two-Fold Approach
1. Classification: Predicting Enrollment Tiers
In this task, the model classifies an Anganwadi center into one of two categories: High Enrollment (1) or Low Enrollment (0).

The Logic: The target is created by calculating the median total enrollment across the dataset. Centers above this median are "High," and those below are "Low."

Decision Making: The computer looks at the K most similar centers and assigns a class based on a majority vote.

2. Regression: Estimating Preschool Counts
In this task, the model predicts a specific number: the total children in the 3 Years to 6 Years age group.

The Logic: Instead of a category, the model calculates the average enrollment of the K nearest neighbors.

Goal: This helps in resource planning by estimating how many children will require preschool services based on current infant and toddler enrollment.

Technical Methodology
Hybrid Feature Selection
KNN is a distance-based algorithm, so it requires numerical inputs. To handle the diverse nature of this data, we use:

Numerical Features: Enrollment figures for infants (0–6 months) and toddlers (7 months – 3 years).

Categorical Feature: The District Name. We use a LabelEncoder to transform text-based district names into unique integers, allowing the algorithm to consider regional similarity.

Data Scaling (Standardization)
Since enrollment numbers vary significantly across age groups, we use a StandardScaler. This transforms all features to the same scale (Mean = 0, Std Dev = 1), preventing larger numbers from skewing the distance calculations.

The Math: Euclidean Distance
Similarity between two data points (p and q) is determined by calculating the straight-line distance between them in a multi-dimensional space:

d(p,q)= 
i=1
∑
n
​
 (p 
i
​
 −q 
i
​
 ) 
2
 

​
 
Performance Analysis: The Impact of K
The choice of K (number of neighbors) is the most critical part of this experiment.

Classification Accuracy
We track how the accuracy score changes as we increase K.

Small K: May lead to Overfitting (picking up on random noise).

Large K: May lead to Underfitting (ignoring local patterns and becoming too "blurry").

Regression MSE (Mean Squared Error)
For regression, we measure the "cost" of our errors using MSE. The lower the MSE, the better our predictions. By plotting MSE vs K, we look for the "elbow" where the error reaches its minimum.

MSE= 
n
1
​
  
i=1
∑
n
​
 (y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
How to Read the Results
img/classification_k_impact.png: Look for the highest point on the blue line. The K value at this peak is the best setting for categorizing new data.

img/regression_k_impact.png: Look for the lowest dip on the red line. This represents the K value that produces the most precise numerical predictions.

Metrics:

Max Accuracy: The highest percentage of centers correctly classified.

Min MSE: The point where the model's numerical guesses were closest to the actual enrollment numbers.
