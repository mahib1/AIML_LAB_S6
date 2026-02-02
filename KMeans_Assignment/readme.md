# MGNREGA District Clustering Project: Understanding Rural Development Through Data

## What is this project?

This project uses a branch of **Artificial Intelligence (AI)** called **Machine Learning** to group 726 districts in India based on how they use the MGNREGA (Mahatma Gandhi National Rural Employment Guarantee Act) program. Instead of looking at districts one by one, we use a technique called **K-Means Clustering** to find patterns in how money is spent and how much work is actually provided to families.

---

## The Sorting Hat for Districts

Imagine you have 700 different delicious and otherwise, chocolates, but they are all mixed up in one giant box. You want to group them, but you don't know the themes. You decide to group them by two things: **How many pieces they have** and **What flavour is most common.**

In this project, we do the same thing with districts. We look at:

1. **The Paycheck:** How much total money the district spent.
2. **The Hard Work:** How many days of work an average family got.
3. **The Farming Focus:** How much of that work was for farming.

The computer acts like a "Sorting Hat." It looks at these three numbers for every district and puts similar districts into the same "House" (Cluster). This helps us see which districts are "Super Producers" and which ones might need more help.

---

## Multi-Dimensional Pattern Recognition

This project utilizes **Unsupervised Learning**. Unlike supervised learning (where we tell the computer what the answer is), unsupervised learning finds hidden structures in "unlabeled" data.

### The Science: K-Means Clustering

We represent each district as a point in a 3D space. The coordinates are:

*  $x = $ Average days of employment
*  $y = $ Total expenditure
*  $z = $ Agricultural focus percentage

### The Math: Euclidean Distance

The algorithm works by minimizing the distance between points and their group center (centroid). It calculates the **Euclidean Distance**:

$$ 
d(p, q) = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2 + (p_z - q_z)^2}
$$
1. **Initialization:** The computer picks 4 random points as "centers."
2. **Assignment:** Every district is assigned to the nearest center.
3. **Update:** The center moves to the average location of all its assigned districts.
4. **Repeat:** This continues until the groups stop changing.

We used the **Elbow Method** to find the perfect number of groups (K). By plotting the "tightness" of the groups (Inertia) against the number of clusters, we look for the "elbow" in the graph where adding more groups doesn't help much anymore.

---

## Strategic Policy & Resource Allocation

From a governance and socio-economic perspective, this analysis is a tool for **Performance Benchmarking**.

In a massive program like MGNREGA, looking at national averages hides local failures and successes. By clustering these districts, we move from "one-size-fits-all" policy to **Targeted Intervention**.

* **Cluster Analysis:** We found that some districts spend massive amounts of money but provide fewer days of work per household compared to smaller, more efficient districts.
* **Asset Creation:** By including "Agricultural Expenditure" as a feature, we distinguish between districts using MGNREGA as a temporary "safety net" (just paying wages) versus those using it for "capital formation" (building irrigation and improving soil).
* **Operational Efficiency:** This data allows a policy-maker to identify "outlier" districts that are underperforming relative to their peers in the same cluster, signaling a need for an audit or administrative support rather than just more funding.

---

## How to read the results

1. **`img/elbow_method.png`**: Shows why we chose 4 groups. Note that the elbow is almost non-visible for this data, meaning that the K-Means Clustering might not have been the best approach for this problem after all!
2. **`img/cluster_visualization.png`**: A map of districts showing the trade-off between spending and actual work delivered.
3. **`mgnrega_clustered_districts.csv`**: The master list. Look for your district and see which "Cluster" it fell into!