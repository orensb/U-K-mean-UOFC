# Unsupervised K-Means Clustering Algorithm 
## by KRISTINA P. SINAGA ANDMIIN-SHENYANG
This project analyzes and compares the performance of the Unsupervised K-Means (U-K-Means) clustering algorithm proposed by Sinaga and Yang with other clustering algorithms, particularly the WUOFC algorithm.

[https://ieeexplore.ieee.org/document/9072123?denied=](https://ieeexplore.ieee.org/document/9072123)

## Project Overview
- To understand and implement the U-K-Means algorithm
- To compare U-K-Means with UOFC and other benchmark algorithms
- To validate the results on both synthetic and real-world datasets

## Algorithm Background
#### U-K-Means:

- Innovative K-mean algorithm.
- Unsupervised clustering method that automatically determines the optimal number of clusters without requiring initialization or parameter selection
- Starting with each data point as its own cluster and converging to the optimal number based on the data structure.
- Utilizes an entropy-based penalty term and a competition schema to iteratively discard extra clusters (Hard Clustring)

<img src="https://github.com/user-attachments/assets/3584f0e4-a32f-4ab1-8488-d933d260d29f" width="600">

where: Z(ik) = the neighborhood relationship between point to center ; X(i) - data point ; A(k) - centre   

#### WUOFC:

- Extension of the Fuzzy-Cmean (algorithm based on membership matrix concept - Soft Clustring)
- starting from one cluster (randomly choosen) and increase it until a value of Kmax.
- Determines the optimal number of cluster
- Introduces a new "imaginary" centroid far from existing centroids to allow detection of distinct features and creation of new clusters.
- Use Eucliden and Exponential distance for calculation of memberships.
  


## Datasets:

Synthetic datasets:
   -  6 2D Gaussians mixture
   -  9 2D non-Gaussian distributions in hexagon

Real-world datasets:
   - Iris
   - Seeds
   - Sonar





## Validate Measures

 1. **Silhouette Width (SW):**
    provides a way to measure how similar an object is to its own cluster compared to other clusters.
    <p align="center">
    $SW = \frac{1}{N} \sum_{i=1}^{N} \left( s(i) = \frac{a(i) - b(i)}{\max \{ a(i), b(i) \} } \right)$
    </p>

    the result of SW is between [-1,1] when 1 means that a point is far from the neighbor cluster
3. **Bayesian Information Criterion (BIC):**  
   Based on the likelihood function and penalty for the number of parameters → balance between the big likelihood (similarity of a point to its cluster) and the number of clusters.

   <p align="center">
     BIC = -2 ln(L) + k ln(n)
   </p>

   Where L is the likelihood of a point with the probability of the centroid, and k is the (number of clusters × dimensions).





