import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

from sklearn.metrics import davies_bouldin_score


import numpy as np

def initialize_centroids(X, K):
    # Initialize centroids using a heuristic method (e.g., k-means++)
    centroids = []
    centroids.append(np.mean(X, axis=0))  # Start with a single centroid as the mean of all data points
    return np.array(centroids)
def compute_membership_exp(X, u, p, q):
    """
    Compute membership matrix using exponential distance
    X: data matrix (n_samples x n_features)
    u: initial membership matrix (n_clusters x n_samples)
    p: cluster centers (n_clusters x n_features)
    q: fuzziness parameter
    """
    n_samples = X.shape[0]
    n_clusters = p.shape[0]
    iter_max = 10000
    eps = 1e-4
    
    for iter in range(iter_max):
        # 1. Calculate Fuzzy covariance matrices for each cluster
        F_k = []
        a_k = np.sum(u, axis=1)  # Sum of memberships for each cluster
        
        for k in range(n_clusters):
            diff = X - p[k]  # (n_samples x n_features)
            # Calculate weighted covariance matrix
            F_k_sum = np.zeros((X.shape[1], X.shape[1]))
            for i in range(n_samples):
                diff_i = diff[i].reshape(-1, 1)
                F_k_sum += u[k, i] * (diff_i @ diff_i.T)
            F_k.append(F_k_sum / a_k[k])
        
        # 2. Calculate exponential distances
        d_squared = np.zeros((n_clusters, n_samples))
        for k in range(n_clusters):
            for i in range(n_samples):
                diff = (X[i] - p[k]).reshape(-1, 1)
                try:
                    F_k_inv = np.linalg.pinv(F_k[k])  # Use pseudoinverse for stability
                    det_term = np.sqrt(np.linalg.det(F_k[k])) / a_k[k]
                    exp_term = np.exp(0.5 * diff.T @ F_k_inv @ diff)
                    d_squared[k, i] = det_term * exp_term[0, 0]
                except:
                    d_squared[k, i] = np.inf
        
        # 3. Update membership matrix
        d_powered = d_squared**(1/(q-1))
        new_u = d_powered / np.sum(d_powered, axis=0)
        
        # 4. Update cluster centers
        new_p = np.zeros_like(p)
        for k in range(n_clusters):
            new_p[k] = np.sum(new_u[k].reshape(-1, 1) * X, axis=0) / np.sum(new_u[k])
        
        # Check convergence
        if np.max(np.abs(new_u - u)) < eps:
            break
            
        u = new_u
        p = new_p
        
    return p, u



def compute_centroids_euclidean(X, p, q, img_distance):
    # X needs to be nxd
    # u needs to be kxn
    # p needs to be kxd
    # q is the fuzziness scalar
    d = np.zeros((p.shape[0], X.shape[0]))
    # print(f" shape of d is {d.shape}")
    u_prev = np.zeros((p.shape[0]+1, X.shape[0]))
    iter = 0
    while iter<10000:
        for k in range(p.shape[0]):
            d[k,:] = np.linalg.norm(X-p[k,:], axis=1)**(2/(q-1))

            # first iteration with (1,1000) we get same d and d_sum
            d_sum = np.sum(d, axis=0)
        # for the first iteration we need to add the img_distance to the d matrix
        if iter == 0:
          d = np.vstack([d, (img_distance*np.ones(X.shape[0])**(1/(q-1))).reshape(1,-1)])

        new_u  = (1/d) / np.sum(1/d, axis=0)
        #updating the centroids
        new_p = ((X.T @ new_u.T)/np.sum(new_u, axis=1)).T
        if ((np.max(np.abs(new_u - u_prev))) < 1e-4) and iter>1:
            break

        # update the previous u and p for next iteration
        u_prev = new_u
        p = new_p
        iter += 1

    return new_p, new_u



def wuofc_algorithm(X, K_max, q=1.5, epsilon=1e-4):
    """
    Weighted Unsupervised Optimal Fuzzy Clustering algorithm
    
    Parameters:
        X: Input data of shape (n_samples, n_features)
        K_max: Maximum number of clusters to try
        q: Fuzzifier parameter (default=1.5)
        epsilon: Convergence threshold (default=1e-4)
        normalize_scores: Whether to normalize validation scores (default=True)
    """
    
    # Initialize centroids using a heuristic method
    centroids = initialize_centroids(X, 1)  # Start with K=1
    silhouette_score_list = []
    score_CH_list = []
    db_score_list = []   

    best_db = 3
    best_ch = -1
    best_score = -1
    best_cntrs = None
    best_membership = None
    best_K = 0
#    Compute the covariance matrix for the entire dataset
    covariance_matrix = np.cov(X.T)
    # Calculate the sum of the diagonal elements of the covariance matrix
    img_distance = 10 * np.sum(np.diag(covariance_matrix))
    # print(f"x size is {X.shape}")
    # print(f"centroid is {centroids}")

    for K in range(7):
        # Initialize membershiEp degrees size of K x N
        new_p , new_u = compute_centroids_euclidean(X, centroids, q, img_distance)
        print(new_u)
        
        new_p , new_u = compute_membership_exp(X,new_u,new_p,q)
        print(K)
        print(new_u)
        centroids = new_p
        cluster_membership = np.argmax(new_u, axis=0)
        unique_labels = np.unique(cluster_membership)
        # if len(unique_labels) > 1:
        score = silhouette_score(X, cluster_membership)
        silhouette_score_list.append(score)


        score_CH = calinski_harabasz_score(X, cluster_membership)/10
        score_CH_list.append(score_CH)

        db_score = davies_bouldin_score(X, cluster_membership)
        db_score_list.append(db_score)

        if len(unique_labels) > 1 and score > best_score:
            best_score = score

        if score_CH > best_ch:
            best_ch = score_CH
            best_cntrs = new_p
            best_membership = new_u
            best_K = K+2
        if db_score < best_db:
            best_db = db_score
       
        
  

    for index, score in enumerate(silhouette_score_list, start=1):
        print(f"Cluster number {index+1} SW is: {score}")
    print(f"\n best SW is: {best_score} for cluster: {best_K}")

    print("\nCH Scores by Line:")
    for index, score in enumerate(score_CH_list, start=1):
        print(f"Cluster number {index+1}: CH is {score}")
    print(f"\n best CH is: {best_ch} for cluster: {best_K}  ")

    print("\nDB Scores by Line:")
    for index, score in enumerate(db_score_list, start=1):
        print(f"Cluster number {index+1}: DB is {score}")
    print(f"\n best DB is: {best_db} for cluster: {best_K} ")


    return best_cntrs, best_membership, best_K, silhouette_score_list




        







