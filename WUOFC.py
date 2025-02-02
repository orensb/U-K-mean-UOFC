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
    # X needs to be nxd
    # u needs to be kxn
    # p needs to be kxd
    # q is the fuzziness scalar

    #step 1: creating a list of covariances
    iter = 0
    while iter<10000:
        # print(f" p exp function is {p}")
        list_of_covs = []
        for k in range(p.shape[0]):
            sum = 0
            for i in range(u.shape[1]):
                sum+= u[k,i]* (X[i,:]-p[k, :]).reshape(-1,1) @ (X[i,:]-p[k, :]).reshape(1,-1)
            sum/= np.sum(u[k,:])
            list_of_covs.append(sum)

        #step 2: computing the membership matrix

        d_squared = np.zeros_like(u)
        for k in range(p.shape[0]):
            for i in range(u.shape[1]):
                exp_term = np.exp(((X[i,:]-p[k, :]).reshape(1,-1) @ np.linalg.inv(list_of_covs[k]) @ 
                                (X[i,:]-p[k, :]).reshape(-1,1))/2)
                d_squared[k,i] = np.linalg.det(list_of_covs[k])**(1/2) * exp_term / np.sum(u[k,:])

        fuzzed_d_squared = d_squared**(1/q-1)
        new_u = fuzzed_d_squared / np.sum(fuzzed_d_squared, axis=0)
        new_p = ((X.T @ new_u.T)/np.sum(new_u, axis=1)).T
        # print(f" new_u is exponential {new_u}")

        if ((np.max(np.abs(new_u - u))) < 1e-4):
            break
        u = new_u
        p = new_p
        iter += 1

    return new_p, new_u


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
    # Initialize centroids using a heuristic method
    centroids = initialize_centroids(X, 1)  # Start with K=1
    # print(f" centroids are {centroids}")
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

    for K in range(K_max):
        # Initialize membership degrees size of K x N
        new_p , new_u = compute_centroids_euclidean(X, centroids, q, img_distance)

        centroids = new_p
        cluster_membership = np.argmax(new_u, axis=0)
        
        score = silhouette_score(X, cluster_membership)
        silhouette_score_list.append(score)

        score_CH = calinski_harabasz_score(X, cluster_membership)/10
        score_CH_list.append(score_CH)

        db_score = davies_bouldin_score(X, cluster_membership)
        db_score_list.append(db_score)

        if score > best_score:
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




        







