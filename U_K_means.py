import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def calc_neighbourhood_matrix(X, clust_cen, alpha, gamma):
    u = np.zeros((X.shape[0], clust_cen.shape[0]))
    u_relative = np.zeros((X.shape[0], clust_cen.shape[0]))
    for k in range(clust_cen.shape[0]):
        X_minus_a_k = X - clust_cen[k, :].reshape(1, -1)
        squared_normed_x_minus_a_k = LA.norm(X_minus_a_k, axis=1, ord=2)**2
        metric = squared_normed_x_minus_a_k - gamma * np.log(alpha[k])
        u_relative[:, k] = metric
    # fill in 1 for every column in the index of the minimal value of u_relative
    u[np.arange(X.shape[0]), np.argmin(u_relative, axis=1)] = 1
    return u, u_relative

def U_K_means(X, t_max =100, tol=1e-10):
    # n is the number of data points
    # d is the dimension of the data points
    # k is the number of clusters
    # X is an nxd matrix
    # u is an nxk matrix
    # clust_cen is a kxd matrix


    #X is the data matrix - every row is a data point
    clust_cen = X.copy()
    #i'm checking how to get out of the first iterarion bug
    #randomly sample 1/2 of the data points to be the initial cluster centers
    clust_cen = X[np.random.choice(X.shape[0], int(np.floor((99/100)*X.shape[0])), replace=False), :]
    no_of_clusters = clust_cen.shape[0]


    alpha = np.ones(no_of_clusters)/no_of_clusters

    
    err = 10
    gamma = 1
    beta = 1
    err = tol + 1
    u_history = []
    clust_cen_history = []

    c_history = [clust_cen.shape[1]]
    t = 0
    while no_of_clusters > 1 and err >= tol and t < t_max:
        t += 1

        #step 2: calculate the neighbourhood matrix
        u, u_relative = calc_neighbourhood_matrix(X, clust_cen, alpha, gamma)

        #step 3: compute gamma
        gamma = np.exp(-no_of_clusters/250) #250 is a constant


        #step 4: update alpha

   
        normalized_sum_u_i_k = (np.sum(u, axis=0) / X.shape[0]).reshape(-1)
        alpha_entropy = -np.sum(alpha * np.log(alpha))
        temp_1 = (beta/gamma)* alpha * (np.log(alpha) + alpha_entropy)

        new_alpha = normalized_sum_u_i_k + temp_1


        #step 5: update beta

        eta = np.min(np.array([1, 1/(t**np.floor((X.shape[1]/2)-1))]))

        term_1 = (1/clust_cen.shape[0])*np.sum(np.exp(-eta*X.shape[0]*np.abs(new_alpha - alpha)))

        term_2_1 = 1 - np.max(normalized_sum_u_i_k)
        term_2_2 = -np.max(alpha)*np.sum(np.log(alpha))

        beta = np.min(np.array([term_1, term_2_1/term_2_2]))

        #step 6 update number of clusters:
        indices_to_keep = new_alpha >= 1/X.shape[0] 
        new_alpha = new_alpha[indices_to_keep]
        u_relative = u_relative[:, indices_to_keep]
        no_of_clusters = new_alpha.reshape(-1).shape[0]
        c_history.append(no_of_clusters)
        if t >= 60:
            if c_history[t] == c_history[t-60]:
                beta = 0
        
        #re normalize alpha and update it
        alpha = new_alpha/np.sum(new_alpha)
        #re normalize every row in u_relative
        u_relative = u_relative/np.sum(u_relative, axis=1).reshape(-1, 1)
        assert u_relative.shape[1] == no_of_clusters
        u = np.zeros((X.shape[0], no_of_clusters))
        u[np.arange(X.shape[0]), np.argmin(u_relative, axis=1)] = 1

        u_history.append(u)


        #step 7: update cluster centers
        new_clust_cen = np.dot(u.T, X)/np.sum(u, axis=0).reshape(-1, 1)

        clust_cen_history.append(new_clust_cen)

        err = new_clust_cen - clust_cen[indices_to_keep, :]
        norm_err = LA.norm(err, axis = 1, ord=2)
        err = np.max(norm_err)

        clust_cen = new_clust_cen

    return clust_cen, u, alpha, u_history, clust_cen_history







    
