import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import skfuzzy as fuzz
import time
from U_K_means import U_K_means as ukm
from WUOFC import wuofc_algorithm



def create_data(mus_list, sigmas_list, no_samples, prior_probs=None):
    dict_of_arrays = {}
    for idx, moments in enumerate(zip(mus_list, sigmas_list)):
        mu, sigma = moments
        dict_of_arrays[f"{mu},{sigma}"] = np.random.multivariate_normal(mu, sigma
                                                                        ,no_samples//len(mus_list) if (prior_probs is None) else int(prior_probs[idx]*no_samples))
    return dict_of_arrays


def plot_data(dict_of_arrays, dims=2,ax_to_plot=None, mus_list=None, expected_mus_list = None):
    list_of_colors = ['red', 'blue', 'green', 'yellow', 'black', "pink", "purple", "orange", "brown", "grey"]

    if dims == 2:
        for idx, value in enumerate(dict_of_arrays.values()):
            ax_to_plot.scatter(value[:, 0], value[:, 1],s=2, color=list_of_colors[idx], label=f"Cluster {idx+1}")
            
        ax_to_plot.set_title("Data")
        if mus_list is not None:
            for idx, mu in enumerate(mus_list):
                ax_to_plot.scatter(mu[0], mu[1], color='k', marker='x', label=f"Centroid {idx+1}")
        if expected_mus_list is not None:
            for idx, mu in enumerate(expected_mus_list):
                ax_to_plot.scatter(mu[0], mu[1], color='pink', marker='x', s=10, label=f"Expected Centroid {idx+1}")
        # ax_to_plot.legend()


    if dims == 3:
        for idx, value in enumerate(dict_of_arrays.values()):
            ax_to_plot.scatter(value[:, 0], value[:, 1], value[:, 2])



def fuzzy_cmeans(dict_of_arrays, no_groups, beta = 2, error=0.005, maxiter=1000, ax_to_plot=None):
    list_of_colors = ['purple', 'orange', 'pink', 'brown', 'grey']
    data = np.vstack(dict_of_arrays.values()).T

    cntr, u, _, _ ,_ ,_, _ = fuzz.cluster.cmeans(data, no_groups, m=beta, error=error, maxiter=maxiter)
    cluster_membership = np.argmax(u, axis=0)

    if ax_to_plot is not None:
        for j in range(no_groups):
            ax_to_plot.scatter(data[1,cluster_membership == j],
            data[0,cluster_membership == j], color=list_of_colors[j], label = f"Cluster {j+1}")
            ax_to_plot.scatter(cntr[j, 1], cntr[j, 0], color='k', marker='x', s=100, label=f"Centroid {j+1}")
            # ax_to_plot.legend()
            ax_to_plot.set_title(f"Fuzzy K-Means with {no_groups} clusters")
    
    return cntr, cluster_membership


def compute_membership_exp(X, u, p, q):
    # X needs to be nxd
    # u needs to be kxn
    # p needs to be kxd
    # q is the fuzziness scalar

    #step 1: creating a list of covariances

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

    fuzzed_d_squared = d_squared**(1/1-q)
    new_u = fuzzed_d_squared / np.sum(fuzzed_d_squared, axis=0)
    new_p = ((X.T @ u.T)/np.sum(u, axis=1)).T


    return new_p, new_u


def compute_centroids_euclidean(X, p, q, img_distance):
    # X needs to be nxd
    # u needs to be kxn
    # p needs to be kxd
    # q is the fuzziness scalar

    d = np.zeros((p.shape[0], X.shape[1]))
    for k in range(p.shape[0]):
        d[k,:] = np.linalg.norm(X-p[k,:], axis=1)**(2/(1-q))
        u  = d / np.sum(d, axis=0)
    new_u = np.vstack([u, img_distance*np.ones(u.shape[1]).reshape(1,-1)])
    new_p = ((X.T @ u.T)/np.sum(u, axis=1)).T

    return new_p, new_u


def plot_evaluation_ukm(data, clust_cen_history, u_history):
    list_of_colors = ['red', 'blue', 'green', 'white', 'black', "cyan", "purple", "orange"
                        , "brown", "olive", "orchid", "indigo", "pink", "magenta", "lime", "teal",
                           "navy", "maroon", "coral"]
    indices_for_plotting = np.linspace(0, len(clust_cen_history) - 1, 6).astype(int)
    indices_for_plotting = indices_for_plotting.tolist()

    fig, axes = plt.subplots(len(indices_for_plotting)//2, 2, figsize=(40, 40))
    axes = axes.flatten()
    for ax_idx, idx in enumerate(indices_for_plotting):
        if clust_cen_history[idx].shape[0] <= 10:
            list_of_clustering = []
            for cluster in range(clust_cen_history[idx].shape[0]):
                list_of_clustering.append(data[u_history[idx][:,cluster] == 1])
            for j, sub_data in enumerate(list_of_clustering):
                axes[ax_idx].scatter(sub_data[:,0], sub_data[:,1],s=10, c=list_of_colors[j])
        else:
            axes[ax_idx].scatter(data[:,0], data[:,1],s=10, c='blue')
        axes[ax_idx].scatter(clust_cen_history[idx][:,0], clust_cen_history[idx][:,1],
                        c='red', marker='x', s=500)
        axes[ax_idx].set_title(f"U-K-means result at iteration {(idx if idx >=0 else len(clust_cen_history) + idx)}, No. of clusters: {clust_cen_history[idx].shape[0]}",
                            fontsize=30)
        axes[ax_idx].set_facecolor('lightgrey')

    fig.suptitle("Evaluation of U-K-means clustering", fontsize=20)
    fig.tight_layout()


def calculate_AV_AR_WUOFC(data_dict, data, data_mus, true_c, iterations=1, t_max=30):
    """data_dict: dictionary of arrays, where each array is a cluster
       data: the data matrix
       data_mus: the list of true means of every cluster"""
    
    iterations = iterations
    AR_sum = 0
    time_sum = 0
    no_of_successes = 0
    for iter in range(iterations):
        start_time = time.time()
        clust_cen, u,  u_history, clust_cen_history = wuofc_algorithm(data, K_max=10, q=1.5, epsilon=1e-4)
        time_taken = time.time() - start_time
        u = u.T




        mus_idx_to_group_idx = {}
        for idx, data_mu in enumerate(data_mus):
            mus_idx_to_group_idx[idx] = (np.argmin(np.linalg.norm(clust_cen - data_mu, axis=1)))

        sum = 0
        idx_finding_points = 0
        for mu_idx, temp_data in enumerate(data_dict.values()):
            relevant_u = u[idx_finding_points:idx_finding_points + temp_data.shape[0],:]
            clustered_u = np.argmax(relevant_u, axis=1)
            true_group_idx = mus_idx_to_group_idx[mu_idx]
            sum += np.sum(clustered_u == true_group_idx)
            idx_finding_points += temp_data.shape[0]

        if clust_cen.shape[0] == true_c:
            no_of_successes += 1
            AR_sum += sum/idx_finding_points
            time_sum += time_taken
    AR_avg = AR_sum/no_of_successes
    time_avg = time_sum/no_of_successes
    suceess_ratio = no_of_successes/iterations

    #round  to 4 decimal places
    AR_avg = round(AR_avg, 4)
    time_avg = round(time_avg, 4)
    success_pct = round(suceess_ratio*100, 2)

    print(f"Average AR: {AR_avg}")
    print(f"Average Time: {time_avg}")
    print(f"Success percentage: {success_pct}%")
    return AR_avg, time_avg, success_pct





def calculate_AV_AR(data_dict, data, data_mus, true_c, iterations=25, t_max=30):
    """data_dict: dictionary of arrays, where each array is a cluster
       data: the data matrix
       data_mus: the list of true means of every cluster"""
    
    iterations = iterations
    AR_sum = 0
    time_sum = 0
    no_of_successes = 0
    for iter in range(iterations):
        start_time = time.time()
        clust_cen, u, alpha, u_history, clust_cen_history = ukm(data, t_max=t_max)
        time_taken = time.time() - start_time
   


        mus_idx_to_group_idx = {}
        for idx, data_mu in enumerate(data_mus):
            mus_idx_to_group_idx[idx] = (np.argmin(np.linalg.norm(clust_cen - data_mu, axis=1)))

        sum = 0
        idx_finding_points = 0
        for mu_idx, temp_data in enumerate(data_dict.values()):
            relevant_u = u[idx_finding_points:idx_finding_points + temp_data.shape[0],:]
            clustered_u = np.argmax(relevant_u, axis=1)
            true_group_idx = mus_idx_to_group_idx[mu_idx]
            sum += np.sum(clustered_u == true_group_idx)
            idx_finding_points += temp_data.shape[0]

        if clust_cen.shape[0] == true_c:
            no_of_successes += 1
            AR_sum += sum/idx_finding_points
            time_sum += time_taken
    AR_avg = AR_sum/no_of_successes
    time_avg = time_sum/no_of_successes
    suceess_ratio = no_of_successes/iterations

    #round  to 4 decimal places
    AR_avg = round(AR_avg, 4)
    time_avg = round(time_avg, 4)
    success_pct = round(suceess_ratio*100, 2)

    print(f"Average AR: {AR_avg}")
    print(f"Average Time: {time_avg}")
    print(f"Success percentage: {success_pct}%")
    return AR_avg, time_avg, success_pct

