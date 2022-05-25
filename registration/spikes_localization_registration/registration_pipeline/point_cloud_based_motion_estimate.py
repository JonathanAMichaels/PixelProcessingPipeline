"""
Three-dimensional spike localization and improved motion correction for Neuropixels recordings

Code for point-cloud-based motion estimation

Input: {x, y, z} localization estimate, spike times, amplitudes, geometry
Output: point-cloud-based motion estimation
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, notebook
import torch
import os

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

from scipy.stats import norm
from scipy.ndimage import shift
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import lsqr
from scipy.stats import zscore
from scipy.sparse import csr_matrix

from collections import defaultdict

def estimate_motion(x, y, z,
                    times, 
                    amps, 
                    geomarray,
                    min_ptp=6,
                    denoise_params={'n_neighbors':5, 'avg_dist_thresh': 25},
                    cluster_distance=35,
                    reg_win_num=10, # set to 1 for rigid
                    grid=None, # can manually set grid initialization
                    grid_step=25,
                    icp_max_iter=100,
                    distance_thresh=50, # maximum distance to consider when calculating icp update
                    iteration_num=1, # supports 1 iteration for now
                    robust_mode=True,
                    subsampling_rate=1.,
                    error_sigma=0.1,
                    time_sigma=1000,
                    robust_regression_sigma=1):
    """
    function that wraps around all functions for motion estimation
    generates point-clouds + preprocessing -> decentralized registration

    """
    
    # preprocessing: outlier removal + hierarchical clustering
    clustered_pcs = preprocess(x, y, z, times, amps,
                              min_ptp=min_ptp, denoise_params=denoise_params,
                              cluster_distance=cluster_distance)
                              
    # run icp + decentralized registration
    total_shift = run_icp(clustered_pcs, reg_win_num, geomarray, grid,
                                             grid_step = grid_step,
                                             subsampling_rate=subsampling_rate,
                                             icp_max_iter=icp_max_iter, 
                                             distance_thresh=distance_thresh,
                                             error_sigma=error_sigma, 
                                             time_sigma=time_sigma,
                                             robust_regression_sigma = robust_regression_sigma)
    
    np.save('total_shift.npy', total_shift)
    
    return total_shift

def run_icp(pcs, win_num, geomarray, grid, grid_step=25,
            subsampling_rate=1, icp_max_iter=100, 
            distance_thresh=50, error_sigma=0.1,
            time_sigma=1,robust_regression_sigma=1, robust_regression_n_iters=20):
    
    T = len(pcs)
    D = int(geomarray[:,1].ptp())
    
    # get subsampling matrix, if rate < 1
    if subsampling_rate < 1:
        S = get_subsampling_matrix(subsampling_rate, T)
    else:
        S = np.ones((T,T))
        
    displacement_matrix = np.zeros((win_num,T,T))
    loss = defaultdict(list)

    # get windows
    space = int(D//(win_num+1))
    locs = np.linspace(space, D-space, win_num, dtype=np.int32) # centers
    scale = D/(2*win_num)
    window_list = []
    for i in range(win_num):
        window = get_gaussian_window(D, T, locs[i], scale=scale)
        window_list.append(window)
    window_sum = np.sum(np.asarray(window_list), axis=0)
    
    # get grid
    if grid is None:
        grid = np.arange(-D/5,D/5,grid_step)
        
    for w in range(win_num):
        for i in notebook.tqdm(range(T)):
            pc_i = pcs[i].copy()
            features_i = pc_i[:,:3]
            
            #################
            # Give Gaussian window weights here
            #################
            
            w_i = norm.pdf(features_i[:,2], loc=locs[w], scale=scale)
            ptp_i = pc_i[:,-2]
            n_i = pc_i[:,-1]
            nn_i= NearestNeighbors()
            nn_i.fit(features_i)
            for j in range(T):
                if S[i,j] == 0:
                    continue

                pc_j = pcs[j].copy()
                features_j = pc_j[:,:3]
                
                w_j = norm.pdf(features_j[:,2], loc=locs[w], scale=scale)
                ptp_j = pc_j[:,-2]
                n_j = pc_j[:,-1]
                nn_j= NearestNeighbors()
                nn_j.fit(features_j)

                best_grid = 0
                best_error = np.Inf
                for g in range(grid.shape[0]):
                    d = grid[g]

                    features_j_sh = features_j+np.asarray([[0,0,d]])
                    features_i_sh = features_i-np.asarray([[0,0,d]])
                    
                    distance_sum = 0
                    distance_counts = 0
                    distance_arr, neighbor_idx = get_nearest_neighbor_distance(nn_j,features_i_sh)
                    weight = n_i*ptp_i+n_j[neighbor_idx[:,0]]*ptp_j[neighbor_idx[:,0]]
                    
                    if win_num > 1:
                        w_weight = w_i + w_j[neighbor_idx[:,0]]
                    else:
                        w_weight = 1

                    distance_sum += (weight*w_weight*distance_arr).sum() / (weight*w_weight).sum()

                    neighbor_idx = np.unique(neighbor_idx[:,0])
                    ptp_j_neigh = ptp_j[neighbor_idx]
                    n_j_neigh = n_j[neighbor_idx]
                    features_j_neigh = features_j_sh[neighbor_idx]
                    w_j_neigh = w_j[neighbor_idx]

                    distance_arr, neighbor_idx = get_nearest_neighbor_distance(nn_i,features_j_neigh)
                    kept_idx = np.where(distance_arr <= distance_thresh)[0]
                    neighbor_idx = neighbor_idx[kept_idx][:,0]

                    weight = n_i[neighbor_idx]*ptp_i[neighbor_idx]+n_j_neigh[kept_idx]*ptp_j_neigh[kept_idx]
                    
                    if win_num > 1:
                        w_weight = w_i[neighbor_idx] + w_j_neigh[kept_idx]
                    else:
                        w_weight = 1

                    distance_sum += (weight*w_weight*distance_arr[kept_idx]).sum() / (weight*w_weight).sum()

                    error = distance_sum
                    if error < best_error:
                        best_grid = grid[g]
                        best_error = error

                d = best_grid
                prev_distance = np.Inf
                for k in range(icp_max_iter):
                    features_j_sh = features_j+np.asarray([[0,0,d]])
                    features_i_sh = features_i-np.asarray([[0,0,d]])

                    distance_sum = 0
                    distance_counts = 0
                    distance_arr, neighbor_idx = get_nearest_neighbor_distance(nn_j,features_i_sh)
                    weight = n_i*ptp_i+n_j[neighbor_idx[:,0]]*ptp_j[neighbor_idx[:,0]]
                    if win_num > 1:
                        w_weight = w_i + w_j[neighbor_idx[:,0]]
                    else:
                        w_weight = 1
                    distance_sum += (weight*w_weight*distance_arr).sum() / (weight*w_weight).sum()

                    neighbor_idx = np.unique(neighbor_idx[:,0])
                    ptp_j_neigh = ptp_j[neighbor_idx]
                    n_j_neigh = n_j[neighbor_idx]
                    features_j_neigh = features_j_sh[neighbor_idx]
                    w_j_neigh = w_j[neighbor_idx]

                    distance_arr, neighbor_idx = get_nearest_neighbor_distance(nn_i,features_j_neigh)
                    kept_idx = np.where(distance_arr <= distance_thresh)[0]
                    neighbor_idx = neighbor_idx[kept_idx][:,0]

                    weight = n_i[neighbor_idx]*ptp_i[neighbor_idx]+n_j_neigh[kept_idx]*ptp_j_neigh[kept_idx]
                    if win_num > 1:
                        w_weight = w_i[neighbor_idx] + w_j_neigh[kept_idx]
                    else:
                        w_weight = 1
                    distance_sum += (weight*w_weight*distance_arr[kept_idx]).sum() / (weight*w_weight).sum()

                    z_diff = features_i[neighbor_idx][:,2]-features_j_neigh[kept_idx][:,2]
                    distance = (weight*w_weight*z_diff).sum()/(weight*w_weight).sum()

                    d += distance
                    
                    loss[(w,i,j)].append((distance_sum, d))

                    if np.abs(prev_distance - distance_sum) < 1e-4:
                        break

                    prev_distance = distance_sum

                displacement_matrix[w,i,j] = d
                
    error_mat = np.zeros((win_num,T,T))
    for w in range(win_num):
        for i in notebook.tqdm(range(T)):
            for j in range(T):
                if S[i,j] == 0:
                    continue
                error_mat[w,i,j] = loss[(w,i,j)][-1][0]
                
    ps = np.zeros((win_num, T))
    for w in range(win_num):
        error_mat_S = error_mat[w][np.where(S != 0)]
        W1 = np.exp(-((error_mat_S-error_mat_S.min())/(error_mat_S.max()-error_mat_S.min()))/error_sigma)

        W2 = np.exp(-squareform(pdist(np.arange(error_mat[w].shape[0])[:,None]))/time_sigma)
        W2 = W2[np.where(S != 0)]

        W = (W2*W1)[:,None]

        I, J = np.where(S != 0)
        V = displacement_matrix[w][np.where(S != 0)]
        M = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),I)))
        N = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),J)))
        A = M - N
        idx = np.ones(A.shape[0]).astype(bool)
        for i in tqdm(range(robust_regression_n_iters)):
            p = lsqr(A[idx].multiply(W[idx]), V[idx]*W[idx][:,0])[0]
            idx = np.where(np.abs(zscore(A@p-V)) <= robust_regression_sigma)
        ps[w] = p
        
    total_shift = np.zeros((D,T))
    for i, w in enumerate(window_list):
        total_shift += w * ps[i][None]
    total_shift /= window_sum
    
    return total_shift
                              
def get_nearest_neighbor_distance(A_nn,B,n_neighbors=1):
    return A_nn.kneighbors(B,n_neighbors=n_neighbors,return_distance=True)

def preprocess(x, y, z, times, amps, min_ptp, denoise_params, cluster_distance):
    n_neighbors = denoise_params['n_neighbors']
    avg_dist_thresh = denoise_params['avg_dist_thresh']
    
    ptp_select = np.where(amps > min_ptp)
    x = x[ptp_select]
    y = y[ptp_select]
    z = z[ptp_select]
    amps = amps[ptp_select]
    times = times[ptp_select]
    
    T = int(np.ceil(max(times)))
    
    # bin times
    spike_indices = dict()
    for i in range(T):
        indices = np.logical_and(times >= i, times < i+1)
        spike_indices[i] = indices
        
    # remove outliers
    features = []
    for i in notebook.tqdm(range(T)):
        features_i = np.concatenate((x[spike_indices[i]][:,None],
                               y[spike_indices[i]][:,None],
                               z[spike_indices[i]][:,None],
                               amps[spike_indices[i]][:,None]), axis=-1)
        nn= NearestNeighbors()
        nn.fit(features_i[:,:-1])
        distance_arr, neighbor_idx = get_nearest_neighbor_distance(nn,features_i[:,:-1],n_neighbors=n_neighbors)
        distance_arr = distance_arr.sum(1)/(n_neighbors-1)

        kept_idx = np.where(distance_arr < avg_dist_thresh)
        features.append(features_i[kept_idx])
        
    # hierarchical clustering
    clustered_features = []
    for i in notebook.tqdm(range(T)):
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=cluster_distance)

        points_c = features[i][:,:-1]
        clustering.fit(points_c)

        points = features[i]

        clusters = np.unique(clustering.labels_)
        means = []
        for i in range(clusters.shape[0]):
            n_points = np.where(clustering.labels_ == i)[0].astype(int).sum()
            if n_points > 1:
                mean = points[np.where(clustering.labels_ == i)].mean(0)
            else:
                mean = points[np.where(clustering.labels_ == i)][0]
            means.append(np.hstack([mean,n_points]))
        means = np.asarray(means)
        clustered_features.append(means)
        
    return clustered_features

def get_gaussian_window(height, width, loc, scale=1):
    """
    Gets Gaussian windows along the depth of the raster

    """
    window = np.zeros((height,width))
    for i in range(height):
        window[i] = norm.pdf(i, loc=loc, scale=scale)
    return window / window.max()

def calc_displacement(displacement, n_iter = 1000):
    """
    Calculates the displacement estimate given the displacement matrix (decentralized registration)

    """
    p = torch.zeros(displacement.shape[0]).cuda()
    displacement = torch.from_numpy(displacement).cuda().float()
    n_batch = displacement.shape[0]
    pprev = p.clone()
    for i in notebook.tqdm(range(n_iter)):
        repeat1 = p.repeat_interleave(n_batch).reshape((n_batch, n_batch))
        repeat2 = p.repeat_interleave(n_batch).reshape((n_batch, n_batch)).T
        mat_norm = displacement + repeat1 - repeat2
        p += 2*(torch.sum(displacement-torch.diag(displacement), dim=1) - (n_batch-1)*p)/torch.norm(mat_norm)
        del mat_norm
        del repeat1
        del repeat2
        if torch.allclose(pprev, p):
            break
        else:
            del pprev
            pprev = p.clone()
    disp = np.asarray(p.cpu())
    del p
    del pprev
    del displacement
    torch.cuda.empty_cache()
    return disp

def calc_displacement_robust(displacement_matrix, 
                             error_mat, 
                             S, 
                             error_sigma, 
                             time_sigma, 
                             robust_regression_sigma, n_iter=20):
    error_mat_S = error_mat[np.where(S != 0)]
    W1 = np.exp(-((error_mat_S-error_mat_S.min())/(error_mat_S.max()-error_mat_S.min()))/error_sigma)

    W2 = np.exp(-squareform(pdist(np.arange(error_mat.shape[0])[:,None]))/time_sigma)
    W2 = W2[np.where(S != 0)]

    W = (W2*W1)[:,None]
    
    I, J = np.where(S != 0)
    V = displacement_matrix[np.where(S != 0)]
    M = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),I)))
    N = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),J)))
    A = M - N
    idx = np.ones(A.shape[0]).astype(bool)
    for i in notebook.tqdm(range(n_iter)):
        p = lsqr(A[idx].multiply(W[idx]), V[idx]*W[idx][:,0])[0]
        idx = np.where(np.abs(zscore(A@p-V)) <= robust_regression_sigma)
    return p

def get_subsampling_matrix(rate, T):
    T = int(T) # make sure it's int
    S = np.zeros((T,T))
    p = np.log(T)/T
    while S.sum() < T*T*p:
        r = np.random.permutation(np.arange(T))
        S[np.arange(T),r] = 1
    S = np.bitwise_or(S.astype(bool), S.T.astype(bool)).astype(int)
    return S
