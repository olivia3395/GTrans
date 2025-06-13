

### some code are from https://github.com/ahxt/g-mixup


### some code are from https://github.com/HongtengXu/SGWB-Graphon

import copy
import cv2
import numpy as np
import torch

from skimage.restoration import denoise_tv_chambolle
from typing import List, Tuple


def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()


def align_graphs(graphs: List[np.ndarray],
                 padding: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending


        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree
            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph
            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            # normalized_node_degree = np.ones(sorted_node_degree.shape) / sorted_node_degree.shape[0]
            # normalized_node_degrees.append(normalized_node_degree)
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

    return aligned_graphs, normalized_node_degrees, max_num, min_num


def align_graphs_centrality(graphs: List[np.ndarray],
                 padding: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree
            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph
            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            # normalized_node_degree = np.ones(sorted_node_degree.shape) / sorted_node_degree.shape[0]
            # normalized_node_degrees.append(normalized_node_degree)
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

    return aligned_graphs, normalized_node_degrees, max_num, min_num




def guess_rank(matrix: torch.Tensor) -> int:
    """
    A function to guess the rank of a matrix
    :param matrix: a torch.Tensor matrix
    :return:
    """
    n = matrix.size(0)
    m = matrix.size(1)
    epsilon = torch.sum(matrix != 0) / ((n * m) ** 0.5)

    u, s, v = torch.svd(matrix, compute_uv=False)
    max_num = min([100, s.size(0)])
    s = s[:max_num]
    s, _ = torch.sort(s, descending=True)
    diff_s1 = s[:-1] - s[1:]
    diff_s1 = diff_s1 / torch.mean(diff_s1[-10:])
    r1 = torch.zeros(1)
    gamma = 0.05
    while r1.item() <= 0:
        cost = torch.zeros(diff_s1.size(0))
        for i in range(diff_s1.size(0)):
            cost[i] = gamma * torch.max(diff_s1[i:]) + i + 1

        idx = torch.argmin(cost)
        r1 = torch.argmax(idx)
        gamma += 0.05

    cost = torch.zeros(diff_s1.size(0))
    for i in range(diff_s1.size(0)):
        cost[i] = s[i + 1] + ((i + 1) * epsilon ** 0.5) * s[0] / epsilon

    idx = torch.argmin(cost)
    r2 = torch.max(idx)
    return max([r1.item(), r2.item()])



def averaging_graphs(aligned_graphs: List[np.ndarray], trans: List[np.ndarray], ws: np.ndarray) -> np.ndarray:
    """
    sum_k w_k * (Tk @ Gk @ Tk')
    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param trans: a list of (Nb, Ni) transport matrices
    :param ws: (K, ) weights
    :return: averaged_graph: a (Nb, Nb) adjacency matrix
    """
    averaged_graph = 0
    for k in range(ws.shape[0]):
        averaged_graph += ws[k] * (trans[k] @ aligned_graphs[k] @ trans[k].T)
    return averaged_graph


def proximal_ot(cost: np.ndarray,
                p1: np.ndarray,
                p2: np.ndarray,
                iters: int,
                beta: float,
                error_bound: float = 1e-10,
                prior: np.ndarray = None) -> np.ndarray:
    """
    min_{T in Pi(p1, p2)} <cost, T> + beta * KL(T | prior)

    :param cost: (n1, n2) cost matrix
    :param p1: (n1, 1) source distribution
    :param p2: (n2, 1) target distribution
    :param iters: the number of Sinkhorn iterations
    :param beta: the weight of proximal term
    :param error_bound: the relative error bound
    :param prior: the prior of optimal transport matrix T, if it is None, the proximal term degrades to Entropy term
    :return:
        trans: a (n1, n2) optimal transport matrix
    """
    if prior is not None:
        kernel = np.exp(-cost / beta) * prior
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    a = np.ones(p1.shape) / p1.shape[0]
    K_sum = K.sum()
    if K_sum == 0 or np.isnan(K_sum):
        print("🚨 K matrix invalid — early exit from OT.")
        return np.ones((p1.shape[0], p2.shape[0])) / (p1.shape[0] * p2.shape[0])

    b = []
    i = 0

    while relative_error > error_bound and i < iters:
        b = p2 / (np.matmul(kernel.T, a))
        a_new = p1 / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = copy.deepcopy(a_new)
        i += 1
    trans = np.matmul(a, b.T) * kernel
    return trans

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity

import torch
from typing import List
import torch
from typing import List

import numpy as np
import torch
from typing import List

def est_nbdsmooth_like_r_numpy(A: np.ndarray):
    """
    Core single-graph smoothing using R's est.nbdsmooth logic.
    """
    n = A.shape[0]
    A_sq = (A @ A) / n

    # Step 1: Structural dissimilarity
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.abs(A_sq[i] - A_sq[j])
            diff[i] = 0
            diff[j] = 0
            D[i, j] = D[j, i] = np.max(diff)

    # Step 2: Bandwidth (theoretical default)
    h = np.sqrt(np.log(n) / n)

    # Step 3: Local average based on dissimilarity quantile
    P_hat = np.zeros((n, n))
    for i in range(n):
        threshold = np.quantile(D[i], h)
        neighbors = np.where(D[i] < threshold)[0]
        if len(neighbors) == 0:
            neighbors = [i]
        P_hat[i, :] = A[neighbors, :].mean(axis=0)

    # Step 4: Symmetrize
    P_hat = 0.5 * (P_hat + P_hat.T)
    return P_hat



def neighborhood_smoothing(
    graphs: List[torch.Tensor],
    device: str = 'cuda'
) -> torch.Tensor:
    """
    R-style neighborhood smoothing for a list of adjacency matrices.
    Applies est.nbdsmooth_like_r logic to each graph and returns the average.

    Args:
        graphs: list of (N_i, N_i) adjacency matrices (torch.Tensor)
        device: device for final tensor output

    Returns:
        smoothed_avg: (N_max, N_max) average of smoothed matrices
    """
    smoothed_list = []
    max_nodes = max([adj.shape[0] for adj in graphs])

    for adj in graphs:
        A = adj.cpu().numpy()
        P_hat = est_nbdsmooth_like_r_numpy(A)  # Apply R-style smoothing

        # Pad to max_nodes
        padded = np.zeros((max_nodes, max_nodes))
        n = A.shape[0]
        padded[:n, :n] = P_hat

        smoothed_tensor = torch.tensor(padded, dtype=torch.float32, device=device)
        smoothed_list.append(smoothed_tensor)

    smoothed_avg = torch.stack(smoothed_list).mean(dim=0)
    return smoothed_avg





import numpy as np
import ot

def compute_ot_matrix(
    P_s: np.ndarray,
    P_t: np.ndarray,
    epsilon: float = 1e-5,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute Gromov-Wasserstein OT matrix using POT's built-in gromov_wasserstein2 (like R code).

    Parameters:
        P_s: (n_s, n_s) source graph structure matrix (e.g. smoothed adjacency or graphon)
        P_t: (n_t, n_t) target graph structure matrix
        epsilon: entropic regularization coefficient
        verbose: whether to print progress info

    Returns:
        pi: (n_s, n_t) optimal transport plan matrix (NumPy array)
    """
    n_s = P_s.shape[0]
    n_t = P_t.shape[0]

    # Uniform marginal distributions
    p = np.ones(n_s) / n_s
    q = np.ones(n_t) / n_t

    # Compute GW and get log
    gw_distance_squared, log = ot.gromov.gromov_wasserstein2(
        C1=P_s, C2=P_t, p=p, q=q,
        loss_fun='square_loss',
        epsilon=epsilon,
        verbose=verbose,
        log=True
    )

    pi = log['T']  # optimal transport plan
    return pi




def estimate_graphon(P_s: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Transfer source graphon estimation to target using OT matrix.
    
    Parameters:
        P_s: (n_s, n_s) source graphon's connection probability matrix
        pi:  (n_s, n_t) optimal transport matrix between source and target nodes
    
    Returns:
        P_t_trans: (n_t, n_t) transferred structure on target side
    """
    if isinstance(P_s, np.ndarray):
        P_s = torch.tensor(P_s, dtype=torch.float32, device=pi.device)
    return pi.T @ P_s @ pi


def residual_smoothing(A_t, P_t_trans, k: int = 10) -> np.ndarray:
    """
    Perform residual correction by smoothing the residual matrix: A_t - P_t_trans.

    Parameters:
        A_t: (n, n) adjacency matrix of target graph (np.ndarray or torch.Tensor)
        P_t_trans: (n, n) transferred structure matrix from source (np.ndarray or torch.Tensor)
        k: number of neighbors used for neighborhood smoothing

    Returns:
        P_t_res: (n, n) smoothed residual structure (as np.ndarray)
    """
   
    if torch.is_tensor(A_t):
        A_t = A_t.detach().cpu().numpy()
    if torch.is_tensor(P_t_trans):
        P_t_trans = P_t_trans.detach().cpu().numpy()

    l
    R_t = A_t - P_t_trans

    
    R_t = np.clip(R_t, 0, 1)


    P_t_res = neighborhood_smoothing(R_t, k=k)
    
    return P_t_res


import numpy as np




def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon





def guess_rank(matrix: torch.Tensor) -> int:
    """
    A function to guess the rank of a matrix
    :param matrix: a torch.Tensor matrix
    :return:
    """
    n = matrix.size(0)
    m = matrix.size(1)
    epsilon = torch.sum(matrix != 0) / ((n * m) ** 0.5)

    u, s, v = torch.svd(matrix, compute_uv=False)
    max_num = min([100, s.size(0)])
    s = s[:max_num]
    s, _ = torch.sort(s, descending=True)
    diff_s1 = s[:-1] - s[1:]
    diff_s1 = diff_s1 / torch.mean(diff_s1[-10:])
    r1 = torch.zeros(1)
    gamma = 0.05
    while r1.item() <= 0:
        cost = torch.zeros(diff_s1.size(0))
        for i in range(diff_s1.size(0)):
            cost[i] = gamma * torch.max(diff_s1[i:]) + i + 1

        idx = torch.argmin(cost)
        r1 = torch.argmax(idx)
        gamma += 0.05

    cost = torch.zeros(diff_s1.size(0))
    for i in range(diff_s1.size(0)):
        cost[i] = s[i + 1] + ((i + 1) * epsilon ** 0.5) * s[0] / epsilon

    idx = torch.argmin(cost)
    r2 = torch.max(idx)
    return max([r1.item(), r2.item()])



def averaging_graphs(aligned_graphs: List[np.ndarray], trans: List[np.ndarray], ws: np.ndarray) -> np.ndarray:
    """
    sum_k w_k * (Tk @ Gk @ Tk')
    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param trans: a list of (Nb, Ni) transport matrices
    :param ws: (K, ) weights
    :return: averaged_graph: a (Nb, Nb) adjacency matrix
    """
    averaged_graph = 0
    for k in range(ws.shape[0]):
        averaged_graph += ws[k] * (trans[k] @ aligned_graphs[k] @ trans[k].T)
    return averaged_graph


def proximal_ot(cost: np.ndarray,
                p1: np.ndarray,
                p2: np.ndarray,
                iters: int,
                beta: float,
                error_bound: float = 1e-10,
                prior: np.ndarray = None) -> np.ndarray:
    """
    min_{T in Pi(p1, p2)} <cost, T> + beta * KL(T | prior)

    :param cost: (n1, n2) cost matrix
    :param p1: (n1, 1) source distribution
    :param p2: (n2, 1) target distribution
    :param iters: the number of Sinkhorn iterations
    :param beta: the weight of proximal term
    :param error_bound: the relative error bound
    :param prior: the prior of optimal transport matrix T, if it is None, the proximal term degrades to Entropy term
    :return:
        trans: a (n1, n2) optimal transport matrix
    """
    if prior is not None:
        kernel = np.exp(-cost / beta) * prior
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    a = np.ones(p1.shape) / p1.shape[0]
    b = []
    i = 0

    while relative_error > error_bound and i < iters:
        b = p2 / (np.matmul(kernel.T, a))
        a_new = p1 / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = copy.deepcopy(a_new)
        i += 1
    trans = np.matmul(a, b.T) * kernel
    return trans


def node_cost_st(cost_s: np.ndarray, cost_t: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate invariant cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
    """
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    f1_st = np.repeat((cost_s ** 2) @ p_s, n_t, axis=1)
    f2_st = np.repeat(((cost_t ** 2) @ p_t).T, n_s, axis=0)
    cost_st = f1_st + f2_st
    return cost_st


def gw_cost(cost_s: np.ndarray, cost_t: np.ndarray, trans: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate the cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        trans: (n_s, n_t) array, the learned optimal transport between two graphs
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost: (n_s, n_t) array, the estimated cost between the nodes in two graphs
    """
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t)
    return cost_st - 2 * (cost_s @ trans @ cost_t.T)


def est_LG_like(A, K=None):
    """
    Reimplementation of est.LG using largest gap on sorted empirical degrees
    and histogram estimation, following R::graphon package behavior exactly.

    Parameters:
        A (ndarray): (n, n) binary symmetric adjacency matrix
        K (int): number of blocks (if None, use K=2 by default)

    Returns:
        P_hat (ndarray): (n, n) estimated edge probability matrix
        z (ndarray): (n,) block assignments
    """
    A = A.copy()
    np.fill_diagonal(A, 0)
    n = A.shape[0]

    # Step 1: sort nodes by degree
    degrees = A.sum(axis=1)
    sorted_idx = np.argsort(degrees)
    degrees_sorted = degrees[sorted_idx]

    # Step 2: largest gap criterion to get K blocks
    if K is None:
        K = 2
    gaps = degrees_sorted[1:] - degrees_sorted[:-1]
    top_gaps = np.argsort(gaps)[-(K - 1):]
    top_gaps = np.sort(top_gaps + 1)  # get split indices
    split_points = np.concatenate([[0], top_gaps, [n]])

    # Step 3: assign blocks based on splits
    z = np.zeros(n, dtype=int)
    for k in range(K):
        block_indices = sorted_idx[split_points[k]:split_points[k + 1]]
        z[block_indices] = k

    # Step 4: histogram estimator
    P_hat = np.zeros((n, n))
    for a in range(K):
        idx_a = np.where(z == a)[0]
        for b in range(a, K):
            idx_b = np.where(z == b)[0]
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            sub = A[np.ix_(idx_a, idx_b)]
            if a == b:
                m = len(idx_a)
                if m <= 1:
                    val = 0
                else:
                    mask = ~np.eye(m, dtype=bool)
                    val = sub[mask].sum() / (m * (m - 1))
            else:
                val = sub.sum() / (len(idx_a) * len(idx_b))

            for i in idx_a:
                for j in idx_b:
                    if i != j:
                        P_hat[i, j] = val
                        P_hat[j, i] = val

    np.fill_diagonal(P_hat, 0)
    return P_hat, z



def LG_smoothing(
    graphs: List[torch.Tensor],
    device: str = 'cuda',
    K: int = None
) -> torch.Tensor:
    """
    Apply est_LG_like (block-based histogram smoothing) to a list of adjacency matrices.
    Returns the padded average graphon estimate across graphs.

    Args:
        graphs: list of (N_i, N_i) adjacency matrices (torch.Tensor)
        device: device for final tensor output
        K: number of blocks (optional, default None → use largest gap)

    Returns:
        smoothed_avg: (N_max, N_max) average of smoothed matrices
    """
    smoothed_list = []
    max_nodes = max([adj.shape[0] for adj in graphs])

    for adj in graphs:
        A = adj.cpu().numpy()
        P_hat, _ = est_LG_like(A, K=K)  # Use block-based estimator

        # Pad to max_nodes
        padded = np.zeros((max_nodes, max_nodes))
        n = A.shape[0]
        padded[:n, :n] = P_hat

        smoothed_tensor = torch.tensor(padded, dtype=torch.float32, device=device)
        smoothed_list.append(smoothed_tensor)

    smoothed_avg = torch.stack(smoothed_list).mean(dim=0)
    return smoothed_avg


import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import norm

def random_init_neighbors(n, size):
    """Generate random neighborhood sets."""
    neighbors = []
    for i in range(n):
        indices = np.random.choice([j for j in range(n) if j != i], size=size, replace=False)
        neighbors.append(indices)
    return neighbors

def euclidean_neighbors(P_hat, size):
    """Get nearest neighbors in Euclidean space based on current P_hat."""
    D = cdist(P_hat, P_hat, metric='euclidean')
    neighbors = []
    for i in range(P_hat.shape[0]):
        # Take the closest 'size' excluding self
        nearest = np.argsort(D[i])[1:size+1]
        neighbors.append(nearest)
    return neighbors

def estimate_with_neighbors(A, S, quick=True):
    """Estimate connection probabilities given neighbors."""
    n = A.shape[0]
    P_hat = np.zeros((n, n))

    if quick:
        for i in range(n):
            indices = S[i]
            P_hat[i, :] = A[indices, :].mean(axis=0)
        P_hat = 0.5 * (P_hat + P_hat.T)
    else:
        for i in range(n):
            for j in range(n):
                overlap = len(set(S[i]) & set(S[j]))
                total = len(S[i]) ** 2 - overlap
                if total > 0:
                    P_hat[i, j] = A[np.ix_(S[i], S[j])].sum() / total
    return P_hat

def ICE_estimator(A, C_it=1.0, C_est=1.0, delta_0=1e-4, rounds=None, verbose=False):
    """
    ICE estimator: Iterative Connecting Probability Estimation for Networks.
    
    Args:
        A: (n, n) binary symmetric adjacency matrix
        C_it: constant multiplier for neighbor size in iterations
        C_est: constant multiplier for final estimation
        delta_0: convergence threshold for relative Frobenius norm
        rounds: number of fixed rounds; if None, use delta threshold
        verbose: whether to print iteration deltas
    
    Returns:
        P_final: (n, n) final estimated connection probability matrix
        P_list: list of intermediate estimates
        S_final: final neighborhood sets used in estimation
    """
    A = A.copy()
    np.fill_diagonal(A, 0)
    n = A.shape[0]

    size_it = int(np.round(C_it * np.sqrt(n * np.log(n))))
    size_est = int(np.round(C_est * np.sqrt(n * np.log(n))))

    # Step 1: random init
    S = random_init_neighbors(n, size_it)
    P_hat_0 = estimate_with_neighbors(A, S, quick=True)

    P_list = [P_hat_0]
    delta = np.inf
    m = 0

    if rounds is not None:
        for m in range(rounds):
            S = euclidean_neighbors(P_list[m], size_it)
            P_next = estimate_with_neighbors(A, S, quick=True)
            delta = norm(P_next - P_list[m], ord='fro') / norm(P_list[m], ord='fro')
            if verbose:
                print(f"[ICE] Round {m+1}, delta = {delta:.6f}")
            P_list.append(P_next)
    else:
        while delta > delta_0:
            S = euclidean_neighbors(P_list[m], size_it)
            P_next = estimate_with_neighbors(A, S, quick=True)
            delta = norm(P_next - P_list[m], ord='fro') / norm(P_list[m], ord='fro')
            if verbose:
                print(f"[ICE] Round {m+1}, delta = {delta:.6f}")
            P_list.append(P_next)
            m += 1

    # Final estimation with larger neighborhood
    S_final = euclidean_neighbors(P_list[-1], size_est)
    P_final = estimate_with_neighbors(A, S_final, quick=True)

    return P_final, P_list, S_final



import torch
import numpy as np
from typing import List

def ICE_smoothing(
    graphs: List[torch.Tensor],
    device: str = 'cuda',
    C_it: float = 1.0,
    C_est: float = 1.0,
    delta_0: float = 1e-4,
    rounds: int = 10
) -> torch.Tensor:
    """
    Apply ICE_estimator to a list of adjacency matrices.
    Returns the padded average graphon estimate across graphs.

    Args:
        graphs: list of (N_i, N_i) adjacency matrices (torch.Tensor)
        device: device for final tensor output
        C_it: iteration neighbor multiplier
        C_est: final neighbor multiplier
        delta_0: stopping threshold (if rounds is None)
        rounds: number of fixed rounds (optional)

    Returns:
        smoothed_avg: (N_max, N_max) average of smoothed matrices
    """
    smoothed_list = []
    max_nodes = max([adj.shape[0] for adj in graphs])

    for adj in graphs:
        A = adj.cpu().numpy()
        P_hat, _, _ = ICE_estimator(
            A,
            C_it=C_it,
            C_est=C_est,
            delta_0=delta_0,
            rounds=rounds,
            verbose=False
        )

        # Pad to max_nodes
        padded = np.zeros((max_nodes, max_nodes))
        n = A.shape[0]
        padded[:n, :n] = P_hat

        smoothed_tensor = torch.tensor(padded, dtype=torch.float32, device=device)
        smoothed_list.append(smoothed_tensor)

    smoothed_avg = torch.stack(smoothed_list).mean(dim=0)
    return smoothed_avg


