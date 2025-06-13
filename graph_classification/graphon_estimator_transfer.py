


### some code are from https://github.com/HongtengXu/SGWB-Graphon

### some code are from https://github.com/ahxt/g-mixup




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


# transfer_graphon_estimate

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
    # Step 1: Convert both to numpy if they are torch.Tensor
    if torch.is_tensor(A_t):
        A_t = A_t.detach().cpu().numpy()
    if torch.is_tensor(P_t_trans):
        P_t_trans = P_t_trans.detach().cpu().numpy()

    # Step 2: Compute residual
    R_t = A_t - P_t_trans

    # Step 3: Clip residual to [0, 1] for smoothing stability
    R_t = np.clip(R_t, 0, 1)

    # Step 4: Neighborhood smoothing on residual
    P_t_res = neighborhood_smoothing(R_t, k=k)
    
    return P_t_res


import numpy as np

def aux_nbdsmooth(A: np.ndarray) -> np.ndarray:
    """Structure dissimilarity matrix D from R-style aux_nbdsmooth"""
    N = A.shape[0]
    D = np.zeros((N, N))
    A_sq = (A @ A) / N

    for i in range(N - 1):
        for j in range(i + 1, N):
            tgt = np.abs(A_sq[i] - A_sq[j])
            tgt[i] = 0
            tgt[j] = 0
            val = np.max(tgt)
            D[i, j] = val
            D[j, i] = val
    return D

def debias_with_structural_kernel(R_t_numpy: np.ndarray, h: float = None) -> np.ndarray:
    """
    Apply structure-aware smoothing to residual matrix using aux_nbdsmooth-style kernel.

    R_t_numpy: residual matrix (n, n)
    h: threshold quantile for defining local neighborhoods (e.g., sqrt(log n / n))
    """
    N = R_t_numpy.shape[0]
    D = aux_nbdsmooth(R_t_numpy)
    if h is None:
        h = np.sqrt(np.log(N) / N)

    kernel_mat = np.zeros((N, N))

    for i in range(N):
        threshold = np.quantile(D[i], h)
        kernel_mat[i] = (D[i] < threshold).astype(float)

    # Row normalization
    row_sums = kernel_mat.sum(axis=1, keepdims=True) + 1e-10
    kernel_mat = kernel_mat / row_sums

    # Symmetrized smoothing
    P = kernel_mat @ R_t_numpy
    delta_hat = 0.5 * (P + P.T)
    return delta_hat

################



