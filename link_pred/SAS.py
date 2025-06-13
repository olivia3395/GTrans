#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx as nx
import numpy as np
import scipy.io
from scipy.io import loadmat
import os

from sklearn.metrics import roc_auc_score,average_precision_score,mean_squared_error
from typing import Dict,List,Tuple
import torch
import pandas as pd
from scipy.io import mmread

import re  
import networkx as nx


# In[3]:


def read_edge_file(filepath, delimiter=None):
    """
    Safely load edgelist from .edges/.txt files by ignoring invalid rows.
    """
    import pandas as pd
    edges = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            # Split by space, tab, or comma
            if delimiter:
                parts = line.split(delimiter)
            else:
                parts = re.split(r'[\s,]+', line)
            if len(parts) < 2:
                continue
            try:
                i, j = int(parts[0]), int(parts[1])
                edges.append((i, j))
            except ValueError:
                continue
    return nx.from_edgelist(edges)


# In[4]:


def load_real_world_dataset(name, base_path):

    if name == "miserables":
        G =  nx.les_miserables_graph()
    elif name == "Karate Club":
        G = nx.karate_club_graph()
    elif name == "dolphins":
        path = os.path.join(base_path, "dolphins.gml")
        G = nx.read_gml(path)

    elif name == "jazz":
        path = os.path.join(base_path, "jazz.net")
        G = nx.read_pajek(path)

    elif name == "as-22july06":
        path = os.path.join(base_path, "as-22july06.gml")
        G = nx.read_gml(path)

    elif name == "netscience":
        path = os.path.join(base_path, "netscience.gml")
        G = nx.read_gml(path)

    elif name == "polbooks":
        path = os.path.join(base_path, "polbooks.gml")
        G = nx.read_gml(path)

    elif name == "power":
        path = os.path.join(base_path, "power.gml")
        G = nx.read_gml(path)

    elif name == "lesmis":
        path = os.path.join(base_path, "lesmis.gml")
        G = nx.read_gml(path)

    elif name == "polblogs":
        path = os.path.join(base_path, "polblogs.gml")
        G = nx.read_gml(path)

    elif name == "karate":
        G = nx.karate_club_graph()

    elif name == "football":
        path = os.path.join(base_path, "football.gml")
        G = nx.read_gml(path)

    elif name == "asia":
        path = os.path.join(base_path, "lasftm_asia", "lastfm_asia_edges.csv")
        edges = np.loadtxt(path, delimiter=',', skiprows=1, dtype=int)
        G = nx.Graph()
        G.add_edges_from(edges)

    # === NEW edge file formats ===
  
   
   
    elif name == "wiki_vote":
        path = os.path.join(base_path, "soc-wiki-Vote.mtx")
        mat = mmread(path).tocoo()
        G = nx.Graph()
        G.add_edges_from(zip(mat.row, mat.col))
        
    elif name == "advogato":
        path = os.path.join(base_path, "soc-advogato.edges")
        G = read_edge_file(path)

    elif name == "firm":
        path = os.path.join(base_path, "soc-firm-hi-tech.txt")
        G = read_edge_file(path)

    elif name == "hamster":
        path = os.path.join(base_path, "soc-hamsterster.edges")
        G = read_edge_file(path)

    elif name == "tribes":
        path = os.path.join(base_path, "soc-tribes.edges")
        G = read_edge_file(path)

    elif name == "fb_tv":
        path = os.path.join(base_path, "fb-pages-tvshow.edges")
        G = read_edge_file(path, delimiter=',')


    else:
        raise ValueError(f"❌ Unknown dataset name: {name}")

    # Create binary, symmetric adjacency matrix
    adj = nx.to_numpy_array(G)
    adj[adj > 0] = 1
    np.fill_diagonal(adj, 0)
    return adj


# In[5]:


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




# In[6]:


def is_binary_adjacency_matrix(A):
    return isinstance(A, np.ndarray) and A.ndim == 2 and set(np.unique(A)).issubset({0, 1})

def sum3(A_list, row_idx, col_idx, graph_idx):
    """ Sum adjacency matrices over selected dimensions """
    G = np.zeros_like(A_list[0])
    for idx in graph_idx:
        G += A_list[idx]
    return G

def histogram3D(A_list, B):
    """ Estimate 3D histogram for block model """
    K = len(B)
    n = A_list[0].shape[0]
    H = np.zeros((K, K))
    count = np.zeros((K, K))

    for A in A_list:
        for i in range(K):
            for j in range(K):
                block = A[np.ix_(B[i], B[j])]
                H[i, j] += np.sum(block)
                count[i, j] += len(B[i]) * len(B[j])

    H = H / np.maximum(count, 1)  # avoid division by 0
    P = np.zeros((n, n))
    for i in range(K):
        for j in range(K):
            P[np.ix_(B[i], B[j])] = H[i, j]
    return {'H': H, 'P': P}

def est_LG(A, K=2):
    """
    Estimate graphon using Largest Gap method for block detection.
    
    Parameters:
        A: numpy array (n x n), binary adjacency matrix
        K: int, number of blocks
    
    Returns:
        dict with keys:
            - 'H': histogram matrix (K x K)
            - 'P': estimated probability matrix (n x n)
            - 'B': list of node indices for each cluster
    """
    if isinstance(A, list):
        if not all(is_binary_adjacency_matrix(mat) for mat in A):
            raise ValueError("est_LG: All matrices in list must be binary adjacency matrices.")
        vecA = A
    elif is_binary_adjacency_matrix(A):
        vecA = [A]
    else:
        raise ValueError("est_LG: Input A must be a binary adjacency matrix or list of them.")

    n = vecA[0].shape[0]
    K = int(round(K))
    if K < 1 or K > n:
        raise ValueError("est_LG: The number of blocks K must be between 1 and n.")

    G = sum3(vecA, range(n), range(n), range(len(vecA)))
    Deg = np.sum(G - np.diag(np.diag(G)), axis=1)
    DegNorm = Deg / (n - 1)
    idx_sorted = np.argsort(DegNorm)
    DegNorm_sorted = DegNorm[idx_sorted]
    DegDiff = np.diff(DegNorm_sorted)

    gap_indices = np.argsort(DegDiff)[::-1][:K-1]
    cut_points = sorted(gap_indices)
    boundaries = [0] + [i + 1 for i in cut_points] + [n]

    B = []
    for k in range(K):
        B.append(list(idx_sorted[boundaries[k]:boundaries[k+1]]))

    result = histogram3D(vecA, B)
    result['B'] = B
    return result

from scipy.stats import bernoulli

np.random.seed(42)
n = 50
u = np.random.rand(n)
W_true = np.outer(u, u)
A = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        A[i, j] = A[j, i] = bernoulli.rvs(W_true[i, j])
np.fill_diagonal(A, 0)

result = est_LG(A, K=2)
P_hat = result["P"]


# In[7]:


def evaluate_LG_link_prediction(
    A: np.ndarray,
    test_ratio: float = 0.1,
    K: int = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate LG method on a single graph for link prediction using AUC, AP, RMSE.

    Returns:
        Dictionary with AUC, AP, RMSE
    """
    np.random.seed(seed)
    n = A.shape[0]
    A_masked = A.copy()
    triu_idx = np.triu_indices(n, k=1)
    all_edges = list(zip(triu_idx[0], triu_idx[1]))
    pos_edges = [(i, j) for (i, j) in all_edges if A[i, j] > 0]
    neg_edges = [(i, j) for (i, j) in all_edges if A[i, j] == 0]

    test_size = int(len(pos_edges) * test_ratio)
    test_idx = np.random.choice(len(pos_edges), size=test_size, replace=False)
    test_edges = [pos_edges[i] for i in test_idx]

    for i, j in test_edges:
        A_masked[i, j] = 0
        A_masked[j, i] = 0

    # LG estimate
    P_hat = est_LG(A_masked)["P"]

    # Sample equal number of negatives
    np.random.shuffle(neg_edges)
    sampled_neg = neg_edges[:test_size]

    y_true = [1] * test_size + [0] * test_size
    y_score = [P_hat[i, j] for (i, j) in test_edges + sampled_neg]

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    rmse = mean_squared_error(y_true, y_score)

    return {"AUC": round(auc, 4), "AP": round(ap, 4), "RMSE": round(rmse, 4)}

def batch_evaluate_LG_on_datasets(
    dataset_names: list,
    base_path: str,
    K: int = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate LG metrics on a list of datasets.

    Returns:
        Dictionary of {dataset_name: {"AUC": ..., "AP": ..., "RMSE": ...}}
    """
    results = {}
    for name in dataset_names:
        try:
            A = load_real_world_dataset(name, base_path)
            metrics = evaluate_LG_link_prediction(A, K=K)
            results[name] = metrics
        except Exception as e:
            results[name] = f"❌ Error: {str(e)}"
    return results




# In[8]:


def batch_evaluate_LG_on_datasets_multiple_trials(
    dataset_names,
    base_path,
    n_trials=10
) -> pd.DataFrame:
    all_results = []
    for name in dataset_names:
        for trial in range(n_trials):
            seed = 42 + trial
            try:
                metrics = evaluate_LG_link_prediction(
                    load_real_world_dataset(name, base_path),
                    seed=seed
                )
                metrics.update({"Dataset": name, "Trial": trial})
                all_results.append(metrics)
            except Exception as e:
                all_results.append({"Dataset": name, "Trial": trial, "Error": str(e)})
    return pd.DataFrame(all_results)

# Run USVD evaluation 10 times with threshold 0.6
LG_results_multi = batch_evaluate_LG_on_datasets_multiple_trials(
   ##  dataset_names=["dolphins", "karate", "football", "firm", "hamster", "tribes", "wiki_vote"],
    dataset_names=["dolphins", "karate", "football", "firm","miserables","Karate Club"],
    base_path=path_all,
    n_trials=50
)

print(LG_results_multi)


# In[9]:



if "Error" not in LG_results_multi.columns:
    summary_LG = LG_results_multi.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)
else:
    filtered = LG_results_multi[~LG_results_multi["Error"].notnull()]
    summary_LG = filtered.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)


print(summary_LG)


# In[ ]:




