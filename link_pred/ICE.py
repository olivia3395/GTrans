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
from scipy.spatial.distance import cdist
from numpy.linalg import norm


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
            P_list.append(P_next)
    else:
        while delta > delta_0:
            S = euclidean_neighbors(P_list[m], size_it)
            P_next = estimate_with_neighbors(A, S, quick=True)
            delta = norm(P_next - P_list[m], ord='fro') / norm(P_list[m], ord='fro')
            P_list.append(P_next)
            m += 1

    # Final estimation with larger neighborhood
    S_final = euclidean_neighbors(P_list[-1], size_est)
    P_final = estimate_with_neighbors(A, S_final, quick=True)

    return P_final, P_list, S_final



# In[6]:


def evaluate_ICE_link_prediction(
    A: np.ndarray,
    test_ratio: float = 0.1,
    C_it: float = 1.0,
    C_est: float = 1.0,
    rounds: int = 10,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate ICE method on a single graph for link prediction using AUC, AP, RMSE.

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

    # ICE estimate
    P_hat, _, _ = ICE_estimator(A_masked, C_it=C_it, C_est=C_est, rounds=rounds)

    # Sample equal number of negatives
    np.random.shuffle(neg_edges)
    sampled_neg = neg_edges[:test_size]

    y_true = [1] * test_size + [0] * test_size
    y_score = [P_hat[i, j] for (i, j) in test_edges + sampled_neg]

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    rmse = mean_squared_error(y_true, y_score)

    return {"AUC": round(auc, 4), "AP": round(ap, 4), "RMSE": round(rmse, 4)}

def batch_evaluate_ICE_on_datasets(
    dataset_names: list,
    base_path: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate ICE metrics on a list of datasets.

    Returns:
        Dictionary of {dataset_name: {"AUC": ..., "AP": ..., "RMSE": ...}}
    """
    results = {}
    for name in dataset_names:
        try:
            A = load_real_world_dataset(name, base_path)
            metrics = evaluate_ICE_link_prediction(A)
            results[name] = metrics
        except Exception as e:
            results[name] = f"❌ Error: {str(e)}"
    return results


# In[7]:


def batch_evaluate_ICE_on_datasets_multiple_trials(
    dataset_names,
    base_path,
    n_trials=10
) -> pd.DataFrame:
    all_results = []
    for name in dataset_names:
        for trial in range(n_trials):
            seed = 42 + trial
            try:
                metrics = evaluate_ICE_link_prediction(
                    load_real_world_dataset(name, base_path),
                    seed=seed
                )
                metrics.update({"Dataset": name, "Trial": trial})
                all_results.append(metrics)
            except Exception as e:
                all_results.append({"Dataset": name, "Trial": trial, "Error": str(e)})
    return pd.DataFrame(all_results)

# Run USVD evaluation 10 times with threshold 0.6
ICE_results_multi = batch_evaluate_ICE_on_datasets_multiple_trials(
    dataset_names=["dolphins", "karate", "football", "firm","miserables","Karate Club"],
    base_path=path_all,
    n_trials=50
)

print(ICE_results_multi)


# In[13]:


if "Error" not in ICE_results_multi.columns:
    summary_ICE = ICE_results_multi.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)
else:
    filtered = ICE_results_multi[~ICE_results_multi["Error"].notnull()]
    summary_ICE = filtered.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)


print(summary_ICE)


# In[ ]:




