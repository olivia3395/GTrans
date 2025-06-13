#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

import copy
import cv2
from skimage.restoration import denoise_tv_chambolle


# In[8]:


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


# In[9]:


def load_real_world_dataset(name, base_path):

    if name == "miserables":
        G =  nx.les_miserables_graph()
    elif name == "Karate Club":
        G_karate = nx.karate_club_graph()
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


# In[10]:


def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of adjacency matrices into a torch tensor.

    Args:
        graphs: List of (N, N) numpy arrays

    Returns:
        tensor of shape (num_graphs, N, N)
    """
    graph_list = [torch.tensor(g, dtype=torch.float32) for g in graphs]
    return torch.stack(graph_list, dim=0)



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



# In[11]:


def evaluate_USVD_link_prediction(
    A: np.ndarray,
    test_ratio: float = 0.1,
    threshold: float = 0.2,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate Universal SVD (USVD) method on a single graph for link prediction using AUC, AP, RMSE.

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

    # Apply Universal SVD estimation
    P_hat = universal_svd([A_masked], threshold=threshold)
    P_hat = np.clip(P_hat, 0, 1)

    # Sample equal number of negatives
    np.random.shuffle(neg_edges)
    sampled_neg = neg_edges[:test_size]

    y_true = [1] * test_size + [0] * test_size
    y_score = [P_hat[i, j] for (i, j) in test_edges + sampled_neg]

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    rmse = mean_squared_error(y_true, y_score)

    return {"AUC": round(auc, 4), "AP": round(ap, 4), "RMSE": round(rmse, 4)}

def batch_evaluate_USVD_on_datasets(
    dataset_names: list,
    base_path: str,
    threshold: float = 0.2
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate USVD metrics on a list of datasets.

    Returns:
        Dictionary of {dataset_name: {"AUC": ..., "AP": ..., "RMSE": ...}}
    """
    results = {}
    for name in dataset_names:
        try:
            A = load_real_world_dataset(name, base_path)
            metrics = evaluate_USVD_link_prediction(A, threshold=threshold)
            results[name] = metrics
        except Exception as e:
            results[name] = f"❌ Error: {str(e)}"
    return results


# In[ ]:


def batch_evaluate_USVD_on_datasets_multiple_trials(
    dataset_names,
    base_path,
    threshold=0.6,
    n_trials=10
) -> pd.DataFrame:
    all_results = []
    for name in dataset_names:
        for trial in range(n_trials):
            seed = 42 + trial
            try:
                metrics = evaluate_USVD_link_prediction(
                    load_real_world_dataset(name, base_path),
                    threshold=threshold,
                    seed=seed
                )
                metrics.update({"Dataset": name, "Trial": trial})
                all_results.append(metrics)
            except Exception as e:
                all_results.append({"Dataset": name, "Trial": trial, "Error": str(e)})
    return pd.DataFrame(all_results)

# Run USVD evaluation 10 times with threshold 0.6
usvd_results_multi = batch_evaluate_USVD_on_datasets_multiple_trials(
    dataset_names=["dolphins", "karate", "football", "firm","Karate Club","miserables"],
    base_path=path_all,
    threshold=0.6,
    n_trials=50
)

print(usvd_results_multi)


# In[ ]:


if "Error" not in usvd_results_multi.columns:
    summary_usvd = usvd_results_multi.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)
else:
    filtered = usvd_results_multi[~usvd_results_multi["Error"].notnull()]
    summary_usvd = filtered.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)


print(summary_usvd)

