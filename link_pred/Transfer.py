#!/usr/bin/env python
# coding: utf-8

# In[177]:


import networkx as nx
import numpy as np
import scipy.io
from scipy.io import loadmat
import os

from sklearn.metrics import roc_auc_score,average_precision_score,mean_squared_error
from typing import Dict
import torch
import pandas as pd
from scipy.io import mmread

import re  
import networkx as nx
import ot 


# In[179]:


def est_nbdsmooth_like_r(A: np.ndarray):
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


# In[181]:


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


# In[183]:


def load_real_world_dataset(name, base_path):
    if name == "miserables":
        G =  nx.les_miserables_graph()
            
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


# In[185]:


def evaluate_TL_NS_link_prediction_fixed(
    A_t: np.ndarray,
    P_s: np.ndarray,
    test_ratio: float = 0.1,
    ot_eps: float = 1e-5,
    seed: int = 42,
    delta=0.12
) -> Dict[str, float]:
    """
    Evaluate transfer learning-based link prediction using a given P_s, OT, and residual debiasing.
    Fix: no label matching; use R_t = NS(P_t_init - P_t_trans)

    Args:
        A_t: target adjacency matrix (n x n)
        P_s: single source graphon matrix
        test_ratio: fraction of positive edges to hide
        h: NS bandwidth for residual smoothing
        ot_eps: epsilon for OT solver
        seed: random seed

    Returns:
        AUC / AP / RMSE evaluation
    """
    np.random.seed(seed)
    n = A_t.shape[0]
    A_masked = A_t.copy()

    # Step 1: mask test edges
    triu_idx = np.triu_indices(n, k=1)
    all_edges = list(zip(triu_idx[0], triu_idx[1]))
    pos_edges = [(i, j) for (i, j) in all_edges if A_t[i, j] > 0]
    neg_edges = [(i, j) for (i, j) in all_edges if A_t[i, j] == 0]

    test_size = int(len(pos_edges) * test_ratio)
    test_idx = np.random.choice(len(pos_edges), size=test_size, replace=False)
    test_edges = [pos_edges[i] for i in test_idx]

    for i, j in test_edges:
        A_masked[i, j] = 0
        A_masked[j, i] = 0

    # Step 2: target graphon (initial estimate)
    A_tensor = torch.tensor(A_masked, dtype=torch.float32)
    P_t_ini = est_nbdsmooth_like_r(A_tensor.numpy())

    # Step 3: compute OT and transfer
    p_s = np.ones(P_s.shape[0]) / P_s.shape[0]
    p_t = np.ones(P_t_ini.shape[0]) / P_t_ini.shape[0]
    ot_eps = 1
    pi = ot.gromov.entropic_gromov_wasserstein(
        C1=P_s, C2=P_t_ini, p=p_s, q=p_t,
        loss_fun="square_loss", epsilon=ot_eps, verbose=False
    )
    
    
    
    print("Max value of unnormalized pi:", np.max(pi))
    print("Min value of unnormalized pi:", np.min(pi))
   
    pi_norm = pi / pi.sum(axis=1, keepdims=True)
                    # Normalize column-wise
    pi_norm = pi_norm / pi_norm.sum(axis=0, keepdims=True)
                    # Now use normalized pi
    pi = pi_norm
    
    print("Max value of normalized pi:", np.max(pi))
    print("Min value of normalized pi:", np.min(pi))
    
 
    
    # Gromov-Wasserstein squared cost
    cost = ot.gromov.entropic_gromov_wasserstein2(P_s, P_t_ini, p_s, p_t,
                                         loss_fun="square_loss", epsilon=ot_eps)
    gw_distance = np.sqrt(cost)  # true GW distance

    print(f"[📏 GW Distance] GW = {gw_distance:.4f} (√cost)")
   ###  print(f"[GW² Cost] {cost:.4f}")


    # Step 4: Decide if we apply debias based on fixed ε
    P_t_trans = pi.T @ P_s @ pi
    P_t_trans_smoothed = est_nbdsmooth_like_r(P_t_trans)

    if gw_distance > delta:
        # Apply debiasing
        P_t_1 = P_t_trans_smoothed
        np.fill_diagonal(P_t_1, 0)
        ## R_t = np.clip(P_t_ini - P_t_trans_smoothed, 0, 1)
        R_t = P_t_ini - P_t_trans_smoothed
        R_t_smoothed = est_nbdsmooth_like_r(R_t)
        ## P_t_final = np.clip(P_t_trans + R_t_smoothed, 0, 1)
        ## P_t_final = np.clip(P_t_trans_smoothed + R_t_smoothed, 0, 1)
        P_t_final =  P_t_1 + R_t_smoothed
        print("✅ Debias triggered.")
    else:
        # No debiasing
        ## P_t_final =  P_t_trans_smoothed
        
        ### P_t_final  = np.clip(P_t_trans_smoothed, 0, 1)
        P_t_final  = P_t_trans_smoothed
        print("🚫 No debias (transfer only).")


    np.fill_diagonal(P_t_final, 0)

    P_t_final = np.clip(P_t_final, 0, 1)

        # Step 5: Evaluate
    np.random.shuffle(neg_edges)
    sampled_neg = neg_edges[:test_size]
    y_true = [1] * test_size + [0] * test_size
    y_score = [P_t_final[i, j] for (i, j) in test_edges + sampled_neg]
    
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    rmse = mean_squared_error(y_true, y_score)


    return {
        "AUC": round(auc, 4),
        "AP": round(ap, 4),
        "RMSE": round(rmse, 4),
        "GW": round(gw_distance, 4)
    }


# In[187]:


# TL
import pandas as pd

def batch_evaluate_TL_NS_multiple_trials(
    dataset_names,
    base_path,
    source_name="wiki_vote",
    n_trials=10,
    epsilon=1e-5,
    threshold_delta=0.15
) -> pd.DataFrame:
    all_results = []

    try:
        A_s = load_real_world_dataset(source_name, base_path)
        P_s = est_nbdsmooth_like_r(A_s)
    except Exception as e:
        return pd.DataFrame([{"Source Dataset": source_name, "Error": f"Source load failed: {str(e)}"}])

    for name in dataset_names:
        print(f"🔄 Evaluating target dataset: {name}...")  # ✅ NEW: show progress
        try:
            A_t = load_real_world_dataset(name, base_path)
        except Exception as e:
            all_results.append({
                "Dataset": name, "Trial": None, "Source": source_name, "Error": f"Target load failed: {str(e)}"
            })
            continue

        for trial in range(n_trials):
            seed = 42 + trial
            try:
                metrics = evaluate_TL_NS_link_prediction_fixed(
                    A_t=A_t,
                    P_s=P_s,
                    test_ratio=0.1,
                    ot_eps=epsilon,
                    seed=seed,
                    delta=threshold_delta
                )
                metrics.update({"Dataset": name, "Trial": trial, "Source": source_name})
                all_results.append(metrics)
            except Exception as e:
                all_results.append({
                    "Dataset": name,
                    "Trial": trial,
                    "Source": source_name,
                    "Error": str(e)
                })

    return pd.DataFrame(all_results)


# ###  epsilon=1e-5, threshold_delta=0.15

# In[190]:


# Run TL evaluation using wiki_vote as source on all other datasets
transfer_results_multi = batch_evaluate_TL_NS_multiple_trials(
    dataset_names=["dolphins", "karate", "football", "firm","miserables"],
    ## dataset_names= ["hamster"],
    #dataset_names=["karate"],
    base_path=path_all,
    source_name="wiki_vote",
    n_trials=50,
    epsilon=1e-5,
    threshold_delta=0.15
)

print(transfer_results_multi)


# In[191]:


# compute AUC、AP、RMSE(mean and std) of all datasets
if "Error" not in transfer_results_multi.columns:
    summary_transfer_results_multi= transfer_results_multi.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)
else:
    filtered = transfer_results_multi[~transfer_results_multi["Error"].notnull()]
    summary_transfer_results_multi = filtered.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)

print(summary_transfer_results_multi)


# ###  epsilon=0.01, threshold_delta=0.15

# In[193]:


# Run TL evaluation using wiki_vote as source on all other datasets
transfer_results_multi = batch_evaluate_TL_NS_multiple_trials(
    dataset_names=["dolphins", "karate", "football", "firm","miserables"],
    ## dataset_names= ["hamster"],
    #dataset_names=["karate"],
    base_path=path_all,
    source_name="wiki_vote",
    n_trials=50,
    epsilon=0.01,
    threshold_delta=0.15
)

print(transfer_results_multi)


# In[194]:


# compute AUC、AP、RMSE(mean and std) of all datasets
if "Error" not in transfer_results_multi.columns:
    summary_transfer_results_multi= transfer_results_multi.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)
else:
    filtered = transfer_results_multi[~transfer_results_multi["Error"].notnull()]
    summary_transfer_results_multi = filtered.groupby("Dataset")[["AUC", "AP", "RMSE"]].agg(["mean", "std"]).round(4)

print(summary_transfer_results_multi)


# In[ ]:




