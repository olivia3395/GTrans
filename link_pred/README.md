# Link Prediction Methods using Graphon Estimation

This folder contains Python implementations of several **graphon-based link prediction algorithms** used for estimating edge probabilities in real-world and synthetic networks. These methods are evaluated under a masking-based testing protocol on six benchmark datasets. The overall goal is to assess how well graphon estimation (with or without transfer) can recover **missing or future links** in graphs.

---

## Contents

| Script         | Description |
|----------------|-------------|
| `USVD.py`      | Universal Singular Value Decomposition. Performs low-rank SVD on adjacency matrices for link recovery. |
| `NS.py`        | Neighborhood Smoothing. Nonparametric method using local averaging to estimate edge probabilities. |
| `SAS.py`       | Sorted and Smoothed graphon estimator. Reorders nodes by degree and applies smoothing on sorted matrix. |
| `ICE.py`       | Iterative Connecting Probability Estimator. Alternates between estimation and binning. |
| `Transfer.py`  | Our proposed **GTRANS** transfer learning framework using Gromov-Wasserstein transport for structure alignment and estimation debiasing. |

---

## Experimental Setup

We evaluate link prediction performance using **six real-world graphs** spanning various domains:

| Dataset       | #Nodes | #Edges | Avg. Deg. | Density  |
|---------------|--------|--------|-----------|----------|
| dolphins      | 62     | 159    | 5.13      | 0.0841   |
| karate        | 34     | 78     | 4.59      | 0.1390   |
| football      | 115    | 613    | 10.66     | 0.0935   |
| firm          | 33     | 91     | 5.52      | 0.1723   |
| hamster       | 2,426  | 16,630 | 13.71     | 0.0057   |
| wiki-vote (*) | 889    | 2,914  | 6.56      | 0.0074   |

> (*) `wiki-vote` is used as the **source graph** for transfer. All others are treated as target graphs.

We randomly mask 10% of edges in the upper triangle of each adjacency matrix to form a **test set**, then use graphon-based methods to estimate the full probability matrix and compute predictions on the masked entries.

---

## Evaluation Metrics


We follow the AUC evaluation protocol from the literature, where predicted edge probabilities are compared against held-out ground truth links, and the area under the ROC curve (AUC) is computed based on different decision thresholds.


---

## Results Summary

| Dataset   | USVD         | NS            | ICE           | SAS           | **GTRANS (ours)** |
|-----------|--------------|---------------|----------------|----------------|-------------------|
| dolphins  | 0.7235 ± 0.10 | 0.7060 ± 0.09 | 0.7536 ± 0.08  | 0.5066 ± 0.06  | **0.7635 ± 0.0857** |
| firm      | 0.6564 ± 0.12 | 0.6632 ± 0.12 | 0.6426 ± 0.12  | 0.5490 ± 0.08  | **0.7109 ± 0.1213** |
| football  | 0.8532 ± 0.04 | 0.8675 ± 0.03 | 0.8246 ± 0.05  | 0.4456 ± 0.07  | **0.8676 ± 0.0372** |
| karate    | 0.7186 ± 0.15 | 0.7674 ± 0.12 | 0.8043 ± 0.11  | 0.6388 ± 0.11  | **0.8210 ± 0.1035** |
| hamster   | 0.8264 ± 0.00 | **0.9513 ± 0.00** | 0.9331 ± 0.00  | 0.5108 ± 0.00  | 0.9395 ± 0.0047 |

**Figure**: See *Table D7* and *Figure D5* in the paper.

---


## Running the Code

Make sure to install the following dependencies:

```bash
pip install numpy networkx scipy matplotlib scikit-learn POT

```
