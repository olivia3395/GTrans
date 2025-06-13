# Simulation: 

This directory contains a complete suite of **R scripts** for simulating and evaluating our proposed **GTRANS** framework for **graphon estimation under transfer learning**. The experiments investigate performance across different conditions, such as varying **source sample size**, **graphon structural perturbation (λ)**, and **ablation of debiasing mechanisms**.

---

## Script Overview

| Script | Description |
|--------|-------------|
| `function.R` | Core methods for graphon estimation, transport plan computation, debiasing, and MSE evaluation. |
| `auxiliary.R` | Utility functions, including adjacency validation (`is.Adj`, `is.binAdj`), smoothing support (`aux_nbdsmooth`), and tensor operations (`sum3`, `histogram3D`). |
| `network_generate.R` | Synthetic graphon generation and adjacency matrix simulation. |
| `increasingN_MSE.R` | Evaluates estimation MSE as **source sample size increases** (Figure 4). |
| `varyingLambda_MSE.R` | Evaluates estimation MSE under **different λ perturbations** between source and target (Figure 6). |
| `cross_graphon_trans.R` | Evaluates **cross-graphon transferability**, testing all source-target graphon pairs (Table 1). |
| `threshold_delta.R` | Edge-based cross-validation to tune **debiasing threshold δ**. |
| `threshold_epsilon.R` | Hyperparameter tuning for **entropic regularization ε** in Gromov-Wasserstein. |
| `ICE.R` | Reimplementation of **ICE baseline** for direct comparison. |
| `increasingN_ablation.R` | Ablation under increasing N to isolate GTRANS components. |
| `varyingLambda_ablation.R` | Ablation under structural perturbation λ. |

---

## Simulated Experiments

Our simulations target three main questions:

1. **Does GTRANS improve estimation as source sample size increases?**  
   → See: `increasingN_MSE.R`  
   

2. **How does GTRANS perform under varying domain shifts (λ)?**  
   → See: `varyingLambda_MSE.R`  
   

3. **Can GTRANS handle transfer across structurally different graphons?**  
   → See: `cross_graphon_trans.R`  

4. **What is the effect of debiasing?**  
   → See: `increasingN_ablation.R` and `varyingLambda_ablation.R`  
  

---

## Usage


Make sure the following R packages are installed:

```r
install.packages(c("reticulate", "pheatmap", "parallel", "ggplot2", "reshape2"))
```

If you use `reticulate`, ensure Python dependencies (e.g., `transport`, `pot`, `scikit-learn`) are installed in your Python environment.


