# GTrans

# GTRANS: Transfer Learning on Graphon Estimation

![GTRANS Workflow](assets/gtrans_workflow.png)

---

## Overview

**GTRANS** is a novel transfer learning framework for estimating edge connection probabilities under the **graphon model**, specifically designed for **small-sample graphs**. We propose a method that leverages a **large source graph** via **Gromov-Wasserstein Optimal Transport**, with an **adaptive debiasing mechanism** to avoid negative transfer.

Graphon models offer a nonparametric foundation for modeling large random graphs, but estimation accuracy suffers in small graphs. GTRANS transfers structural information from larger graphs to improve estimation in the small target graph, achieving state-of-the-art performance in both simulation and real-world tasks such as **link prediction** and **graph classification**.

---

## Method Highlights

- **Neighborhood Smoothing** to obtain initial estimators of connection probabilities.
- **(Entropic) Gromov-Wasserstein Distance** to align source and target latent structures.
- **Projection and Transfer** of source estimator onto target space via the learned alignment.
- **Adaptive Debiasing Step** to correct target-specific patterns when domain shift is large.
- **Theoretical guarantees** on alignment matrix stability.
- **Plug-in ready** for graph classification and data augmentation tasks.

---

## Experiments

### 🔬 Simulation Results
- Tested on 10 benchmark graphons.
- Robust to **domain shift**, **density shift**, and **source sample size variations**.
- GTRANS outperforms baseline methods: NS, SAS, ICE, and USVT.

### Real-World Applications
- Improved **graph classification** accuracy on:
  - IMDB-BINARY (↑ ~5%)
  - IMDB-MULTI
  - PROTEINS-FULL
- Works with **G-Mixup** for graph augmentation.
- Results generalize to both **social networks** and **biological graphs**.

---

