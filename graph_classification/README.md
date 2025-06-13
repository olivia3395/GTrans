# GTRANS + G-Mixup: 
## This repository provides the full experimental pipeline for:

- Baseline GIN training  
- G-Mixup without transfer  
- G-Mixup + GTRANS (our proposed method)  

---

## Requirements

We recommend Python 3.7. Required packages:

```bash
pip install torch==1.7.1
pip install cudatoolkit==11.0
pip install opencv-python==4.5.3.56
pip install scikit-image==0.18.3
````

To install PyTorch Geometric and its dependencies:

```bash
pip install torch_spline_conv-1.2.0-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.8-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.8-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==1.6.3
```

---

## Dataset

All datasets used in this project are from `torch_geometric.datasets` and will be downloaded automatically to:

```
../dataset/loaded/
```

Supported datasets:

* IMDB-BINARY
* IMDB-MULTI
* COLLAB
* REDDIT-BINARY
* D\&D
* PROTEINS-FULL

---

## Running Experiments

### 🔹 Baseline GIN (No Mixup, No Transfer)

```bash
sh run_vanilla.sh
```

### 🔹 G-Mixup (No Transfer)

```bash
sh run_gmixup.sh
```

Or run manually:

```bash
python gmixup_no_transfer_baseline.py \
  --dataset IMDB-BINARY \
  --gmixup True \
  --method ICE \
  --epoch 200 \
  --log_screen True
```

### 🔹 G-Mixup + GTRANS (Our Method)

```bash
python gmixup_transfer.py \
  --dataset IMDB-BINARY \
  --gmixup True \
  --use_transfer True \
  --epoch 200 \
  --log_screen True \
  --data_path ../dataset/loaded/
```

---

