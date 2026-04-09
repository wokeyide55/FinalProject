# Improved VSF Wrapper for Variable Subset Forecasting

**Course:** 7013 – Advanced Machine Learning
**Base Paper:** [Multi-Variate Time Series Forecasting on Variable Subsets](https://doi.org/10.1145/3534678.3539394), KDD 2022
**Base Model:** [MTGNN](https://github.com/nnzhan/MTGNN)

This project replicates and extends the FDW (Forecast Discrepancy Weighting) method from the KDD 2022 paper. We propose **Improved-FDW**, a three-part enhancement to the inference-time wrapper that operates on top of a frozen MTGNN model in the Variable Subset Forecasting (VSF) setting.

---

## Problem Setting

**Variable Subset Forecasting (VSF):** At inference time, only a subset $S$ (~15%) of the $N$ training variables is observable. The model, trained on all $N$ variables, must still produce accurate forecasts using only the available subset.

---

## Our Improvements over FDW

All improvements are implemented in `improved_wrapper.py` and operate **only at inference time** — the MTGNN model is unchanged.

### Improvement 1 — Cosine Similarity Retrieval
Replaces the original distance with **cosine distance** for retrieving k nearest neighbour windows from the prototype bank, computed on the observed subset $S$ only. Cosine similarity captures time-series shape similarity independent of absolute amplitude.

### Improvement 2 — Adaptive Softmax Temperature
Replaces the fixed global temperature $\tau = 0.1$ with a **per-sample adaptive temperature** equal to the standard deviation of the $k$ neighbours' discrepancy scores. This allows the aggregation sharpness to adapt to the actual quality variance among neighbours for each test instance.

### Improvement 3 — Joint DDW + FDW Weighting
Combines the cosine input-space distance (DDW) and the forecast discrepancy (FDW) into a single weighting signal via linear interpolation:

$$d_{\text{joint},i} = \alpha \cdot \hat{d}_{\cos,i} + (1-\alpha) \cdot \hat{d}_{\text{FDW},i}, \quad \alpha=0.3$$

$$w_i \propto \exp\!\left(-d_{\text{joint},i} / \tau_{\text{adaptive}}\right)$$

---

## Project Structure

```
vsf-time-series/
├── improved_wrapper.py      # Our three improvements (core contribution)
├── run_improved.py          # Unified train/inference script (all methods)
├── run_all.bat              # Windows batch script for full pipeline
├── train_multi_step.py      # Original training script (unchanged)
├── net.py                   # MTGNN model definition (unchanged)
├── trainer.py               # Training engine (unchanged)
├── layer.py                 # Model layers (unchanged)
├── util.py                  # Data utilities (unchanged)
├── generate_training_data.py# Dataset preprocessing
├── requirements.txt         # Python dependencies
├── report.md                # My report
└── data/                    # Dataset directory (not tracked by git)
    ├── SOLAR/
    ├── TRAFFIC/
    ├── METR-LA/
    └── ECG/
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.13+, CUDA recommended.

---

## Data Preparation

**1. SOLAR & TRAFFIC** — Download from [laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data), decompress and place `solar.txt` / `traffic.txt` in `data/`.

**2. METR-LA** — Download `metr-la.h5` from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX). Optionally download `adj_mx.pkl`:
```bash
mkdir -p data/sensor_graph
# place adj_mx.pkl from https://github.com/nnzhan/MTGNN into data/sensor_graph/
```

**3. ECG** — Download ECG5000 from [timeseriesclassification.com](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000), merge TRAIN+TEST `.arff` files and save as `data/ECG_data.csv`.

**4. Generate splits** for each dataset:
```bash
python generate_training_data.py --ds_name solar    --output_dir data/SOLAR   --dataset_filename data/solar.txt
python generate_training_data.py --ds_name traffic  --output_dir data/TRAFFIC --dataset_filename data/traffic.txt
python generate_training_data.py --ds_name metr-la  --output_dir data/METR-LA --dataset_filename data/metr-la.h5
python generate_training_data.py --ds_name ECG      --output_dir data/ECG     --dataset_filename data/ECG_data.csv
```

---

## Training

Train each dataset using `run_improved.py`. Pre-trained checkpoints are available in `saved_models/`.

```bash
# SOLAR
python run_improved.py --data ./data/SOLAR --model_name mtgnn --device cuda:0 \
    --expid 1 --epochs 50 --batch_size 64 --runs 3 \
    --step_size1 2500 --random_node_idx_split_runs 1 \
    --lower_limit_random_node_selections 100 \
    --upper_limit_random_node_selections 100

# ECG
python run_improved.py --data ./data/ECG --model_name mtgnn --device cuda:0 \
    --expid 1 --epochs 100 --batch_size 64 --runs 5 \
    --step_size1 400 --random_node_idx_split_runs 1 \
    --lower_limit_random_node_selections 100 \
    --upper_limit_random_node_selections 100

# METR-LA
python run_improved.py --data ./data/METR-LA --model_name mtgnn --device cuda:0 \
    --expid 1 --epochs 50 --batch_size 64 --runs 3 \
    --step_size1 2500 --random_node_idx_split_runs 1 \
    --lower_limit_random_node_selections 100 \
    --upper_limit_random_node_selections 100

# TRAFFIC
python run_improved.py --data ./data/TRAFFIC --model_name mtgnn --device cuda:0 \
    --expid 1 --epochs 50 --batch_size 32 --runs 3 \
    --step_size1 1000 --random_node_idx_split_runs 1 \
    --lower_limit_random_node_selections 100 \
    --upper_limit_random_node_selections 100
```

Or use the provided batch script (Windows):
```bat
run_all.bat train
```

---

## Inference / Evaluation

Run all four methods (Partial, UW, FDW, Improved) on all datasets at 15% subset:

```bash
# Partial (zero-fill baseline)
python run_improved.py --data ./data/SOLAR --model_name mtgnn --device cuda:0 \
    --expid 1 --epochs 0 --batch_size 64 --runs 3 \
    --random_node_idx_split_runs 20 \
    --lower_limit_random_node_selections 15 \
    --upper_limit_random_node_selections 15 \
    --mask_remaining True

# UW (uniform weighting)
python run_improved.py --data ./data/SOLAR --model_name mtgnn --device cuda:0 \
    --expid 1 --epochs 0 --batch_size 64 --runs 3 \
    --random_node_idx_split_runs 20 \
    --lower_limit_random_node_selections 15 \
    --upper_limit_random_node_selections 15 \
    --borrow_from_train_data True --num_neighbors_borrow 5 \
    --dist_exp_value 0.5 --use_ewp False

# FDW (original paper method)
python run_improved.py --data ./data/SOLAR --model_name mtgnn --device cuda:0 \
    --expid 1 --epochs 0 --batch_size 64 --runs 3 \
    --random_node_idx_split_runs 20 \
    --lower_limit_random_node_selections 15 \
    --upper_limit_random_node_selections 15 \
    --borrow_from_train_data True --num_neighbors_borrow 5 \
    --dist_exp_value 0.5 --neighbor_temp 0.1 --use_ewp True

# Improved (our method)
python run_improved.py --data ./data/SOLAR --model_name mtgnn --device cuda:0 \
    --expid 1 --epochs 0 --batch_size 64 --runs 3 \
    --random_node_idx_split_runs 20 \
    --lower_limit_random_node_selections 15 \
    --upper_limit_random_node_selections 15 \
    --use_improved_wrapper True --num_neighbors_borrow 5 --joint_alpha 0.3
```

Or use the batch script:
```bat
run_all.bat eval
```

---

## Results (Horizon 12, Subset = 15%)

| Dataset | Partial | UW | FDW | **Improved** |
|---------|---------|-----|-----|-------------|
| SOLAR (MAE) | 4.178 | 3.861 | 3.846 | 4.156 |
| SOLAR (RMSE) | 6.070 | 5.808 | 5.792 | 6.109 |
| ECG (MAE) | 3.913 | 3.207 | 3.211 | 3.250 |
| ECG (RMSE) | 6.482 | 5.531 | 5.537 | 5.611 |
| METR-LA (MAE) | 4.623 | 3.772 | 3.763 | 4.217 |
| METR-LA (RMSE) | 9.411 | 7.671 | 7.660 | 8.632 |
| TRAFFIC (MAE) | 18.927 | 11.200 | 11.195 | 11.831 |
| TRAFFIC (RMSE) | 39.167 | 27.660 | 27.654 | 28.657 |


---

## Citation

```bibtex
@inproceedings{10.1145/3534678.3539394,
  author    = {Chauhan, Jatin and Raghuveer, Aravindan and Saket, Rishi and Nandy, Jay and Ravindran, Balaraman},
  title     = {Multi-Variate Time Series Forecasting on Variable Subsets},
  year      = {2022},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3534678.3539394},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages     = {76--86},
  series    = {KDD '22}
}
```