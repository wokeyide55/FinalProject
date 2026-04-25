# improved_wrapper.py
"""
Improved VSF Wrapper - Three enhancements over the original FDW:

  Improvement 1 – Cosine Retrieval:
      Replace the original distance (dist_exp_value=0.5) used in the
      original code with cosine similarity for neighbour retrieval.
      Time-series *shape* similarity matters more than absolute magnitude
      differences, so cosine distance is a better proxy.

  Improvement 2 – Adaptive Temperature:
      The original FDW uses a single global temperature (neighbor_temp=0.1)
      for the softmax over forecast discrepancies.  We replace this with a
      per-sample adaptive temperature equal to the standard deviation of the
      k neighbours' discrepancy scores.  When neighbours are very similar the
      temperature is small (sharp distribution); when quality varies widely the
      temperature is larger (smoother distribution).

  Improvement 3 – Joint Weighting (DDW + FDW fusion):
      The original code chooses *either* DDW (input-distance weights) *or* FDW
      (forecast-discrepancy weights).  We combine both signals:
          w_i  ∝  exp( -[ α·d_input_i  +  (1-α)·d_forecast_i ] / τ_adaptive )
      where d_input is the cosine distance from Improvement 1 and d_forecast
      is the per-step-normalised MAE from the original FDW formula.
      α is a mixing coefficient (default 0.3, tunable).

All three improvements are implemented as standalone functions so they can be
called from run_improved.py with minimal changes to the original pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# Improvement 1: Cosine-similarity based neighbour retrieval
# ─────────────────────────────────────────────────────────────────────────────

def cosine_retrieval(testx, instance_prototypes, idx_current_nodes,
                     num_neighbors, device):
    """
    Find the top-k nearest prototype windows to each test instance using
    cosine distance computed only on the observed subset S.

    Args:
        testx               : (B, C, N, T)  test batch (C=channels, already transposed)
        instance_prototypes : (P, C, N, T)  prototype bank (stored as (P, T, N, C),
                                             caller must transpose before passing)
        idx_current_nodes   : np.ndarray    indices of observed variables S
        num_neighbors       : int           k
        device              : str / torch.device

    Returns:
        cos_dist   : (B, k)   cosine *distance* (1 - similarity) for top-k
        topk_idxs  : (B, k)   prototype indices of the k nearest neighbours
    """
    idx_s = torch.LongTensor(idx_current_nodes).to(device)

    # Extract only subset-S dimensions  →  (B, C*|S|*T) and (P, C*|S|*T)
    test_s  = testx[:, :, idx_s, :]                          # (B, C, |S|, T)
    proto_s = instance_prototypes[:, :, idx_s, :]            # (P, C, |S|, T)

    B = test_s.shape[0]
    P = proto_s.shape[0]

    test_flat  = test_s.reshape(B, -1)                       # (B, D)
    proto_flat = proto_s.reshape(P, -1)                      # (P, D)

    # cosine similarity matrix  (B, P)
    test_norm  = F.normalize(test_flat,  dim=1)              # (B, D)
    proto_norm = F.normalize(proto_flat, dim=1)              # (P, D)
    sim = torch.mm(test_norm, proto_norm.t())                # (B, P)

    cos_dist = 1.0 - sim                                     # distance ∈ [0, 2]

    # topk smallest distances
    topk_dist, topk_idxs = torch.topk(cos_dist, num_neighbors,
                                       dim=-1, largest=False)
    return topk_dist, topk_idxs


def build_neighbour_inputs(testx, instance_prototypes, topk_idxs,
                           idx_current_nodes, num_neighbors, device):
    """
    Given the top-k indices, fill missing variables in each test instance
    with values from the corresponding prototype (same logic as original
    obtain_relevant_data_from_prototypes, but decoupled from distance
    computation so we can swap in our cosine retrieval).

    Returns:
        testx_filled  : (B*k, C, N, T)  one copy per neighbour with missing
                         variables filled from that neighbour
        orig_neighs   : (B*k, C, N, T)  the raw prototype windows (for FDW)
    """
    num_nodes   = testx.shape[2]
    idx_s       = torch.LongTensor(idx_current_nodes).to(device)
    rem_idx     = torch.LongTensor(
                    np.setdiff1d(np.arange(num_nodes), idx_current_nodes)
                  ).to(device)

    B           = testx.shape[0]
    testx_rep   = testx.repeat(num_neighbors, 1, 1, 1)       # (B*k, C, N, T)
    orig_neighs = []

    for j in range(num_neighbors):
        nbs          = instance_prototypes[topk_idxs[:, j].view(-1)]  # (B, C, N, T)
        orig_neighs.append(nbs)
        desired_vals = nbs[:, :, rem_idx, :]
        start, end   = j * B, (j + 1) * B
        _local       = testx_rep[start:end]
        _local[:, :, rem_idx, :] = desired_vals
        testx_rep[start:end]     = _local

    orig_neighs = torch.cat(orig_neighs, dim=0)               # (B*k, C, N, T)
    return testx_rep, orig_neighs


# ─────────────────────────────────────────────────────────────────────────────
# Improvement 2: Adaptive temperature
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_temperature(discrepancy_scores, eps=1e-6):
    """
    Compute a per-sample adaptive softmax temperature from the standard
    deviation of each sample's k neighbour discrepancy scores.

    When neighbours are very homogeneous (std ≈ 0) the temperature is kept
    at eps so the best neighbour wins almost entirely.  When scores vary a
    lot the temperature is larger, giving a smoother aggregation.

    Args:
        discrepancy_scores : (B, k)  raw discrepancy values (lower = better)

    Returns:
        tau : (B, 1)  per-sample temperature
    """
    tau = discrepancy_scores.std(dim=-1, keepdim=True)       # (B, 1)
    tau = torch.clamp(tau, min=eps)
    return tau


# ─────────────────────────────────────────────────────────────────────────────
# Improvement 3: Joint weighting  (DDW + FDW fusion)
# ─────────────────────────────────────────────────────────────────────────────

def joint_weights(cos_dist, forecast_disc, alpha=0.3, eps=1e-6):
    """
    Combine input-space cosine distance and forecast discrepancy into a
    single weight vector using adaptive temperature.

    w_i ∝ exp( -[ α·d_cos_i  +  (1-α)·d_fdw_i ] / τ_adaptive )

    Args:
        cos_dist      : (B, k)   cosine distances from Improvement 1
        forecast_disc : (B, k)   forecast discrepancy MAE from FDW step
        alpha         : float    mixing weight for cosine distance (0 → pure FDW,
                                 1 → pure cosine DDW)

    Returns:
        weights : (B, k, 1, 1)  normalised weights ready for broadcast
    """
    # Normalise each signal to [0,1] per sample so they are on the same scale
    def _minmax(x):
        mn = x.min(dim=-1, keepdim=True).values
        mx = x.max(dim=-1, keepdim=True).values
        return (x - mn) / (mx - mn + eps)

    d_cos = _minmax(cos_dist)          # (B, k)
    d_fdw = _minmax(forecast_disc)     # (B, k)

    combined = alpha * d_cos + (1.0 - alpha) * d_fdw   # (B, k)

    # Adaptive temperature from combined signal
    tau = adaptive_temperature(combined)                # (B, 1)

    weights = F.softmax(-combined / tau, dim=-1)        # (B, k)
    B, k = weights.shape
    return weights.view(B, k, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# FDW discrepancy computation (same formula as original, kept here for clarity)
# ─────────────────────────────────────────────────────────────────────────────

def compute_forecast_discrepancy(preds_split, orig_neighs_forecasts,
                                 num_neighbors, idx_current_nodes, device):
    """
    Replicates obtain_discrepancy_from_neighs from the original util.py,
    but accepts already-split predictions for clarity.

    Args:
        preds_split           : (B, k, |S|, T)  model predictions for each
                                 neighbour-filled input, subset S only
        orig_neighs_forecasts : (B, k, |S|, T)  model predictions for the raw
                                 neighbour windows, subset S only

    Returns:
        disc : (B, k)  mean per-step-normalised absolute error
    """
    T   = preds_split.shape[-1]
    B   = preds_split.shape[0]
    k   = num_neighbors

    # time-step normalisation tensor  (1, 1, 1, T)
    len_tensor = torch.arange(1, T + 1, dtype=torch.float32,
                               device=device).view(1, 1, 1, T)

    disc = torch.abs((preds_split - orig_neighs_forecasts) / len_tensor)
    disc = disc.reshape(B, k, -1).mean(dim=-1)             # (B, k)
    return disc


# ─────────────────────────────────────────────────────────────────────────────
# Prototype bank builder (same as original obtain_instance_prototypes)
# ─────────────────────────────────────────────────────────────────────────────

def build_prototype_bank(x_train, num_prots, device):
    """
    Sample one representative window from each of num_prots equal-sized
    strides of the training set.  Stored transposed as (P, C, N, T) to
    match the format expected by cosine_retrieval.

    x_train shape: (num_train, T, N, C)  – raw npz format from load_dataset
    """
    stride     = max(x_train.shape[0] // num_prots, 1)
    prototypes = []
    for i in range(num_prots):
        chunk = x_train[i * stride: (i + 1) * stride]
        idx   = np.random.randint(0, max(chunk.shape[0], 1))
        prototypes.append(chunk[idx: idx + 1])              # (1, T, N, C)

    prototypes = np.concatenate(prototypes, axis=0)         # (P, T, N, C)
    # Convert to model input format: transpose to (P, C, N, T)
    proto_tensor = torch.FloatTensor(prototypes).to(device) # (P, T, N, C)
    proto_tensor = proto_tensor.permute(0, 3, 2, 1)         # (P, C, N, T)
    print(f"[Prototype bank] shape = {proto_tensor.shape}")
    return proto_tensor
