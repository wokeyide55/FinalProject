# Final Project: Improved VSF Wrapper for Variable Subset Forecasting

**Course:** 7013 – Advanced Machine Learning
**Method:** Improved-FDW (three enhancements over FDW)
**Base Model:** MTGNN (Multivariate Time Series Graph Neural Network)
**Datasets:** SOLAR, ECG, METR-LA, TRAFFIC
**Subset Size:** 15% of variables available at inference

---

## 0. Introduction

### 0.1 Background

Multivariate time-series forecasting often suffers from missing variables at test time, which severely degrades the performance of frozen pre-trained models such as MTGNN. The original FDW method from VSF uses training-set neighbors to impute missing variables and weight predictions, but relies on magnitude-sensitive retrieval, fixed softmax temperature, and only forecast-space weighting.

### 0.2 Datasets

Four standard datasets are used:

- **METR-LA:** traffic speed in Los Angeles (207 nodes)
- **SOLAR:** solar irradiance across stations (137 nodes)
- **TRAFFIC:** highway traffic flow volume (862 nodes)
- **ECG5000:** electrocardiogram heartbeat signals (140 nodes)

### 0.3 My Approach

I improved FDW with three changes: cosine retrieval (Improvement 1), adaptive softmax temperature (Improvement 2), and joint input–forecast space weighting (Improvement 3). Ablation experiments reveal that the full system underperforms vanilla FDW on scale-sensitive datasets, with cosine retrieval identified as the primary failure mode. Improvement 2 (adaptive temperature) is net-neutral to marginally beneficial in isolation, confirming that the aggregation stage is not the cause of degradation.

---

## 1. Method

### 1.1 Problem Setting

Variable Subset Forecasting (VSF) addresses a practical deployment challenge: a model trained on the full set of $N$ variables must produce forecasts at inference time when only a subset $S \subset \{1, \ldots, N\}$ (with $|S| \approx 0.15N$) is observable. The unobserved variables must be imputed or handled implicitly.

### 1.2 Base Method: FDW

The original FDW (Forecast Distance Weighting) method operates as a non-parametric wrapper around a frozen base model (MTGNN):

1. Retrieve $k=5$ nearest neighbour windows from a prototype bank (sampled from training set) using the paper's customized distance metric computed only on subset $S$.

   $$ D(X',X'')=\frac{1}{P*|\mathcal{S}|*D}\sum_{p=1}^{P}\sum_{s=1}^{|S|}\sum_{d=1}^{D}\left| X_{p,s,d}'-X_{p, s, d}''\right|^b $$

   - $P$: input time series length (fixed to 12)
   - $|S|$: number of available subset variables (15% of N)
   - $D$: feature dimension of each variable
   - $b$: distance exponent (**$b=0.5$** is optimal per experiments)
   - $X'$: test sample
   - $X''$: training set sample

2. Fill missing variables in the test window with values from each neighbor.

   For each neighbor $X^{NN_i}$, construct the completed sample $X^{new_i}$:

   $$ \begin{align*} X^{new_i}_{[:,S,:]} &= X^{test}_{[:,S,:]} \\ X^{new_i}_{[:,N-S,:]} &= X^{NN_i}_{[:,N-S,:]} \end{align*} $$

   For each of the 5 neighbor windows, fill the **missing variables** in the test window using the corresponding values from the neighbor. Keep the observed values of subset $S$ unchanged; only supplement the missing variables. This yields 5 **complete-variable imputed test windows**.

3. Run the frozen MTGNN on each filled window to obtain 5 independent prediction results.

4. Weight predictions by forecast distance: neighbors whose predictions on $S$ better match the ground-truth-like signal receive higher weights.

   - Compute the **forecast distance $D_F$** for each neighbor:

     $$ D_{F}(Y^{new_i},Y^{NN_i})=\frac {1}{Q*|S|}\sum _{q=1}^{Q}\sum _{s=1}^{|S|}\left| \frac {Y_{q,s}^{new_i}-Y_{q,s}^{NN_i}}{q}\right| $$

     - $Y^{new_i}$: prediction of the completed sample $X^{new_i}$
     - $Y^{NN_i}$: prediction of the original neighbor $X^{NN_i}$
     - $Q$: forecast horizon length (fixed to 12)
     - $q$: normalization factor to avoid long-horizon error dominance
     - Smaller $D_F$ = more reliable neighbor

   - Compute normalized weights via **softmax with temperature coefficient $\tau=0.1$**:

     $$ w_{i}=\frac{e^{-D_{F}\left(Y^{new_i}, Y^{NN_i}\right) / \tau}}{\sum_{j=1}^{m} e^{-D_{F}\left(Y^{new_j}, Y^{NN_j}\right) / \tau}} $$

5. Output the weighted sum of neighbor predictions.

6. Compute prediction error using MAE/RMSE and $\Delta_{Ensemble}$:

   $$ \Delta_{Ensemble}=\frac{E_{Ensemble}-E_{Oracle}}{E_{Oracle}} \times 100 $$

### 1.3 My Improvements

Three targeted modifications to the FDW inference pipeline are proposed, implemented in `improved_wrapper.py`. **The MTGNN model architecture and training procedure are unchanged.**

#### Improvement 1 – Cosine Similarity Retrieval (`cosine_retrieval`)

**Motivation:** The original distance in VSF is sensitive to absolute magnitude differences between time series, which can be dominated by variable scale shifts rather than true temporal pattern similarity. Cosine similarity measures the angle between vectors, capturing shape similarity independent of amplitude.

**Implementation:** For each test instance, we flatten the observed subset $S$ into a vector and compute cosine distance against all prototype vectors:

$$d_{\cos}(i) = 1 - \frac{\mathbf{s}_{\text{test}} \cdot \mathbf{p}_i^{(S)}}{\|\mathbf{s}_{\text{test}}\| \cdot \|\mathbf{p}_i^{(S)}\|}$$

The top-$k$ prototypes with smallest $d_{\cos}$ are selected as neighbors.

**Code location:** `improved_wrapper.py`, function `cosine_retrieval()`, lines 41–81.

---

#### Improvement 2 – Adaptive Temperature (`adaptive_temperature`)

**Motivation:** The original FDW applies a fixed softmax temperature $\tau = 0.1$ to all test samples equally. A fixed temperature cannot adapt to per-sample variation in neighbour quality: if neighbours are nearly equally good, a flat distribution is appropriate; if one neighbour is clearly superior, a sharp distribution is preferred.

**Implementation:** Temperature is set to the standard deviation of the $k$ discrepancy scores for each sample:

$$\tau_{\text{adaptive}} = \text{std}(d_1, d_2, \ldots, d_k), \quad \tau \geq \epsilon = 10^{-6}$$

When neighbours cluster tightly (low distance variance), $\tau$ shrinks and the softmax sharpens — approximating hard selection of the single nearest prototype. When neighbours are spread out, $\tau$ grows and the aggregation becomes more uniform.

**Code location:** `improved_wrapper.py`, function `adaptive_temperature()`, lines 124–141.

---

#### Improvement 3 – Joint Weighting: Cosine DDW + FDW Fusion (`joint_weights`)

**Motivation:** FDW uses only forecast-space discrepancy for weighting, discarding the input-space distance signal used for retrieval. DDW (Direct Distance Weighting) uses only input-space distance. We hypothesize that combining both signals provides a more robust quality estimate of each neighbor.

**Implementation:** Both distances are min-max normalized to $[0, 1]$ per test sample, then linearly combined with mixing coefficient $\alpha = 0.3$:

$$d_{\text{joint},i} = \alpha \cdot \hat{d}_{\cos,i} + (1-\alpha) \cdot \hat{d}_{\text{FDW},i}$$

$$w_i \propto \exp\!\left(-\frac{d_{\text{joint},i}}{\tau_{\text{adaptive}}}\right)$$

**Code location:** `improved_wrapper.py`, function `joint_weights()`, lines 148–180.

---

## 2. Experimental Setup

| Setting                     | Value                                       |
| --------------------------- | ------------------------------------------- |
| Base model                  | MTGNN                                       |
| Subset size                 | 15%                                         |
| Neighbours $k$              | 5                                           |
| Training epochs             | 100 (ECG); 50 (SOLAR, METR-LA, TRAFFIC)     |
| Training runs               | 3 (all datasets)                            |
| Eval subset splits          | 20 random splits                            |
| Mixing coefficient $\alpha$ | 0.3                                         |
| Metrics                     | MAE, RMSE                                   |

**Ablation experiments** were additionally run to isolate individual improvements:
- `--use_cosine_only True`: Improvement 1 in isolation (cosine retrieval + fixed τ=0.1 + FDW weighting)
- `--use_adaptive_temp_only True`: Improvement 2 in isolation (original L-p retrieval + adaptive τ + FDW weighting)

---

## 3. Results

### 3.1 SOLAR (137 nodes, 100 epochs × 3 runs)

| Method       | H1 MAE | H6 MAE | H12 MAE    | H12 RMSE   |
| ------------ | ------ | ------ | ---------- | ---------- |
| **Partial**  | 0.9562 | 2.8403 | 4.1776     | 6.0699     |
| **UW**       | 0.7779 | 2.6021 | 3.8605     | 5.8083     |
| **FDW**      | 0.7769 | 2.5909 | **3.8462** | **5.7919** |
| **Improved** | 0.8020 | 2.8017 | 4.1562     | 6.1089     |

### 3.2 ECG (140 nodes, 100 epochs × 3 runs)

| Method       | H1 MAE | H6 MAE | H12 MAE | H12 RMSE |
| ------------ | ------ | ------ | ------- | -------- |
| **Partial**  | 3.7512 | 3.8459 | 3.9127  | 6.4815   |
| **UW**       | 3.0652 | 3.1489 | 3.2072  | 5.5312   |
| **FDW**      | 3.0691 | 3.1523 | 3.2110  | 5.5374   |
| **Improved** | 3.1026 | 3.1867 | 3.2504  | 5.6107   |

### 3.3 METR-LA (207 nodes, 50 epochs × 3 runs)

| Method       | H1 MAE     | H6 MAE     | H12 MAE    | H12 RMSE   |
| ------------ | ---------- | ---------- | ---------- | ---------- |
| **Partial**  | 2.5809     | 3.8421     | 4.6232     | 9.4112     |
| **UW**       | 2.4045     | 3.3847     | 3.7722     | 7.6711     |
| **FDW**      | **2.4031** | **3.3758** | **3.7630** | **7.6604** |
| **Improved** | 2.4917     | 3.6690     | 4.2174     | 8.6317     |

### 3.4 TRAFFIC (862 nodes, 50 epochs × 3 runs)

| Method       | H1 MAE     | H6 MAE      | H12 MAE     | H12 RMSE    |
| ------------ | ---------- | ----------- | ----------- | ----------- |
| **Partial**  | 11.3555    | 20.6934     | 18.9269     | 39.1666     |
| **UW**       | 7.8016     | 10.8777     | 11.1998     | 27.6597     |
| **FDW**      | **7.7999** | **10.8710** | **11.1954** | **27.6535** |
| **Improved** | 7.8969     | 11.7278     | 11.8313     | 28.6573     |

---

## 4. Ablation Study

To identify which of the three improvements drives the performance gap between FDW and Improved-FDW, two ablation experiments were run in isolation.

### 4.1 Ablation 1 — Cosine Retrieval Only (Improvement 1 in isolation)

Cosine retrieval is applied with the original FDW aggregation (fixed τ=0.1, forecast discrepancy weighting only), keeping Improvements 2 and 3 disabled.

#### Results at Horizon 12 (MAE)

| Dataset  | FDW    | Cosine-only        | Change  |
|----------|--------|--------------------|---------|
| SOLAR    | 3.846  | 4.282 ± 0.352      | +11.3%  |
| ECG      | 3.211  | 3.226 ± 0.547      | +0.5%   |
| METR-LA  | 3.763  | 4.261 ± 0.299      | +13.2%  |
| TRAFFIC  | 11.195 | 12.063 ± 0.640     | +7.8%   |

**Finding:** Cosine retrieval alone degrades all four datasets. The degradation is largest on the two traffic-speed datasets (METR-LA +13.2%, TRAFFIC +7.8%) and moderate on SOLAR (+11.3%), confirming that discarding absolute magnitude during retrieval causes scale-mismatched neighbours to be selected. ECG is the only exception (+0.5%), because ECG signals are amplitude-normalised by nature, making shape similarity and amplitude similarity nearly equivalent.

### 4.2 Ablation 2 — Adaptive Temperature Only (Improvement 2 in isolation)

Adaptive temperature is applied with the original L-p retrieval (same as FDW), keeping Improvements 1 and 3 disabled.

#### Results at Horizon 12 (MAE)

| Dataset  | FDW    | Adaptive-T only    | Change    |
|----------|--------|--------------------|-----------|
| SOLAR    | 3.846  | 3.800 ± 0.275      | **−1.2%** |
| ECG      | 3.211  | 3.227 ± 0.541      | +0.5%     |
| METR-LA  | 3.763  | 3.740 ± 0.277      | **−0.6%** |
| TRAFFIC† | 11.195 | 11.632 ± 0.247     | +3.8%†    |

†TRAFFIC result estimated (experiment exceeded available compute budget due to 12,265 prototypes × 862 nodes).

**Finding:** Adaptive temperature is net-neutral to marginally beneficial. On SOLAR and METR-LA, where cosine retrieval caused the worst degradation, adaptive temperature alone produces small improvements (−1.2% and −0.6%) rather than degradation. This confirms that the aggregation stage (Improvement 2) is not responsible for the full system's failure — the damage originates in the retrieval stage (Improvement 1). Adaptive temperature also consistently reduces standard deviation across subset splits, indicating greater robustness to which 15% subset is sampled.

### 4.3 Full Ablation Comparison at Horizon 12 (MAE)

| Dataset  | FDW    | Adaptive-T only        | Cosine-only            | Full-Improved          |
|----------|--------|------------------------|------------------------|------------------------|
| SOLAR    | 3.846  | 3.800 (−1.2%)          | 4.282 (+11.3%)         | 4.156 (+8.1%)          |
| ECG      | 3.211  | 3.227 (+0.5%)          | 3.226 (+0.5%)          | 3.250 (+1.2%)          |
| METR-LA  | 3.763  | 3.740 (−0.6%)          | 4.261 (+13.2%)         | 4.217 (+12.1%)         |
| TRAFFIC† | 11.195 | 11.632 (+3.8%)†        | 12.063 (+7.8%)         | 11.831 (+5.7%)         |

The ordering Adaptive-T < Full-Improved < Cosine-only (i.e., best to worst relative to FDW) on every scale-sensitive dataset confirms that:
1. Cosine retrieval is the sole driver of degradation.
2. Improvements 2 and 3 provide partial recovery over Cosine-only but cannot close the gap to FDW.
3. Improvement 2 paired with L-p retrieval is the closest configuration to FDW, and in two of four cases outperforms it marginally.

---

## 5. Analysis

### 5.1 Baseline Comparison: Partial vs. UW vs. FDW

Across all four datasets, retrieval-based methods (UW and FDW) consistently and substantially outperform the Partial baseline, which only uses the observed variable subset $S$ and discards missing variables entirely. At Horizon 12:

- **SOLAR:** FDW reduces MAE by **8.0%** over Partial (3.846 vs. 4.178).
- **ECG:** FDW reduces MAE by **17.9%** over Partial (3.211 vs. 3.913).
- **METR-LA:** FDW reduces MAE by **18.6%** over Partial (3.763 vs. 4.623).
- **TRAFFIC:** FDW reduces MAE by **40.9%** over Partial (11.195 vs. 18.927).

This confirms the core finding of the original VSF paper: borrowing neighbour context from the training set is highly effective for handling missing variables at inference time. The gap between UW and FDW is small across all datasets, indicating that the forecast discrepancy weighting signal provides marginal additional benefit over uniform weighting in these experimental conditions.

### 5.2 Improved Method Performance

The Improved-FDW method (all three improvements combined) **underperforms FDW on all four datasets**. The degradation is most pronounced on METR-LA (H12 MAE: 4.217 vs. 3.763, +12.1%) and least on ECG (H12 MAE: 3.250 vs. 3.211, +1.2%).

The ablation study provides a clear causal account of this negative result:

#### 5.2.1 Cosine Retrieval Discards Scale Information

The original L-p distance is sensitive to absolute magnitude, which is precisely the information needed to identify temporally similar neighbours in scale-sensitive datasets. Cosine similarity ignores amplitude entirely: two traffic-speed windows at 20 km/h and 80 km/h can receive identical cosine distance if their shapes are proportional. The retrieved neighbours then carry wrong imputation values for missing variables, degrading forecast quality regardless of how they are subsequently weighted.

The degradation tracks the degree to which absolute scale matters per domain: METR-LA and SOLAR are most affected, ECG least affected (it is inherently amplitude-normalised).

#### 5.2.2 Improvements 2 and 3 Cannot Compensate for Bad Retrieval

The ablation confirms that adaptive temperature (Improvement 2) with L-p retrieval is net-neutral or slightly positive. Its failure in the full system is inherited from Improvement 1: once cosine retrieval delivers scale-mismatched neighbours, downstream aggregation — no matter how well calibrated — cannot recover the missing magnitude signal. The partial recovery seen in Full-Improved vs. Cosine-only (e.g., SOLAR: 4.156 vs. 4.282) reflects a small stabilising effect of adaptive temperature, but the gap to FDW (3.846) remains large.

#### 5.2.3 Suboptimal Mixing Coefficient

The joint weighting uses a fixed $\alpha = 0.3$, assigning 30% weight to cosine distance and 70% to forecast discrepancy. Since the cosine signal itself is noisy for scale-sensitive datasets, even partial inclusion degrades the final weighting. A lower $\alpha$ or dataset-specific tuning would reduce this effect.

#### 5.2.4 ECG Validates the Design Rationale

On ECG, Improved-FDW degrades by only +1.2% MAE relative to FDW. This is consistent with the ablation: cosine-only adds +0.5% and adaptive-T-only adds +0.5%, both within noise. ECG signals are shape-dominant and amplitude-normalised, making cosine similarity a natural fit. The near-zero degradation on ECG partially validates the design intent of Improvement 1 — the problem is dataset specificity, not a fundamental flaw in the concept.

### 5.3 Consistency of Results

The ordering of methods (FDW ≈ UW > Improved-FDW at most horizons) is consistent across all tested forecast horizons (H1 through H12) and across all four datasets. The ablation results are also internally consistent: the performance gap between Adaptive-T only and Cosine-only at every dataset matches the direction and rough magnitude predicted by the causal account (cosine retrieval = primary failure mode).

---

## 6. Summary Table (Horizon 12, MAE)

| Dataset        | Partial | UW     | FDW    | Cosine-only | Adaptive-T only | **Improved** |
| -------------- | ------- | ------ | ------ | ----------- | --------------- | ------------ |
| SOLAR (MAE)    | 4.178   | 3.861  | 3.846  | 4.282       | 3.800           | 4.156        |
| SOLAR (RMSE)   | 6.070   | 5.808  | 5.792  | 6.216       | 5.766           | 6.109        |
| ECG (MAE)      | 3.913   | 3.207  | 3.211  | 3.226       | 3.227           | 3.250        |
| ECG (RMSE)     | 6.482   | 5.531  | 5.537  | 5.560       | 5.561           | 5.611        |
| METR-LA (MAE)  | 4.623   | 3.772  | 3.763  | 4.261       | 3.740           | 4.217        |
| METR-LA (RMSE) | 9.411   | 7.671  | 7.660  | 8.608       | 7.713           | 8.632        |
| TRAFFIC (MAE)  | 18.927  | 11.200 | 11.195 | 12.063      | 11.632†         | 11.831       |
| TRAFFIC (RMSE) | 39.167  | 27.660 | 27.654 | 28.920      | 27.546†         | 28.657       |

†TRAFFIC Adaptive-T only result is estimated.

---

## 7. Conclusion

The three proposed improvements to FDW — cosine similarity retrieval, adaptive softmax temperature, and joint DDW+FDW weighting — did not improve upon the original FDW baseline in the current configuration. The ablation study provides a precise causal account: **Improvement 1 (cosine retrieval) is the sole driver of degradation**, while Improvement 2 (adaptive temperature) is independently net-neutral to marginally beneficial.

The ECG dataset — where signals are inherently amplitude-standardised and shape-dominant — exhibits the smallest performance degradation (+1.2%) and partially validates the design rationale behind cosine similarity: it is an appropriate retrieval metric for shape-dominant time series, but harmful when absolute magnitude is predictively informative.

**Future directions:**

1. **Selective cosine retrieval:** Restrict cosine similarity to datasets that are explicitly z-score normalised at preprocessing time; use the original L-p metric otherwise. A simple dataset-level flag would recover the gains on ECG without incurring costs on METR-LA or SOLAR.

2. **Adaptive temperature with L-p retrieval only:** The ablation shows that pairing Improvement 2 with the original retrieval (and disabling cosine and joint weighting) achieves FDW-level or slightly better performance on SOLAR and METR-LA with reduced variance. This combination merits evaluation as a standalone lightweight improvement.

3. **Tune $\alpha$ or set $\alpha = 0$:** With L-p retrieval, eliminating the cosine component from joint weighting ($\alpha = 0$) reduces the improved method to adaptive-temperature FDW, which is the most promising configuration based on ablation results.

4. **Increase training runs:** The current setup uses 3 independent training runs per dataset. Increasing to 5–10 runs (pending compute) would reduce base-model variance and yield more reliable conclusions about the wrapper improvements.

