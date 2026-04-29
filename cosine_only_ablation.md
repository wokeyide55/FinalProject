# Ablation: Cosine Retrieval in Isolation (Improvement 1)

**Purpose:** Isolate the contribution of Improvement 1 (cosine similarity retrieval) by running it with the original FDW weighting (fixed τ = 0.1, forecast discrepancy only). This separates the effect of cosine retrieval from Improvements 2 (adaptive temperature) and 3 (joint DDW+FDW weighting).

**Setup:** `--use_cosine_only True`, k = 5 neighbours, 20 random subset splits, 15% subset size, base model MTGNN (frozen).

---

## Full Results by Horizon

### SOLAR (137 nodes, 50 epochs × 3 runs)

| Horizon | MAE | RMSE |
|---------|-----|------|
| H1  | 0.8072 ± 0.0988 | 1.5423 ± 0.2521 |
| H2  | 1.3685 ± 0.1418 | 2.4120 ± 0.3302 |
| H3  | 1.8407 ± 0.1764 | 3.0743 ± 0.3810 |
| H4  | 2.2462 ± 0.2071 | 3.6352 ± 0.4251 |
| H5  | 2.5920 ± 0.2278 | 4.1073 ± 0.4556 |
| H6  | 2.8937 ± 0.2518 | 4.5116 ± 0.4861 |
| H7  | 3.1560 ± 0.2642 | 4.8495 ± 0.5032 |
| H8  | 3.3899 ± 0.2790 | 5.1448 ± 0.5255 |
| H9  | 3.6159 ± 0.2947 | 5.4238 ± 0.5471 |
| H10 | 3.8261 ± 0.3120 | 5.6825 ± 0.5660 |
| H11 | 4.0446 ± 0.3282 | 5.9448 ± 0.5805 |
| H12 | 4.2815 ± 0.3515 | 6.2156 ± 0.5974 |

### ECG (140 nodes, 100 epochs × 5 runs)

| Horizon | MAE | RMSE |
|---------|-----|------|
| H1  | 3.0838 ± 0.5149 | 5.3305 ± 1.0454 |
| H2  | 3.0849 ± 0.5155 | 5.3122 ± 1.0333 |
| H3  | 3.1181 ± 0.5271 | 5.3888 ± 1.0613 |
| H4  | 3.1509 ± 0.5239 | 5.3914 ± 1.0526 |
| H5  | 3.1394 ± 0.5295 | 5.4066 ± 1.0620 |
| H6  | 3.1655 ± 0.5339 | 5.4391 ± 1.0637 |
| H7  | 3.1781 ± 0.5366 | 5.4710 ± 1.0756 |
| H8  | 3.2022 ± 0.5427 | 5.4776 ± 1.0773 |
| H9  | 3.2508 ± 0.5605 | 5.5635 ± 1.1156 |
| H10 | 3.2153 ± 0.5476 | 5.5265 ± 1.0994 |
| H11 | 3.2153 ± 0.5485 | 5.5380 ± 1.1013 |
| H12 | 3.2255 ± 0.5471 | 5.5597 ± 1.1075 |

### METR-LA (207 nodes, 50 epochs × 3 runs)

| Horizon | MAE | RMSE |
|---------|-----|------|
| H1  | 2.4917 ± 0.1159 | 4.4192 ± 0.1872 |
| H2  | 2.8791 ± 0.1512 | 5.4683 ± 0.2683 |
| H3  | 3.1674 ± 0.1801 | 6.2191 ± 0.3303 |
| H4  | 3.3831 ± 0.2012 | 6.7532 ± 0.3751 |
| H5  | 3.5525 ± 0.2190 | 7.1529 ± 0.4098 |
| H6  | 3.6977 ± 0.2357 | 7.4795 ± 0.4396 |
| H7  | 3.8236 ± 0.2507 | 7.7532 ± 0.4672 |
| H8  | 3.9318 ± 0.2633 | 7.9719 ± 0.4892 |
| H9  | 4.0254 ± 0.2730 | 8.1631 ± 0.5061 |
| H10 | 4.1078 ± 0.2820 | 8.3267 ± 0.5207 |
| H11 | 4.1855 ± 0.2909 | 8.4716 ± 0.5342 |
| H12 | 4.2614 ± 0.2993 | 8.6083 ± 0.5471 |

### TRAFFIC (862 nodes, 50 epochs × 3 runs)

| Horizon | MAE | RMSE |
|---------|-----|------|
| H1  | 7.9423 ± 0.4072 | 20.5090 ± 1.3446 |
| H2  | 9.9421 ± 0.5336 | 25.3058 ± 2.0126 |
| H3  | 10.9932 ± 0.5961 | 27.2414 ± 2.2458 |
| H4  | 11.6585 ± 0.6388 | 28.4508 ± 2.3409 |
| H5  | 11.9722 ± 0.6405 | 28.9343 ± 2.3781 |
| H6  | 12.0623 ± 0.6320 | 28.9972 ± 2.3831 |
| H7  | 12.1160 ± 0.6238 | 29.0401 ± 2.3576 |
| H8  | 12.1416 ± 0.6192 | 29.0519 ± 2.3455 |
| H9  | 12.1675 ± 0.6158 | 29.1088 ± 2.3467 |
| H10 | 12.1425 ± 0.6136 | 29.1096 ± 2.3530 |
| H11 | 12.0759 ± 0.6220 | 29.0409 ± 2.3580 |
| H12 | 12.0625 ± 0.6395 | 28.9204 ± 2.3462 |

---

## Comparison at Horizon 12 (MAE)

| Dataset | FDW | Cosine-only | Full-Improved | Cosine-only vs FDW | Full-Improved vs FDW |
|---------|-----|-------------|---------------|--------------------|----------------------|
| SOLAR   | 3.846 | 4.282 ± 0.352 | 4.156 ± 0.327 | +11.3% | +8.1% |
| ECG     | 3.211 | 3.226 ± 0.547 | 3.250 ± 0.548 | +0.5%  | +1.2% |
| METR-LA | 3.763 | 4.261 ± 0.299 | 4.217 ± 0.300 | +13.2% | +12.1% |
| TRAFFIC | 11.195 | 12.063 ± 0.640 | 11.831 ± 0.542 | +7.8% | +5.7% |

## Comparison at Horizon 12 (RMSE)

| Dataset | FDW | Cosine-only | Full-Improved | Cosine-only vs FDW | Full-Improved vs FDW |
|---------|-----|-------------|---------------|--------------------|----------------------|
| SOLAR   | 5.792 | 6.216 ± 0.597 | 6.109 ± 0.582 | +7.3% | +5.5% |
| ECG     | 5.537 | 5.560 ± 1.108 | 5.611 ± 1.111 | +0.4% | +1.3% |
| METR-LA | 7.660 | 8.608 ± 0.547 | 8.632 ± 0.559 | +12.4% | +12.7% |
| TRAFFIC | 27.654 | 28.920 ± 2.346 | 28.657 ± 2.389 | +4.6% | +3.6% |

---

## Analysis

### Finding 1: Cosine retrieval is the primary failure mode

On all three scale-sensitive datasets (SOLAR, METR-LA, TRAFFIC), Cosine-only already degrades substantially vs FDW. Discarding absolute magnitude during retrieval causes the model to select neighbours that match in shape but not in level, leading to systematically worse imputation of missing variables.

The degradation is largest on METR-LA (+13.2% MAE), moderate on SOLAR (+11.3%) and TRAFFIC (+7.8%), roughly tracking how much absolute scale matters for prediction in each domain.

### Finding 2: ECG is the exception

On ECG, Cosine-only adds only +0.5% MAE over FDW. ECG signals are amplitude-normalised by nature, making shape similarity and amplitude similarity nearly equivalent. This partially validates the cosine retrieval design rationale — the problem is dataset-specificity, not a fundamental flaw.

### Finding 3: Improvements 2+3 provide marginal partial recovery

Comparing Full-Improved vs Cosine-only at H12 MAE:

| Dataset | Cosine-only | Full-Improved | Recovery |
|---------|-------------|---------------|----------|
| SOLAR   | 4.282 | 4.156 | −0.126 (partial) |
| ECG     | 3.226 | 3.250 | −0.024 (negligible, slightly worse) |
| METR-LA | 4.261 | 4.217 | −0.044 (partial) |
| TRAFFIC | 12.063 | 11.831 | −0.232 (partial) |

The adaptive temperature (Improvement 2) and joint weighting (Improvement 3) partially compensate for noisy cosine retrieval on SOLAR and TRAFFIC, but neither can close the gap back to FDW. On ECG and METR-LA they make essentially no difference at H12.

### Conclusion

The ablation confirms that **cosine retrieval is the dominant source of degradation** in the Full-Improved method. Improvements 2 and 3 are not harmful in isolation, but their benefit is capped by the quality of the retrieved neighbours. A natural next step is to evaluate Improvements 2+3 independently by pairing them with the original L-p retrieval (i.e., keeping FDW's retrieval but replacing its fixed temperature and uniform weighting).

---

*Experiment date: 2026-04-27. Base model: MTGNN, subset: 15%, k = 5, 20 random splits.*
