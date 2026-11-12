# Model Evaluation & Statistical Comparison Protocol

## 1. Why Accuracy Alone Is Misleading in Retinal Grading

In diabetic retinopathy (DR) datasets such as APTOS 2019 and EyePACS, class frequencies are heavily skewed toward Grade `0` (No DR), while clinically urgent classes (`3` Severe NPDR, `4` Proliferative DR) form a small minority. A degenerate model predicting `0` exclusively can score high overall accuracy while exhibiting $0\%$ sensitivity to sight-threatening disease.

In addition, DR grades `0..4` form an **ordinal disease severity continuum**: confusing Grade `0` with Grade `4` carries far greater clinical consequence than confusing Grade `1` with Grade `2`.

## 2. Classification & Ordinal Metrics (`src/evaluation/classification.py`)

`compute_classification_metrics` reports:

1. **Overall & Imbalance-Aware Summary Metrics:**
   - **Accuracy** and **Balanced Accuracy** (macro-average per-class recall, compensating for class skew).
   - **Macro-F1** and **Weighted-F1** (macro unweighted vs support-weighted harmonic mean of precision and recall).
   - **Unweighted Cohen's $\kappa$** and **Quadratic Weighted Kappa (QWK)** (`compute_qwk`).
2. **Quadratic Weighted Kappa Formula:**
   $$\kappa_{\text{QWK}} = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$
   where $O_{ij}$ is the observed confusion count, $E_{ij}$ is the outer-product expected count under independence, and $C = 5$. Quadratic weighting penalizes multi-grade mistakes quadratically.
3. **Ordinal Error Decomposition (`compute_ordinal_error_breakdown`):**
   - `exact_match_rate`: $| \hat{y} - y | = 0$
   - `one_grade_error_rate`: $| \hat{y} - y | = 1$ (borderline adjacent grade discrepancy)
   - `severe_multigrade_error_rate`: $| \hat{y} - y | \ge 2$ (clinically hazardous under/over-grading)
   - `mean_absolute_grade_error`: $\mathbb{E}[| \hat{y} - y |]$
4. **Per-Class Discrimination:**
   - Precision, recall, F1, support, one-vs-rest **ROC-AUC**, and one-vs-rest **PR-AUC** (Average Precision), safely returning `None` when a class is absent in tiny validation/test subsets.

## 3. Complexity vs Performance Trade-offs

On CPU-only hardware (4-core Intel Core i5, 8 GB RAM), `RetinaX` benchmarks each model family alongside:
- Total and trainable parameter counts (`CustomFundusCNN` ~50K–100K vs `ResNet18` ~11.2M vs `EfficientNet-B0` ~4.0M vs `ViT-Tiny` ~5.5M vs `DINOv2 ViT-S/14` ~22M frozen).
- Single-image CPU inference latency (ms).
- Calibration (`ECE`, `NLL`) and Selective Risk (`AURC`) alongside discrimination (`QWK`, `Macro-F1`).

## 4. Paired Statistical Testing (`src/evaluation/statistical_tests.py`)

Comparing models on a single point estimate can mistake test-set variance for architectural superiority:

- **`paired_bootstrap_ci`:** Draws $B$ stratified/non-parametric resamples with replacement on identical test predictions to produce 95% confidence intervals (`[ci_low, ci_high]`) for QWK, Macro-F1, or Accuracy.
- **`paired_bootstrap_comparison`:** Evaluates paired score differences $\Delta_b = M(\text{Model B})_b - M(\text{Model A})_b$ over identical indices to compute two-sided empirical $p$-values without Gaussian assumptions.
- **`mcnemar_test`:** Tests whether the discordant predictions between two models ($n_{01}$ vs $n_{10}$ on identical images) deviate significantly from symmetry (using exact binomial testing for small discordant counts $n_{01} + n_{10} < 25$ and continuity-corrected $\chi^2$ otherwise).
