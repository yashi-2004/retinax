# Uncertainty Quantification, Calibration & Selective Prediction

## 1. Why Softmax Confidence Is Insufficient

Modern neural networks are frequently **overconfident and miscalibrated** — emitting high softmax probability ($\ge 0.90$) on ambiguous borderline grades, motion-blurred fundi, or external camera cohorts. In medical screening, trusting raw argmax probabilities risks missed severe pathology.

`RetinaX` separates uncertainty modeling into four complementary layers.

## 2. Calibration Metrics & Temperature Scaling (`src/evaluation/calibration.py`)

`compute_calibration_metrics` evaluates probability trustworthiness via:
- **Expected Calibration Error (ECE):** Weighted gap between empirical accuracy and mean confidence across $M = 10$ uniform probability bins:
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$
- **Maximum Calibration Error (MCE):** Worst-case bin calibration gap $\max_m |\text{acc}(B_m) - \text{conf}(B_m)|$.
- **Multiclass Brier Score:** Proper strictly scoring rule $\frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{C} (p_{nk} - y_{nk})^2$.
- **Negative Log Likelihood (NLL):** Cross-entropy of predicted probabilities on ground-truth classes.

### Post-Hoc Temperature Scaling (`TemperatureScaler`)
Optimizes a single scalar temperature $T > 0$ via L-BFGS minimizing NLL strictly on the held-out `cal` split:
$$\hat{p}_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$
Temperature scaling preserves rank ordering (and therefore top-1 accuracy/QWK) while aligning softmax sharpness with true empirical accuracy.

## 3. Stochastic & Ensemble Uncertainty (`src/uncertainty/`)

### 3.1 Monte Carlo Dropout (`src/uncertainty/mc_dropout.py`)
`enable_dropout_only` keeps `BatchNorm` / `LayerNorm` in deterministic `eval` mode while activating `Dropout` layers for $T$ forward passes (`mc_dropout_predict_single`), decomposing softmax samples $\{p^{(t)}\}_{t=1}^T$ (`_decompose_mc_probabilities`) into:
- **Predictive Entropy (Total Uncertainty):** $H(\bar{p}) = -\sum_c \bar{p}_c \log \bar{p}_c$
- **Expected Entropy (Aleatoric Uncertainty):** $\frac{1}{T} \sum_{t=1}^T H(p^{(t)})$ (irreducible input ambiguity / noise)
- **Mutual Information (Epistemic BALD Uncertainty):** $I(y; w \mid x) = H(\bar{p}) - \frac{1}{T} \sum_{t=1}^T H(p^{(t)})$ (model knowledge disagreement)
- **Predictive Variance:** Mean across classes of $\text{Var}_t(p^{(t)}_c)$.

### 3.2 Deep Ensembles (`src/uncertainty/ensembles.py`)
`predict_ensemble_single` and `predict_ensemble_batch` aggregate $M$ independently initialized/trained architectures or seeds with optional per-member fitted temperatures, computing ensemble member agreement alongside decomposed variance, entropy, and mutual information.

## 4. Distribution-Free Split Conformal Prediction (`src/uncertainty/conformal.py`)

`SplitConformalPredictor` fits quantile $\hat{q}$ strictly on the held-out `cal` split using either:
- **LAC (Least Ambiguous Classifier):** Non-conformity score $s(x, y) = 1 - \hat{p}_y(x)$.
- **APS (Adaptive Prediction Sets):** Cumulative sorted softmax mass required to include true label $y$, producing larger prediction sets on ambiguous grades and tight singleton sets on clear examples.

For finite sample $n_{\text{cal}}$, $\hat{q}$ is the $\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil / n_{\text{cal}}$ empirical quantile (clamped at $1.0$ if $n_{\text{cal}} < \lceil 1/\alpha \rceil$, with an explicit small-sample warning emitted).

- **Assumption & Caveat:** Marginal finite-sample coverage guarantee $\mathbb{P}(Y_{\text{test}} \in C(X_{\text{test}})) \ge 1 - \alpha$ holds under **exchangeability** between `cal` and test samples. Under cross-dataset shift (e.g., EyePACS external validation), empirical coverage may drop below $1 - \alpha$.

## 5. Selective Prediction & Abstention (`src/uncertainty/selective_prediction.py`)

- `evaluate_selective_decision` flags an input as **`ABSTAIN — Refer for Ophthalmic Expert Review`** whenever any configurable gate triggers: low confidence (`confidence < min_confidence`), high predictive entropy (`entropy > max_entropy`), wide conformal set (`set_size > max_conformal_set_size`), or degraded fundus quality (`quality_score < min_quality_score`).
- `compute_risk_coverage_curve` sweeps acceptance thresholds from confident subsets (`20%` coverage) to `100%` coverage and integrates **AURC (Area Under Risk–Coverage Curve)** to measure whether uncertainty rankings cleanly separate accurate from error-prone predictions.
