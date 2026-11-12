# Cross-Dataset Generalisation, Domain Shift & OOD Detection

## 1. Sources of Domain Shift in Retinal Photography

Models trained on a single fundus cohort (such as APTOS 2019) routinely suffer performance and calibration degradation when evaluated on external cohorts (such as EyePACS) due to:
1. **Optical & Sensor Differences:** Camera field angle ($30^\circ$–$50^\circ$), resolution, JPEG compression, lens flare, and sensor Bayer color balance.
2. **Population & Prevalence Shift:** Clinic screening populations vs specialist referral centers alter the prior label distribution $P(Y)$.
3. **Acquisition Protocol & Quality Variation:** Ambient pupil dilation, cataract haze, vignetting severity, and ungradable focus artifacts.

## 2. Quantifying Shift Without Data Leakage (`src/evaluation/generalisation.py`)

`compare_cohorts_distribution_shift` evaluates internal held-out `test` predictions side-by-side with external cohort predictions while keeping external labels strictly out of training, hyperparameter tuning, and temperature calibration:

- **Performance & Calibration Delta:** Changes in **QWK**, **Macro-F1**, **Balanced Accuracy**, **ECE**, and **NLL** between internal `test` and external evaluation.
- **Label Prior Shift:** Symmetric **Jensen–Shannon distance** between internal and external $P(Y)$ histograms.
- **Image Covariate Shift:** Per-feature **Wasserstein-1 distance** and **Kolmogorov–Smirnov two-sample statistic ($p$-value)** over unsupervised fundus statistics (`brightness`, `contrast`, `blur_laplacian`, `fov_coverage`, `mean_r`, `mean_g`, `mean_b`) extracted via `inspect_fundus_image`.
- **Confidence vs Accuracy Divergence:** Tracks whether predictive entropy increases appropriately under shift or remains erroneously confident.

## 3. Perturbation Stress Testing (`src/evaluation/robustness.py`)

`evaluate_perturbation_robustness` applies deterministic, severity-indexed ($s \in [0, 1]$) image degradations (`apply_controlled_perturbation`):
- `gaussian_blur` (defocus / media opacity)
- `underexposure` & `overexposure` (illumination failures)
- `low_contrast` (cataract / washed-out fundus)
- `gaussian_noise` (low-light sensor noise)
- `partial_occlusion` (eyelid/lash or dust artifact)

Evaluating QWK and ECE across severity sweeps reveals brittle preprocessing dependencies and sensitivity differences between CNNs, ViTs, and DINOv2 probes.

## 4. Test-Time Colour Normalization Without External Labels

`RetinaX` investigates strictly label-free unsupervised fundus normalization (`fov_clahe_norm` gray-world balance + CLAHE luminance normalization vs `ben_graham` local spatial subtraction in `src/preprocessing/fundus_processing.py`) to test how much cross-device chromatic shift can be attenuated before model inference without fitting on external cohort targets.

## 5. Out-of-Distribution (OOD) Detection Benchmarks (`src/evaluation/generalisation.py`)

`evaluate_ood_detection` benchmarks four post-hoc OOD detectors using **AUROC** and **FPR@95%TPR**:
1. **Inverse Maximum Softmax Probability (`max_softmax_inv`):** $1 - \max_c \hat{p}_c(x)$.
2. **Predictive Entropy (`predictive_entropy`):** $H(\hat{p}(x))$.
3. **Feature-Space Mahalanobis Distance (`mahalanobis_distance`):** Fit with regularized empirical covariance $\Sigma + \epsilon I$ on internal training embeddings (`fit_feature_ood_detector`, `score_feature_ood`).
4. **Cosine 1-NN Embedding Distance (`cosine_knn_distance`):** $1 - \max_{i \in \text{train}} \cos(\phi(x), \phi(x_i))$ over L2-normalized features.

> **Important Limitation:** As explicitly recorded in `evaluate_ood_detection`'s `limitation_note`, no post-hoc OOD score achieves perfect separation in medical imaging; subtle acquisition shift can still trigger high-confidence misclassifications, motivating human-in-the-loop abstention.
