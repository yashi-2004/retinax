# Methodological Limitations, Ethics & Governance

## 1. Clinical Safety Statement

> **CLINICAL_DISCLAIMER (`src/config.py`):**
> *RetinaX is an academic research prototype. It is not a clinically validated diagnostic system and must not be used to diagnose disease, determine treatment or replace professional ophthalmic assessment.*

Automated fundus screening tools must undergo prospective multi-site clinical trials, regulatory clearance (e.g., FDA SaMD / CE MDR), clinician workflow human-factors testing, and ongoing post-deployment drift surveillance before clinical use.

## 2. Synthetic Demonstration Mode vs Real Clinical Data

When local datasets under `data/raw/` (`aptos`, `eyepacs`, `idrid`) have not been staged by the user, `RetinaX` provides a procedural retinal phantom generator (`src/data/synthetic.py`) creating primary and domain-shifted external demonstration cohorts:
- Every record is explicitly flagged `is_synthetic=True`.
- Dashboards, manifests, checkpoints, and reports prominently mark synthetic runs as **SYNTHETIC DEMONSTRATION (Non-Clinical)**.
- Synthetic phantoms validate end-to-end algorithmic plumbing, uncertainty equations, conformal math, and visualization rendering; they **do not** constitute clinical evidence of retinal lesion accuracy.

## 3. Dataset Governance & Demographic Honesty

- **Label Integrity:** Original `0..4` diabetic retinopathy severity grades are strictly preserved (`0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR`); labels are never altered, merged artificially, or relabeled to boost metrics.
- **Zero Data Leakage:** `assign_splits` keeps correlated `patient_id`s (when metadata exists) and exact (`SHA-256`) or near-duplicate (`dHash`) images atomic within a single split (`train`, `val`, `cal`, or `test`).
- **Demographic Fairness Honesty:** Public APTOS 2019 and EyePACS Kaggle releases do not include verified patient age, sex, ethnicity, or diabetes duration metadata. `RetinaX` explicitly **refuses to infer or guess protected demographic attributes from retinal appearance**, documenting subgroup fairness evaluation across demographic strata as an open limitation requiring prospective metadata-linked clinical cohorts.
- **Subgroup Proxy Analysis:** Where camera quality proxies (blur, illumination, contrast, FOV coverage) or synthetic domain indicators exist, performance stratification is reported quantitatively without substituting quality metrics for demographic fairness.

## 4. Technical & Computational Boundaries

- **CPU-Only Design:** Engineered for 4-core CPU / 8 GB RAM execution (`batch_size=8`, lazy loading, frozen foundation backbones). Scaling to full un-subsampled EyePACS (>88,000 high-res images) or end-to-end ViT/DINOv2 fine-tuning would require GPU acceleration.
- **Conformal Exchangeability Breakdown:** Finite-sample Split Conformal coverage ($1 - \alpha$) holds on exchangeable i.i.d. data and can degrade under EyePACS domain shift.
- **Post-Hoc Saliency Faithfulness:** Grad-CAM, Integrated Gradients, and Occlusion reflect feature sensitivity, not causal biological proof or calibrated lesion contours.

## 5. Dataset & Foundation Model Provenance and Licenses

| Asset | Source / Model ID | License / Terms | Notes |
| :--- | :--- | :--- | :--- |
| **APTOS 2019** | Kaggle Competition (`aptos2019-blindness-detection`) | Competition Data Terms (non-commercial research / manual Kaggle auth required) | Primary 5-class supervised cohort; user downloads manually. |
| **EyePACS** | Kaggle Competition (`diabetic-retinopathy-detection`) | Competition Data Terms (manual Kaggle auth required) | External cross-dataset cohort; bounded stratified sampling supported. |
| **IDRiD** | IEEE DataPort / Grand Challenge (`idrid.grand-challenge.org`) | CC-BY 4.0 (research use per challenge terms) | Optional lesion segmentation extension via MONAI `LesionUNetWrapper`. |
| **MedMNIST (RetinaMNIST)** | `medmnist.com` (`RetinaMNIST`) | CC-BY 4.0 | Optional lightweight pipeline/sanity benchmark (low resolution, not primary). |
| **DINOv2 ViT-S/14** | `vit_small_patch14_dinov2.lvd142m` (`timm`) | Apache License 2.0 (Meta Platforms, Inc.) | Self-supervised on LVD-142M natural images; domain gap evaluated empirically. |
| **BiomedCLIP** *(Optional)* | `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | MIT License (Microsoft Research) | Pretrained on PMC-15M biomedical figure-caption pairs; strictly optional dependency. |
| **ImageNet Backbones** | `resnet18`, `efficientnet_b0`, `densenet121`, `vit_tiny_patch16_224` (`timm`) | Apache-2.0 / BSD-3-Clause | Pretrained weights cached locally on first download; offline random-init fallback available. |
