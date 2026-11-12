# Research Methodology & Experimental Protocol

## 1. Central Research Title & Hypotheses

**Title:** *Towards Reliable Retinal Disease Grading: A Comparative Study of Convolutional Neural Networks, Self-Supervised Representations and Vision Foundation Models under Dataset Shift*

**Central Question:** Can data-efficient deep learning and modern visual foundation-model representations improve retinal disease grading while maintaining reliable uncertainty estimates and generalisation across imaging datasets?

### Ten Supporting Questions
1. How do conventional CNNs compare with pretrained CNNs and Vision Transformers?
2. Does transfer learning improve performance when labelled training data is limited?
3. Can self-supervised visual representations improve downstream classification?
4. Do pretrained foundation-model embeddings generalise better than task-specific CNN representations?
5. How does image-quality variation affect classification performance?
6. Can Grad-CAM identify meaningful retinal regions associated with model predictions?
7. Can uncertainty estimation identify potentially unreliable predictions?
8. How well do models generalise to an external retinal imaging dataset?
9. Does calibration improve the reliability of model confidence?
10. Can a model abstain from making predictions when confidence is insufficient?

## 2. Methodological Safeguards

1. **Four-Way Leakage-Free Splits (`train / val / cal / test`):**
   - `train` (65%): Model optimization and data augmentation.
   - `val` (12.5%): Early stopping and hyperparameter selection.
   - `cal` (12.5%): Held-out post-hoc Temperature Scaling and Split Conformal Prediction calibration.
   - `test` (10%): Strictly untouched until final evaluation.
2. **Duplicate & Group Isolation:** Union-find clustering merges shared `patient_id`s (when available, e.g. EyePACS) and exact/perceptual near-duplicates (`SHA-256` + `dHash` Hamming distance $\le 4$) so correlated pairs never cross split boundaries.
3. **No Test-Time Augmentation Leakage:** Augmentation applies strictly to training tensors.
4. **External Validation Purity:** External cohorts (`EyePACS` or shifted validation sets) are never used for weight updates, threshold selection, or temperature optimization.
5. **Statistical Comparison:** Model comparisons use non-parametric paired bootstrap 95% confidence intervals and McNemar's test over identical test samples.

## 3. Structured Experiments (A–J)

| ID | Experiment | Key Variables Compared |
| :--- | :--- | :--- |
| **A** | Foundational Baselines | `CustomFundusCNN` vs `ResNet18` vs `EfficientNet-B0` |
| **B** | Training Strategy | `scratch` vs `frozen` vs `partial_finetune` vs `full_finetune` |
| **C** | Label Efficiency | Nested stratified subsets (`10%`, `25%`, `50%`, `100%` of `train`) |
| **D** | Architecture & SSL | CNN vs ViT (`vit_tiny`) vs SimCLR vs DINOv2 linear probe |
| **E** | Preprocessing Ablation | `raw_resize`, `fov_crop`, `fov_clahe_norm`, `ben_graham` |
| **F** | Class Imbalance | Cross-Entropy vs Class-Weighted CE vs Focal Loss ($\gamma=2$) |
| **G** | Visual Attribution | Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion across correct/wrong/uncertain cases |
| **H** | Calibration | Uncalibrated probabilities vs held-out Temperature Scaling (ECE, Brier, NLL) |
| **I** | Cross-Dataset Shift | Internal `test` vs external cohort (QWK delta, JS shift, OOD AUROC) |
| **J** | Selective Prediction | Risk–Coverage curves, AURC, conformal set sizes, abstention thresholds |
