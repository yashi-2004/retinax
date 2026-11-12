# RetinaX System Architecture

## 1. Design Philosophy

RetinaX enforces strict separation of concerns across data engineering, computer vision preprocessing, neural modeling, trustworthy uncertainty/explainability, experiment tracking, and the Gradio biomedical interface. No paid APIs, cloud GPUs, or proprietary databases are required; all components execute deterministically on a standard CPU-only laptop (e.g., 4-core Intel Core i5, 8 GB RAM).

## 2. Architecture Diagram

```mermaid
flowchart LR
    A[Raw Datasets: APTOS / EyePACS / IDRiD / Synthetic Demo] --> B[Data Governance & Validation: SHA-256, dHash, Corrupt Check]
    B --> C[Leakage-Free Splitting: Stratified + Patient/Duplicate Grouping into train/val/cal/test]
    C --> D[Preprocessing & QA Lab: FOV Crop, CLAHE, Ben Graham, Quality Score]
    D --> E1[Classical CNNs: CustomFundusCNN]
    D --> E2[Transfer Learning: ResNet18, EfficientNet-B0, DenseNet121]
    D --> E3[Vision Transformers & SSL: ViT-Tiny, SimCLR]
    D --> E4[Foundation Models: DINOv2 Frozen Probe, Optional BiomedCLIP]
    E1 & E2 & E3 & E4 --> F[Training & Calibration: Weighted CE / Focal, Temp Scaling, Split Conformal]
    F --> G1[Explainability: Grad-CAM, Grad-CAM++, Captum IG, Occlusion]
    F --> G2[Uncertainty & Abstention: MC Dropout, Ensembles, Risk-Coverage]
    F --> G3[Generalisation & OOD: Cross-Cohort Shift, Perturbations, Mahalanobis/Entropy]
    F & G1 & G2 & G3 --> H[TensorBoard + JSON/CSV Experiment Registry + MD/HTML/PDF Reports]
    H --> I[14-Tab Gradio Biomedical Research Dashboard]
```

## 3. Memory & CPU Optimization Architecture

- **Lazy Image Loading:** `FundusDataset` decodes images on demand from disk (`num_workers=0` safe on macOS).
- **Pre-resized Disk Cache:** Repeated epochs read standardized PNGs cached under `data/processed/images/<config_hash>/`.
- **Frozen Embedding Caching:** High-parameter backbones (DINOv2 ViT-S/14, ViT-Tiny) extract features once to `.npz` archives under `experiments/cache/`, enabling millisecond linear/MLP probes.
- **CPU Thread Control:** `resolve_device_and_threads` tunes PyTorch intra-op threads and disables CUDA AMP cleanly on CPU.

## 4. Module Layout

- `src/data/`: Ingestion, synthetic phantom generator, SHA-256 + dHash duplicate validation, manifest serialization, leakage-free splitting, lazy PyTorch caching.
- `src/preprocessing/`: Classical OpenCV filters, FOV masking, Ben Graham illumination correction, Albumentations clinical augmentations, classical image quality proxy analyzer.
- `src/models/`: `CustomFundusCNN`, `ResNet18Classifier`, `EfficientNetB0Classifier`, `DenseNet121Classifier`, `ViTClassifier`, embedding probes, CPU inference engine, optional MONAI `LesionUNetWrapper`.
- `src/foundation/`: `SimCLREncoderModel` + NT-Xent pretraining, `DINOv2FeatureExtractor`, optional `BiomedCLIPExperiment`, PCA/t-SNE projection.
- `src/training/`: Class-weighted & Focal losses, differential LR optimizers, schedulers, reproducible checkpoints, TensorBoard `Trainer`.
- `src/evaluation/`: Multi-class/ordinal metrics (QWK, OvR AUROC/PR-AUC), Temperature Scaling & ECE, paired bootstrap CIs & McNemar test, perturbation stress & domain shift / OOD detection.
- `src/explainability/`: Hook-based Grad-CAM / Grad-CAM++, Captum Integrated Gradients & Occlusion Sensitivity, FOV attribution diagnostics.
- `src/uncertainty/`: Dropout-only MC inference, Deep Ensembles, Split Conformal Prediction (LAC/APS), Risk–Coverage abstention gates.
- `app/`: Modular 14-tab Gradio interface (`app/main.py`, `app/pages/`, `app/components/`).
