# Dataset Governance, Provenance & Setup

## 1. Dataset Overview & Licensing

| Dataset | Role in RetinaX | Source & Access | License / Terms | Patient IDs |
| :--- | :--- | :--- | :--- | :--- |
| **APTOS 2019** | Primary 5-class DR grading (`0..4`) | Kaggle (`aptos2019-blindness-detection`) | Kaggle Competition Rules (Non-commercial / research) | Unavailable |
| **EyePACS** | External domain-shift validation | Kaggle (`diabetic-retinopathy-detection`) | Kaggle Competition Rules / EyePACS attribution | Available (`<pt>_left`, `<pt>_right`) |
| **IDRiD** | Optional lesion segmentation (MONAI U-Net) | Grand Challenge / IEEE DataPort | Creative Commons Attribution 4.0 (CC BY 4.0) | Unavailable |
| **MedMNIST (RetinaMNIST)** | Optional lightweight pipeline smoke testing | `medmnist.com` | CC BY 4.0 | Unavailable |
| **Synthetic Demo Phantoms** | Out-of-the-box offline verification | Procedural (`src/data/synthetic.py`) | MIT (`is_synthetic=True`, NON-CLINICAL) | Simulated |

## 2. Five-Class DR Severity Taxonomy (Preserved Unmodified)

- `0`: **No DR**
- `1`: **Mild**
- `2`: **Moderate**
- `3`: **Severe**
- `4`: **Proliferative DR (PDR)**

Original ground-truth labels are never remapped, merged, or relabelled to artificially inflate accuracy.

## 3. Manual Setup Instructions (Kaggle Datasets)

Because Kaggle requires rule acceptance and user authentication, RetinaX never circumvents download walls:

1. **APTOS 2019:**
   - Accept competition terms at `https://www.kaggle.com/competitions/aptos2019-blindness-detection`.
   - Place `train.csv` and `train_images/` inside `data/raw/aptos/`.
2. **EyePACS (Subset Recommended for 8 GB RAM):**
   - Accept terms at `https://www.kaggle.com/competitions/diabetic-retinopathy-detection`.
   - Place `trainLabels.csv` and downloaded images inside `data/raw/eyepacs/train/`.
   - Use `python scripts/prepare_data.py --dataset eyepacs --max-samples 500` for bounded stratified sampling.
3. **Synthetic Demonstration Mode (Immediate Offline Testing):**
   - Run `python scripts/prepare_data.py --synthetic-demo`.
   - All manifests and model checkpoints generated from this mode carry `is_synthetic=True`.

## 4. Manifest Schema

Every manifest stored in `data/manifests/` records: `image_id`, `dataset_name`, `relative_path`, `original_label`, `label_name`, `patient_id`, `width`, `height`, `channels`, `aspect_ratio`, `processed_width`, `processed_height`, `sha256`, `phash`, `duplicate_cluster_id`, `is_corrupt`, `validation_error`, `quality_score`, `quality_category`, `quality_issues`, `split`, `is_synthetic`, and `processing_config`.
