"""Dataset discovery, governance metadata, and non-destructive raw ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DR_CLASS_NAMES, PROJECT_ROOT, resolve_path

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class DatasetGovernanceInfo:
    """Governance and access metadata for a supported retinal dataset."""

    name: str
    title: str
    source_url: str
    license_note: str
    role: str
    manual_instructions: str
    patient_id_available: bool


DATASET_DOCS: dict[str, DatasetGovernanceInfo] = {
    "aptos": DatasetGovernanceInfo(
        name="aptos",
        title="APTOS 2019 Blindness Detection",
        source_url="https://www.kaggle.com/competitions/aptos2019-blindness-detection",
        license_note="Kaggle Competition Rules — Academic / non-commercial competition terms.",
        role="Primary 5-class Diabetic Retinopathy supervised classification cohort.",
        manual_instructions=(
            "1. Accept the competition rules at Kaggle APTOS 2019.\n"
            "2. Download the dataset archive manually or via `kaggle competitions download -c aptos2019-blindness-detection`.\n"
            "3. Extract `train.csv` and `train_images/` into `data/raw/aptos/`."
        ),
        patient_id_available=False,
    ),
    "eyepacs": DatasetGovernanceInfo(
        name="eyepacs",
        title="EyePACS Diabetic Retinopathy Detection",
        source_url="https://www.kaggle.com/competitions/diabetic-retinopathy-detection",
        license_note="Kaggle Competition Rules — EyePACS attribution & non-commercial research use.",
        role="External validation dataset for cross-dataset generalisation & domain-shift testing.",
        manual_instructions=(
            "1. Accept competition rules at Kaggle Diabetic Retinopathy Detection.\n"
            "2. Download `trainLabels.csv` (or `retinopathy_solution.csv`) and a subset of images.\n"
            "3. Place CSV and `train/` or `images/` folder into `data/raw/eyepacs/`.\n"
            "Note: RetinaX uses bounded stratified sampling so the entire ~80 GB archive is unnecessary."
        ),
        patient_id_available=True,
    ),
    "idrid": DatasetGovernanceInfo(
        name="idrid",
        title="IDRiD (Indian Diabetic Retinopathy Image Dataset)",
        source_url="https://idrid.grand-challenge.org/",
        license_note="Creative Commons Attribution 4.0 (CC BY 4.0).",
        role="Optional lesion localisation / segmentation extension (MA, HE, EX, SE, OD).",
        manual_instructions=(
            "1. Register on IEEE DataPort / IDRiD Grand Challenge.\n"
            "2. Extract `A. Segmentation/1. Original Images/` and `2. All Segmentation Groundtruths/` into `data/raw/idrid/`."
        ),
        patient_id_available=False,
    ),
    "medmnist": DatasetGovernanceInfo(
        name="medmnist",
        title="MedMNIST v2 — RetinaMNIST",
        source_url="https://medmnist.com/",
        license_note="Creative Commons Attribution 4.0 (CC BY 4.0).",
        role="Optional lightweight educational / pipeline smoke-test dataset (not a replacement for APTOS).",
        manual_instructions=(
            "1. Place `retinamnist.npz` or `retinamnist_224.npz` in `data/raw/medmnist/`, "
            "or install optional `medmnist` package to download it directly."
        ),
        patient_id_available=False,
    ),
    "synthetic_demo": DatasetGovernanceInfo(
        name="synthetic_demo",
        title="Synthetic Demo Phantoms (Non-Clinical)",
        source_url="Local procedural generator (src/data/synthetic.py)",
        license_note="MIT (Local synthetic demonstration phantoms — NOT clinical images).",
        role="Offline software verification & demonstration when Kaggle datasets are unavailable.",
        manual_instructions="Generate locally via `python scripts/prepare_data.py --synthetic-demo`.",
        patient_id_available=True,
    ),
}


def _find_image_file(search_dirs: list[Path], stem_or_name: str) -> Path | None:
    """Resolve an image path matching exact filename or stem across candidate folders."""
    candidate_name = Path(stem_or_name).name
    stem = Path(stem_or_name).stem
    for directory in search_dirs:
        if not directory.exists():
            continue
        direct = directory / candidate_name
        if direct.is_file() and direct.suffix.lower() in SUPPORTED_EXTENSIONS:
            return direct
        for ext in SUPPORTED_EXTENSIONS:
            cand = directory / f"{stem}{ext}"
            if cand.is_file():
                return cand
    return None


def discover_datasets(raw_root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Inspect local `data/raw/` folders and summarize dataset availability."""
    root = resolve_path(raw_root or "data/raw")
    status: dict[str, dict[str, Any]] = {}

    for key, info in DATASET_DOCS.items():
        ds_dir = root / key
        files = [p for p in ds_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"] if ds_dir.exists() else []
        image_files = [p for p in files if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        csv_files = [p for p in files if p.suffix.lower() == ".csv"]

        available = False
        note = "Not found locally."
        if key == "aptos":
            available = (ds_dir / "train.csv").is_file() and len(image_files) > 0
            note = f"Found {len(image_files)} images." if available else "Missing `train.csv` and/or `train_images/`."
        elif key == "eyepacs":
            has_csv = any((ds_dir / c).is_file() for c in ("trainLabels.csv", "retinopathy_solution.csv", "labels.csv"))
            available = has_csv and len(image_files) > 0
            note = f"Found {len(image_files)} images." if available else "Missing labels CSV and/or image directory."
        elif key == "idrid":
            available = len(image_files) > 0
            note = f"Found {len(image_files)} image/mask files." if available else "Optional dataset not downloaded."
        elif key == "medmnist":
            npz_files = [p for p in files if p.suffix.lower() == ".npz"]
            available = len(npz_files) > 0 or len(image_files) > 0
            note = f"Found {len(npz_files)} archive(s)." if available else "Optional dataset not downloaded."
        elif key == "synthetic_demo":
            available = (ds_dir / "primary_labels.csv").is_file() and len(image_files) > 0
            note = f"Synthetic demo ready ({len(image_files)} phantoms, NOT clinical data)." if available else "Run generator to create synthetic demo phantoms."

        status[key] = {
            "name": key,
            "title": info.title,
            "available": available,
            "image_count": len(image_files),
            "csv_count": len(csv_files),
            "path": str(ds_dir),
            "license": info.license_note,
            "role": info.role,
            "patient_id_available": info.patient_id_available,
            "note": note,
            "is_synthetic": key == "synthetic_demo",
        }
    return status


def _bounded_stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    """Memory-efficient stratified sample (preserving class proportions and patient groups)."""
    if max_samples is None or len(df) <= max_samples:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []
    counts = df[label_col].value_counts()
    total = len(df)

    for cls, count in counts.items():
        cls_idx = df.index[df[label_col] == cls].to_numpy()
        target_n = max(1, int(round(max_samples * (count / total))))
        target_n = min(target_n, len(cls_idx))
        chosen = rng.choice(cls_idx, size=target_n, replace=False)
        sampled_indices.extend(chosen.tolist())

    if len(sampled_indices) > max_samples:
        sampled_indices = rng.choice(sampled_indices, size=max_samples, replace=False).tolist()

    return df.loc[sorted(sampled_indices)].reset_index(drop=True)


def load_raw_records(
    dataset_name: str,
    raw_root: str | Path | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    synthetic_cohort: str = "primary",
) -> pd.DataFrame:
    """Load standardized raw dataset records without modifying original files or labels.

    Returns a DataFrame with columns:
      `image_id`, `dataset_name`, `relative_path`, `abs_path`, `original_label`,
      `label_name`, `patient_id`, `is_synthetic`.
    """
    root = resolve_path(raw_root or "data/raw")
    ds_key = dataset_name.lower()
    ds_dir = root / ds_key

    if ds_key == "aptos":
        csv_path = ds_dir / "train.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"APTOS dataset not found at {csv_path}.\n{DATASET_DOCS['aptos'].manual_instructions}"
            )
        df = pd.read_csv(csv_path)
        if "id_code" not in df.columns or "diagnosis" not in df.columns:
            raise ValueError(f"Expected columns ['id_code', 'diagnosis'] in {csv_path}, got {list(df.columns)}")
        search_dirs = [ds_dir / "train_images", ds_dir / "images", ds_dir]
        records = []
        for row in df.itertuples(index=False):
            img_id = str(row.id_code)
            label = int(row.diagnosis)
            img_path = _find_image_file(search_dirs, img_id)
            rel_path = str(img_path.relative_to(PROJECT_ROOT)) if img_path and img_path.is_relative_to(PROJECT_ROOT) else (str(img_path) if img_path else "")
            records.append(
                {
                    "image_id": img_id,
                    "dataset_name": "aptos",
                    "relative_path": rel_path,
                    "abs_path": str(img_path) if img_path else "",
                    "original_label": label,
                    "label_name": DR_CLASS_NAMES.get(label, f"Unknown({label})"),
                    "patient_id": None,
                    "is_synthetic": False,
                }
            )
        out_df = pd.DataFrame(records)
        return _bounded_stratified_sample(out_df, "original_label", max_samples, seed)

    if ds_key == "eyepacs":
        csv_candidates = [ds_dir / "trainLabels.csv", ds_dir / "retinopathy_solution.csv", ds_dir / "labels.csv"]
        csv_path = next((p for p in csv_candidates if p.is_file()), None)
        if csv_path is None:
            raise FileNotFoundError(
                f"EyePACS labels CSV not found in {ds_dir}.\n{DATASET_DOCS['eyepacs'].manual_instructions}"
            )
        df = pd.read_csv(csv_path)
        id_col = "image" if "image" in df.columns else ("id_code" if "id_code" in df.columns else df.columns[0])
        lbl_col = "level" if "level" in df.columns else ("diagnosis" if "diagnosis" in df.columns else df.columns[1])
        search_dirs = [ds_dir / "train", ds_dir / "images", ds_dir / "test", ds_dir]

        # Filter to images actually downloaded locally if user staged a subset
        records = []
        for row in df.itertuples(index=False):
            img_id = str(getattr(row, id_col))
            label = int(getattr(row, lbl_col))
            img_path = _find_image_file(search_dirs, img_id)
            if img_path is None:
                continue
            patient_id = img_id.rsplit("_", 1)[0] if "_" in img_id else None
            rel_path = str(img_path.relative_to(PROJECT_ROOT)) if img_path.is_relative_to(PROJECT_ROOT) else str(img_path)
            records.append(
                {
                    "image_id": img_id,
                    "dataset_name": "eyepacs",
                    "relative_path": rel_path,
                    "abs_path": str(img_path),
                    "original_label": label,
                    "label_name": DR_CLASS_NAMES.get(label, f"Unknown({label})"),
                    "patient_id": patient_id,
                    "is_synthetic": False,
                }
            )
        if not records:
            raise FileNotFoundError(
                f"Labels CSV {csv_path} found, but no matching image files exist in {search_dirs}."
            )
        out_df = pd.DataFrame(records)
        return _bounded_stratified_sample(out_df, "original_label", max_samples, seed)

    if ds_key in ("synthetic_demo", "synthetic_primary", "synthetic_external"):
        syn_dir = root / "synthetic_demo"
        cohort = "external" if (ds_key == "synthetic_external" or synthetic_cohort == "external") else "primary"
        csv_path = syn_dir / f"{cohort}_labels.csv"
        img_dir = syn_dir / f"{cohort}_images"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Synthetic demo cohort '{cohort}' not found at {csv_path}. "
                "Generate it via `generate_synthetic_demo_dataset()`."
            )
        df = pd.read_csv(csv_path)
        records = []
        for row in df.itertuples(index=False):
            img_path = img_dir / str(row.filename)
            label = int(row.diagnosis)
            rel_path = str(img_path.relative_to(PROJECT_ROOT)) if img_path.is_relative_to(PROJECT_ROOT) else str(img_path)
            records.append(
                {
                    "image_id": str(row.id_code),
                    "dataset_name": f"synthetic_{cohort}",
                    "relative_path": rel_path,
                    "abs_path": str(img_path),
                    "original_label": label,
                    "label_name": DR_CLASS_NAMES.get(label, f"Unknown({label})"),
                    "patient_id": str(row.patient_id) if pd.notna(row.patient_id) else None,
                    "is_synthetic": True,
                }
            )
        out_df = pd.DataFrame(records)
        return _bounded_stratified_sample(out_df, "original_label", max_samples, seed)

    raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {list(DATASET_DOCS.keys())}")
