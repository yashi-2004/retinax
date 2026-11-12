"""Unit tests for data ingestion, validation, splitting, preprocessing, and quality assessment."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.config import DR_CLASS_NAMES
from src.data.caching import FundusDataset, normalize_rgb_tensor
from src.data.ingestion import load_raw_records
from src.data.manifests import build_manifest, load_manifest, save_manifest
from src.data.splitting import create_label_efficiency_subsets, verify_no_split_leakage
from src.data.synthetic import render_synthetic_fundus_phantom
from src.data.validation import compute_dhash, hamming_distance_hex, validate_image_file
from src.preprocessing.augmentation import build_augmentation_pipeline, generate_augmentation_grid
from src.preprocessing.enhancement import (
    apply_clahe,
    apply_gaussian_blur,
    apply_histogram_equalization,
    apply_median_filter,
    convert_color_space,
    extract_green_channel,
)
from src.preprocessing.fundus_processing import (
    crop_to_fov,
    extract_fov_mask,
    inspect_fundus_image,
    preprocess_fundus_image,
)
from src.preprocessing.quality_assessment import QualityProxyClassifier, assess_fundus_quality, extract_quality_features


def test_image_loading_validation_and_corrupt_handling(tmp_path: Path) -> None:
    # Valid image
    arr = render_synthetic_fundus_phantom(grade=2, seed=10, size=128)
    valid_path = tmp_path / "valid.png"
    Image.fromarray(arr).save(valid_path)
    res_ok = validate_image_file(valid_path, label=2)
    assert res_ok["is_corrupt"] is False
    assert res_ok["width"] == 128 and res_ok["height"] == 128 and res_ok["channels"] == 3
    assert len(res_ok["sha256"]) == 64 and len(res_ok["phash"]) == 16

    # Corrupt file
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"not_a_real_jpeg_stream")
    res_bad = validate_image_file(corrupt_path, label=0)
    assert res_bad["is_corrupt"] is True
    assert "Corrupt" in res_bad["validation_error"]

    # Unsupported extension
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("hello")
    res_ext = validate_image_file(txt_path, label=0)
    assert res_ext["is_corrupt"] is True
    assert "Unsupported" in res_ext["validation_error"]

    # Too-small image dimensions
    tiny_path = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(tiny_path)
    res_tiny = validate_image_file(tiny_path, label=0, min_dim=32)
    assert res_tiny["is_corrupt"] is True
    assert "below minimum" in res_tiny["validation_error"]

    # Invalid DR label
    res_lbl = validate_image_file(valid_path, label=9)
    assert res_lbl["is_corrupt"] is True


def test_missing_real_dataset_raises_informative_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="APTOS dataset not found"):
        load_raw_records("aptos", raw_root=tmp_path)


def test_duplicate_detection_and_split_leakage_prevention(tmp_path: Path, synthetic_workspace: dict) -> None:
    manifest_p = synthetic_workspace["manifest_primary"]
    audit = synthetic_workspace["split_audit"]
    assert audit["leakage_free"] is True
    assert audit["patient_id_overlaps"] == 0
    assert audit["duplicate_cluster_overlaps"] == 0
    assert set(manifest_p["original_label"].unique()).issubset(set(DR_CLASS_NAMES.keys()))

    # Explicitly test exact duplicate clustering across copied image
    sample_row = manifest_p.iloc[0].to_dict()
    dup_img = tmp_path / "dup_copy.png"
    shutil.copy(sample_row["abs_path"], dup_img)
    dup_df = pd.DataFrame(
        [
            {"image_id": "orig_1", "dataset_name": "test", "relative_path": "", "abs_path": sample_row["abs_path"], "original_label": 1, "label_name": "Mild", "patient_id": None, "is_synthetic": True},
            {"image_id": "copy_1", "dataset_name": "test", "relative_path": "", "abs_path": str(dup_img), "original_label": 1, "label_name": "Mild", "patient_id": None, "is_synthetic": True},
        ]
    )
    m_dup, summary = build_manifest(dup_df)
    assert summary["duplicate_pairs_detected"] >= 1
    assert m_dup["duplicate_cluster_id"].nunique() == 1
    assert verify_no_split_leakage(m_dup)["leakage_free"] is True


def test_manifest_persistence_and_label_efficiency_subsets(tmp_path: Path, synthetic_workspace: dict) -> None:
    manifest_p = synthetic_workspace["manifest_primary"]
    csv_p, _ = save_manifest(manifest_p, name="pytest_syn", manifests_dir=tmp_path)
    reloaded = load_manifest(csv_p)
    assert len(reloaded) == len(manifest_p)

    subsets = create_label_efficiency_subsets(manifest_p, fractions=(0.25, 0.50, 1.00), seed=42)
    assert len(subsets[0.25]) <= len(subsets[0.50]) <= len(subsets[1.00])
    assert set(subsets[0.25]["image_id"]).issubset(set(subsets[0.50]["image_id"]))


def test_preprocessing_enhancement_and_augmentation_pipelines() -> None:
    img = render_synthetic_fundus_phantom(grade=3, seed=7, size=160)
    stats = inspect_fundus_image(img)
    assert stats["width"] == 160 and 0.3 < stats["fov_coverage"] < 1.0

    fov_mask = extract_fov_mask(img)
    assert fov_mask.dtype == bool and fov_mask.any()
    cropped = crop_to_fov(img)
    assert cropped.shape[0] == cropped.shape[1]

    for preset in ("raw_resize", "fov_crop", "fov_clahe_norm", "ben_graham", "green_channel", "color_norm"):
        out = preprocess_fundus_image(img, preset=preset, image_size=128)
        assert out.shape == (128, 128, 3) and out.dtype == np.uint8

    assert apply_histogram_equalization(img).shape == img.shape
    assert apply_clahe(img).shape == img.shape
    assert apply_gaussian_blur(img).shape == img.shape
    assert apply_median_filter(img).shape == img.shape
    assert extract_green_channel(img).shape == img.shape
    for space in ("RGB", "GRAY", "LAB", "HSV", "GREEN", "GREEN_CLAHE"):
        assert convert_color_space(img, space).shape == img.shape

    for pol in ("none", "light_clinical", "moderate_clinical", "domain_shift_robust"):
        aug_fn = build_augmentation_pipeline(policy=pol, image_size=128)
        aug_out = aug_fn(preprocess_fundus_image(img, preset="fov_crop", image_size=128))
        assert aug_out.shape == (128, 128, 3)

    grid = generate_augmentation_grid(preprocess_fundus_image(img, preset="fov_crop", image_size=128), num_samples=3, image_size=128)
    assert len(grid) == 3


def test_quality_assessment_and_dataset_lazy_loading(synthetic_workspace: dict) -> None:
    sharp_img = render_synthetic_fundus_phantom(grade=1, seed=15, size=160)
    blurry_img = apply_gaussian_blur(sharp_img, kernel_size=11, sigma=4.5)

    qa_sharp = assess_fundus_quality(sharp_img)
    qa_blur = assess_fundus_quality(blurry_img)
    assert qa_sharp["quality_score"] > qa_blur["quality_score"]
    assert qa_sharp["metrics"]["blur_laplacian"] > qa_blur["metrics"]["blur_laplacian"]

    # Proxy classifier sanity check
    feats = [extract_quality_features(sharp_img), extract_quality_features(blurry_img),
             extract_quality_features(sharp_img * 0.3), extract_quality_features(sharp_img)]
    clf = QualityProxyClassifier().fit(feats, [1, 0, 0, 1])
    probs = clf.predict_proba(feats)
    assert probs.shape == (4, 2) and len(clf.feature_weights()) == 9

    # Lazy FundusDataset tensor check
    ds = FundusDataset(synthetic_workspace["manifest_primary"], image_size=128)
    tensor, label, img_id = ds[0]
    assert tensor.shape == (3, 128, 128) and label in {0, 1, 2, 3, 4} and isinstance(img_id, str)
    assert normalize_rgb_tensor(sharp_img, mode="zscore_fov").shape == (3, 160, 160)
    assert hamming_distance_hex(compute_dhash(sharp_img), compute_dhash(sharp_img)) == 0
