"""Fundus image processing, classical enhancement, clinical augmentation, and quality assessment."""

from src.preprocessing.augmentation import build_augmentation_pipeline, generate_augmentation_grid
from src.preprocessing.enhancement import (
    apply_clahe,
    apply_gaussian_blur,
    apply_histogram_equalization,
    apply_median_filter,
    apply_unsharp_mask,
    contrast_stretch,
    convert_color_space,
    extract_green_channel,
)
from src.preprocessing.fundus_processing import (
    apply_circular_fov_mask,
    ben_graham_preprocess,
    crop_to_fov,
    extract_fov_mask,
    gray_world_color_norm,
    inspect_fundus_image,
    preprocess_fundus_image,
)
from src.preprocessing.quality_assessment import (
    QUALITY_FEATURE_NAMES,
    QualityProxyClassifier,
    assess_fundus_quality,
    extract_quality_features,
)

__all__ = [
    "QUALITY_FEATURE_NAMES",
    "QualityProxyClassifier",
    "apply_circular_fov_mask",
    "apply_clahe",
    "apply_gaussian_blur",
    "apply_histogram_equalization",
    "apply_median_filter",
    "apply_unsharp_mask",
    "assess_fundus_quality",
    "ben_graham_preprocess",
    "build_augmentation_pipeline",
    "contrast_stretch",
    "convert_color_space",
    "crop_to_fov",
    "extract_fov_mask",
    "extract_green_channel",
    "extract_quality_features",
    "generate_augmentation_grid",
    "gray_world_color_norm",
    "inspect_fundus_image",
    "preprocess_fundus_image",
]
