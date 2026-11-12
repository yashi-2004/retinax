"""Fundus image quality assessment using classical metrics and an optional proxy classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing.fundus_processing import extract_fov_mask, load_rgb_array

QUALITY_FEATURE_NAMES: list[str] = [
    "blur_laplacian",
    "blur_tenengrad",
    "brightness_mean",
    "clipped_dark_frac",
    "clipped_bright_frac",
    "contrast_rms",
    "fov_coverage",
    "illumination_imbalance",
    "artifact_edge_ratio",
]


def extract_quality_features(image_source: str | Path | Image.Image | np.ndarray) -> dict[str, float]:
    """Extract classical statistical quality indicators inside the retinal field of view."""
    img = load_rgb_array(image_source)
    # Standardize analysis resolution for scale-consistent Laplacian & Tenengrad metrics
    img_std = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_std, cv2.COLOR_RGB2GRAY)
    fov = extract_fov_mask(img_std, threshold=10)
    fg = gray[fov] if fov.any() else gray.ravel()

    # 1. Focus / Sharpness inside FOV
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_laplacian = float(np.var(lap[fov])) if fov.any() else float(np.var(lap))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = gx**2 + gy**2
    blur_tenengrad = float(np.mean(grad_mag_sq[fov])) if fov.any() else float(np.mean(grad_mag_sq))

    # 2. Exposure & Contrast inside FOV
    brightness_mean = float(np.mean(fg))
    contrast_rms = float(np.std(fg))
    clipped_dark_frac = float(np.mean(fg < 18))
    clipped_bright_frac = float(np.mean(fg > 240))

    # 3. Field-of-view coverage
    fov_coverage = float(np.mean(fov))

    # 4. Quadrant illumination imbalance / occlusion proxy
    h, w = gray.shape
    quads = [
        gray[: h // 2, : w // 2][fov[: h // 2, : w // 2]],
        gray[: h // 2, w // 2 :][fov[: h // 2, w // 2 :]],
        gray[h // 2 :, : w // 2][fov[h // 2 :, : w // 2]],
        gray[h // 2 :, w // 2 :][fov[h // 2 :, w // 2 :]],
    ]
    quad_means = [float(np.mean(q)) for q in quads if len(q) > 25]
    illumination_imbalance = float(np.std(quad_means)) if len(quad_means) >= 2 else 0.0

    # 5. Sharp sensor / glare / occlusion edge density near extremes
    extreme_mask = ((gray < 8) | (gray > 248)) & fov
    artifact_edge_ratio = float(np.mean(extreme_mask))

    return {
        "blur_laplacian": round(blur_laplacian, 3),
        "blur_tenengrad": round(blur_tenengrad, 3),
        "brightness_mean": round(brightness_mean, 3),
        "clipped_dark_frac": round(clipped_dark_frac, 4),
        "clipped_bright_frac": round(clipped_bright_frac, 4),
        "contrast_rms": round(contrast_rms, 3),
        "fov_coverage": round(fov_coverage, 4),
        "illumination_imbalance": round(illumination_imbalance, 3),
        "artifact_edge_ratio": round(artifact_edge_ratio, 4),
    }


def assess_fundus_quality(
    image_source: str | Path | Image.Image | np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute composite image quality score [0, 100], category, detected issues, and actions.

    Note: Classical image statistics are technical proxies and do NOT substitute for
    ophthalmic clinical gradability standards (e.g., macular visibility or optic disc focus).
    """
    th = {
        "blur_laplacian_threshold": 45.0,
        "min_brightness": 32.0,
        "max_brightness": 215.0,
        "min_contrast_rms": 18.0,
        "min_fov_coverage": 0.35,
        "max_illumination_imbalance": 35.0,
    }
    if thresholds:
        th.update(thresholds)

    feats = extract_quality_features(image_source)
    issues: list[str] = []
    recommendations: list[str] = []
    penalty = 0.0

    if feats["blur_laplacian"] < th["blur_laplacian_threshold"]:
        ratio = max(0.0, 1.0 - feats["blur_laplacian"] / max(th["blur_laplacian_threshold"], 1.0))
        penalty += 32.0 * ratio + 8.0
        issues.append(f"Blur / defocus (Laplacian var={feats['blur_laplacian']:.1f})")
        recommendations.append("Flag prediction uncertainty; avoid relying on fine microaneurysm grading.")

    if feats["brightness_mean"] < th["min_brightness"] or feats["clipped_dark_frac"] > 0.30:
        penalty += 24.0
        issues.append(f"Underexposed / dark FOV (mean={feats['brightness_mean']:.1f})")
        recommendations.append("Evaluate gentle CLAHE or illumination normalization carefully.")

    if feats["brightness_mean"] > th["max_brightness"] or feats["clipped_bright_frac"] > 0.12:
        penalty += 24.0
        issues.append(f"Overexposed / glare clipping (bright_frac={feats['clipped_bright_frac']:.2f})")
        recommendations.append("Inspect for camera flash artifact or lens haze; consider abstention.")

    if feats["contrast_rms"] < th["min_contrast_rms"]:
        penalty += 18.0
        issues.append(f"Low retinal contrast (RMS={feats['contrast_rms']:.1f})")
        recommendations.append("Compare raw vs LAB CLAHE (clip=2.0) in Preprocessing Lab.")

    if feats["fov_coverage"] < th["min_fov_coverage"]:
        penalty += 22.0
        issues.append(f"Poor or truncated FOV (coverage={feats['fov_coverage']:.2f})")
        recommendations.append("Apply tight FOV cropping or review for severe framing occlusion.")

    if feats["illumination_imbalance"] > th["max_illumination_imbalance"]:
        penalty += 14.0
        issues.append(f"Uneven illumination / partial occlusion (std={feats['illumination_imbalance']:.1f})")
        recommendations.append("Test Ben Graham illumination subtraction and inspect occlusion sensitivity.")

    if feats["artifact_edge_ratio"] > 0.08:
        penalty += 10.0
        issues.append(f"Extreme intensity artifact pixels ({feats['artifact_edge_ratio']*100:.1f}%)")
        recommendations.append("Inspect image visually for eyelashes, dust, or sensor clipping.")

    score = float(np.clip(100.0 - penalty, 0.0, 100.0))
    if score >= 70.0:
        category = "Good"
        if not recommendations:
            recommendations.append("Nominal quality for downstream baseline & uncertainty evaluation.")
    elif score >= 40.0:
        category = "Suboptimal / Usable"
    else:
        category = "Poor / Review Recommended"

    return {
        "quality_score": round(score, 2),
        "quality_category": category,
        "issues": issues,
        "recommendations": recommendations,
        "metrics": feats,
        "limitation_note": (
            "Classical statistical quality metrics reflect sharpness, exposure, and contrast proxies; "
            "they do not guarantee clinical gradability of the macula or optic nerve head."
        ),
    }


class QualityProxyClassifier:
    """Lightweight interpretable logistic regression classifier over classical quality features."""

    def __init__(self, random_state: int = 42) -> None:
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state)),
            ]
        )
        self.is_fitted = False

    def fit(self, feature_dicts: list[dict[str, float]], labels: list[int]) -> "QualityProxyClassifier":
        """Fit binary or ordinal quality classifier (e.g., 1=Acceptable, 0=Low Quality)."""
        X = np.array([[d[k] for k in QUALITY_FEATURE_NAMES] for d in feature_dicts], dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, feature_dicts: list[dict[str, float]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("QualityProxyClassifier must be fitted before calling predict_proba.")
        X = np.array([[d[k] for k in QUALITY_FEATURE_NAMES] for d in feature_dicts], dtype=np.float32)
        return self.pipeline.predict_proba(X)

    def feature_weights(self) -> dict[str, float]:
        if not self.is_fitted:
            return {}
        coefs = self.pipeline.named_steps["clf"].coef_[0]
        return {name: round(float(w), 4) for name, w in zip(QUALITY_FEATURE_NAMES, coefs)}
