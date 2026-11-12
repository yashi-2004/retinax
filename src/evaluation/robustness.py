"""Perturbation robustness stress tests and image-quality-stratified evaluation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.caching import load_rgb_image, normalize_rgb_tensor
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics
from src.preprocessing.fundus_processing import preprocess_fundus_image


def apply_controlled_perturbation(
    image_rgb: np.ndarray,
    perturbation: str,
    severity: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Apply deterministic degradation to study model sensitivity under stress."""
    sev = float(np.clip(severity, 0.0, 1.0))
    if sev == 0.0:
        return image_rgb.copy()

    img = image_rgb.astype(np.float32)
    kind = perturbation.lower()
    rng = np.random.default_rng(seed)

    if kind == "gaussian_blur":
        k = int(round(3 + sev * 10)) | 1
        return cv2.GaussianBlur(image_rgb, (k, k), sigmaX=0.5 + sev * 3.0)
    if kind == "underexposure":
        scale = 1.0 - 0.65 * sev
        return np.clip(img * scale, 0, 255).astype(np.uint8)
    if kind == "overexposure":
        shift = 110.0 * sev
        return np.clip(img * (1.0 + 0.35 * sev) + shift, 0, 255).astype(np.uint8)
    if kind == "low_contrast":
        mean_val = float(img.mean())
        alpha = 1.0 - 0.72 * sev
        return np.clip((img - mean_val) * alpha + mean_val, 0, 255).astype(np.uint8)
    if kind == "gaussian_noise":
        noise = rng.normal(0.0, 42.0 * sev, size=img.shape).astype(np.float32)
        return np.clip(img + noise, 0, 255).astype(np.uint8)
    if kind == "partial_occlusion":
        out = image_rgb.copy()
        h, w = out.shape[:2]
        rad = int(max(4, min(h, w) * (0.12 + 0.25 * sev)))
        cy, cx = int(h * 0.32), int(w * 0.32)
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (rad**2)
        out[mask] = 0
        return out
    raise ValueError(f"Unsupported perturbation '{perturbation}'.")


def evaluate_perturbation_robustness(
    model: nn.Module,
    manifest_df: pd.DataFrame,
    perturbation: str = "gaussian_blur",
    severities: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0),
    preprocess_preset: str = "fov_clahe_norm",
    image_size: int = 224,
    normalisation: str = "imagenet",
    max_samples: int = 60,
) -> list[dict[str, Any]]:
    """Evaluate classification & calibration degradation across controlled perturbation severities."""
    model.eval()
    sub = manifest_df[~manifest_df["is_corrupt"].astype(bool)].head(max_samples).reset_index(drop=True)
    if len(sub) == 0:
        return []

    y_true = sub["original_label"].astype(int).to_numpy()
    raw_images = [load_rgb_image(p) for p in sub["abs_path"]]
    results: list[dict[str, Any]] = []

    for sev in severities:
        tensors = []
        for idx, img in enumerate(raw_images):
            pert = apply_controlled_perturbation(img, perturbation=perturbation, severity=sev, seed=42 + idx)
            proc = preprocess_fundus_image(pert, preset=preprocess_preset, image_size=image_size)
            tensors.append(normalize_rgb_tensor(proc, mode=normalisation))
        batch_t = torch.stack(tensors, dim=0)
        with torch.inference_mode():
            logits = model(batch_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        cls_m = compute_classification_metrics(y_true, probs)
        cal_m = compute_calibration_metrics(y_true, probs)
        results.append(
            {
                "perturbation": perturbation,
                "severity": float(sev),
                "n_samples": int(len(y_true)),
                "accuracy": cls_m["accuracy"],
                "macro_f1": cls_m["macro_f1"],
                "quadratic_weighted_kappa": cls_m["quadratic_weighted_kappa"],
                "ece": cal_m["ece"],
                "mean_confidence": round(float(np.max(probs, axis=1).mean()), 4),
            }
        )
    return results


def evaluate_quality_stratified(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray,
    quality_categories: list[str] | np.ndarray,
) -> dict[str, Any]:
    """Compare model metrics across image quality strata and quantify filtering impact."""
    yt = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    cats = np.asarray([str(c) for c in quality_categories])

    overall_cls = compute_classification_metrics(yt, probs)
    overall_cal = compute_calibration_metrics(yt, probs)

    strata: dict[str, Any] = {}
    for cat in ["Good", "Suboptimal / Usable", "Poor / Review Recommended"]:
        mask = cats == cat
        count = int(mask.sum())
        if count == 0:
            continue
        cls_m = compute_classification_metrics(yt[mask], probs[mask])
        cal_m = compute_calibration_metrics(yt[mask], probs[mask])
        class_dist = {str(k): int(v) for k, v in zip(*np.unique(yt[mask], return_counts=True))}
        strata[cat] = {
            "count": count,
            "fraction_of_dataset": round(count / max(len(yt), 1), 4),
            "class_distribution": class_dist,
            "accuracy": cls_m["accuracy"],
            "macro_f1": cls_m["macro_f1"],
            "quadratic_weighted_kappa": cls_m["quadratic_weighted_kappa"],
            "ece": cal_m["ece"],
        }

    # Quantify impact of excluding Poor images
    non_poor = cats != "Poor / Review Recommended"
    filtered_summary = None
    if 0 < int(non_poor.sum()) < len(yt):
        f_cls = compute_classification_metrics(yt[non_poor], probs[non_poor])
        f_cal = compute_calibration_metrics(yt[non_poor], probs[non_poor])
        filtered_summary = {
            "retained_fraction": round(float(non_poor.mean()), 4),
            "excluded_count": int((~non_poor).sum()),
            "accuracy_after_filtering": f_cls["accuracy"],
            "qwk_after_filtering": f_cls["quadratic_weighted_kappa"],
            "ece_after_filtering": f_cal["ece"],
        }

    return {
        "overall": {
            "count": int(len(yt)),
            "accuracy": overall_cls["accuracy"],
            "quadratic_weighted_kappa": overall_cls["quadratic_weighted_kappa"],
            "ece": overall_cal["ece"],
        },
        "by_quality_category": strata,
        "filtering_impact": filtered_summary,
    }
