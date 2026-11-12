"""Cross-dataset domain shift analysis and out-of-distribution (OOD) detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics
from src.preprocessing.fundus_processing import inspect_fundus_image


def _compute_cohort_image_stats(df: pd.DataFrame, max_samples: int = 80) -> pd.DataFrame:
    valid = df[~df["is_corrupt"].astype(bool)].head(max_samples)
    rows: list[dict[str, float]] = []
    for row in valid.itertuples(index=False):
        stats = inspect_fundus_image(row.abs_path)
        rows.append(
            {
                "brightness": stats["brightness_mean"],
                "contrast": stats["contrast_rms"],
                "blur_laplacian": stats["blur_laplacian"],
                "fov_coverage": stats["fov_coverage"],
                "mean_r": stats["mean_rgb"][0],
                "mean_g": stats["mean_rgb"][1],
                "mean_b": stats["mean_rgb"][2],
            }
        )
    return pd.DataFrame(rows)


def _predictive_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-8, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def compare_cohorts_distribution_shift(
    internal_df: pd.DataFrame,
    external_df: pd.DataFrame,
    internal_labels: np.ndarray,
    internal_probs: np.ndarray,
    external_labels: np.ndarray,
    external_probs: np.ndarray,
    max_image_samples: int = 80,
) -> dict[str, Any]:
    """Quantify classification, calibration, label, and image-level domain shifts."""
    int_cls = compute_classification_metrics(internal_labels, internal_probs)
    ext_cls = compute_classification_metrics(external_labels, external_probs)
    int_cal = compute_calibration_metrics(internal_labels, internal_probs)
    ext_cal = compute_calibration_metrics(external_labels, external_probs)

    # Label distribution shift (Jensen-Shannon divergence)
    int_dist = np.bincount(np.asarray(internal_labels, dtype=np.int64), minlength=5).astype(np.float64)
    ext_dist = np.bincount(np.asarray(external_labels, dtype=np.int64), minlength=5).astype(np.float64)
    int_dist = (int_dist + 1e-6) / (int_dist.sum() + 5e-6)
    ext_dist = (ext_dist + 1e-6) / (ext_dist.sum() + 5e-6)
    js_label_shift = float(jensenshannon(int_dist, ext_dist))

    # Image characteristics shift
    int_img_stats = _compute_cohort_image_stats(internal_df, max_samples=max_image_samples)
    ext_img_stats = _compute_cohort_image_stats(external_df, max_samples=max_image_samples)
    feature_shifts: dict[str, dict[str, float]] = {}
    for col in int_img_stats.columns:
        if len(int_img_stats) >= 2 and len(ext_img_stats) >= 2:
            w_dist = float(wasserstein_distance(int_img_stats[col], ext_img_stats[col]))
            ks_stat, ks_p = ks_2samp(int_img_stats[col], ext_img_stats[col])
            feature_shifts[col] = {
                "internal_mean": round(float(int_img_stats[col].mean()), 3),
                "external_mean": round(float(ext_img_stats[col].mean()), 3),
                "wasserstein_distance": round(w_dist, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(ks_p), 4),
            }

    int_ent = float(np.mean(_predictive_entropy(internal_probs)))
    ext_ent = float(np.mean(_predictive_entropy(external_probs)))
    int_conf = float(np.mean(np.max(internal_probs, axis=1)))
    ext_conf = float(np.mean(np.max(external_probs, axis=1)))

    return {
        "internal_metrics": {
            "accuracy": int_cls["accuracy"],
            "macro_f1": int_cls["macro_f1"],
            "quadratic_weighted_kappa": int_cls["quadratic_weighted_kappa"],
            "ece": int_cal["ece"],
            "brier_score": int_cal["brier_score"],
            "mean_confidence": round(int_conf, 4),
            "mean_entropy": round(int_ent, 4),
        },
        "external_metrics": {
            "accuracy": ext_cls["accuracy"],
            "macro_f1": ext_cls["macro_f1"],
            "quadratic_weighted_kappa": ext_cls["quadratic_weighted_kappa"],
            "ece": ext_cal["ece"],
            "brier_score": ext_cal["brier_score"],
            "mean_confidence": round(ext_conf, 4),
            "mean_entropy": round(ext_ent, 4),
        },
        "performance_delta_external_minus_internal": {
            "accuracy": round(ext_cls["accuracy"] - int_cls["accuracy"], 4),
            "macro_f1": round(ext_cls["macro_f1"] - int_cls["macro_f1"], 4),
            "quadratic_weighted_kappa": round(
                ext_cls["quadratic_weighted_kappa"] - int_cls["quadratic_weighted_kappa"], 4
            ),
            "ece": round(ext_cal["ece"] - int_cal["ece"], 4),
            "mean_entropy": round(ext_ent - int_ent, 4),
        },
        "label_distribution_js_divergence": round(js_label_shift, 4),
        "internal_class_proportions": np.round(int_dist, 4).tolist(),
        "external_class_proportions": np.round(ext_dist, 4).tolist(),
        "image_characteristic_shifts": feature_shifts,
        "internal_feature_table": int_img_stats,
        "external_feature_table": ext_img_stats,
    }


def fit_feature_ood_detector(train_embeddings: np.ndarray, reg_eps: float = 1e-3) -> dict[str, Any]:
    """Fit global & L2-normalized centroid + precision matrix for OOD scoring."""
    embs = np.asarray(train_embeddings, dtype=np.float64)
    if embs.ndim != 2 or len(embs) < 2:
        raise ValueError("Need at least 2 training embeddings to fit feature OOD detector.")

    mean = np.mean(embs, axis=0)
    centered = embs - mean
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov_reg = cov + reg_eps * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov_reg)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / np.clip(norms, 1e-8, None)

    return {
        "mean": mean,
        "inv_cov": inv_cov,
        "train_norm_embs": embs_norm,
    }


def score_feature_ood(detector: dict[str, Any], query_embeddings: np.ndarray) -> dict[str, np.ndarray]:
    """Compute Mahalanobis distance and 1-NN cosine distance for query embeddings."""
    q = np.asarray(query_embeddings, dtype=np.float64)
    diff = q - detector["mean"][None, :]
    maha_sq = np.einsum("nd,dd,nd->n", diff, detector["inv_cov"], diff)
    maha = np.sqrt(np.clip(maha_sq, 0.0, None))

    q_norm = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    cos_sim = q_norm @ detector["train_norm_embs"].T
    max_cos_sim = np.max(cos_sim, axis=1)
    cos_dist = 1.0 - max_cos_sim

    return {"mahalanobis_distance": maha, "cosine_knn_distance": cos_dist}


def _auroc_and_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    """Evaluate binary OOD detection where higher score indicates OOD."""
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    if len(np.unique(y)) < 2:
        return {"auroc": 0.5, "fpr_at_95_tpr": 1.0}
    auroc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    idx = int(np.searchsorted(tpr, 0.95, side="left"))
    idx = min(idx, len(fpr) - 1)
    fpr95 = float(fpr[idx])
    return {"auroc": round(auroc, 4), "fpr_at_95_tpr": round(fpr95, 4)}


def evaluate_ood_detection(
    id_probs: np.ndarray,
    ood_probs: np.ndarray,
    id_embeddings: np.ndarray | None = None,
    ood_embeddings: np.ndarray | None = None,
    train_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Benchmark multiple post-hoc OOD detectors (MSP, Entropy, Mahalanobis, Cosine k-NN)."""
    id_p = np.asarray(id_probs, dtype=np.float64)
    ood_p = np.asarray(ood_probs, dtype=np.float64)

    msp_id = 1.0 - np.max(id_p, axis=1)
    msp_ood = 1.0 - np.max(ood_p, axis=1)

    ent_id = _predictive_entropy(id_p)
    ent_ood = _predictive_entropy(ood_p)

    results: dict[str, Any] = {
        "max_softmax_inv": _auroc_and_fpr95(msp_id, msp_ood),
        "predictive_entropy": _auroc_and_fpr95(ent_id, ent_ood),
        "limitation_note": (
            "No post-hoc OOD detector achieves perfect separation in medical imaging; "
            "high-confidence failures under subtle acquisition shift remain an open safety risk."
        ),
    }

    if train_embeddings is not None and id_embeddings is not None and ood_embeddings is not None:
        detector = fit_feature_ood_detector(train_embeddings)
        id_feat_scores = score_feature_ood(detector, id_embeddings)
        ood_feat_scores = score_feature_ood(detector, ood_embeddings)
        results["mahalanobis_distance"] = _auroc_and_fpr95(
            id_feat_scores["mahalanobis_distance"],
            ood_feat_scores["mahalanobis_distance"],
        )
        results["cosine_knn_distance"] = _auroc_and_fpr95(
            id_feat_scores["cosine_knn_distance"],
            ood_feat_scores["cosine_knn_distance"],
        )

    return results
