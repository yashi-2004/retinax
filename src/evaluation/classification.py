"""Comprehensive medical image classification metrics including ordinal and per-class evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.config import DR_CLASS_NAMES


def compute_qwk(y_true: np.ndarray | list[int], y_pred: np.ndarray | list[int], num_classes: int = 5) -> float:
    """Compute Quadratic Weighted Kappa safely even on single-class subsets."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if len(yt) == 0:
        return 0.0
    if len(np.unique(yt)) < 2 and len(np.unique(yp)) < 2:
        return 1.0 if np.array_equal(yt, yp) else 0.0
    val = cohen_kappa_score(yt, yp, labels=list(range(num_classes)), weights="quadratic")
    return 0.0 if np.isnan(val) else float(val)


def compute_ordinal_error_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Quantify exact match vs 1-grade vs >=2-grade severe severity errors."""
    if len(y_true) == 0:
        return {
            "exact_match_rate": 0.0,
            "one_grade_error_rate": 0.0,
            "severe_multigrade_error_rate": 0.0,
            "mean_absolute_grade_error": 0.0,
        }
    diffs = np.abs(np.asarray(y_pred, dtype=np.int64) - np.asarray(y_true, dtype=np.int64))
    return {
        "exact_match_rate": round(float(np.mean(diffs == 0)), 4),
        "one_grade_error_rate": round(float(np.mean(diffs == 1)), 4),
        "severe_multigrade_error_rate": round(float(np.mean(diffs >= 2)), 4),
        "mean_absolute_grade_error": round(float(np.mean(diffs)), 4),
    }


def compute_classification_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray,
    num_classes: int = 5,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Calculate comprehensive multi-class and ordinal medical classification metrics."""
    yt = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != len(yt):
        raise ValueError(f"Expected y_prob shape ({len(yt)}, C), got {probs.shape}")

    yp = np.argmax(probs, axis=1)
    names = class_names or DR_CLASS_NAMES
    labels = list(range(num_classes))

    acc = float(accuracy_score(yt, yp)) if len(yt) else 0.0
    bal_acc = float(balanced_accuracy_score(yt, yp)) if len(np.unique(yt)) >= 2 else acc
    kappa = (
        float(cohen_kappa_score(yt, yp, labels=labels))
        if (len(np.unique(yt)) >= 2 or len(np.unique(yp)) >= 2)
        else (1.0 if np.array_equal(yt, yp) else 0.0)
    )
    if np.isnan(kappa):
        kappa = 0.0
    qwk = compute_qwk(yt, yp, num_classes=num_classes)

    cm = confusion_matrix(yt, yp, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(np.float64), row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0)

    prec, rec, f1, support = precision_recall_fscore_support(
        yt, yp, labels=labels, average=None, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, average="weighted", zero_division=0
    )

    per_class: dict[str, dict[str, Any]] = {}
    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    total_samples = int(cm.sum())

    for c in labels:
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        fp = float(cm[:, c].sum() - tp)
        tn = float(total_samples - tp - fn - fp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None

        y_bin = (yt == c).astype(np.int64)
        c_probs = probs[:, c] if c < probs.shape[1] else np.zeros_like(yt, dtype=np.float64)
        c_roc_auc = None
        c_pr_auc = None
        if len(np.unique(y_bin)) == 2:
            c_roc_auc = float(roc_auc_score(y_bin, c_probs))
            c_pr_auc = float(average_precision_score(y_bin, c_probs))
            roc_aucs.append(c_roc_auc)
            pr_aucs.append(c_pr_auc)

        per_class[str(c)] = {
            "class_name": names.get(c, f"Class {c}"),
            "support": int(support[c]),
            "precision": round(float(prec[c]), 4),
            "recall_sensitivity": round(float(rec[c]), 4),
            "specificity": round(float(specificity), 4) if specificity is not None else None,
            "f1": round(float(f1[c]), 4),
            "roc_auc": round(c_roc_auc, 4) if c_roc_auc is not None else None,
            "pr_auc": round(c_pr_auc, 4) if c_pr_auc is not None else None,
        }

    macro_roc_auc = float(np.mean(roc_aucs)) if roc_aucs else None
    macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else None

    return {
        "n_samples": int(len(yt)),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_prec), 4),
        "weighted_recall": round(float(weighted_rec), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "cohens_kappa": round(kappa, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "macro_roc_auc_ovr": round(macro_roc_auc, 4) if macro_roc_auc is not None else None,
        "macro_pr_auc_ovr": round(macro_pr_auc, 4) if macro_pr_auc is not None else None,
        "ordinal_errors": compute_ordinal_error_breakdown(yt, yp),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
    }


def compute_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 5,
    class_names: dict[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute one-vs-rest ROC and Precision-Recall curve points per class."""
    yt = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    names = class_names or DR_CLASS_NAMES
    roc_data: dict[str, Any] = {}
    pr_data: dict[str, Any] = {}

    for c in range(num_classes):
        y_bin = (yt == c).astype(np.int64)
        if len(np.unique(y_bin)) < 2 or c >= probs.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        prec, rec, _ = precision_recall_curve(y_bin, probs[:, c])
        roc_data[names.get(c, str(c))] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(roc_auc_score(y_bin, probs[:, c])), 4),
        }
        pr_data[names.get(c, str(c))] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "ap": round(float(average_precision_score(y_bin, probs[:, c])), 4),
        }

    return {"roc": roc_data, "pr": pr_data}
