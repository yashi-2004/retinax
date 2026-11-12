"""Selective prediction, risk-coverage trade-off analysis, and abstention decision logic."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.classification import compute_qwk


def evaluate_selective_decision(
    probabilities: np.ndarray,
    predictive_entropy: float | None = None,
    conformal_set_size: int | None = None,
    quality_score: float | None = None,
    min_confidence: float = 0.45,
    max_entropy: float = 1.25,
    max_conformal_set_size: int = 3,
    min_quality_score: float = 30.0,
) -> dict[str, Any]:
    """Decide whether to emit a severity prediction or abstain for specialist review."""
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs / np.maximum(probs.sum(), 1e-8)
    confidence = float(np.max(probs))
    pred_cls = int(np.argmax(probs))

    if predictive_entropy is None:
        p_clip = np.clip(probs, 1e-8, 1.0)
        predictive_entropy = float(-np.sum(p_clip * np.log(p_clip)))

    reasons: list[str] = []
    if confidence < min_confidence:
        reasons.append(f"Low confidence ({confidence:.2f} < {min_confidence:.2f})")
    if predictive_entropy > max_entropy:
        reasons.append(f"High predictive entropy ({predictive_entropy:.2f} > {max_entropy:.2f})")
    if conformal_set_size is not None and conformal_set_size > max_conformal_set_size:
        reasons.append(f"Ambiguous conformal set (size {conformal_set_size} > {max_conformal_set_size})")
    if quality_score is not None and quality_score < min_quality_score:
        reasons.append(f"Poor image quality score ({quality_score:.1f} < {min_quality_score:.1f})")

    abstain = len(reasons) > 0
    return {
        "abstain": abstain,
        "status": "ABSTAIN — Refer for Ophthalmic Expert Review" if abstain else "ACCEPTED (Research Output)",
        "predicted_class": pred_cls,
        "confidence": round(confidence, 4),
        "predictive_entropy": round(float(predictive_entropy), 4),
        "reasons": reasons if reasons else ["Confidence, entropy, and quality gates satisfied."],
    }


def compute_risk_coverage_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty_scores: np.ndarray | None = None,
    coverages: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute Risk-Coverage curve (selective error vs coverage) and AURC.

    Args:
        uncertainty_scores: Higher score means MORE uncertain (e.g. entropy or 1 - confidence).
            Defaults to `1.0 - max_softmax_prob`.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    yp = np.argmax(probs, axis=1)
    errors = (yp != yt).astype(np.float64)

    if uncertainty_scores is None:
        unc = 1.0 - np.max(probs, axis=1)
    else:
        unc = np.asarray(uncertainty_scores, dtype=np.float64)

    # Sort from MOST confident (lowest uncertainty) to LEAST confident
    order = np.argsort(unc, kind="mergesort")
    sorted_errors = errors[order]
    sorted_yt = yt[order]
    sorted_yp = yp[order]
    n = len(yt)

    if n == 0:
        return {"aurc": 0.0, "points": []}

    cov_grid = coverages if coverages is not None else np.linspace(0.2, 1.0, 9)
    points: list[dict[str, float]] = []
    risks: list[float] = []
    covs: list[float] = []

    for cov in cov_grid:
        k = max(1, min(n, int(round(float(cov) * n))))
        sel_risk = float(np.mean(sorted_errors[:k]))
        sel_acc = 1.0 - sel_risk
        sel_qwk = compute_qwk(sorted_yt[:k], sorted_yp[:k])
        actual_cov = k / n
        covs.append(actual_cov)
        risks.append(sel_risk)
        points.append(
            {
                "coverage": round(actual_cov, 4),
                "n_retained": int(k),
                "selective_risk": round(sel_risk, 4),
                "selective_accuracy": round(sel_acc, 4),
                "selective_qwk": round(sel_qwk, 4),
            }
        )

    aurc = float(np.trapz(risks, covs) / max(covs[-1] - covs[0], 1e-8)) if len(covs) > 1 else float(risks[0])
    return {
        "aurc": round(aurc, 4),
        "full_coverage_risk": round(float(np.mean(errors)), 4),
        "points": points,
    }
