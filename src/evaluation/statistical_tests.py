"""Statistical confidence intervals and paired hypothesis tests for model comparisons."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.stats import binomtest, chi2


def paired_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute non-parametric bootstrap confidence interval for a classification metric."""
    yt = np.asarray(y_true)
    probs = np.asarray(y_prob)
    point = float(metric_fn(yt, probs))
    if len(yt) < 2:
        return {"estimate": round(point, 4), "ci_low": round(point, 4), "ci_high": round(point, 4), "std": 0.0}

    rng = np.random.default_rng(seed)
    n = len(yt)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(float(metric_fn(yt[idx], probs[idx])))

    alpha = 1.0 - confidence
    lo = float(np.percentile(scores, 100.0 * alpha / 2.0))
    hi = float(np.percentile(scores, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "estimate": round(point, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "std": round(float(np.std(scores)), 4),
    }


def paired_bootstrap_comparison(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap comparison (Model B minus Model A) on identical test samples."""
    yt = np.asarray(y_true)
    pa = np.asarray(y_prob_a)
    pb = np.asarray(y_prob_b)
    diff_obs = float(metric_fn(yt, pb) - metric_fn(yt, pa))
    if len(yt) < 2:
        return {"diff_b_minus_a": round(diff_obs, 4), "ci_low": round(diff_obs, 4), "ci_high": round(diff_obs, 4), "p_value": 1.0}

    rng = np.random.default_rng(seed)
    n = len(yt)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(metric_fn(yt[idx], pb[idx]) - metric_fn(yt[idx], pa[idx])))

    arr = np.asarray(diffs, dtype=np.float64)
    p_val = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
    p_val = min(max(p_val, 0.0), 1.0)
    return {
        "diff_b_minus_a": round(diff_obs, 4),
        "ci_low": round(float(np.percentile(arr, 2.5)), 4),
        "ci_high": round(float(np.percentile(arr, 97.5)), 4),
        "p_value": round(p_val, 4),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, Any]:
    """McNemar's test comparing paired classifier accuracy on identical samples."""
    yt = np.asarray(y_true)
    corr_a = np.asarray(y_pred_a) == yt
    corr_b = np.asarray(y_pred_b) == yt

    n_11 = int(np.sum(corr_a & corr_b))
    n_10 = int(np.sum(corr_a & (~corr_b)))
    n_01 = int(np.sum((~corr_a) & corr_b))
    n_00 = int(np.sum((~corr_a) & (~corr_b)))

    discordant = n_10 + n_01
    if discordant == 0:
        return {
            "n_both_correct": n_11,
            "n_a_only_correct": n_10,
            "n_b_only_correct": n_01,
            "n_both_wrong": n_00,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "exact_degenerate",
        }

    if discordant < 25:
        res = binomtest(min(n_10, n_01), n=discordant, p=0.5, alternative="two-sided")
        return {
            "n_both_correct": n_11,
            "n_a_only_correct": n_10,
            "n_b_only_correct": n_01,
            "n_both_wrong": n_00,
            "statistic": round(float(min(n_10, n_01)), 4),
            "p_value": round(float(res.pvalue), 4),
            "method": "exact_binomial",
        }

    stat = ((abs(n_10 - n_01) - 1.0) ** 2) / float(discordant)
    p_val = float(chi2.sf(stat, df=1))
    return {
        "n_both_correct": n_11,
        "n_a_only_correct": n_10,
        "n_b_only_correct": n_01,
        "n_both_wrong": n_00,
        "statistic": round(float(stat), 4),
        "p_value": round(p_val, 4),
        "method": "continuity_corrected_chi2",
    }
