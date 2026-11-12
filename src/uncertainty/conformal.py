"""Split Conformal Prediction (LAC and APS) for distribution-free uncertainty sets."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import DR_CLASS_NAMES


class SplitConformalPredictor:
    r"""Split Conformal Classification producing valid prediction sets under exchangeability.

    Supports:
      - ``lac``: Least Ambiguous Set-Valued Classifier, conformity score $1 - \hat{p}_y(x)$.
      - ``aps``: Adaptive Prediction Sets (Romano et al., 2020), cumulative sorted probability mass.

    Assumptions & Limitations:
      Marginal coverage guarantee $P(Y \in C(X)) \ge 1 - \alpha$ assumes calibration and test
      samples are exchangeable. Under external dataset shift or non-random image quality drop,
      empirical coverage may fall below the nominal $1 - \alpha$ target.
    """

    def __init__(self, alpha: float = 0.10, method: str = "aps", seed: int = 42) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        method = method.lower()
        if method not in {"lac", "aps"}:
            raise ValueError(f"Unsupported conformal method '{method}'. Choose 'lac' or 'aps'.")

        self.alpha = float(alpha)
        self.method = method
        self.seed = int(seed)
        self.q_hat: float | None = None
        self.n_cal: int = 0

    def _compute_aps_scores(self, probs: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        """Compute APS cumulative score matrix [N, C] or ground-truth scores [N]."""
        p = np.asarray(probs, dtype=np.float64)
        p = np.clip(p, 0.0, 1.0)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)

        order = np.argsort(-p, axis=1)
        sorted_p = np.take_along_axis(p, order, axis=1)
        cum_p = np.cumsum(sorted_p, axis=1)

        # Map cumulative scores back to original class index [N, C]
        score_mat = np.zeros_like(p)
        rows = np.arange(len(p))[:, None]
        score_mat[rows, order] = cum_p

        if labels is None:
            return score_mat
        y = np.asarray(labels, dtype=np.int64)
        return score_mat[np.arange(len(y)), y]

    def fit(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> dict[str, Any]:
        """Calibrate conformal threshold q_hat on held-out calibration split (`cal`)."""
        probs = np.asarray(cal_probs, dtype=np.float64)
        labels = np.asarray(cal_labels, dtype=np.int64)
        n = len(labels)
        if n < 1:
            raise ValueError("Need at least 1 calibration sample for split conformal prediction.")

        if self.method == "lac":
            scores = 1.0 - probs[np.arange(n), labels]
        else:
            scores = self._compute_aps_scores(probs, labels)

        # Finite-sample conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        self.n_cal = n

        min_recommended = int(np.ceil(1.0 / max(self.alpha, 1e-3)))
        out = {
            "method": self.method,
            "alpha": self.alpha,
            "target_coverage": round(1.0 - self.alpha, 4),
            "n_calibration": n,
            "q_hat": round(self.q_hat, 5),
        }
        if n < min_recommended:
            out["warning"] = (
                f"Small calibration split (n={n} < {min_recommended}): conformal quantile saturates at max score."
            )
        return out

    def predict_sets(self, probs: np.ndarray) -> list[list[int]]:
        """Return conformal prediction set C(x) <= {0..C-1} for each sample."""
        if self.q_hat is None:
            raise RuntimeError("SplitConformalPredictor must be fitted before calling predict_sets.")

        p = np.asarray(probs, dtype=np.float64)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
        n, c = p.shape
        sets: list[list[int]] = []

        if self.method == "lac":
            threshold = 1.0 - self.q_hat
            for i in range(n):
                included = [cls for cls in range(c) if p[i, cls] >= threshold]
                if not included:
                    # Fallback singleton if numerical threshold leaves empty set (or flag empty if strictly desired)
                    included = [int(np.argmax(p[i]))]
                sets.append(included)
        else:
            # APS inclusion: include top-sorted classes until cumulative mass exceeds q_hat
            order = np.argsort(-p, axis=1)
            sorted_p = np.take_along_axis(p, order, axis=1)
            cum_p = np.cumsum(sorted_p, axis=1)
            for i in range(n):
                included: list[int] = []
                for rank in range(c):
                    cls_id = int(order[i, rank])
                    included.append(cls_id)
                    if cum_p[i, rank] >= self.q_hat:
                        break
                sets.append(sorted(included))
        return sets

    def predict_single_set(self, probs_1d: np.ndarray) -> dict[str, Any]:
        """Return formatted prediction set for a single sample."""
        pred_set = self.predict_sets(np.asarray(probs_1d, dtype=np.float64).reshape(1, -1))[0]
        return {
            "prediction_set_indices": pred_set,
            "prediction_set_labels": [DR_CLASS_NAMES.get(c, str(c)) for c in pred_set],
            "set_size": len(pred_set),
            "target_coverage": round(1.0 - self.alpha, 4),
            "method": self.method,
            "q_hat": round(float(self.q_hat or 0.0), 5),
            "assumption_note": (
                "Conformal coverage holds under exchangeability with the calibration cohort; "
                "domain shift or image degradation can invalidate nominal guarantees."
            ),
        }

    def evaluate(self, test_probs: np.ndarray, test_labels: np.ndarray) -> dict[str, Any]:
        """Compute empirical marginal coverage, average set size, singleton rate, and per-class coverage."""
        sets = self.predict_sets(test_probs)
        yt = np.asarray(test_labels, dtype=np.int64)
        covered = np.array([int(yt[i] in sets[i]) for i in range(len(yt))], dtype=np.float64)
        sizes = np.array([len(s) for s in sets], dtype=np.float64)

        per_class_cov: dict[str, float | None] = {}
        for c in range(test_probs.shape[1]):
            mask = yt == c
            per_class_cov[DR_CLASS_NAMES.get(c, str(c))] = (
                round(float(covered[mask].mean()), 4) if mask.any() else None
            )

        return {
            "method": self.method,
            "alpha": self.alpha,
            "target_coverage": round(1.0 - self.alpha, 4),
            "empirical_coverage": round(float(covered.mean()), 4),
            "average_set_size": round(float(sizes.mean()), 4),
            "singleton_rate": round(float(np.mean(sizes == 1)), 4),
            "empty_set_rate": round(float(np.mean(sizes == 0)), 4),
            "per_class_coverage": per_class_cov,
            "n_test": int(len(yt)),
        }
