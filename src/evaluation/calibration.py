"""Probability calibration metrics (ECE, Brier, NLL) and Temperature Scaling."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def compute_calibration_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE), MCE, multiclass Brier Score, and NLL."""
    yt = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    if len(yt) == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "nll": 0.0, "bins": []}

    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == yt).astype(np.float64)

    # Multiclass Brier Score: mean over samples of sum_k (p_k - y_k)^2
    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes, dtype=np.float64)[yt]
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # Negative Log Likelihood
    true_probs = probs[np.arange(len(yt)), yt]
    nll = float(-np.mean(np.log(true_probs)))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins_info: list[dict[str, Any]] = []

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count > 0:
            bin_acc = float(accuracies[mask].mean())
            bin_conf = float(confidences[mask].mean())
            gap = abs(bin_acc - bin_conf)
            ece += (count / len(yt)) * gap
            mce = max(mce, gap)
        else:
            bin_acc = None
            bin_conf = None
            gap = 0.0
        bins_info.append(
            {
                "bin_lower": round(float(lo), 3),
                "bin_upper": round(float(hi), 3),
                "bin_midpoint": round(float((lo + hi) / 2.0), 3),
                "count": count,
                "accuracy": round(bin_acc, 4) if bin_acc is not None else None,
                "confidence": round(bin_conf, 4) if bin_conf is not None else None,
                "gap": round(float(gap), 4),
            }
        )

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(brier_score, 4),
        "nll": round(nll, 4),
        "bins": bins_info,
    }


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling T > 0 to logits and return softmax probabilities."""
    t = max(float(temperature), 1e-4)
    scaled = np.asarray(logits, dtype=np.float64) / t
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling calibrated strictly on held-out calibration/validation logits."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([np.log(max(init_temperature, 1e-3))], dtype=torch.float32))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temp).clamp(min=0.05, max=20.0).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp).clamp(min=0.05, max=20.0)
        return logits / temp

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 80) -> dict[str, Any]:
        """Optimize scalar temperature to minimize NLL without changing top-1 predictions."""
        logits_t = torch.tensor(np.asarray(logits, dtype=np.float32))
        labels_t = torch.tensor(np.asarray(labels, dtype=np.int64))

        before_probs = F.softmax(logits_t, dim=-1).numpy()
        before_cal = compute_calibration_metrics(labels, before_probs)

        optimizer = optim.LBFGS([self.log_temp], lr=0.1, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(_closure)
        with torch.no_grad():
            after_probs = F.softmax(self.forward(logits_t), dim=-1).numpy()
        after_cal = compute_calibration_metrics(labels, after_probs)

        return {
            "temperature": round(self.temperature, 4),
            "before": before_cal,
            "after": after_cal,
        }
