"""Deep Ensemble predictive uncertainty aggregation across multiple trained models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.uncertainty.mc_dropout import _decompose_mc_probabilities


def predict_ensemble_single(
    models: list[nn.Module],
    input_tensor: torch.Tensor,
    temperatures: list[float] | None = None,
) -> dict[str, Any]:
    """Aggregate predictions across M ensemble members for a single image tensor."""
    if not models:
        raise ValueError("Ensemble requires at least 1 trained model.")
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    temps = temperatures or [1.0] * len(models)
    member_probs: list[np.ndarray] = []
    member_preds: list[int] = []

    with torch.inference_mode():
        for model, temp in zip(models, temps):
            model.eval()
            logits = model(input_tensor)
            probs = F.softmax(logits / max(float(temp), 1e-4), dim=-1).cpu().numpy()
            member_probs.append(probs)
            member_preds.append(int(np.argmax(probs[0])))

    stack = np.stack(member_probs, axis=0)  # [M, 1, C]
    stats = _decompose_mc_probabilities(stack)
    mean_p = stats["mean_probabilities"][0]
    pred_cls = int(np.argmax(mean_p))
    agreement = float(np.mean([1.0 if p == pred_cls else 0.0 for p in member_preds]))

    return {
        "mean_probabilities": mean_p,
        "predicted_class": pred_cls,
        "confidence": float(mean_p[pred_cls]),
        "predictive_variance": float(stats["predictive_variance"][0]),
        "predictive_entropy": float(stats["predictive_entropy"][0]),
        "expected_entropy": float(stats["expected_entropy"][0]),
        "mutual_information": float(stats["mutual_information"][0]),
        "member_predictions": member_preds,
        "member_agreement": round(agreement, 4),
        "n_members": len(models),
    }


def predict_ensemble_batch(
    models: list[nn.Module],
    dataloader: DataLoader,
    temperatures: list[float] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run batch Deep Ensemble evaluation across M models."""
    if not models:
        raise ValueError("Ensemble requires at least 1 model.")
    target_device = torch.device(device)
    temps = temperatures or [1.0] * len(models)

    xs: list[torch.Tensor] = []
    ys: list[np.ndarray] = []
    for batch in dataloader:
        xs.append(batch[0])
        ys.append(batch[1].numpy())
    full_x = torch.cat(xs, dim=0).to(target_device)
    labels = np.concatenate(ys, axis=0)
    bs = dataloader.batch_size or 8

    member_outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for model, temp in zip(models, temps):
            model.to(target_device).eval()
            chunks = []
            for start in range(0, len(full_x), bs):
                logits = model(full_x[start : start + bs])
                probs = F.softmax(logits / max(float(temp), 1e-4), dim=-1).cpu().numpy()
                chunks.append(probs)
            member_outputs.append(np.concatenate(chunks, axis=0))

    stats = _decompose_mc_probabilities(np.stack(member_outputs, axis=0))
    return {
        **stats,
        "labels": labels,
        "predictions": np.argmax(stats["mean_probabilities"], axis=1),
        "n_members": len(models),
    }
