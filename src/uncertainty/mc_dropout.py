"""Monte Carlo Dropout stochastic inference and epistemic/aleatoric decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def enable_dropout_only(model: nn.Module) -> int:
    """Keep BatchNorm/LayerNorm in eval mode while enabling Dropout layers for MC sampling."""
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            if getattr(module, "p", 0.0) > 0.0:
                module.train()
                count += 1
    return count


def _decompose_mc_probabilities(pass_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose [T, N, C] stochastic softmax samples into mean, variance, entropy, and MI."""
    p = np.clip(pass_probs, 1e-8, 1.0)
    mean_probs = np.mean(p, axis=0)  # [N, C]
    pred_variance = np.mean(np.var(p, axis=0), axis=1)  # [N]
    pred_entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, 1e-8, 1.0)), axis=1)
    expected_entropy = -np.mean(np.sum(p * np.log(p), axis=2), axis=0)
    mutual_info = np.clip(pred_entropy - expected_entropy, 0.0, None)
    return {
        "mean_probabilities": mean_probs,
        "predictive_variance": pred_variance,
        "predictive_entropy": pred_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_info,
    }


def mc_dropout_predict_single(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 12,
    temperature: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run T stochastic forward passes on a single image tensor [1, C, H, W]."""
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    torch.manual_seed(seed)
    n_active = enable_dropout_only(model)
    passes: list[np.ndarray] = []

    with torch.inference_mode():
        for _ in range(max(int(n_passes), 2)):
            logits = model(input_tensor)
            scaled = logits / max(float(temperature), 1e-4)
            probs = F.softmax(scaled, dim=-1).cpu().numpy()
            passes.append(probs)

    model.eval()
    pass_arr = np.stack(passes, axis=0)  # [T, 1, C]
    stats = _decompose_mc_probabilities(pass_arr)
    mean_p = stats["mean_probabilities"][0]
    pred_cls = int(np.argmax(mean_p))

    return {
        "mean_probabilities": mean_p,
        "predicted_class": pred_cls,
        "confidence": float(mean_p[pred_cls]),
        "predictive_variance": float(stats["predictive_variance"][0]),
        "predictive_entropy": float(stats["predictive_entropy"][0]),
        "expected_entropy": float(stats["expected_entropy"][0]),
        "mutual_information": float(stats["mutual_information"][0]),
        "n_passes": int(len(passes)),
        "active_dropout_layers": int(n_active),
        "pass_samples": pass_arr[:, 0, :],
    }


def mc_dropout_predict_batch(
    model: nn.Module,
    dataloader: DataLoader,
    n_passes: int = 10,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Compute MC Dropout predictive mean & uncertainty estimates across a DataLoader."""
    torch.manual_seed(seed)
    target_device = torch.device(device)
    model.to(target_device)

    xs: list[torch.Tensor] = []
    ys: list[np.ndarray] = []
    for batch in dataloader:
        xs.append(batch[0])
        ys.append(batch[1].numpy())
    if not xs:
        return {"mean_probabilities": np.empty((0, 5)), "labels": np.empty((0,), dtype=np.int64)}

    full_x = torch.cat(xs, dim=0).to(target_device)
    labels = np.concatenate(ys, axis=0)

    n_active = enable_dropout_only(model)
    passes: list[np.ndarray] = []
    bs = dataloader.batch_size or 8

    with torch.inference_mode():
        for _ in range(max(int(n_passes), 2)):
            pass_chunks: list[np.ndarray] = []
            for start in range(0, len(full_x), bs):
                logits = model(full_x[start : start + bs])
                scaled = logits / max(float(temperature), 1e-4)
                pass_chunks.append(F.softmax(scaled, dim=-1).cpu().numpy())
            passes.append(np.concatenate(pass_chunks, axis=0))

    model.eval()
    stats = _decompose_mc_probabilities(np.stack(passes, axis=0))
    return {
        **stats,
        "labels": labels,
        "predictions": np.argmax(stats["mean_probabilities"], axis=1),
        "n_passes": int(len(passes)),
        "active_dropout_layers": int(n_active),
    }
