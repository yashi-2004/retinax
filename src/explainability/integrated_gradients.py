"""Captum-backed Integrated Gradients and Occlusion Sensitivity explainability."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Occlusion


def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    n_steps: int = 24,
    smooth_sigma: float = 1.5,
) -> dict[str, Any]:
    """Compute pixel-level Integrated Gradients attribution via Captum."""
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    model.eval()

    with torch.inference_mode():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    cls_idx = int(np.argmax(probs)) if target_class is None else int(target_class)

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)
    x = input_tensor.clone().detach().requires_grad_(True)
    attr, delta = ig.attribute(
        x,
        baselines=baseline,
        target=cls_idx,
        n_steps=int(n_steps),
        return_convergence_delta=True,
    )

    # Aggregate channel attributions via L2/abs energy and smooth slightly for clinical readability
    attr_map = attr.detach().abs().sum(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    if smooth_sigma > 0:
        attr_map = cv2.GaussianBlur(attr_map, (0, 0), sigmaX=float(smooth_sigma))
    max_val = float(np.percentile(attr_map, 99.5))
    if max_val > 1e-8:
        attr_map = np.clip((attr_map - float(attr_map.min())) / max(max_val - float(attr_map.min()), 1e-8), 0.0, 1.0)
    else:
        attr_map = np.zeros_like(attr_map, dtype=np.float32)

    return {
        "heatmap": attr_map,
        "target_class": cls_idx,
        "predicted_class": int(np.argmax(probs)),
        "probabilities": probs,
        "convergence_delta": round(float(delta.abs().mean().item()), 6),
        "method": "integrated_gradients",
    }


def compute_occlusion_sensitivity(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
    patch_size: int | None = None,
    stride: int | None = None,
) -> dict[str, Any]:
    """Compute Captum Occlusion Sensitivity (CPU-friendly windowing)."""
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    model.eval()
    h, w = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
    p_sz = patch_size if patch_size is not None else max(16, h // 8)
    st = stride if stride is not None else max(8, p_sz // 2)

    with torch.inference_mode():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    cls_idx = int(np.argmax(probs)) if target_class is None else int(target_class)

    occ = Occlusion(model)
    attr = occ.attribute(
        input_tensor,
        strides=(3, st, st),
        target=cls_idx,
        sliding_window_shapes=(3, p_sz, p_sz),
        baselines=0.0,
    )
    # Positive attribution means occluding that region reduces target logit (important evidence)
    attr_map = F.relu(attr).mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    max_val = float(attr_map.max())
    if max_val > 1e-8:
        attr_map = np.clip((attr_map - float(attr_map.min())) / max(max_val - float(attr_map.min()), 1e-8), 0.0, 1.0)
    else:
        attr_map = np.zeros_like(attr_map, dtype=np.float32)

    return {
        "heatmap": attr_map,
        "target_class": cls_idx,
        "predicted_class": int(np.argmax(probs)),
        "probabilities": probs,
        "method": "occlusion_sensitivity",
    }
