"""Hook-based Grad-CAM and Grad-CAM++ supporting CNNs and Vision Transformers."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _reshape_vit_tokens(tensor: torch.Tensor) -> torch.Tensor:
    """Convert 3D ViT token tensor [B, N_tokens, D] into 4D spatial map [B, D, H, W]."""
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim != 3:
        raise RuntimeError(f"Expected 3D or 4D activation tensor for Grad-CAM, got shape {tensor.shape}")

    b, n_tok, d = tensor.shape
    # Common prefix token counts: 1 (CLS) or 2 (CLS + distil) or 5 (DINOv2 registers)
    for prefix in (1, 0, 2, 5):
        rem = n_tok - prefix
        side = int(round(math.sqrt(rem)))
        if rem > 0 and side * side == rem:
            spatial = tensor[:, prefix:, :].reshape(b, side, side, d)
            return spatial.permute(0, 3, 1, 2).contiguous()

    raise RuntimeError(f"Cannot reshape non-square token sequence length {n_tok} into spatial map.")


class GradCAMExplainer:
    """Computes Grad-CAM or Grad-CAM++ attribution maps without modifying model weights."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_target_layer"):
            self.target_layer = model.get_target_layer()
        else:
            raise ValueError("Model does not implement `get_target_layer()` and no `target_layer` was supplied.")

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        method: str = "gradcam",
        output_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Generate normalized [0, 1] spatial attribution heatmap for a single image tensor [1, C, H, W]."""
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.shape[0] != 1:
            raise ValueError("GradCAMExplainer expects batch size 1.")

        method_key = method.lower()
        if method_key not in {"gradcam", "gradcam_pp"}:
            raise ValueError(f"Unsupported method '{method}'. Choose 'gradcam' or 'gradcam_pp'.")

        was_training = self.model.training
        self.model.eval()
        self._activations = None
        self._gradients = None

        def _fwd_hook(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            self._activations = out
            # Register hook directly on tensor output so frozen backbones still capture gradients safely
            out.register_hook(self._save_grad)

        handle = self.target_layer.register_forward_hook(_fwd_hook)
        try:
            # Ensure autograd graph flows even if backbone weights are frozen
            x = input_tensor.clone().detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]

            cls_idx = int(torch.argmax(logits, dim=1).item()) if target_class is None else int(target_class)
            score = logits[0, cls_idx]
            score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("Grad-CAM hooks failed to capture activations/gradients.")

            acts = _reshape_vit_tokens(self._activations.detach())
            grads = _reshape_vit_tokens(self._gradients.detach())

            if method_key == "gradcam":
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * acts, dim=1).squeeze(0)
            else:
                # Grad-CAM++ (Chattopadhay et al., 2018)
                grads_sq = grads.pow(2)
                grads_cube = grads.pow(3)
                sum_acts_cube = torch.sum(acts * grads_cube, dim=(2, 3), keepdim=True)
                denom = 2.0 * grads_sq + sum_acts_cube + 1e-8
                alpha = grads_sq / denom
                positive_grads = F.relu(score.exp().detach() * grads)
                weights = torch.sum(alpha * positive_grads, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * acts, dim=1).squeeze(0)

            cam = F.relu(cam)
            cam_np = cam.cpu().numpy().astype(np.float32)
            max_val = float(cam_np.max())
            if max_val > 1e-8:
                cam_np = (cam_np - float(cam_np.min())) / max(max_val - float(cam_np.min()), 1e-8)
            else:
                cam_np = np.zeros_like(cam_np, dtype=np.float32)

            target_h, target_w = output_size if output_size is not None else (int(input_tensor.shape[-2]), int(input_tensor.shape[-1]))
            heatmap = cv2.resize(cam_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            heatmap = np.clip(heatmap, 0.0, 1.0)
        finally:
            handle.remove()
            if was_training:
                self.model.train()

        return {
            "heatmap": heatmap,
            "target_class": cls_idx,
            "predicted_class": int(np.argmax(probs)),
            "probabilities": probs,
            "method": method_key,
        }

    def _save_grad(self, grad: torch.Tensor) -> None:
        self._gradients = grad
