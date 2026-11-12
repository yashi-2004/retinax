"""Optional IDRiD lesion segmentation module using MONAI 2D U-Net (CPU-compatible)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import UNet


class LesionUNetWrapper(nn.Module):
    """Lightweight 2D MONAI U-Net for optional microaneurysm/haemorrhage/exudate segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        strides: tuple[int, ...] = (2, 2, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=1,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_segmentation_metrics(
    pred_logits: torch.Tensor | np.ndarray,
    target_mask: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute Dice coefficient, IoU (Jaccard), precision, recall, and pixel sensitivity."""
    logits_np = pred_logits.detach().cpu().numpy() if isinstance(pred_logits, torch.Tensor) else np.asarray(pred_logits)
    target_np = target_mask.detach().cpu().numpy() if isinstance(target_mask, torch.Tensor) else np.asarray(target_mask)

    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -30.0, 30.0)))
    pred_bin = (probs >= threshold).astype(np.float64).ravel()
    gt_bin = (target_np >= 0.5).astype(np.float64).ravel()

    tp = float(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = float(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = float(np.sum((pred_bin == 0) & (gt_bin == 1)))

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "pixel_sensitivity": round(recall, 4),
    }


def soft_dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Combined BCEWithLogits + soft Dice loss for highly imbalanced tiny lesion masks."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets.float())
    probs = torch.sigmoid(logits)
    inter = (probs * targets.float()).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.float().sum(dim=(1, 2, 3)) + 1e-6
    dice_loss = 1.0 - (2.0 * inter + 1e-6) / denom
    return 0.5 * bce + 0.5 * dice_loss.mean()
