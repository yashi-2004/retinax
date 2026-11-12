"""Classification, imbalance-aware, and self-supervised contrastive loss functions."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: np.ndarray | list[int],
    num_classes: int = 5,
    mode: str = "inverse_sqrt",
) -> torch.Tensor:
    """Compute class balancing weights safely without divide-by-zero on absent classes."""
    arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(arr, minlength=num_classes).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)

    if mode == "inverse":
        weights = len(arr) / (num_classes * safe_counts)
    elif mode == "inverse_sqrt":
        weights = np.sqrt(len(arr) / (num_classes * safe_counts))
    elif mode == "effective_num":
        beta = 0.99
        effective = 1.0 - np.power(beta, safe_counts)
        weights = (1.0 - beta) / np.maximum(effective, 1e-6)
    else:
        raise ValueError(f"Unsupported weight mode '{mode}'.")

    # Zero-out weights only if strictly absent? Better to keep median weight so loss remains finite
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., 2017) for severe diabetic retinopathy class imbalance."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha.float() if alpha is not None else None)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal_term = torch.pow(1.0 - pt, self.gamma)
        loss = focal_term * ce
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class NTXentLoss(nn.Module):
    """Normalized Temperature-Scaled Cross-Entropy (NT-Xent) loss for SimCLR."""

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        if batch_size < 2:
            raise ValueError("NTXentLoss requires batch_size >= 2 to form negative pairs.")

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        sim = torch.matmul(z, z.T) / max(self.temperature, 1e-4)  # [2B, 2B]
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)

        # Positive pair indices: i <-> i + B
        pos_idx = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z.device),
                torch.arange(0, batch_size, device=z.device),
            ],
            dim=0,
        )
        return F.cross_entropy(sim, pos_idx)


def build_loss_function(
    loss_name: str = "weighted_ce",
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Instantiate classification loss function cleanly."""
    name = loss_name.lower()
    weights = class_weights.to(device) if class_weights is not None else None

    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if name == "focal":
        return FocalLoss(alpha=weights, gamma=focal_gamma)
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=float(label_smoothing))
    raise ValueError(f"Unknown loss '{loss_name}'. Valid: ['ce', 'weighted_ce', 'focal', 'label_smoothing']")
