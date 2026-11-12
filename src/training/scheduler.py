"""Learning-rate scheduler factory."""

from __future__ import annotations

import torch.optim as optim


def build_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    epochs: int = 15,
    min_lr: float = 1e-6,
) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None:
    """Create learning-rate scheduler from configuration name."""
    sched = name.lower()
    if sched == "none":
        return None
    if sched == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    if sched == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=min_lr)
    if sched == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    raise ValueError(f"Unsupported scheduler '{name}'. Valid: ['cosine', 'plateau', 'step', 'none']")
