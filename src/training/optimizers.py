"""Optimizer builder supporting differential learning rates for transfer learning."""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    backbone_lr_mult: float = 0.25,
) -> optim.Optimizer:
    """Build Adam, AdamW, or SGD optimizer over trainable parameters.

    When both `backbone`/`features` and `head`/`classifier` have trainable weights,
    uses differential learning rate (`lr * backbone_lr_mult` for backbone).
    """
    opt_name = name.lower()
    head_params = []
    backbone_params = []

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param_name.startswith(("head.", "classifier.", "net.")):
            head_params.append(param)
        else:
            backbone_params.append(param)

    if head_params and backbone_params:
        param_groups = [
            {"params": backbone_params, "lr": float(lr) * float(backbone_lr_mult)},
            {"params": head_params, "lr": float(lr)},
        ]
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": trainable, "lr": float(lr)}]

    if opt_name == "adamw":
        return optim.AdamW(param_groups, lr=float(lr), weight_decay=float(weight_decay))
    if opt_name == "adam":
        return optim.Adam(param_groups, lr=float(lr), weight_decay=float(weight_decay))
    if opt_name == "sgd":
        return optim.SGD(param_groups, lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))
    raise ValueError(f"Unsupported optimizer '{name}'. Valid: ['adam', 'adamw', 'sgd']")
