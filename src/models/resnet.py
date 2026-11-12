"""Transfer-learning ResNet18 classifier supporting scratch, frozen, partial, and full fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _load_resnet18_backbone(pretrained: bool) -> models.ResNet:
    if not pretrained:
        return models.resnet18(weights=None)
    try:
        return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load pretrained ImageNet weights for ResNet18 (check local torch cache or network). "
            "Use `mode='scratch'` (`pretrained=False`) for offline random-init baseline."
        ) from exc


class ResNet18Classifier(nn.Module):
    """ResNet18 wrapper with configurable fine-tuning regime and MC-Dropout head."""

    VALID_MODES = {"scratch", "frozen", "partial_finetune", "full_finetune"}

    def __init__(
        self,
        num_classes: int = 5,
        mode: str = "partial_finetune",
        dropout: float = 0.3,
        pretrained: bool | None = None,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Valid: {sorted(self.VALID_MODES)}")

        use_pretrained = (mode != "scratch") if pretrained is None else bool(pretrained)
        self.mode = mode
        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout)

        backbone = _load_resnet18_backbone(use_pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding_dim = int(in_features)

        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embedding_dim, self.num_classes),
        )
        self._apply_freeze_policy()

    def _apply_freeze_policy(self) -> None:
        if self.mode in {"scratch", "full_finetune"}:
            for p in self.backbone.parameters():
                p.requires_grad = True
        elif self.mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif self.mode == "partial_finetune":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

        for p in self.head.parameters():
            p.requires_grad = True

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_embeddings(x)
        return self.head(emb)

    def get_target_layer(self) -> nn.Module:
        """Return final residual block for Grad-CAM."""
        return self.backbone.layer4[-1]
