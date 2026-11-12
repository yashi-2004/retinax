"""Unified model factory, parameter accounting, and lightweight embedding probes."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.custom_cnn import CustomFundusCNN
from src.models.efficientnet import DenseNet121Classifier, EfficientNetB0Classifier
from src.models.resnet import ResNet18Classifier
from src.models.vision_transformer import ViTClassifier


class LinearProbeClassifier(nn.Module):
    """Lightweight linear probe over precomputed frozen backbone/foundation embeddings."""

    def __init__(self, embedding_dim: int, num_classes: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class MLPProbeClassifier(nn.Module):
    """Lightweight 2-layer MLP downstream classifier over frozen embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.embedding_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), self.num_classes),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


def count_parameters(model: nn.Module) -> dict[str, Any]:
    """Return total parameters, trainable parameters, and estimated FP32 memory footprint."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total * 4.0) / (1024.0 * 1024.0)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "estimated_fp32_mb": round(float(size_mb), 3),
    }


def build_model(
    model_name: str,
    num_classes: int = 5,
    cfg: dict[str, Any] | None = None,
    mode_override: str | None = None,
    pretrained_override: bool | None = None,
) -> nn.Module:
    """Instantiate a RetinaX architecture cleanly from config and name."""
    name = model_name.lower()
    models_cfg = (cfg or {}).get("models", {})

    if name == "custom_cnn":
        c_cfg = models_cfg.get("custom_cnn", {})
        return CustomFundusCNN(
            num_classes=num_classes,
            num_blocks=int(c_cfg.get("num_blocks", 4)),
            base_channels=int(c_cfg.get("base_channels", 16)),
            channel_multiplier=int(c_cfg.get("channel_multiplier", 2)),
            kernel_size=int(c_cfg.get("kernel_size", 3)),
            stride=int(c_cfg.get("stride", 1)),
            padding=c_cfg.get("padding"),
            use_batchnorm=bool(c_cfg.get("use_batchnorm", True)),
            pooling=str(c_cfg.get("pooling", "max")),
            dropout=float(c_cfg.get("dropout", 0.3)),
            fc_dims=c_cfg.get("fc_dims", [128]),
            activation=str(c_cfg.get("activation", "relu")),
        )

    if name == "resnet18":
        r_cfg = models_cfg.get("resnet18", {})
        return ResNet18Classifier(
            num_classes=num_classes,
            mode=mode_override or str(r_cfg.get("mode", "partial_finetune")),
            dropout=float(r_cfg.get("dropout", 0.3)),
            pretrained=pretrained_override,
        )

    if name == "efficientnet_b0":
        e_cfg = models_cfg.get("efficientnet_b0", {})
        return EfficientNetB0Classifier(
            num_classes=num_classes,
            mode=mode_override or str(e_cfg.get("mode", "partial_finetune")),
            dropout=float(e_cfg.get("dropout", 0.3)),
            pretrained=pretrained_override,
        )

    if name == "densenet121":
        d_cfg = models_cfg.get("densenet121", {})
        return DenseNet121Classifier(
            num_classes=num_classes,
            mode=mode_override or str(d_cfg.get("mode", "frozen")),
            dropout=float(d_cfg.get("dropout", 0.3)),
            pretrained=pretrained_override,
        )

    if name in {"vit", "vit_tiny", "deit_tiny", "vit_small"}:
        v_cfg = models_cfg.get("vit", {})
        backbone_map = {
            "vit_tiny": "vit_tiny_patch16_224",
            "deit_tiny": "deit_tiny_patch16_224",
            "vit_small": "vit_small_patch16_224",
            "vit": str(v_cfg.get("backbone", "vit_tiny_patch16_224")),
        }
        return ViTClassifier(
            backbone_name=backbone_map[name],
            num_classes=num_classes,
            mode=mode_override or str(v_cfg.get("mode", "frozen")),
            dropout=float(v_cfg.get("dropout", 0.2)),
            pretrained=pretrained_override,
        )

    raise ValueError(
        f"Unknown model_name '{model_name}'. Supported: "
        "['custom_cnn', 'resnet18', 'efficientnet_b0', 'densenet121', 'vit_tiny', 'deit_tiny', 'vit_small']"
    )
