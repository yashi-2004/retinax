"""EfficientNet-B0 and optional DenseNet121 transfer learning classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EfficientNetB0Classifier(nn.Module):
    """Lightweight EfficientNet-B0 transfer learning architecture for CPU experimentation."""

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

        if use_pretrained:
            try:
                backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load pretrained EfficientNet-B0 weights. "
                    "Use `mode='scratch'` (`pretrained=False`) for offline random-init baseline."
                ) from exc
        else:
            backbone = models.efficientnet_b0(weights=None)

        in_features = backbone.classifier[1].in_features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.embedding_dim = int(in_features)

        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embedding_dim, self.num_classes),
        )
        self._apply_freeze_policy()

    def _apply_freeze_policy(self) -> None:
        if self.mode in {"scratch", "full_finetune"}:
            for p in self.features.parameters():
                p.requires_grad = True
        elif self.mode == "frozen":
            for p in self.features.parameters():
                p.requires_grad = False
        elif self.mode == "partial_finetune":
            for p in self.features.parameters():
                p.requires_grad = False
            # Unfreeze final two feature stages (stages 7 and 8)
            for idx in (-2, -1):
                for p in self.features[idx].parameters():
                    p.requires_grad = True

        for p in self.head.parameters():
            p.requires_grad = True

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.avgpool(feat)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_embeddings(x)
        return self.head(emb)

    def get_target_layer(self) -> nn.Module:
        return self.features[-1]


class DenseNet121Classifier(nn.Module):
    """Optional DenseNet121 baseline for parameter/complexity vs accuracy trade-offs."""

    VALID_MODES = {"scratch", "frozen", "partial_finetune", "full_finetune"}

    def __init__(
        self,
        num_classes: int = 5,
        mode: str = "frozen",
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

        if use_pretrained:
            try:
                backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            except Exception as exc:
                raise RuntimeError("Unable to load pretrained DenseNet121 weights.") from exc
        else:
            backbone = models.densenet121(weights=None)

        in_features = backbone.classifier.in_features
        self.features = backbone.features
        self.embedding_dim = int(in_features)
        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embedding_dim, self.num_classes),
        )
        self._apply_freeze_policy()

    def _apply_freeze_policy(self) -> None:
        if self.mode in {"scratch", "full_finetune"}:
            for p in self.features.parameters():
                p.requires_grad = True
        elif self.mode == "frozen":
            for p in self.features.parameters():
                p.requires_grad = False
        elif self.mode == "partial_finetune":
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.features.denseblock4.parameters():
                p.requires_grad = True
            for p in self.features.norm5.parameters():
                p.requires_grad = True

        for p in self.head.parameters():
            p.requires_grad = True

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = F.relu(feat, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_embeddings(x)
        return self.head(emb)

    def get_target_layer(self) -> nn.Module:
        return self.features.norm5
