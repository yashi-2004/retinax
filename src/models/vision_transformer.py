"""Vision Transformer (ViT) classifier and patch-token analysis using timm."""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTClassifier(nn.Module):
    """Lightweight Vision Transformer wrapper for CPU-friendly retinal classification."""

    VALID_MODES = {"scratch", "frozen", "partial_finetune", "full_finetune"}

    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        num_classes: int = 5,
        mode: str = "frozen",
        dropout: float = 0.2,
        pretrained: bool | None = None,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Valid: {sorted(self.VALID_MODES)}")

        use_pretrained = (mode != "scratch") if pretrained is None else bool(pretrained)
        self.backbone_name = backbone_name
        self.mode = mode
        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout)

        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=use_pretrained,
                num_classes=0,  # Returns pooled CLS / token embedding
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to instantiate Vision Transformer '{backbone_name}' (pretrained={use_pretrained}). "
                "If offline without cached HuggingFace weights, run once with network access or use `mode='scratch'`."
            ) from exc

        self.embedding_dim = int(getattr(self.backbone, "num_features", 192))
        self.expected_img_size = int(getattr(getattr(self.backbone, "patch_embed", None), "img_size", (224, 224))[0])
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
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is not None and len(blocks) >= 2:
                for p in blocks[-2:].parameters():
                    p.requires_grad = True
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad = True

        for p in self.head.parameters():
            p.requires_grad = True

    def _align_resolution(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.expected_img_size or x.shape[-1] != self.expected_img_size:
            return F.interpolate(
                x,
                size=(self.expected_img_size, self.expected_img_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x_aligned = self._align_resolution(x)
        return self.backbone(x_aligned)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_embeddings(x)
        return self.head(emb)

    def get_target_layer(self) -> nn.Module:
        """Return final transformer block norm1 for token-reshaped Grad-CAM."""
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            return blocks[-1].norm1
        return self.backbone

    def extract_cls_patch_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial cosine similarity between CLS token and patch tokens at final block."""
        x_aligned = self._align_resolution(x)
        tokens = self.backbone.forward_features(x_aligned)
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise RuntimeError("Unexpected ViT token tensor shape.")
        num_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        cls_tok = tokens[:, 0:1, :]
        patch_toks = tokens[:, num_prefix:, :]
        sim = F.cosine_similarity(cls_tok, patch_toks, dim=-1)  # [B, N_patches]
        grid = int(round(sim.shape[1] ** 0.5))
        return sim.reshape(sim.shape[0], grid, grid)

    def architecture_summary(self) -> dict[str, Any]:
        patch_embed = getattr(self.backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", (16, 16))
        grid_size = getattr(patch_embed, "grid_size", (14, 14))
        blocks = getattr(self.backbone, "blocks", [])
        return {
            "architecture": self.backbone_name,
            "mode": self.mode,
            "expected_input_size": self.expected_img_size,
            "patch_size": tuple(int(v) for v in patch_size) if isinstance(patch_size, (tuple, list)) else (int(patch_size), int(patch_size)),
            "patch_grid": tuple(int(v) for v in grid_size) if isinstance(grid_size, (tuple, list)) else (int(grid_size), int(grid_size)),
            "num_transformer_blocks": len(blocks),
            "embedding_dim": self.embedding_dim,
            "total_params": int(sum(p.numel() for p in self.parameters())),
            "trainable_params": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }
