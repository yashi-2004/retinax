"""DINOv2 self-supervised foundation model feature extraction and linear/MLP probing.

Provenance & License:
- Model family: Meta AI DINOv2 (`vit_small_patch14_dinov2.lvd142m`, ~22M parameters, 384-d).
- License: Apache License 2.0 (Meta Platforms, Inc. / timm weights via HuggingFace Hub).
- Hardware strategy: Frozen feature extraction at 224x224 + disk caching + lightweight linear/MLP probes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.classifier import MLPProbeClassifier

DINOV2_PROVENANCE = {
    "model_id": "vit_small_patch14_dinov2.lvd142m",
    "source": "timm / HuggingFace Hub (Meta AI Research DINOv2)",
    "license": "Apache-2.0",
    "embedding_dim": 384,
    "pretraining_objective": "Self-supervised discriminative self-distillation on LVD-142M curated natural images",
    "domain_caveat": (
        "Pretrained on natural images (LVD-142M), not retinal fundus photography; "
        "transferability to DR microlesions is investigated empirically."
    ),
}


class DINOv2FeatureExtractor(nn.Module):
    """Frozen DINOv2 ViT-S/14 visual representation extractor (CPU-friendly at 224x224)."""

    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        img_size: int = 224,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.img_size = int(img_size)
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=self.img_size,
            )
        except Exception as exc:
            raise RuntimeError(
                f"DINOv2 weights for '{backbone_name}' could not be loaded (check local cache or internet access). "
                f"Original error: {exc}"
            ) from exc

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.embedding_dim = int(getattr(self.backbone, "num_features", 384))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            return self.backbone(x)


def train_embedding_probe(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    eval_emb: np.ndarray,
    probe_type: str = "logistic",
    num_classes: int = 5,
    seed: int = 42,
    epochs: int = 40,
) -> dict[str, Any]:
    """Train a lightweight classifier probe (Logistic Regression or 2-layer MLP) on frozen embeddings."""
    probe_kind = probe_type.lower()

    if probe_kind == "logistic":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(train_emb, train_labels)
        probs = pipe.predict_proba(eval_emb)
        # Align probability columns if a rare class was absent during training
        clf_classes = pipe.named_steps["clf"].classes_
        full_probs = np.zeros((len(eval_emb), num_classes), dtype=np.float64)
        for col_idx, cls_id in enumerate(clf_classes):
            if 0 <= int(cls_id) < num_classes:
                full_probs[:, int(cls_id)] = probs[:, col_idx]
        # Re-normalize defensively
        row_sums = full_probs.sum(axis=1, keepdims=True)
        full_probs = np.divide(full_probs, np.maximum(row_sums, 1e-8))
        return {
            "probe_type": "logistic_regression",
            "model": pipe,
            "eval_probabilities": full_probs,
            "eval_logits": np.log(np.clip(full_probs, 1e-7, 1.0)),
        }

    if probe_kind == "mlp":
        torch.manual_seed(seed)
        mlp = MLPProbeClassifier(embedding_dim=train_emb.shape[1], hidden_dim=128, num_classes=num_classes)
        opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-3)
        x_tr = torch.tensor(train_emb, dtype=torch.float32)
        y_tr = torch.tensor(train_labels, dtype=torch.long)
        x_ev = torch.tensor(eval_emb, dtype=torch.float32)

        mlp.train()
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logits = mlp(x_tr)
            loss = F.cross_entropy(logits, y_tr)
            loss.backward()
            opt.step()

        mlp.eval()
        with torch.inference_mode():
            ev_logits = mlp(x_ev)
            ev_probs = F.softmax(ev_logits, dim=-1).numpy()
        return {
            "probe_type": "mlp_probe",
            "model": mlp,
            "eval_probabilities": ev_probs,
            "eval_logits": ev_logits.numpy(),
        }

    raise ValueError(f"Unsupported probe_type '{probe_type}'. Choose 'logistic' or 'mlp'.")
