"""SimCLR-style self-supervised contrastive learning for retinal representations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.config import resolve_device_and_threads, resolve_path
from src.data.caching import load_rgb_image, normalize_rgb_tensor
from src.models.custom_cnn import CustomFundusCNN
from src.models.resnet import ResNet18Classifier
from src.preprocessing.fundus_processing import preprocess_fundus_image
from src.training.losses import NTXentLoss


def build_simclr_view_transform(image_size: int = 128) -> A.Compose:
    """Create stochastic fundus-safe contrastive view augmentation pipeline."""
    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.70, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, border_mode=0, p=0.6),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.04, p=0.8),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        ]
    )


class SimCLRPairDataset(Dataset):
    """Generates positive pairs `(view_1, view_2)` from unlabelled retinal images."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        image_size: int = 128,
        preprocess_preset: str = "fov_crop",
        normalisation: str = "imagenet",
    ) -> None:
        self.df = manifest_df[~manifest_df["is_corrupt"].astype(bool)].reset_index(drop=True)
        self.image_size = int(image_size)
        self.preprocess_preset = preprocess_preset
        self.normalisation = normalisation
        self.view_transform = build_simclr_view_transform(image_size=self.image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        raw = load_rgb_image(row["abs_path"])
        base = preprocess_fundus_image(raw, preset=self.preprocess_preset, image_size=self.image_size)
        v1 = self.view_transform(image=base.copy())["image"]
        v2 = self.view_transform(image=base.copy())["image"]
        t1 = normalize_rgb_tensor(v1, mode=self.normalisation)
        t2 = normalize_rgb_tensor(v2, mode=self.normalisation)
        return t1, t2


class SimCLREncoderModel(nn.Module):
    """Encoder + non-linear MLP projection head for SimCLR contrastive pretraining."""

    def __init__(
        self,
        encoder_type: str = "custom_cnn",
        projection_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == "custom_cnn":
            self.encoder = CustomFundusCNN(num_classes=num_classes, num_blocks=4, base_channels=16, fc_dims=[128])
        elif self.encoder_type == "resnet18":
            self.encoder = ResNet18Classifier(num_classes=num_classes, mode="scratch", pretrained=False)
        else:
            raise ValueError(f"Unsupported SimCLR encoder '{encoder_type}'. Choose 'custom_cnn' or 'resnet18'.")

        emb_dim = int(self.encoder.embedding_dim)
        self.embedding_dim = emb_dim
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, int(projection_dim)),
        )
        self.downstream_head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(emb_dim, num_classes),
        )

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.extract_embeddings(x)

    def forward_contrastive(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.extract_embeddings(x)
        z = self.projector(h)
        return h, z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downstream supervised classification forward pass using learned encoder."""
        h = self.extract_embeddings(x)
        return self.downstream_head(h)

    def get_target_layer(self) -> nn.Module:
        return self.encoder.get_target_layer()

    def set_encoder_frozen(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze
        for p in self.downstream_head.parameters():
            p.requires_grad = True


def train_simclr_ssl(
    ssl_model: SimCLREncoderModel,
    unlabelled_df: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 16,
    image_size: int = 128,
    lr: float = 1e-3,
    temperature: float = 0.5,
    preprocess_preset: str = "fov_crop",
    run_id: str = "simclr_ssl",
    checkpoint_path: str | Path | None = None,
    tb_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train SimCLR encoder contrastively on unlabelled training retinal images."""
    hw = resolve_device_and_threads(cfg)
    device = hw["device"]
    ssl_model.to(device)

    dataset = SimCLRPairDataset(
        unlabelled_df,
        image_size=image_size,
        preprocess_preset=preprocess_preset,
    )
    if len(dataset) < 2:
        raise ValueError("SimCLR training requires at least 2 valid images.")

    eff_bs = min(batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=eff_bs, shuffle=True, drop_last=(len(dataset) > eff_bs), num_workers=0)

    params = list(ssl_model.encoder.parameters()) + list(ssl_model.projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    criterion = NTXentLoss(temperature=temperature)

    tb_path = resolve_path(tb_dir or f"experiments/tensorboard/{run_id}")
    tb_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_path))

    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    try:
        for epoch in range(1, epochs + 1):
            ssl_model.train()
            loss_sum = 0.0
            n_items = 0
            for v1, v2 in loader:
                if v1.size(0) < 2:
                    continue
                v1, v2 = v1.to(device), v2.to(device)
                optimizer.zero_grad(set_to_none=True)
                _, z1 = ssl_model.forward_contrastive(v1)
                _, z2 = ssl_model.forward_contrastive(v2)
                loss = criterion(z1, z2)
                loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                bs = v1.size(0)
                loss_sum += float(loss.item()) * bs
                n_items += bs

            ep_loss = loss_sum / max(n_items, 1)
            history.append({"epoch": epoch, "nt_xent_loss": round(ep_loss, 4)})
            writer.add_scalar("SSL/NT_Xent_Loss", ep_loss, epoch)
    finally:
        writer.flush()
        writer.close()

    ckpt_out = resolve_path(checkpoint_path or f"checkpoints/{run_id}.pt")
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "simclr_encoder",
            "encoder_type": ssl_model.encoder_type,
            "state_dict": ssl_model.state_dict(),
            "history": history,
            "image_size": image_size,
        },
        ckpt_out,
    )

    return {
        "run_id": run_id,
        "checkpoint_path": str(ckpt_out),
        "tensorboard_dir": str(tb_path),
        "epochs": epochs,
        "final_nt_xent_loss": history[-1]["nt_xent_loss"] if history else None,
        "duration_seconds": round(time.perf_counter() - t0, 2),
        "history": history,
    }
