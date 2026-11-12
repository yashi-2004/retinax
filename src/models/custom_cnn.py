"""Configurable from-scratch Convolutional Neural Network for retinal disease grading."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unsupported activation '{name}'. Valid: ['relu', 'gelu', 'leaky_relu']")


class CustomFundusCNN(nn.Module):
    """Modular from-scratch CNN demonstrating convolution, receptive fields, pooling, and regularization."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        num_blocks: int = 4,
        base_channels: int = 16,
        channel_multiplier: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        use_batchnorm: bool = True,
        pooling: str = "max",
        dropout: float = 0.3,
        fc_dims: list[int] | tuple[int, ...] = (128,),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_classes = int(num_classes)
        self.num_blocks = int(num_blocks)
        self.base_channels = int(base_channels)
        self.channel_multiplier = int(channel_multiplier)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding if padding is not None else kernel_size // 2)
        self.use_batchnorm = bool(use_batchnorm)
        self.pooling_type = pooling.lower()
        self.dropout_rate = float(dropout)
        self.fc_dims = [int(d) for d in fc_dims]
        self.activation_name = activation.lower()

        blocks: list[nn.Module] = []
        self.block_specs: list[dict[str, Any]] = []
        cur_in = in_channels
        cur_out = base_channels

        for b_idx in range(num_blocks):
            layers: list[nn.Module] = [
                nn.Conv2d(
                    cur_in,
                    cur_out,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=not self.use_batchnorm,
                )
            ]
            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(cur_out))
            layers.append(_make_activation(self.activation_name))
            if self.pooling_type == "max":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif self.pooling_type == "avg":
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                raise ValueError(f"Unsupported pooling '{pooling}'. Choose 'max' or 'avg'.")

            block = nn.Sequential(*layers)
            blocks.append(block)
            self.block_specs.append(
                {
                    "block": b_idx + 1,
                    "in_channels": cur_in,
                    "out_channels": cur_out,
                    "kernel_size": self.kernel_size,
                    "stride": self.stride,
                    "padding": self.padding,
                    "pool": f"{self.pooling_type}2d(2,2)",
                }
            )
            cur_in = cur_out
            cur_out = min(cur_out * self.channel_multiplier, 512)

        self.features = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = cur_in

        head_layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = self.embedding_dim
        for hidden_dim in self.fc_dims:
            if self.dropout_rate > 0:
                head_layers.append(nn.Dropout(p=self.dropout_rate))
            head_layers.append(nn.Linear(prev_dim, hidden_dim))
            head_layers.append(_make_activation(self.activation_name))
            prev_dim = hidden_dim
        if self.dropout_rate > 0:
            head_layers.append(nn.Dropout(p=self.dropout_rate))
        head_layers.append(nn.Linear(prev_dim, self.num_classes))
        self.classifier = nn.Sequential(*head_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return intermediate feature maps after each convolutional block."""
        maps: list[torch.Tensor] = []
        h = x
        for block in self.features:
            h = block(h)
            maps.append(h)
        return maps

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global-average-pooled representation prior to classifier MLP."""
        h = x
        for block in self.features:
            h = block(h)
        pooled = self.global_pool(h)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_embeddings(x)
        return self.classifier(emb)

    def get_target_layer(self) -> nn.Module:
        """Return final Conv2d layer for Grad-CAM / Grad-CAM++ attribution."""
        last_block = self.features[-1]
        for layer in reversed(last_block):
            if isinstance(layer, nn.Conv2d):
                return layer
        return last_block

    def compute_receptive_field(self, input_size: int = 224) -> list[dict[str, Any]]:
        """Analytically compute receptive field size, jump, and spatial resolution per block."""
        rf = 1
        jump = 1
        spatial = int(input_size)
        summary: list[dict[str, Any]] = []

        for spec in self.block_specs:
            # Conv layer
            k = spec["kernel_size"]
            s = spec["stride"]
            p = spec["padding"]
            rf = rf + (k - 1) * jump
            jump = jump * s
            spatial = (spatial + 2 * p - k) // s + 1

            # Pooling layer (2x2, stride 2)
            rf = rf + (2 - 1) * jump
            jump = jump * 2
            spatial = max(1, (spatial - 2) // 2 + 1)

            summary.append(
                {
                    "block": spec["block"],
                    "channels": spec["out_channels"],
                    "receptive_field_px": int(rf),
                    "effective_stride": int(jump),
                    "output_spatial": f"{spatial}x{spatial}",
                }
            )
        return summary

    def architecture_summary(self, input_size: int = 224) -> dict[str, Any]:
        """Return parameter counts and theoretical receptive field breakdown."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture": "CustomFundusCNN",
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "embedding_dim": int(self.embedding_dim),
            "receptive_fields": self.compute_receptive_field(input_size=input_size),
        }
