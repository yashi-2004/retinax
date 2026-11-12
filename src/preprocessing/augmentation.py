"""Clinically plausible fundus image augmentation policies using Albumentations."""

from __future__ import annotations

from typing import Callable

import albumentations as A
import numpy as np


def build_augmentation_pipeline(
    policy: str = "light_clinical",
    image_size: int = 224,
    seed: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a configurable augmentation callable `(RGB uint8 HWC) -> (RGB uint8 HWC)`.

    Policies avoid extreme geometric shears or unnatural colour inversions that
    could distort optic disc anatomy or mimic haemorrhage/exudate pathology.
    """
    policy = policy.lower()
    if policy == "none":
        return lambda img: img

    if policy == "light_clinical":
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
            ]
        )
    elif policy == "moderate_clinical":
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.25),
                A.Rotate(limit=25, border_mode=0, p=0.6),
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.88, 1.0), p=0.45),
                A.RandomBrightnessContrast(brightness_limit=0.16, contrast_limit=0.16, p=0.6),
                A.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.10, hue=0.02, p=0.4),
            ]
        )
    elif policy == "domain_shift_robust":
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=25, border_mode=0, p=0.6),
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.86, 1.0), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.22, p=0.7),
                A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.18, hue=0.03, p=0.6),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.GaussNoise(var_limit=(8.0, 28.0), p=1.0),
                    ],
                    p=0.35,
                ),
            ]
        )
    else:
        raise ValueError(
            f"Unknown augmentation policy '{policy}'. "
            "Valid: ['none', 'light_clinical', 'moderate_clinical', 'domain_shift_robust']"
        )

    if seed is not None:
        np.random.seed(seed)

    def _apply(image_rgb: np.ndarray) -> np.ndarray:
        return transform(image=image_rgb)["image"]

    return _apply


def generate_augmentation_grid(
    image_rgb: np.ndarray,
    policy: str = "moderate_clinical",
    num_samples: int = 4,
    image_size: int = 224,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate deterministic augmented previews alongside the original image."""
    np.random.seed(seed)
    aug_fn = build_augmentation_pipeline(policy=policy, image_size=image_size)
    return [aug_fn(image_rgb.copy()) for _ in range(num_samples)]
