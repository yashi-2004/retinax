"""Lazy PyTorch datasets, disk preprocessing cache, and frozen embedding cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from src.config import resolve_path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def normalize_rgb_tensor(
    image_rgb: np.ndarray,
    mode: str = "imagenet",
) -> torch.Tensor:
    """Convert HWC uint8/float RGB image into normalized CHW float32 PyTorch tensor."""
    arr = np.asarray(image_rgb, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0

    if mode == "imagenet":
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    elif mode == "zscore_fov":
        mask = arr.mean(axis=-1) > 0.03
        if mask.any():
            mean = arr[mask].mean(axis=0)
            std = np.clip(arr[mask].std(axis=0), 1e-4, None)
            arr = (arr - mean) / std
        else:
            arr = (arr - 0.5) / 0.5
    elif mode == "zero_one":
        pass
    else:
        raise ValueError(f"Unsupported normalisation mode '{mode}'.")

    chw = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(chw)


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Safely load an image from disk as RGB uint8 NumPy array."""
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"), dtype=np.uint8)


def ensure_processed_cache(
    manifest_df: pd.DataFrame,
    preprocess_fn: Callable[[np.ndarray], np.ndarray],
    preprocess_cfg: dict[str, Any],
    cache_root: str | Path | None = None,
) -> Path:
    """Pre-resize and preprocess images once into `data/processed/images/<hash>/`."""
    base_dir = resolve_path(cache_root or "data/processed/images")
    cfg_id = _config_hash(preprocess_cfg)
    target_dir = base_dir / cfg_id
    target_dir.mkdir(parents=True, exist_ok=True)

    meta_file = target_dir / "preprocess_config.json"
    if not meta_file.exists():
        with meta_file.open("w", encoding="utf-8") as handle:
            json.dump(preprocess_cfg, handle, indent=2)

    valid_df = manifest_df[~manifest_df["is_corrupt"].astype(bool)]
    for row in valid_df.itertuples(index=False):
        cached_file = target_dir / f"{row.image_id}.png"
        if cached_file.exists():
            continue
        raw_img = load_rgb_image(row.abs_path)
        proc_img = preprocess_fn(raw_img)
        Image.fromarray(np.clip(proc_img, 0, 255).astype(np.uint8)).save(cached_file, format="PNG")

    return target_dir


class FundusDataset(Dataset):
    """Memory-efficient lazy-loading PyTorch Dataset for retinal fundus manifests."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        image_size: int = 224,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        normalisation: str = "imagenet",
        disk_cache_dir: str | Path | None = None,
    ) -> None:
        self.df = manifest_df[~manifest_df["is_corrupt"].astype(bool)].reset_index(drop=True)
        self.image_size = int(image_size)
        self.preprocess_fn = preprocess_fn
        self.augment_fn = augment_fn
        self.normalisation = normalisation
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_item_image(self, row: pd.Series) -> np.ndarray:
        if self.disk_cache_dir is not None:
            cached = self.disk_cache_dir / f"{row['image_id']}.png"
            if cached.exists():
                return load_rgb_image(cached)

        img = load_rgb_image(row["abs_path"])
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)
        else:
            pil = Image.fromarray(img).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            img = np.asarray(pil, dtype=np.uint8)
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img = self._load_item_image(row)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            pil = Image.fromarray(img).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            img = np.asarray(pil, dtype=np.uint8)
        if self.augment_fn is not None:
            img = self.augment_fn(img)
        tensor = normalize_rgb_tensor(img, mode=self.normalisation)
        label = int(row["original_label"])
        image_id = str(row["image_id"])
        return tensor, label, image_id


def save_embeddings(
    cache_name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    image_ids: list[str],
    splits: list[str],
    metadata: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Persist extracted foundation/backbone embeddings to disk (`experiments/cache/`)."""
    out_dir = resolve_path(cache_dir or "experiments/cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cache_name}.npz"
    np.savez_compressed(
        out_path,
        embeddings=embeddings.astype(np.float32),
        labels=labels.astype(np.int64),
        image_ids=np.array(image_ids, dtype=object),
        splits=np.array(splits, dtype=object),
        metadata=json.dumps(metadata or {}),
    )
    return out_path


def load_embeddings(cache_name_or_path: str | Path, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Load persisted embedding archive from `experiments/cache/`."""
    cand = Path(cache_name_or_path)
    if not cand.is_file():
        cand = resolve_path(cache_dir or "experiments/cache") / f"{cache_name_or_path}.npz"
    if not cand.is_file():
        raise FileNotFoundError(f"Embedding cache not found: {cand}")
    data = np.load(cand, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"],
        "image_ids": data["image_ids"].tolist(),
        "splits": data["splits"].tolist(),
        "metadata": json.loads(str(data["metadata"])),
    }
