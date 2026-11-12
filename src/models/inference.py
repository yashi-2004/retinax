"""Lightweight CPU-compatible inference engine for single images and dataloaders."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from src.config import DR_CLASS_NAMES
from src.data.caching import normalize_rgb_tensor
from src.preprocessing.fundus_processing import load_rgb_array, preprocess_fundus_image


def predict_single_image(
    model: nn.Module,
    image_source: str | Path | Image.Image | np.ndarray,
    preprocess_preset: str = "fov_clahe_norm",
    image_size: int = 224,
    normalisation: str = "imagenet",
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run deterministic CPU-safe single-image inference with latency tracking."""
    model.eval()
    target_device = torch.device(device)
    model.to(target_device)

    raw_img = load_rgb_array(image_source)
    proc_img = preprocess_fundus_image(raw_img, preset=preprocess_preset, image_size=image_size)
    tensor = normalize_rgb_tensor(proc_img, mode=normalisation).unsqueeze(0).to(target_device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        scaled_logits = logits / max(float(temperature), 1e-4)
        probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()
        embedding = None
        if hasattr(model, "extract_embeddings"):
            embedding = model.extract_embeddings(tensor).squeeze(0).cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    return {
        "preprocessed_image": proc_img,
        "input_tensor": tensor.cpu(),
        "logits": logits.squeeze(0).cpu().numpy(),
        "scaled_logits": scaled_logits.squeeze(0).cpu().numpy(),
        "probabilities": probs,
        "prob_dict": {DR_CLASS_NAMES[i]: round(float(probs[i]), 5) for i in range(len(probs))},
        "predicted_class": pred_class,
        "predicted_label": DR_CLASS_NAMES.get(pred_class, str(pred_class)),
        "confidence": confidence,
        "temperature": float(temperature),
        "inference_ms": round(float(latency_ms), 2),
        "embedding": embedding,
    }


def predict_batch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
    extract_embeddings: bool = False,
) -> dict[str, Any]:
    """Run batched inference over a DataLoader, collecting logits, probs, labels, and timing."""
    model.eval()
    target_device = torch.device(device)
    model.to(target_device)

    all_logits: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids: list[str] = []
    all_embs: list[np.ndarray] = []

    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, ids = batch
                all_ids.extend([str(i) for i in ids])
            else:
                x, y = batch
                all_ids.extend([str(i) for i in range(len(y))])
            x = x.to(target_device)
            logits = model(x)
            scaled = logits / max(float(temperature), 1e-4)
            probs = F.softmax(scaled, dim=-1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

            if extract_embeddings and hasattr(model, "extract_embeddings"):
                all_embs.append(model.extract_embeddings(x).cpu().numpy())

    total_sec = time.perf_counter() - t0
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 5), dtype=np.float32)
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 5), dtype=np.float32)
    labels_arr = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
    n_items = max(len(labels_arr), 1)

    return {
        "logits": logits_arr,
        "probabilities": probs_arr,
        "predictions": np.argmax(probs_arr, axis=1) if len(probs_arr) else np.empty((0,), dtype=np.int64),
        "labels": labels_arr,
        "image_ids": all_ids,
        "embeddings": np.concatenate(all_embs, axis=0) if all_embs else None,
        "total_seconds": round(float(total_sec), 3),
        "ms_per_image": round(float(total_sec * 1000.0 / n_items), 2),
    }
