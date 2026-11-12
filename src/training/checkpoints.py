"""Reproducible checkpoint persistence and model reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config import DR_CLASS_NAMES, resolve_path
from src.models.classifier import build_model


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    model_name: str,
    model_kwargs: dict[str, Any] | None = None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    seed: int = 42,
    temperature: float = 1.0,
    conformal_qhat: float | None = None,
    preprocessing_preset: str = "fov_clahe_norm",
    image_size: int = 224,
    dataset_name: str = "unknown",
    is_synthetic: bool = False,
) -> Path:
    """Save a reproducible model checkpoint bundle including provenance and calibration params."""
    out_path = resolve_path(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_kwargs": model_kwargs or {},
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "config": config or {},
        "seed": int(seed),
        "class_names": DR_CLASS_NAMES,
        "temperature": float(temperature),
        "conformal_qhat": float(conformal_qhat) if conformal_qhat is not None else None,
        "preprocessing_preset": preprocessing_preset,
        "image_size": int(image_size),
        "dataset_name": dataset_name,
        "is_synthetic": bool(is_synthetic),
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    model_override: nn.Module | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load checkpoint payload and reconstruct model weights safely on target device."""
    ckpt_path = resolve_path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=torch.device(device))
    if model_override is not None:
        model = model_override
    else:
        model_name = str(payload["model_name"])
        model_kwargs = dict(payload.get("model_kwargs") or {})
        cfg = payload.get("config") or {}
        # Never force network download when loading saved checkpoint weights!
        if model_name != "custom_cnn":
            model_kwargs.setdefault("pretrained_override", False)
        model = build_model(model_name, cfg=cfg, **model_kwargs)

    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, payload


def list_checkpoints(checkpoints_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Discover saved local `.pt` checkpoints with lightweight metadata inspection."""
    base = resolve_path(checkpoints_dir or "checkpoints")
    if not base.exists():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.pt")):
        try:
            meta = torch.load(path, map_location="cpu")
            entries.append(
                {
                    "name": path.stem,
                    "path": str(path),
                    "model_name": meta.get("model_name", "unknown"),
                    "dataset_name": meta.get("dataset_name", "unknown"),
                    "is_synthetic": bool(meta.get("is_synthetic", False)),
                    "epoch": int(meta.get("epoch", 0)),
                    "image_size": int(meta.get("image_size", 224)),
                    "preprocessing_preset": meta.get("preprocessing_preset", "fov_clahe_norm"),
                    "temperature": float(meta.get("temperature", 1.0)),
                    "metrics": meta.get("metrics", {}),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                }
            )
        except Exception as exc:
            entries.append({"name": path.stem, "path": str(path), "error": str(exc)})
    return entries
