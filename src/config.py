"""Central configuration, constants, and hardware safety utilities for RetinaX."""

from __future__ import annotations

import copy
import os
import platform
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DR_CLASS_NAMES: dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

DR_CLASS_LIST: list[str] = [DR_CLASS_NAMES[i] for i in range(5)]

CLINICAL_DISCLAIMER: str = (
    "RetinaX is an academic research prototype. It is not a clinically validated "
    "diagnostic system and must not be used to diagnose disease, determine treatment "
    "or replace professional ophthalmic assessment."
)

RESEARCH_TITLE: str = (
    "Towards Reliable Retinal Disease Grading: A Comparative Study of Convolutional "
    "Neural Networks, Self-Supervised Representations and Vision Foundation Models "
    "under Dataset Shift"
)

CENTRAL_RESEARCH_QUESTION: str = (
    "Can data-efficient deep learning and modern visual foundation-model representations "
    "improve retinal disease grading while maintaining reliable uncertainty estimates "
    "and generalisation across imaging datasets?"
)

SUPPORTING_RESEARCH_QUESTIONS: list[str] = [
    "How do conventional CNNs compare with pretrained CNNs and Vision Transformers?",
    "Does transfer learning improve performance when labelled training data is limited?",
    "Can self-supervised visual representations improve downstream classification?",
    "Do pretrained foundation-model embeddings generalise better than task-specific CNN representations?",
    "How does image-quality variation affect classification performance?",
    "Can Grad-CAM identify meaningful retinal regions associated with model predictions?",
    "Can uncertainty estimation identify potentially unreliable predictions?",
    "How well do models generalise to an external retinal imaging dataset?",
    "Does calibration improve the reliability of model confidence?",
    "Can a model abstain from making predictions when confidence is insufficient?",
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    config_path: str | Path | None = None,
    preset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load default YAML configuration and optionally merge a preset override."""
    default_file = PROJECT_ROOT / "configs" / "default.yaml"
    with default_file.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if config_path is not None:
        custom_file = Path(config_path)
        if not custom_file.is_absolute():
            custom_file = PROJECT_ROOT / custom_file
        with custom_file.open("r", encoding="utf-8") as handle:
            custom_cfg = yaml.safe_load(handle) or {}
        cfg = _deep_merge(cfg, custom_cfg)

    if preset_path is not None:
        preset_file = Path(preset_path)
        if not preset_file.is_absolute():
            preset_file = PROJECT_ROOT / preset_file
        with preset_file.open("r", encoding="utf-8") as handle:
            preset_cfg = yaml.safe_load(handle) or {}
        cfg = _deep_merge(cfg, preset_cfg)

    return cfg


def resolve_path(relative_or_abs: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(relative_or_abs)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device_and_threads(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Determine safe execution device (CPU-first) and thread count."""
    import torch

    requested = "auto"
    num_threads = 4
    if cfg:
        requested = str(cfg.get("project", {}).get("device", "auto")).lower()
        num_threads = int(cfg.get("project", {}).get("num_threads", 4))

    cpu_count = os.cpu_count() or 2
    safe_threads = max(1, min(num_threads, cpu_count))
    torch.set_num_threads(safe_threads)

    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        amp_supported = True
    elif requested == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
        amp_supported = True
    else:
        device = torch.device("cpu")
        amp_supported = False

    return {
        "device": device,
        "device_str": str(device),
        "amp_supported": amp_supported,
        "num_threads": safe_threads,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def get_hardware_summary() -> dict[str, Any]:
    """Collect a lightweight hardware/runtime snapshot for reports and UI."""
    import torch

    summary = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_cores": os.cpu_count() or 1,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "execution_mode": "CPU-only" if not torch.cuda.is_available() else "CUDA",
        "mixed_precision_enabled": bool(torch.cuda.is_available()),
    }
    return summary
