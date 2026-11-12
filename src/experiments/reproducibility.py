"""Deterministic seeding, configuration hashing, and code revision capture."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from typing import Any

import numpy as np
import torch

from src.config import PROJECT_ROOT


def set_global_seed(seed: int = 42, deterministic_algorithms: bool = False) -> None:
    """Fix RNG seeds across Python, NumPy, and PyTorch for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def compute_config_hash(cfg: dict[str, Any]) -> str:
    """Compute stable SHA-256 prefix over an experiment configuration dictionary."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def get_git_revision() -> str:
    """Capture current git short commit SHA if available, without failing outside git."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "uncommitted"
    except Exception:
        return "uncommitted"
