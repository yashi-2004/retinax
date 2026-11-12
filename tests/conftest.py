"""Shared pytest fixtures for RetinaX unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.data.ingestion import load_raw_records
from src.data.manifests import build_manifest
from src.data.splitting import assign_splits
from src.data.synthetic import generate_synthetic_demo_dataset
from src.preprocessing.quality_assessment import assess_fundus_quality


@pytest.fixture(scope="session")
def synthetic_workspace(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Create a temporary synthetic primary + external cohort for fast CPU testing."""
    base = tmp_path_factory.mktemp("retinax_syn")
    syn_dir = base / "synthetic_demo"
    generate_synthetic_demo_dataset(
        output_dir=syn_dir,
        num_primary=35,
        num_external=15,
        image_size=128,
        seed=42,
    )

    # Emulate load_raw_records over tmp directory by pointing raw_root=base
    raw_primary = load_raw_records("synthetic_primary", raw_root=base, seed=42)
    raw_external = load_raw_records("synthetic_external", raw_root=base, seed=42)

    cfg = load_config(preset_path="configs/lightweight_cpu.yaml")
    manifest_p, _ = build_manifest(raw_primary, processing_config=cfg["preprocessing"], quality_fn=assess_fundus_quality)
    manifest_p, audit_p = assign_splits(manifest_p, seed=42)

    manifest_e, _ = build_manifest(raw_external, processing_config=cfg["preprocessing"], quality_fn=assess_fundus_quality)
    manifest_e["split"] = "external_val"

    return {
        "root": base,
        "cfg": cfg,
        "manifest_primary": manifest_p,
        "manifest_external": manifest_e,
        "split_audit": audit_p,
    }
