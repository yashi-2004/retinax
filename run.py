"""Root Gradio entrypoint for Hugging Face Spaces, Docker, and cloud CPU hosting."""

from __future__ import annotations

import os
from pathlib import Path

from src.config import load_config, resolve_path
from src.data.ingestion import load_raw_records
from src.data.manifests import build_manifest, save_manifest
from src.data.splitting import assign_splits
from src.data.synthetic import generate_synthetic_demo_dataset
from src.preprocessing.quality_assessment import assess_fundus_quality
from app.main import create_app


def _bootstrap_demo_if_empty() -> None:
    """Populate clearly-labelled synthetic demo manifests on a fresh cloud/container launch."""
    manifests_dir = resolve_path("data/manifests")
    if manifests_dir.exists() and any(manifests_dir.glob("*_manifest.csv")):
        return

    cfg = load_config()
    generate_synthetic_demo_dataset(num_primary=60, num_external=30, seed=42)
    max_hamming = int(cfg.get("dataset", {}).get("near_duplicate_hamming_threshold", 4))
    proc_cfg = cfg.get("preprocessing", {})

    raw_primary = load_raw_records("synthetic_primary", seed=42)
    p_df, p_sum = build_manifest(raw_primary, processing_config=proc_cfg, quality_fn=assess_fundus_quality, max_hamming=max_hamming)
    sp = cfg.get("dataset", {}).get("splits", {})
    p_df, audit = assign_splits(
        p_df,
        train_ratio=float(sp.get("train_ratio", 0.65)),
        val_ratio=float(sp.get("val_ratio", 0.125)),
        cal_ratio=float(sp.get("cal_ratio", 0.125)),
        test_ratio=float(sp.get("test_ratio", 0.10)),
        seed=42,
    )
    p_sum["split_audit"] = audit
    save_manifest(p_df, name="synthetic_primary", summary=p_sum)

    raw_ext = load_raw_records("synthetic_external", seed=42)
    e_df, e_sum = build_manifest(raw_ext, processing_config=proc_cfg, quality_fn=assess_fundus_quality, max_hamming=max_hamming)
    e_df["split"] = "external_val"
    e_df.loc[e_df["is_corrupt"].astype(bool), "split"] = "excluded_corrupt"
    e_sum["split_audit"] = {"external_validation_only": True, "count": int((e_df["split"] == "external_val").sum())}
    save_manifest(e_df, name="synthetic_external", summary=e_sum)


_bootstrap_demo_if_empty()
demo = create_app()

if __name__ == "__main__":
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=host, server_port=port, share=False, show_error=True)
