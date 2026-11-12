"""CLI script to ingest, validate, quality-assess, split, and manifest retinal datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.ingestion import discover_datasets, load_raw_records
from src.data.manifests import build_manifest, save_manifest
from src.data.splitting import assign_splits
from src.data.synthetic import generate_synthetic_demo_dataset
from src.preprocessing.quality_assessment import assess_fundus_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinaX Data Ingestion, QA & Leakage-Free Splitting")
    parser.add_argument("--dataset", type=str, default="auto",
                        help="Dataset key: 'auto', 'aptos', 'eyepacs', 'synthetic_primary', 'synthetic_external'")
    parser.add_argument("--synthetic-demo", action="store_true",
                        help="Generate clearly-labelled synthetic demonstration phantoms locally.")
    parser.add_argument("--num-primary", type=int, default=100, help="Primary synthetic phantom count.")
    parser.add_argument("--num-external", type=int, default=50, help="External shifted synthetic phantom count.")
    parser.add_argument("--max-samples", type=int, default=None, help="Bounded stratified sample cap.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    if args.synthetic_demo:
        syn_res = generate_synthetic_demo_dataset(
            num_primary=args.num_primary,
            num_external=args.num_external,
            seed=args.seed,
        )
        print("Generated synthetic demo datasets (NON-CLINICAL):", json.dumps(syn_res, indent=2))

    status = discover_datasets()
    ds_name = args.dataset
    if ds_name == "auto":
        if status["aptos"]["available"]:
            ds_name = "aptos"
        elif status["synthetic_demo"]["available"]:
            ds_name = "synthetic_primary"
        else:
            generate_synthetic_demo_dataset(num_primary=args.num_primary, num_external=args.num_external, seed=args.seed)
            ds_name = "synthetic_primary"

    print(f"Preparing manifest for dataset='{ds_name}'...")
    raw_df = load_raw_records(ds_name, max_samples=args.max_samples, seed=args.seed)
    manifest_df, summary = build_manifest(
        raw_df,
        processing_config=cfg.get("preprocessing", {}),
        quality_fn=assess_fundus_quality,
        max_hamming=int(cfg.get("dataset", {}).get("near_duplicate_hamming_threshold", 4)),
    )

    if ds_name in ("eyepacs", "synthetic_external"):
        manifest_df["split"] = "external_val"
        manifest_df.loc[manifest_df["is_corrupt"].astype(bool), "split"] = "excluded_corrupt"
        audit = {"external_validation_only": True, "count": int((manifest_df["split"] == "external_val").sum())}
    else:
        sp = cfg.get("dataset", {}).get("splits", {})
        manifest_df, audit = assign_splits(
            manifest_df,
            train_ratio=float(sp.get("train_ratio", 0.65)),
            val_ratio=float(sp.get("val_ratio", 0.125)),
            cal_ratio=float(sp.get("cal_ratio", 0.125)),
            test_ratio=float(sp.get("test_ratio", 0.10)),
            seed=args.seed,
        )

    summary["split_audit"] = audit
    csv_path, meta_path = save_manifest(manifest_df, name=ds_name, summary=summary)
    print(f"Saved manifest: {csv_path}")
    print(f"Saved summary:  {meta_path}")
    print("Split audit:", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
