"""Execute reproducible RetinaX research experiment suites (Experiments A-J)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.manifests import load_manifest
from src.experiments.ablation import run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinaX Ablation & Research Suite Runner")
    parser.add_argument("--manifest", type=str, default="synthetic_primary")
    parser.add_argument("--external-manifest", type=str, default="synthetic_external")
    parser.add_argument("--experiments", nargs="+", default=["A", "C", "E", "F"],
                        help="Subset of experiment keys to run (A, C, E, F)")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(preset_path="configs/lightweight_cpu.yaml")
    manifest_df = load_manifest(args.manifest)
    try:
        ext_df = load_manifest(args.external_manifest)
    except FileNotFoundError:
        ext_df = None

    results = run_experiment_suite(
        manifest_df=manifest_df,
        cfg=cfg,
        external_manifest_df=ext_df,
        experiments_to_run=tuple(args.experiments),
        pretrained_override=(False if args.no_pretrained else None),
        epochs=args.epochs,
        image_size=args.image_size,
        seed=args.seed,
    )
    compact = [
        {
            "run_id": r["run_id"],
            "acc": r["metrics"]["test"]["accuracy"],
            "qwk": r["metrics"]["test"]["quadratic_weighted_kappa"],
            "macro_f1": r["metrics"]["test"]["macro_f1"],
            "ece": r["metrics"]["calibration_after"]["ece"],
        }
        for r in results
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
