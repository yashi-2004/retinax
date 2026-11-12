"""CLI training entrypoint for supervised models and SimCLR self-supervised pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.manifests import load_manifest
from src.experiments.ablation import run_supervised_experiment
from src.foundation.ssl_simclr import SimCLREncoderModel, train_simclr_ssl


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinaX Reproducible Model Trainer")
    parser.add_argument("--manifest", type=str, default="synthetic_primary", help="Manifest name or path")
    parser.add_argument("--external-manifest", type=str, default=None, help="Optional external validation manifest")
    parser.add_argument("--model", type=str, default="custom_cnn",
                        help="custom_cnn, resnet18, efficientnet_b0, densenet121, vit_tiny, simclr")
    parser.add_argument("--mode", type=str, default=None, help="scratch, frozen, partial_finetune, full_finetune")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet weight download")
    parser.add_argument("--preset-cfg", type=str, default="configs/lightweight_cpu.yaml")
    parser.add_argument("--preprocess-preset", type=str, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(preset_path=args.preset_cfg if args.preset_cfg else None)
    manifest_df = load_manifest(args.manifest)
    ext_df = load_manifest(args.external_manifest) if args.external_manifest else None

    if args.model.lower() == "simclr":
        train_df = manifest_df[manifest_df["split"] == "train"]
        ssl_cfg = cfg.get("models", {}).get("simclr", {})
        img_sz = int(args.image_size or cfg.get("preprocessing", {}).get("image_size", 128))
        ssl_model = SimCLREncoderModel(
            encoder_type=str(ssl_cfg.get("encoder", "custom_cnn")),
            projection_dim=int(ssl_cfg.get("projection_dim", 64)),
        )
        res = train_simclr_ssl(
            ssl_model=ssl_model,
            unlabelled_df=train_df,
            epochs=int(args.epochs or ssl_cfg.get("ssl_epochs", 5)),
            batch_size=int(args.batch_size or ssl_cfg.get("batch_size", 16)),
            image_size=img_sz,
            temperature=float(ssl_cfg.get("temperature", 0.5)),
            run_id=args.run_id or f"simclr_s{args.seed}",
            cfg=cfg,
        )
        print(json.dumps(res, indent=2))
        return

    res = run_supervised_experiment(
        manifest_df=manifest_df,
        model_name=args.model,
        run_id=args.run_id,
        experiment_group="cli_train",
        cfg=cfg,
        mode_override=args.mode,
        pretrained_override=(False if args.no_pretrained else None),
        external_manifest_df=ext_df,
        preprocess_preset=args.preprocess_preset,
        loss_name=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )
    summary = {
        "run_id": res["run_id"],
        "checkpoint_path": res["checkpoint_path"],
        "test_accuracy": res["metrics"]["test"]["accuracy"],
        "test_qwk": res["metrics"]["test"]["quadratic_weighted_kappa"],
        "test_macro_f1": res["metrics"]["test"]["macro_f1"],
        "ece_before": res["metrics"]["calibration_before"]["ece"],
        "ece_after": res["metrics"]["calibration_after"]["ece"],
        "conformal_coverage": res["metrics"]["conformal_test"]["empirical_coverage"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
