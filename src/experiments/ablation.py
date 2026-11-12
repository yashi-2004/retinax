"""Reproducible end-to-end training, calibration, conformal, and domain-shift experiment runners."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.caching import FundusDataset
from src.data.splitting import create_label_efficiency_subsets
from src.evaluation.calibration import TemperatureScaler, apply_temperature_scaling, compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics, compute_qwk
from src.evaluation.generalisation import compare_cohorts_distribution_shift
from src.evaluation.robustness import evaluate_quality_stratified
from src.evaluation.statistical_tests import paired_bootstrap_ci
from src.experiments.reproducibility import set_global_seed
from src.experiments.tracking import ExperimentRegistry
from src.models.classifier import build_model, count_parameters
from src.models.inference import predict_batch
from src.preprocessing.augmentation import build_augmentation_pipeline
from src.preprocessing.fundus_processing import preprocess_fundus_image
from src.training.checkpoints import save_checkpoint
from src.training.trainer import Trainer
from src.uncertainty.conformal import SplitConformalPredictor
from src.uncertainty.selective_prediction import compute_risk_coverage_curve


def run_supervised_experiment(
    manifest_df: pd.DataFrame,
    model_name: str = "custom_cnn",
    run_id: str | None = None,
    experiment_group: str = "baseline",
    cfg: dict[str, Any] | None = None,
    mode_override: str | None = None,
    pretrained_override: bool | None = None,
    train_subset_df: pd.DataFrame | None = None,
    external_manifest_df: pd.DataFrame | None = None,
    preprocess_preset: str | None = None,
    augmentation_policy: str | None = None,
    loss_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    image_size: int | None = None,
    lr: float | None = None,
    seed: int = 42,
    notes: str = "",
) -> dict[str, Any]:
    """Run an end-to-end leakage-free experiment: train -> val select -> cal calibrate -> test evaluate."""
    cfg = cfg or {}
    set_global_seed(seed)

    p_cfg = cfg.get("preprocessing", {})
    t_cfg = cfg.get("training", {})
    u_cfg = cfg.get("uncertainty", {})

    img_size = int(image_size or p_cfg.get("image_size", 224))
    preset = str(preprocess_preset or p_cfg.get("preset", "fov_clahe_norm"))
    aug_pol = str(augmentation_policy or p_cfg.get("augmentation_policy", "light_clinical"))
    norm_mode = str(p_cfg.get("normalisation", "imagenet"))
    ep = int(epochs if epochs is not None else t_cfg.get("epochs", 6))
    bs = int(batch_size if batch_size is not None else t_cfg.get("batch_size", 8))
    loss_fn_name = str(loss_name or t_cfg.get("loss", "weighted_ce"))

    train_df = train_subset_df if train_subset_df is not None else manifest_df[manifest_df["split"] == "train"]
    val_df = manifest_df[manifest_df["split"] == "val"]
    cal_df = manifest_df[manifest_df["split"] == "cal"]
    test_df = manifest_df[manifest_df["split"] == "test"]

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Manifest must contain non-empty `train`, `val`, and `test` splits.")

    def _pre_fn(arr: np.ndarray) -> np.ndarray:
        return preprocess_fundus_image(arr, preset=preset, image_size=img_size)

    aug_fn = build_augmentation_pipeline(policy=aug_pol, image_size=img_size, seed=seed)

    train_ds = FundusDataset(train_df, image_size=img_size, preprocess_fn=_pre_fn, augment_fn=aug_fn, normalisation=norm_mode)
    val_ds = FundusDataset(val_df, image_size=img_size, preprocess_fn=_pre_fn, augment_fn=None, normalisation=norm_mode)
    cal_source_df = cal_df if len(cal_df) >= 5 else val_df
    cal_ds = FundusDataset(cal_source_df, image_size=img_size, preprocess_fn=_pre_fn, augment_fn=None, normalisation=norm_mode)
    test_ds = FundusDataset(test_df, image_size=img_size, preprocess_fn=_pre_fn, augment_fn=None, normalisation=norm_mode)

    model_kwargs: dict[str, Any] = {}
    if mode_override is not None:
        model_kwargs["mode_override"] = mode_override
    if pretrained_override is not None:
        model_kwargs["pretrained_override"] = pretrained_override

    model = build_model(model_name, num_classes=5, cfg=cfg, **model_kwargs)
    complexity = count_parameters(model)

    dataset_name = str(manifest_df["dataset_name"].iloc[0]) if len(manifest_df) else "unknown"
    is_synthetic = bool(manifest_df["is_synthetic"].any()) if "is_synthetic" in manifest_df.columns else False
    eff_run_id = run_id or f"{experiment_group}_{model_name}_{preset}_s{seed}"

    trainer = Trainer(
        model=model,
        model_name=model_name,
        model_kwargs=model_kwargs,
        cfg=cfg,
        run_id=eff_run_id,
    )
    train_summary = trainer.fit(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=ep,
        batch_size=bs,
        lr=lr,
        loss_name=loss_fn_name,
        seed=seed,
        dataset_name=dataset_name,
        preprocessing_preset=preset,
        image_size=img_size,
        is_synthetic=is_synthetic,
    )

    # 1. Calibration & Conformal fitting strictly on held-out calibration split (`cal`)
    cal_loader = DataLoader(cal_ds, batch_size=bs, shuffle=False, num_workers=0)
    cal_out = predict_batch(model, cal_loader, temperature=1.0)
    temp_scaler = TemperatureScaler()
    cal_fit_info = temp_scaler.fit(cal_out["logits"], cal_out["labels"])
    opt_temp = float(cal_fit_info["temperature"])

    cal_probs_scaled = apply_temperature_scaling(cal_out["logits"], opt_temp)
    alpha = float(u_cfg.get("conformal_alpha", 0.10))
    conf_method = str(u_cfg.get("conformal_method", "aps"))
    conformal = SplitConformalPredictor(alpha=alpha, method=conf_method, seed=seed)
    conf_fit_info = conformal.fit(cal_probs_scaled, cal_out["labels"])

    # Update checkpoint with fitted temperature & conformal threshold
    save_checkpoint(
        model=model,
        checkpoint_path=train_summary["checkpoint_path"],
        model_name=model_name,
        model_kwargs=model_kwargs,
        epoch=train_summary["best_epoch"],
        metrics=train_summary["best_val_metrics"],
        config=cfg,
        seed=seed,
        temperature=opt_temp,
        conformal_qhat=conformal.q_hat,
        preprocessing_preset=preset,
        image_size=img_size,
        dataset_name=dataset_name,
        is_synthetic=is_synthetic,
    )

    # 2. Strictly held-out Test evaluation
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_raw = predict_batch(model, test_loader, temperature=1.0, extract_embeddings=True)
    test_probs_uncal = test_raw["probabilities"]
    test_probs_cal = apply_temperature_scaling(test_raw["logits"], opt_temp)
    test_labels = test_raw["labels"]

    test_cls = compute_classification_metrics(test_labels, test_probs_cal)
    cal_before = compute_calibration_metrics(test_labels, test_probs_uncal)
    cal_after = compute_calibration_metrics(test_labels, test_probs_cal)
    conf_eval = conformal.evaluate(test_probs_cal, test_labels)

    entropies = -np.sum(np.clip(test_probs_cal, 1e-8, 1.0) * np.log(np.clip(test_probs_cal, 1e-8, 1.0)), axis=1)
    rc_curve = compute_risk_coverage_curve(test_labels, test_probs_cal, uncertainty_scores=entropies)

    qwk_ci = paired_bootstrap_ci(
        test_labels,
        test_probs_cal,
        metric_fn=lambda yt, pr: compute_qwk(yt, np.argmax(pr, axis=1)),
        n_bootstrap=300,
        seed=seed,
    )

    quality_strat = None
    if "quality_category" in test_df.columns and test_df["quality_category"].notna().all():
        quality_strat = evaluate_quality_stratified(
            test_labels, test_probs_cal, test_df["quality_category"].tolist()
        )

    # 3. Optional External Validation (never tuned on external labels)
    external_summary = None
    if external_manifest_df is not None and len(external_manifest_df) > 0:
        ext_valid = external_manifest_df[~external_manifest_df["is_corrupt"].astype(bool)]
        if len(ext_valid) > 0:
            ext_ds = FundusDataset(ext_valid, image_size=img_size, preprocess_fn=_pre_fn, augment_fn=None, normalisation=norm_mode)
            ext_loader = DataLoader(ext_ds, batch_size=bs, shuffle=False, num_workers=0)
            ext_out = predict_batch(model, ext_loader, temperature=opt_temp, extract_embeddings=True)
            external_summary = compare_cohorts_distribution_shift(
                internal_df=test_df,
                external_df=ext_valid,
                internal_labels=test_labels,
                internal_probs=test_probs_cal,
                external_labels=ext_out["labels"],
                external_probs=ext_out["probabilities"],
                max_image_samples=min(40, len(ext_valid)),
            )
            # Drop large DataFrames from JSON metrics serialization
            external_summary.pop("internal_feature_table", None)
            external_summary.pop("external_feature_table", None)

    metrics_bundle = {
        "model_complexity": complexity,
        "training_seconds": train_summary["training_seconds"],
        "best_epoch": train_summary["best_epoch"],
        "epochs_ran": train_summary["epochs_ran"],
        "inference_ms_per_image": test_raw["ms_per_image"],
        "val_best": train_summary["best_val_metrics"],
        "test": test_cls,
        "test_qwk_bootstrap_95ci": qwk_ci,
        "temperature_scaling": {
            "fitted_temperature": opt_temp,
            "calibration_split_size": len(cal_source_df),
            "calibration_split_name": "cal" if len(cal_df) >= 5 else "val_fallback",
        },
        "calibration_before": cal_before,
        "calibration_after": cal_after,
        "conformal_fit": conf_fit_info,
        "conformal_test": conf_eval,
        "selective_prediction": rc_curve,
        "quality_stratified": quality_strat,
        "external_generalisation": external_summary,
    }

    preds_df = pd.DataFrame(
        {
            "image_id": test_raw["image_ids"],
            "y_true": test_labels,
            "y_pred": np.argmax(test_probs_cal, axis=1),
            "confidence": np.round(np.max(test_probs_cal, axis=1), 5),
            "entropy": np.round(entropies, 5),
            **{f"prob_{c}": np.round(test_probs_cal[:, c], 5) for c in range(test_probs_cal.shape[1])},
        }
    )

    run_config = {
        "run_id": eff_run_id,
        "experiment_group": experiment_group,
        "model_name": model_name,
        "mode_override": mode_override,
        "pretrained_override": pretrained_override,
        "preprocessing_preset": preset,
        "augmentation_policy": aug_pol,
        "loss": loss_fn_name,
        "image_size": img_size,
        "batch_size": bs,
        "epochs": ep,
        "seed": seed,
        "train_n": int(len(train_df)),
        "val_n": int(len(val_df)),
        "cal_n": int(len(cal_source_df)),
        "test_n": int(len(test_df)),
    }

    registry = ExperimentRegistry(cfg.get("paths", {}).get("experiments_dir", "experiments"))
    run_dir = registry.log_run(
        run_id=eff_run_id,
        experiment_group=experiment_group,
        model_name=model_name,
        dataset_name=dataset_name,
        is_synthetic=is_synthetic,
        config=run_config,
        metrics=metrics_bundle,
        history=train_summary["history"],
        predictions_df=preds_df,
        checkpoint_path=train_summary["checkpoint_path"],
        tensorboard_dir=train_summary["tensorboard_dir"],
        notes=notes,
    )

    return {
        "run_id": eff_run_id,
        "run_dir": str(run_dir),
        "checkpoint_path": train_summary["checkpoint_path"],
        "metrics": metrics_bundle,
        "history": train_summary["history"],
        "predictions_df": preds_df,
    }


def run_experiment_suite(
    manifest_df: pd.DataFrame,
    cfg: dict[str, Any],
    external_manifest_df: pd.DataFrame | None = None,
    experiments_to_run: tuple[str, ...] = ("A", "E", "F"),
    pretrained_override: bool | None = None,
    epochs: int = 3,
    image_size: int = 128,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run selected structured research ablations cleanly on identical splits."""
    results: list[dict[str, Any]] = []

    if "A" in experiments_to_run:
        # Experiment A: Custom CNN vs ResNet18 vs EfficientNet-B0
        for m_name in ("custom_cnn", "resnet18", "efficientnet_b0"):
            mode = None if m_name == "custom_cnn" else ("scratch" if pretrained_override is False else "frozen")
            res = run_supervised_experiment(
                manifest_df=manifest_df,
                model_name=m_name,
                run_id=f"expA_{m_name}_s{seed}",
                experiment_group="ExpA_Foundational_Baseline",
                cfg=cfg,
                mode_override=mode,
                pretrained_override=pretrained_override,
                external_manifest_df=external_manifest_df,
                epochs=epochs,
                image_size=image_size,
                seed=seed,
            )
            results.append(res)

    if "C" in experiments_to_run:
        # Experiment C: Label Efficiency (25%, 50%, 100%)
        subsets = create_label_efficiency_subsets(manifest_df, fractions=(0.25, 0.50, 1.00), seed=seed)
        for frac, sub_df in subsets.items():
            pct = int(round(frac * 100))
            res = run_supervised_experiment(
                manifest_df=manifest_df,
                model_name="custom_cnn",
                run_id=f"expC_custom_cnn_pct{pct}_s{seed}",
                experiment_group="ExpC_Label_Efficiency",
                cfg=cfg,
                train_subset_df=sub_df,
                epochs=epochs,
                image_size=image_size,
                seed=seed,
                notes=f"Label-efficiency subset {pct}% ({len(sub_df)} train samples).",
            )
            results.append(res)

    if "E" in experiments_to_run:
        # Experiment E: Preprocessing Ablation
        for preset in ("raw_resize", "fov_crop", "fov_clahe_norm", "ben_graham"):
            res = run_supervised_experiment(
                manifest_df=manifest_df,
                model_name="custom_cnn",
                run_id=f"expE_prep_{preset}_s{seed}",
                experiment_group="ExpE_Preprocessing_Ablation",
                cfg=cfg,
                preprocess_preset=preset,
                epochs=epochs,
                image_size=image_size,
                seed=seed,
            )
            results.append(res)

    if "F" in experiments_to_run:
        # Experiment F: Class Imbalance Loss Ablation
        for loss_k in ("ce", "weighted_ce", "focal"):
            res = run_supervised_experiment(
                manifest_df=manifest_df,
                model_name="custom_cnn",
                run_id=f"expF_loss_{loss_k}_s{seed}",
                experiment_group="ExpF_Class_Imbalance",
                cfg=cfg,
                loss_name=loss_k,
                epochs=epochs,
                image_size=image_size,
                seed=seed,
            )
            results.append(res)

    return results
