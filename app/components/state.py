"""Shared state helpers and end-to-end 10-step inference workflow for Gradio UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DR_CLASS_NAMES, resolve_path
from src.data.manifests import load_manifest
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.integrated_gradients import compute_integrated_gradients, compute_occlusion_sensitivity
from src.explainability.visualisation import EXPLAINABILITY_DISCLAIMER, compute_fov_attribution_ratio, overlay_heatmap
from src.models.inference import predict_single_image
from src.preprocessing.fundus_processing import inspect_fundus_image, load_rgb_array, preprocess_fundus_image
from src.preprocessing.quality_assessment import assess_fundus_quality
from src.training.checkpoints import list_checkpoints, load_checkpoint
from src.uncertainty.conformal import SplitConformalPredictor
from src.uncertainty.mc_dropout import mc_dropout_predict_single
from src.uncertainty.selective_prediction import evaluate_selective_decision


def get_available_manifests() -> list[str]:
    m_dir = resolve_path("data/manifests")
    if not m_dir.exists():
        return []
    return sorted(p.stem.replace("_manifest", "") for p in m_dir.glob("*_manifest.csv"))


def load_active_manifest(manifest_name: str | None = None) -> pd.DataFrame:
    names = get_available_manifests()
    if not names:
        return pd.DataFrame()
    chosen = manifest_name if manifest_name in names else names[0]
    return load_manifest(chosen)


def get_checkpoint_choices() -> list[tuple[str, str]]:
    """Return list of `(display_label, path)` pairs for Gradio Dropdowns."""
    ckpts = list_checkpoints()
    choices: list[tuple[str, str]] = []
    for c in ckpts:
        if "error" in c:
            continue
        tag = " [SYNTHETIC]" if c.get("is_synthetic") else ""
        qwk = c.get("metrics", {}).get("val_qwk", "N/A")
        label = f"{c['name']} ({c['model_name']}, val_QWK={qwk}){tag}"
        choices.append((label, c["path"]))
    return choices


def run_full_clinical_research_inference(
    image_arr: np.ndarray | str | Path | None,
    checkpoint_path: str | None,
    preprocess_preset: str = "fov_clahe_norm",
    xai_method: str = "gradcam",
    mc_passes: int = 10,
    min_confidence: float = 0.45,
    max_entropy: float = 1.25,
) -> dict[str, Any]:
    """Execute the full 10-step trustworthy inference workflow on a user/dataset image."""
    if image_arr is None:
        return {"ok": False, "error": "Please upload or select a retinal fundus image."}

    # 1 & 2: Validate and inspect image
    try:
        rgb = load_rgb_array(image_arr)
        if rgb.shape[0] < 32 or rgb.shape[1] < 32:
            return {"ok": False, "error": f"Image dimensions {rgb.shape[1]}x{rgb.shape[0]} below minimum 32x32."}
    except Exception as exc:
        return {"ok": False, "error": f"Invalid or corrupt image input: {exc}"}

    inspection = inspect_fundus_image(rgb)
    # 3: Quality assessment
    qa = assess_fundus_quality(rgb)

    if not checkpoint_path or not Path(checkpoint_path).is_file():
        return {
            "ok": False,
            "error": (
                "No trained model checkpoint selected or available. "
                "Train a model in the Custom CNN / Transfer Learning tabs or run:\n"
                "`python scripts/prepare_data.py --synthetic-demo && python scripts/train.py --model custom_cnn`"
            ),
            "inspection": inspection,
            "quality": qa,
        }

    # Load checkpoint cleanly
    model, ckpt_meta = load_checkpoint(checkpoint_path, device="cpu")
    img_size = int(ckpt_meta.get("image_size", 224))
    temperature = float(ckpt_meta.get("temperature", 1.0))

    # 4 & 5 & 6: Preprocess & deterministic inference
    det_res = predict_single_image(
        model=model,
        image_source=rgb,
        preprocess_preset=preprocess_preset,
        image_size=img_size,
        temperature=temperature,
    )
    proc_img = det_res["preprocessed_image"]
    input_tensor = det_res["input_tensor"]

    # 7: MC Dropout uncertainty & Conformal Prediction Set
    mc_res = mc_dropout_predict_single(
        model=model,
        input_tensor=input_tensor,
        n_passes=int(mc_passes),
        temperature=temperature,
    )
    conformal_info = None
    q_hat = ckpt_meta.get("conformal_qhat")
    if q_hat is not None:
        cp = SplitConformalPredictor(alpha=0.10, method="aps")
        cp.q_hat = float(q_hat)
        conformal_info = cp.predict_single_set(mc_res["mean_probabilities"])

    # 8: Explainability visualization
    method_key = xai_method.lower()
    if method_key in {"gradcam", "gradcam_pp"}:
        explainer = GradCAMExplainer(model)
        xai_res = explainer.generate(input_tensor, target_class=det_res["predicted_class"], method=method_key)
    elif method_key == "integrated_gradients":
        xai_res = compute_integrated_gradients(model, input_tensor, target_class=det_res["predicted_class"], n_steps=16)
    elif method_key == "occlusion":
        xai_res = compute_occlusion_sensitivity(model, input_tensor, target_class=det_res["predicted_class"])
    else:
        explainer = GradCAMExplainer(model)
        xai_res = explainer.generate(input_tensor, target_class=det_res["predicted_class"], method="gradcam")

    heatmap_rgb, overlay_rgb = overlay_heatmap(proc_img, xai_res["heatmap"], alpha=0.45)
    fov_mass = compute_fov_attribution_ratio(proc_img, xai_res["heatmap"])

    # 9: Abstention logic
    abstention = evaluate_selective_decision(
        probabilities=mc_res["mean_probabilities"],
        predictive_entropy=mc_res["predictive_entropy"],
        conformal_set_size=(conformal_info["set_size"] if conformal_info else None),
        quality_score=qa["quality_score"],
        min_confidence=float(min_confidence),
        max_entropy=float(max_entropy),
    )

    return {
        "ok": True,
        "checkpoint_meta": ckpt_meta,
        "inspection": inspection,
        "quality": qa,
        "preprocessed_image": proc_img,
        "deterministic": det_res,
        "mc_dropout": mc_res,
        "conformal": conformal_info,
        "abstention": abstention,
        "heatmap_rgb": heatmap_rgb,
        "overlay_rgb": overlay_rgb,
        "fov_attribution_ratio": fov_mass,
        "xai_disclaimer": EXPLAINABILITY_DISCLAIMER,
        "class_names": DR_CLASS_NAMES,
    }
