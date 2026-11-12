"""Explainable AI Laboratory: Grad-CAM, Grad-CAM++, Integrated Gradients, and Occlusion."""

from __future__ import annotations

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.components.state import (
    get_available_manifests,
    get_checkpoint_choices,
    load_active_manifest,
    run_full_clinical_research_inference,
)
from src.explainability.visualisation import EXPLAINABILITY_DISCLAIMER, build_explanation_figure

TAB_TITLE = "10. Explainability Lab"


def _run_xai_lab(
    uploaded_img: np.ndarray | None,
    manifest_name: str | None,
    sample_id: str | None,
    ckpt_path: str | None,
    method: str,
    preset: str,
) -> tuple[plt.Figure, str]:
    source = uploaded_img
    actual_cls = None
    if source is None and manifest_name and sample_id:
        df = load_active_manifest(manifest_name)
        hit = df[df["image_id"].astype(str) == str(sample_id)]
        if len(hit) > 0:
            source = str(hit.iloc[0]["abs_path"])
            actual_cls = int(hit.iloc[0]["original_label"])

    out = run_full_clinical_research_inference(
        image_arr=source,
        checkpoint_path=ckpt_path,
        preprocess_preset=preset,
        xai_method=method,
        mc_passes=6,
    )
    if not out.get("ok"):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, str(out.get("error")), ha="center", va="center", wrap=True)
        ax.axis("off")
        return fig, f"**Notice:** {out.get('error')}"

    det = out["deterministic"]
    # Build visual panel using computed overlay & heatmap
    # Recover normalized [0,1] heatmap intensity from overlay/heatmap or regenerate cleanly
    from src.explainability.gradcam import GradCAMExplainer
    from src.explainability.integrated_gradients import compute_integrated_gradients, compute_occlusion_sensitivity
    from src.training.checkpoints import load_checkpoint

    model, _ = load_checkpoint(ckpt_path, device="cpu")
    tensor = det["input_tensor"]
    m_key = method.lower()
    if m_key in {"gradcam", "gradcam_pp"}:
        hm = GradCAMExplainer(model).generate(tensor, target_class=det["predicted_class"], method=m_key)["heatmap"]
    elif m_key == "integrated_gradients":
        hm = compute_integrated_gradients(model, tensor, target_class=det["predicted_class"], n_steps=16)["heatmap"]
    else:
        hm = compute_occlusion_sensitivity(model, tensor, target_class=det["predicted_class"])["heatmap"]

    fig = build_explanation_figure(
        image_rgb=out["preprocessed_image"],
        heatmap=hm,
        probabilities=det["probabilities"],
        predicted_class=det["predicted_class"],
        actual_class=actual_cls,
        method_label=method.upper(),
    )

    msg = (
        f"**Attribution FOV Concentration:** `{out['fov_attribution_ratio']*100:.1f}%` of attribution mass falls inside the retinal FOV.\n\n"
        f"> **Post-Hoc Explainability Caveat:** {EXPLAINABILITY_DISCLAIMER}"
    )
    return fig, msg


def build_tab() -> None:
    manifests = get_available_manifests()
    ckpts = get_checkpoint_choices()
    gr.Markdown(f"### Explainable AI Laboratory\n> {EXPLAINABILITY_DISCLAIMER}")

    with gr.Row():
        with gr.Column():
            upload_img = gr.Image(type="numpy", label="Upload Retinal Image")
            manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Or Pick Manifest")
            sample_dd = gr.Dropdown(choices=[], label="Sample ID")
        with gr.Column():
            ckpt_dd = gr.Dropdown(choices=ckpts, value=(ckpts[0][1] if ckpts else None), label="Trained Checkpoint")
            refresh_btn = gr.Button("Refresh Checkpoints")
            method_dd = gr.Dropdown(
                choices=["gradcam", "gradcam_pp", "integrated_gradients", "occlusion"],
                value="gradcam",
                label="Attribution Method",
            )
            preset_dd = gr.Dropdown(
                choices=["raw_resize", "fov_crop", "fov_clahe_norm", "ben_graham"],
                value="fov_clahe_norm",
                label="Preprocessing",
            )
            run_btn = gr.Button("Generate Attribution Explanation", variant="primary")

    xai_plot = gr.Plot(label="Visual Explanation Panel")
    xai_md = gr.Markdown("")

    def _update_samples(m_name: str):
        df = load_active_manifest(m_name)
        ids = df[~df["is_corrupt"].astype(bool)]["image_id"].astype(str).head(50).tolist() if not df.empty else []
        return gr.update(choices=ids, value=(ids[0] if ids else None))

    def _refresh_ckpts():
        ch = get_checkpoint_choices()
        return gr.update(choices=ch, value=(ch[0][1] if ch else None))

    manifest_dd.change(fn=_update_samples, inputs=[manifest_dd], outputs=[sample_dd])
    refresh_btn.click(fn=_refresh_ckpts, inputs=[], outputs=[ckpt_dd])
    run_btn.click(
        fn=_run_xai_lab,
        inputs=[upload_img, manifest_dd, sample_dd, ckpt_dd, method_dd, preset_dd],
        outputs=[xai_plot, xai_md],
    )
