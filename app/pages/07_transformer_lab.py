"""Vision Transformer (ViT) Laboratory: patchification, attention blocks, and CPU probing."""

from __future__ import annotations

import json

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.components.state import get_available_manifests, load_active_manifest
from src.config import load_config
from src.data.caching import normalize_rgb_tensor
from src.experiments.ablation import run_supervised_experiment
from src.models.vision_transformer import ViTClassifier
from src.preprocessing.fundus_processing import load_rgb_array, preprocess_fundus_image
from src.reporting.figures import plot_training_curves

TAB_TITLE = "7. Vision Transformer Lab"


def _inspect_vit_architecture(backbone_name: str, mode: str, pretrained: bool) -> str:
    try:
        model = ViTClassifier(backbone_name=backbone_name, mode=mode, pretrained=pretrained)
        return f"```json\n{json.dumps(model.architecture_summary(), indent=2)}\n```"
    except Exception as exc:
        return f"**Offline / Weight Cache Notice:** {exc}"


def _run_vit_experiment(
    manifest_name: str | None,
    backbone_name: str,
    mode: str,
    pretrained: bool,
    epochs: int,
) -> tuple[str, plt.Figure]:
    df = load_active_manifest(manifest_name)
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No manifest loaded.", ha="center", va="center")
        ax.axis("off")
        return "No active manifest loaded.", fig

    cfg = load_config(preset_path="configs/lightweight_cpu.yaml")
    model_key = "vit_tiny" if "vit_tiny" in backbone_name else ("deit_tiny" if "deit_tiny" in backbone_name else "vit_small")
    try:
        res = run_supervised_experiment(
            manifest_df=df,
            model_name=model_key,
            run_id=f"ui_{model_key}_{mode}_e{int(epochs)}",
            experiment_group="UI_Vision_Transformer",
            cfg=cfg,
            mode_override=mode,
            pretrained_override=bool(pretrained),
            epochs=int(epochs),
            image_size=224,
        )
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"ViT execution halted:\n{exc}", ha="center", va="center")
        ax.axis("off")
        return f"**Notice:** {exc}", fig

    fig = plot_training_curves(res["history"], title=f"Vision Transformer ({backbone_name}, {mode})")
    m = res["metrics"]
    msg = (
        f"ViT experiment `{res['run_id']}` finished in **{m['training_seconds']}s**.\n"
        f"- **Test Accuracy:** `{m['test']['accuracy']}` | **QWK:** `{m['test']['quadratic_weighted_kappa']}` | "
        f"**Latency:** `{m['inference_ms_per_image']} ms/img`"
    )
    return msg, fig


def _visualize_patch_similarity(
    uploaded_img: np.ndarray | None,
    backbone_name: str,
    pretrained: bool,
) -> tuple[np.ndarray | None, plt.Figure, str]:
    if uploaded_img is None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Upload an image to inspect CLS-Patch cosine map.", ha="center", va="center")
        ax.axis("off")
        return None, fig, "Please upload a retinal image."

    try:
        model = ViTClassifier(backbone_name=backbone_name, mode=("frozen" if pretrained else "scratch"), pretrained=pretrained)
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, str(exc), ha="center", va="center")
        ax.axis("off")
        return None, fig, str(exc)

    rgb = load_rgb_array(uploaded_img)
    proc = preprocess_fundus_image(rgb, preset="fov_clahe_norm", image_size=224)
    tensor = normalize_rgb_tensor(proc, mode="imagenet").unsqueeze(0)
    with np.errstate(all="ignore"):
        sim_grid = model.extract_cls_patch_similarity(tensor).squeeze(0).detach().numpy()

    fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=120)
    im = ax.imshow(sim_grid, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"CLS-to-Patch Similarity Grid ({sim_grid.shape[0]}x{sim_grid.shape[1]})", fontsize=9.5)
    ax.axis("off")
    plt.tight_layout()
    return proc, fig, f"Extracted {sim_grid.shape[0]}x{sim_grid.shape[1]} = {sim_grid.size} patch tokens (16x16px each)."


def build_tab() -> None:
    manifests = get_available_manifests()
    gr.Markdown(
        "### Vision Transformer (ViT) Research Laboratory\n"
        "Investigates patchification, positional embeddings, global self-attention, and CPU latency vs CNNs."
    )

    with gr.Row():
        with gr.Column():
            bb_dd = gr.Dropdown(
                choices=["vit_tiny_patch16_224", "deit_tiny_patch16_224", "vit_small_patch16_224"],
                value="vit_tiny_patch16_224",
                label="ViT Backbone",
            )
            mode_dd = gr.Dropdown(choices=["frozen", "partial_finetune", "scratch"], value="frozen", label="Training Regime")
            pre_chk = gr.Checkbox(value=False, label="Load Pretrained timm Weights (Requires local cache or internet)")
            inspect_btn = gr.Button("Inspect Patchification & Transformer Specs", variant="secondary")
            spec_md = gr.Markdown(_inspect_vit_architecture("vit_tiny_patch16_224", "scratch", False))

        with gr.Column():
            manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Manifest")
            epochs_sl = gr.Slider(1, 10, value=2, step=1, label="Epochs (CPU-Friendly)")
            run_btn = gr.Button("Run ViT Experiment", variant="primary")
            status_md = gr.Markdown("")
            curves_plot = gr.Plot(label="ViT Training Curves")

    gr.Markdown("#### Patch Token Spatial Similarity Preview")
    with gr.Row():
        patch_img = gr.Image(type="numpy", label="Upload Fundus Image")
        patch_btn = gr.Button("Visualize Patch Token Grid", variant="secondary")
        proc_out = gr.Image(type="numpy", label="224x224 Preprocessed Input")
        patch_plot = gr.Plot(label="CLS-Patch Cosine Map")
    patch_md = gr.Markdown("")

    inspect_btn.click(fn=_inspect_vit_architecture, inputs=[bb_dd, mode_dd, pre_chk], outputs=[spec_md])
    run_btn.click(fn=_run_vit_experiment, inputs=[manifest_dd, bb_dd, mode_dd, pre_chk, epochs_sl], outputs=[status_md, curves_plot])
    patch_btn.click(fn=_visualize_patch_similarity, inputs=[patch_img, bb_dd, pre_chk], outputs=[proc_out, patch_plot, patch_md])
