"""Transfer Learning & Unified Research Inference Laboratory page."""

from __future__ import annotations

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.components.state import (
    get_available_manifests,
    get_checkpoint_choices,
    load_active_manifest,
    run_full_clinical_research_inference,
)
from src.config import load_config
from src.experiments.ablation import run_supervised_experiment
from src.reporting.figures import plot_training_curves

TAB_TITLE = "6. Transfer Learning & Inference"


def _train_transfer_model(
    manifest_name: str | None,
    arch_name: str,
    mode: str,
    use_pretrained: bool,
    loss_name: str,
    epochs: int,
    image_size: int,
) -> tuple[str, plt.Figure, dict]:
    df = load_active_manifest(manifest_name)
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No manifest loaded.", ha="center", va="center")
        ax.axis("off")
        return "No active manifest found.", fig, gr.update()

    cfg = load_config(preset_path="configs/lightweight_cpu.yaml")
    try:
        res = run_supervised_experiment(
            manifest_df=df,
            model_name=arch_name,
            run_id=f"ui_{arch_name}_{mode}_e{int(epochs)}",
            experiment_group="UI_Transfer_Learning",
            cfg=cfg,
            mode_override=mode,
            pretrained_override=bool(use_pretrained),
            loss_name=loss_name,
            epochs=int(epochs),
            image_size=int(image_size),
        )
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"Training halted:\n{exc}", ha="center", va="center")
        ax.axis("off")
        return f"**Error:** {exc}", fig, gr.update()

    fig = plot_training_curves(res["history"], title=f"{arch_name} ({mode}) Curves")
    m = res["metrics"]
    msg = (
        f"Run `{res['run_id']}` completed in **{m['training_seconds']}s**.\n\n"
        f"- **Test QWK:** `{m['test']['quadratic_weighted_kappa']}` | "
        f"**Macro F1:** `{m['test']['macro_f1']}` | **ECE:** `{m['calibration_after']['ece']}`\n"
        f"- **Inference Latency:** `{m['inference_ms_per_image']} ms/img`"
    )
    choices = get_checkpoint_choices()
    return msg, fig, gr.update(choices=choices, value=res["checkpoint_path"])


def _run_inference_ui(
    uploaded_img: np.ndarray | None,
    manifest_name: str | None,
    sample_id: str | None,
    ckpt_path: str | None,
    preset: str,
) -> tuple[np.ndarray | None, str, pd.DataFrame]:
    source = uploaded_img
    if source is None and manifest_name and sample_id:
        df = load_active_manifest(manifest_name)
        hit = df[df["image_id"].astype(str) == str(sample_id)]
        if len(hit) > 0:
            source = str(hit.iloc[0]["abs_path"])

    out = run_full_clinical_research_inference(
        image_arr=source,
        checkpoint_path=ckpt_path,
        preprocess_preset=preset,
        xai_method="gradcam",
        mc_passes=8,
    )
    if not out.get("ok"):
        return None, f"**Notice:** {out.get('error')}", pd.DataFrame()

    det = out["deterministic"]
    mc = out["mc_dropout"]
    qa = out["quality"]
    ab = out["abstention"]
    ckpt_meta = out["checkpoint_meta"]
    syn_tag = " [Trained on Synthetic Demo Data]" if ckpt_meta.get("is_synthetic") else ""

    summary = (
        f"### {ab['status']}\n\n"
        f"- **Model:** `{ckpt_meta.get('model_name')}`{syn_tag} | **Latency:** `{det['inference_ms']} ms`\n"
        f"- **Predicted Grade:** `{det['predicted_label']}` (Temp-Scaled Prob: `{det['confidence']*100:.1f}%`, T=`{det['temperature']:.2f}`)\n"
        f"- **MC Uncertainty:** Entropy=`{mc['predictive_entropy']:.3f}`, Epistemic MI=`{mc['mutual_information']:.4f}`\n"
        f"- **Image Quality:** `{qa['quality_category']}` (`{qa['quality_score']}/100`)\n"
        f"- **Gate Reasons:** {'; '.join(ab['reasons'])}"
    )
    prob_df = pd.DataFrame(
        [{"Class": k, "Probability": round(float(v), 4)} for k, v in det["prob_dict"].items()]
    )
    return out["overlay_rgb"], summary, prob_df


def build_tab() -> None:
    manifests = get_available_manifests()
    ckpts = get_checkpoint_choices()
    gr.Markdown("### Transfer Learning Experiments & Single-Image Inference Workflow")

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### A. Train / Compare Transfer Architecture")
            manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Manifest")
            arch_dd = gr.Dropdown(choices=["resnet18", "efficientnet_b0", "densenet121"], value="resnet18", label="Architecture")
            mode_dd = gr.Dropdown(choices=["scratch", "frozen", "partial_finetune", "full_finetune"], value="frozen", label="Fine-Tuning Regime")
            pretrained_chk = gr.Checkbox(value=False, label="Use Pretrained ImageNet Weights (Requires local cache or internet)")
            loss_dd = gr.Dropdown(choices=["ce", "weighted_ce", "focal"], value="weighted_ce", label="Imbalance Loss")
            epochs_sl = gr.Slider(1, 15, value=3, step=1, label="Epochs")
            sz_dd = gr.Dropdown(choices=[128, 224], value=128, label="Image Size")
            train_btn = gr.Button("Train Transfer Model", variant="primary")
            train_md = gr.Markdown("")
            train_plot = gr.Plot(label="Training Curves")

        with gr.Column():
            gr.Markdown("#### B. 10-Step Trustworthy Research Inference")
            ckpt_dd = gr.Dropdown(choices=ckpts, value=(ckpts[0][1] if ckpts else None), label="Trained Checkpoint")
            refresh_ckpt_btn = gr.Button("Refresh Checkpoints")
            inf_img = gr.Image(type="numpy", label="Upload Fundus Image")
            sample_dd = gr.Dropdown(choices=[], label="Or Pick Sample from Manifest")
            preset_dd = gr.Dropdown(choices=["raw_resize", "fov_crop", "fov_clahe_norm", "ben_graham"], value="fov_clahe_norm", label="Preprocessing")
            infer_btn = gr.Button("Run Full Research Inference", variant="primary")
            overlay_out = gr.Image(type="numpy", label="Grad-CAM Overlay")
            inf_md = gr.Markdown("")
            prob_tbl = gr.Dataframe(interactive=False, label="Severity Probability Distribution")

    def _update_samples(m_name: str):
        df = load_active_manifest(m_name)
        ids = df[~df["is_corrupt"].astype(bool)]["image_id"].astype(str).head(50).tolist() if not df.empty else []
        return gr.update(choices=ids, value=(ids[0] if ids else None))

    def _refresh_ckpts():
        ch = get_checkpoint_choices()
        return gr.update(choices=ch, value=(ch[0][1] if ch else None))

    manifest_dd.change(fn=_update_samples, inputs=[manifest_dd], outputs=[sample_dd])
    refresh_ckpt_btn.click(fn=_refresh_ckpts, inputs=[], outputs=[ckpt_dd])
    train_btn.click(
        fn=_train_transfer_model,
        inputs=[manifest_dd, arch_dd, mode_dd, pretrained_chk, loss_dd, epochs_sl, sz_dd],
        outputs=[train_md, train_plot, ckpt_dd],
    )
    infer_btn.click(
        fn=_run_inference_ui,
        inputs=[inf_img, manifest_dd, sample_dd, ckpt_dd, preset_dd],
        outputs=[overlay_out, inf_md, prob_tbl],
    )
