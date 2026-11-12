"""Custom CNN Foundations Lab: receptive fields, hyperparameters, and CPU training."""

from __future__ import annotations

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.components.state import get_available_manifests, load_active_manifest
from src.config import load_config
from src.experiments.ablation import run_supervised_experiment
from src.models.custom_cnn import CustomFundusCNN
from src.reporting.figures import plot_training_curves

TAB_TITLE = "5. Custom CNN Foundations"


def _inspect_cnn_architecture(
    num_blocks: int,
    base_channels: int,
    kernel_size: int,
    dropout: float,
    pooling: str,
    activation: str,
    image_size: int,
) -> tuple[str, pd.DataFrame]:
    model = CustomFundusCNN(
        num_blocks=int(num_blocks),
        base_channels=int(base_channels),
        kernel_size=int(kernel_size),
        dropout=float(dropout),
        pooling=pooling,
        activation=activation,
    )
    summary = model.architecture_summary(input_size=int(image_size))
    md = (
        f"**Architecture:** `CustomFundusCNN` | "
        f"**Total Params:** `{summary['total_params']:,}` | "
        f"**Trainable Params:** `{summary['trainable_params']:,}` | "
        f"**Penultimate Embedding Dim:** `{summary['embedding_dim']}`"
    )
    return md, pd.DataFrame(summary["receptive_fields"])


def _train_custom_cnn_ui(
    manifest_name: str | None,
    num_blocks: int,
    base_channels: int,
    kernel_size: int,
    dropout: float,
    pooling: str,
    activation: str,
    image_size: int,
    epochs: int,
    lr: float,
    loss_name: str,
    preset: str,
) -> tuple[str, plt.Figure]:
    df = load_active_manifest(manifest_name)
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No manifest loaded.", ha="center", va="center")
        ax.axis("off")
        return "No active manifest found. Initialize datasets in Overview first.", fig

    cfg = load_config(preset_path="configs/lightweight_cpu.yaml")
    cfg["models"]["custom_cnn"].update(
        {
            "num_blocks": int(num_blocks),
            "base_channels": int(base_channels),
            "kernel_size": int(kernel_size),
            "dropout": float(dropout),
            "pooling": pooling,
            "activation": activation,
        }
    )
    res = run_supervised_experiment(
        manifest_df=df,
        model_name="custom_cnn",
        run_id=f"ui_custom_cnn_b{int(num_blocks)}_e{int(epochs)}",
        experiment_group="UI_CustomCNN",
        cfg=cfg,
        preprocess_preset=preset,
        loss_name=loss_name,
        epochs=int(epochs),
        image_size=int(image_size),
        lr=float(lr),
    )
    fig = plot_training_curves(res["history"], title=f"Custom CNN Curves ({res['run_id']})")
    m = res["metrics"]
    msg = (
        f"Training finished in **{m['training_seconds']}s** (best epoch `{m['best_epoch']}`).\n\n"
        f"- **Checkpoint:** `{res['checkpoint_path']}`\n"
        f"- **Test Accuracy:** `{m['test']['accuracy']}` | **Balanced Acc:** `{m['test']['balanced_accuracy']}`\n"
        f"- **Test QWK:** `{m['test']['quadratic_weighted_kappa']}` | **Macro F1:** `{m['test']['macro_f1']}`\n"
        f"- **Calibration ECE (Before -> After):** `{m['calibration_before']['ece']} -> {m['calibration_after']['ece']}`"
    )
    return msg, fig


def build_tab() -> None:
    manifests = get_available_manifests()
    gr.Markdown("### Classical Deep Learning Foundations — Custom CNN from Scratch")

    with gr.Row():
        with gr.Column():
            num_blocks = gr.Slider(2, 5, value=3, step=1, label="Convolutional Blocks")
            base_ch = gr.Dropdown(choices=[8, 16, 32], value=16, label="Base Filter Count")
            k_sz = gr.Dropdown(choices=[3, 5], value=3, label="Kernel Size")
            dropout = gr.Slider(0.0, 0.6, value=0.25, step=0.05, label="Dropout Rate")
            pooling = gr.Dropdown(choices=["max", "avg"], value="max", label="Pooling Layer")
            act = gr.Dropdown(choices=["relu", "gelu", "leaky_relu"], value="relu", label="Activation")
            img_sz = gr.Dropdown(choices=[128, 224], value=128, label="Input Resolution (CPU-Friendly)")
            inspect_btn = gr.Button("Inspect Receptive Field & Params", variant="secondary")
        with gr.Column():
            manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Training Manifest")
            preset_dd = gr.Dropdown(choices=["raw_resize", "fov_crop", "fov_clahe_norm", "ben_graham"], value="fov_clahe_norm", label="Preprocessing")
            loss_dd = gr.Dropdown(choices=["ce", "weighted_ce", "focal", "label_smoothing"], value="weighted_ce", label="Loss Function")
            epochs_sl = gr.Slider(1, 15, value=3, step=1, label="Training Epochs")
            lr_num = gr.Number(value=0.0005, label="Learning Rate")
            train_btn = gr.Button("Train Custom CNN Baseline", variant="primary")

    arch_md = gr.Markdown("")
    rf_table = gr.Dataframe(interactive=False, label="Layer-by-Layer Theoretical Receptive Field")
    train_status = gr.Markdown("")
    curves_plot = gr.Plot(label="Training & Validation Curves")

    init_md, init_rf = _inspect_cnn_architecture(3, 16, 3, 0.25, "max", "relu", 128)
    arch_md.value = init_md
    rf_table.value = init_rf

    inspect_btn.click(
        fn=_inspect_cnn_architecture,
        inputs=[num_blocks, base_ch, k_sz, dropout, pooling, act, img_sz],
        outputs=[arch_md, rf_table],
    )
    train_btn.click(
        fn=_train_custom_cnn_ui,
        inputs=[manifest_dd, num_blocks, base_ch, k_sz, dropout, pooling, act, img_sz, epochs_sl, lr_num, loss_dd, preset_dd],
        outputs=[train_status, curves_plot],
    )
