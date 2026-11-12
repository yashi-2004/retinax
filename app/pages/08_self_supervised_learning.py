"""Self-Supervised Learning Laboratory: SimCLR contrastive pretraining & linear evaluation."""

from __future__ import annotations

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from app.components.state import get_available_manifests, load_active_manifest
from src.config import load_config
from src.data.caching import FundusDataset
from src.evaluation.classification import compute_classification_metrics
from src.foundation.dinov2 import train_embedding_probe
from src.foundation.embeddings import extract_embeddings_from_model
from src.foundation.ssl_simclr import SimCLREncoderModel, train_simclr_ssl
from src.preprocessing.fundus_processing import preprocess_fundus_image

TAB_TITLE = "8. Self-Supervised Learning"


def _run_simclr_pipeline(
    manifest_name: str | None,
    encoder_type: str,
    ssl_epochs: int,
    batch_size: int,
    temperature: float,
    image_size: int,
) -> tuple[str, plt.Figure]:
    df = load_active_manifest(manifest_name)
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No manifest loaded.", ha="center", va="center")
        ax.axis("off")
        return "No active manifest loaded.", fig

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    if len(train_df) < 4 or len(test_df) < 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        return "Need at least 4 train and 2 test images.", fig

    cfg = load_config(preset_path="configs/lightweight_cpu.yaml")
    ssl_model = SimCLREncoderModel(encoder_type=encoder_type, projection_dim=64)
    ssl_res = train_simclr_ssl(
        ssl_model=ssl_model,
        unlabelled_df=train_df,
        epochs=int(ssl_epochs),
        batch_size=int(batch_size),
        image_size=int(image_size),
        temperature=float(temperature),
        run_id=f"ui_simclr_{encoder_type}_e{int(ssl_epochs)}",
        cfg=cfg,
    )

    # Downstream frozen linear evaluation on learned SSL features vs random init check
    img_sz = int(image_size)
    pre_fn = lambda arr: preprocess_fundus_image(arr, preset="fov_clahe_norm", image_size=img_sz)
    tr_loader = DataLoader(FundusDataset(train_df, image_size=img_sz, preprocess_fn=pre_fn), batch_size=8, shuffle=False)
    te_loader = DataLoader(FundusDataset(test_df, image_size=img_sz, preprocess_fn=pre_fn), batch_size=8, shuffle=False)

    tr_feats = extract_embeddings_from_model(ssl_model, tr_loader)
    te_feats = extract_embeddings_from_model(ssl_model, te_loader)
    probe = train_embedding_probe(tr_feats["embeddings"], tr_feats["labels"], te_feats["embeddings"], probe_type="logistic")
    cls_m = compute_classification_metrics(te_feats["labels"], probe["eval_probabilities"])

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=125)
    fig.patch.set_facecolor("#FFFFFF")
    hist = ssl_res["history"]
    ax.plot([h["epoch"] for h in hist], [h["nt_xent_loss"] for h in hist], marker="o", color="#0D9488", linewidth=2.0)
    ax.set_title("SimCLR Contrastive Pretraining Loss (NT-Xent)", fontsize=10, fontweight="bold", color="#0F172A")
    ax.set_xlabel("SSL Epoch")
    ax.set_ylabel("NT-Xent Contrastive Loss")
    ax.grid(linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    msg = (
        f"SimCLR pretraining finished in **{ssl_res['duration_seconds']}s**.\n\n"
        f"- **Final NT-Xent Loss:** `{ssl_res['final_nt_xent_loss']}`\n"
        f"- **Checkpoint:** `{ssl_res['checkpoint_path']}`\n"
        f"- **Downstream Linear Probe on Frozen SSL Embeddings:**\n"
        f"  - Test Accuracy: `{cls_m['accuracy']}` | Balanced Acc: `{cls_m['balanced_accuracy']}` | "
        f"Macro F1: `{cls_m['macro_f1']}` | QWK: `{cls_m['quadratic_weighted_kappa']}`"
    )
    return msg, fig


def build_tab() -> None:
    manifests = get_available_manifests()
    gr.Markdown(
        "### Self-Supervised Contrastive Representation Learning (SimCLR)\n"
        "Learns visual representations from positive augmented retinal views without class labels, "
        "followed by downstream linear probing on frozen representations."
    )
    with gr.Row():
        with gr.Column():
            manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Manifest")
            enc_dd = gr.Dropdown(choices=["custom_cnn", "resnet18"], value="custom_cnn", label="Encoder Backbone")
            epochs_sl = gr.Slider(1, 20, value=4, step=1, label="SSL Pretraining Epochs")
            bs_dd = gr.Dropdown(choices=[8, 16, 32], value=16, label="Contrastive Batch Size")
            temp_sl = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="NT-Xent Temperature")
            sz_dd = gr.Dropdown(choices=[128, 224], value=128, label="Image Size")
            run_btn = gr.Button("Run SimCLR Pretraining & Linear Probe", variant="primary")
        with gr.Column():
            status_md = gr.Markdown("")
            ssl_plot = gr.Plot(label="Contrastive NT-Xent Convergence")

    run_btn.click(
        fn=_run_simclr_pipeline,
        inputs=[manifest_dd, enc_dd, epochs_sl, bs_dd, temp_sl, sz_dd],
        outputs=[status_md, ssl_plot],
    )
