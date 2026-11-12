"""Vision Foundation Models Laboratory: DINOv2 linear probing, 2D embedding projection, and optional BiomedCLIP."""

from __future__ import annotations

import json

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from app.components.state import get_available_manifests, load_active_manifest
from src.data.caching import FundusDataset, save_embeddings
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics
from src.foundation.biomedical_clip import check_biomedclip_availability
from src.foundation.dinov2 import DINOV2_PROVENANCE, DINOv2FeatureExtractor, train_embedding_probe
from src.foundation.embeddings import compute_2d_projection, extract_embeddings_from_model
from src.preprocessing.fundus_processing import preprocess_fundus_image
from src.reporting.figures import plot_embedding_projection

TAB_TITLE = "9. Foundation Models"


def _run_dinov2_probe(
    manifest_name: str | None,
    probe_type: str,
    proj_method: str,
    use_pretrained: bool,
    max_samples: int,
) -> tuple[str, plt.Figure]:
    df = load_active_manifest(manifest_name)
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No manifest loaded.", ha="center", va="center")
        ax.axis("off")
        return "No active manifest found.", fig

    try:
        extractor = DINOv2FeatureExtractor(img_size=224, pretrained=bool(use_pretrained))
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "DINOv2 weights unavailable offline.\nSee status panel.", ha="center", va="center")
        ax.axis("off")
        return f"**DINOv2 Weight Notice:** {exc}", fig

    train_df = df[df["split"] == "train"].head(int(max_samples))
    test_df = df[df["split"] == "test"].head(int(max_samples))
    if len(train_df) < 3 or len(test_df) < 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        return "Manifest needs non-empty train & test splits.", fig

    pre_fn = lambda arr: preprocess_fundus_image(arr, preset="fov_clahe_norm", image_size=224)
    tr_loader = DataLoader(FundusDataset(train_df, image_size=224, preprocess_fn=pre_fn), batch_size=8, shuffle=False)
    te_loader = DataLoader(FundusDataset(test_df, image_size=224, preprocess_fn=pre_fn), batch_size=8, shuffle=False)

    tr_out = extract_embeddings_from_model(extractor, tr_loader)
    te_out = extract_embeddings_from_model(extractor, te_loader)

    save_embeddings(
        cache_name=f"dinov2_{manifest_name}_test",
        embeddings=te_out["embeddings"],
        labels=te_out["labels"],
        image_ids=te_out["image_ids"],
        splits=["test"] * len(te_out["labels"]),
        metadata={"pretrained": bool(use_pretrained)},
    )

    probe = train_embedding_probe(tr_out["embeddings"], tr_out["labels"], te_out["embeddings"], probe_type=probe_type)
    cls_m = compute_classification_metrics(te_out["labels"], probe["eval_probabilities"])
    cal_m = compute_calibration_metrics(te_out["labels"], probe["eval_probabilities"])

    all_emb = np.concatenate([tr_out["embeddings"], te_out["embeddings"]], axis=0)
    all_lbl = np.concatenate([tr_out["labels"], te_out["labels"]], axis=0)
    proj = compute_2d_projection(all_emb, labels=all_lbl, method=proj_method)
    fig = plot_embedding_projection(
        proj["coords_2d"],
        all_lbl,
        title=f"DINOv2 ({proj_method.upper()}) | Silhouette={proj['silhouette_score']}",
    )

    msg = (
        f"### DINOv2 Frozen Representation + `{probe['probe_type']}`\n"
        f"- **Pretrained Weights Loaded:** `{bool(use_pretrained)}` (Dim: `{extractor.embedding_dim}`)\n"
        f"- **Test Accuracy:** `{cls_m['accuracy']}` | **Balanced Acc:** `{cls_m['balanced_accuracy']}` | "
        f"**Macro F1:** `{cls_m['macro_f1']}` | **QWK:** `{cls_m['quadratic_weighted_kappa']}`\n"
        f"- **ECE:** `{cal_m['ece']}` | **Silhouette Score:** `{proj['silhouette_score']}`"
    )
    return msg, fig


def build_tab() -> None:
    manifests = get_available_manifests()
    bclip_status = check_biomedclip_availability()

    gr.Markdown(
        "### Vision Foundation Models Laboratory\n"
        "Evaluates frozen general-purpose self-supervised foundation representations (DINOv2) and "
        "optional biomedical vision-language embeddings (BiomedCLIP) without GPU or paid APIs."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### DINOv2 Provenance & License")
            gr.Markdown(f"```json\n{json.dumps(DINOV2_PROVENANCE, indent=2)}\n```")
            gr.Markdown("#### Optional BiomedCLIP Status")
            gr.Markdown(f"```json\n{json.dumps(bclip_status, indent=2)}\n```")

        with gr.Column():
            manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Manifest")
            probe_dd = gr.Dropdown(choices=["logistic", "mlp"], value="logistic", label="Downstream Classifier Head")
            proj_dd = gr.Dropdown(choices=["pca", "tsne"], value="pca", label="2D Projection Method")
            pre_chk = gr.Checkbox(value=False, label="Use Pretrained DINOv2 Weights (Requires local cache or internet)")
            max_sl = gr.Slider(10, 150, value=60, step=10, label="Max Samples per Split (CPU Memory Guard)")
            run_btn = gr.Button("Extract Frozen DINOv2 Features & Probe", variant="primary")

    status_md = gr.Markdown("")
    emb_plot = gr.Plot(label="2D Representation Space")

    run_btn.click(
        fn=_run_dinov2_probe,
        inputs=[manifest_dd, probe_dd, proj_dd, pre_chk, max_sl],
        outputs=[status_md, emb_plot],
    )
