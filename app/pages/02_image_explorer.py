"""Image Explorer page: manifest inspection, image upload, RGB histograms, green channel, and FOV."""

from __future__ import annotations

import json
from typing import Any

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.components.state import get_available_manifests, load_active_manifest
from src.preprocessing.enhancement import extract_green_channel
from src.preprocessing.fundus_processing import inspect_fundus_image, load_rgb_array, preprocess_fundus_image
from src.reporting.figures import plot_class_distribution

TAB_TITLE = "2. Image Explorer"


def _load_manifest_view(manifest_name: str) -> tuple[pd.DataFrame, plt.Figure, str, list[str]]:
    df = load_active_manifest(manifest_name)
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No manifest loaded. Initialize datasets in Overview tab.", ha="center", va="center")
        ax.axis("off")
        return pd.DataFrame(), fig, "No manifest found.", []

    fig = plot_class_distribution(df, title=f"Severity Distribution — {manifest_name}")
    cols = [c for c in ["image_id", "dataset_name", "original_label", "label_name", "split", "width", "height", "quality_score", "quality_category", "is_synthetic"] if c in df.columns]
    valid = df[~df["is_corrupt"].astype(bool)]
    img_choices = valid["image_id"].astype(str).head(50).tolist()
    syn_note = " (SYNTHETIC NON-CLINICAL DATA)" if bool(df.get("is_synthetic", pd.Series([False])).any()) else ""
    summary = (
        f"**Manifest:** `{manifest_name}`{syn_note} | **Total:** {len(df)} | "
        f"**Valid:** {len(valid)} | **Corrupt:** {int(df['is_corrupt'].sum())} | "
        f"**Duplicate Groups:** {int(df.get('is_duplicate_group', pd.Series([False])).sum())}"
    )
    return df[cols].head(50), fig, summary, img_choices


def _inspect_selected_image(
    uploaded_img: np.ndarray | None,
    manifest_name: str,
    sample_id: str | None,
    preset: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, plt.Figure, str]:
    rgb = None
    source_meta: dict[str, Any] = {}

    if uploaded_img is not None:
        rgb = load_rgb_array(uploaded_img)
        source_meta["source"] = "User Upload"
    elif manifest_name and sample_id:
        df = load_active_manifest(manifest_name)
        match = df[df["image_id"].astype(str) == str(sample_id)]
        if len(match) > 0:
            row = match.iloc[0]
            rgb = load_rgb_array(row["abs_path"])
            source_meta = {
                "source": "Manifest Sample",
                "image_id": str(row["image_id"]),
                "label": f"{row['original_label']} ({row['label_name']})",
                "split": str(row["split"]),
                "is_synthetic": bool(row["is_synthetic"]),
            }

    if rgb is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Upload an image or pick a manifest sample.", ha="center", va="center")
        ax.axis("off")
        return None, None, None, fig, "No image selected."

    stats = inspect_fundus_image(rgb)
    proc = preprocess_fundus_image(rgb, preset=preset, image_size=224)
    green = extract_green_channel(proc, apply_clahe_boost=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=120)
    fig.patch.set_facecolor("#FFFFFF")
    bins = np.linspace(0, 255, 64)
    hists = stats["histograms"]
    ax.plot(bins, hists["r"], color="#DC2626", label="Red", linewidth=1.4)
    ax.plot(bins, hists["g"], color="#0D9488", label="Green", linewidth=1.6)
    ax.plot(bins, hists["b"], color="#2563EB", label="Blue", linewidth=1.4)
    ax.plot(bins, hists["luminance"], color="#0F172A", linestyle="--", label="Luminance", linewidth=1.2)
    ax.set_title("Colour Channel & Luminance Histograms", fontsize=10, fontweight="bold", color="#0F172A")
    ax.set_xlabel("Pixel Intensity [0-255]", fontsize=8.5)
    ax.set_ylabel("Pixel Count", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    info = {**source_meta, **{k: v for k, v in stats.items() if k != "histograms"}}
    return rgb, proc, green, fig, f"```json\n{json.dumps(info, indent=2)}\n```"


def build_tab() -> None:
    manifests = get_available_manifests()
    gr.Markdown("### Dataset Manifest & Retinal Image Explorer")
    with gr.Row():
        manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Active Manifest")
        refresh_m_btn = gr.Button("Refresh Manifests")

    manifest_summary = gr.Markdown("")
    with gr.Row():
        class_plot = gr.Plot(label="Class Distribution")
        manifest_table = gr.Dataframe(interactive=False, label="Manifest Preview (Top 50 Rows)")

    gr.Markdown("#### Single Image Inspection")
    with gr.Row():
        upload_img = gr.Image(type="numpy", label="Upload Retinal Image (Optional)")
        with gr.Column():
            sample_dd = gr.Dropdown(choices=[], label="Or Select Sample ID from Manifest")
            preset_dd = gr.Dropdown(
                choices=["raw_resize", "fov_crop", "fov_clahe_norm", "ben_graham", "green_channel", "color_norm"],
                value="fov_clahe_norm",
                label="Preview Preprocessing Preset",
            )
            inspect_btn = gr.Button("Inspect Image", variant="primary")

    with gr.Row():
        orig_out = gr.Image(label="Original RGB", type="numpy")
        proc_out = gr.Image(label="Processed View (224x224)", type="numpy")
        green_out = gr.Image(label="Green Channel + CLAHE", type="numpy")

    with gr.Row():
        hist_plot = gr.Plot(label="RGB & Luminance Histograms")
        meta_md = gr.Markdown("")

    def _on_manifest_change(m_name: str):
        tbl, fig, summ, ids = _load_manifest_view(m_name)
        return tbl, fig, summ, gr.update(choices=ids, value=(ids[0] if ids else None))

    def _refresh_list():
        ms = get_available_manifests()
        return gr.update(choices=ms, value=(ms[0] if ms else None))

    refresh_m_btn.click(fn=_refresh_list, inputs=[], outputs=[manifest_dd])
    manifest_dd.change(fn=_on_manifest_change, inputs=[manifest_dd], outputs=[manifest_table, class_plot, manifest_summary, sample_dd])
    inspect_btn.click(
        fn=_inspect_selected_image,
        inputs=[upload_img, manifest_dd, sample_dd, preset_dd],
        outputs=[orig_out, proc_out, green_out, hist_plot, meta_md],
    )
