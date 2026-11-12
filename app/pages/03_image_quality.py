"""Image Quality Assessment Laboratory page."""

from __future__ import annotations

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.components.state import get_available_manifests, load_active_manifest
from src.preprocessing.fundus_processing import load_rgb_array
from src.preprocessing.quality_assessment import assess_fundus_quality

TAB_TITLE = "3. Image Quality Lab"


def _analyze_single_quality(
    uploaded_img: np.ndarray | None,
    manifest_name: str | None,
    sample_id: str | None,
    blur_thresh: float,
    min_bright: float,
    max_bright: float,
) -> tuple[np.ndarray | None, str, pd.DataFrame]:
    rgb = None
    if uploaded_img is not None:
        rgb = load_rgb_array(uploaded_img)
    elif manifest_name and sample_id:
        df = load_active_manifest(manifest_name)
        hit = df[df["image_id"].astype(str) == str(sample_id)]
        if len(hit) > 0:
            rgb = load_rgb_array(hit.iloc[0]["abs_path"])

    if rgb is None:
        return None, "Upload an image or select a manifest sample.", pd.DataFrame()

    res = assess_fundus_quality(
        rgb,
        thresholds={
            "blur_laplacian_threshold": float(blur_thresh),
            "min_brightness": float(min_bright),
            "max_brightness": float(max_bright),
        },
    )
    issues_md = "\n".join(f"- {i}" for i in res["issues"]) if res["issues"] else "- None detected"
    recs_md = "\n".join(f"- {r}" for r in res["recommendations"])
    report_md = (
        f"### Quality Category: **{res['quality_category']}** (Score: **{res['quality_score']} / 100**)\n\n"
        f"**Detected Issues:**\n{issues_md}\n\n"
        f"**Processing Recommendations:**\n{recs_md}\n\n"
        f"> *Methodological Caveat:* {res['limitation_note']}"
    )
    feat_df = pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in res["metrics"].items()]
    )
    return rgb, report_md, feat_df


def _analyze_cohort_quality(manifest_name: str | None) -> tuple[plt.Figure, pd.DataFrame]:
    df = load_active_manifest(manifest_name)
    if df.empty or "quality_score" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No manifest quality records found.", ha="center", va="center")
        ax.axis("off")
        return fig, pd.DataFrame()

    valid = df[~df["is_corrupt"].astype(bool)].copy()
    counts = valid["quality_category"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=125)
    fig.patch.set_facecolor("#FFFFFF")
    axes[0].hist(valid["quality_score"].dropna(), bins=15, color="#0D9488", edgecolor="#0F766E", alpha=0.85)
    axes[0].set_title("Quality Score Distribution", fontsize=10, fontweight="bold", color="#0F172A")
    axes[0].set_xlabel("Quality Score [0-100]")
    axes[0].set_ylabel("Count")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].bar(counts.index.tolist(), counts.values.tolist(), color=["#0D9488", "#D97706", "#DC2626"][: len(counts)])
    axes[1].set_title("Quality Categories", fontsize=10, fontweight="bold", color="#0F172A")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    plt.tight_layout()

    cols = [c for c in ["image_id", "original_label", "label_name", "quality_score", "quality_category", "quality_issues"] if c in valid.columns]
    return fig, valid[cols].sort_values("quality_score", ascending=True).head(30)


def build_tab() -> None:
    manifests = get_available_manifests()
    gr.Markdown("### Fundus Image Quality Assessment Laboratory")
    with gr.Row():
        with gr.Column():
            upload_img = gr.Image(type="numpy", label="Upload Fundus Image")
            manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Or Pick Manifest")
            sample_dd = gr.Dropdown(choices=[], label="Sample ID")
        with gr.Column():
            blur_slider = gr.Slider(10, 150, value=45, step=5, label="Blur Laplacian Threshold")
            min_b_slider = gr.Slider(5, 80, value=32, step=1, label="Min Brightness Threshold")
            max_b_slider = gr.Slider(160, 250, value=215, step=5, label="Max Brightness Threshold")
            run_qa_btn = gr.Button("Run Quality Assessment", variant="primary")
            cohort_btn = gr.Button("Analyze Manifest Quality Composition", variant="secondary")

    with gr.Row():
        img_out = gr.Image(type="numpy", label="Assessed Image")
        qa_md = gr.Markdown("")
        qa_table = gr.Dataframe(interactive=False, label="Classical Quality Feature Vector")

    with gr.Row():
        cohort_plot = gr.Plot(label="Manifest Quality Distribution")
        lowest_tbl = gr.Dataframe(interactive=False, label="Lowest-Quality Flagged Samples")

    def _update_samples(m_name: str):
        df = load_active_manifest(m_name)
        ids = df[~df["is_corrupt"].astype(bool)]["image_id"].astype(str).head(50).tolist() if not df.empty else []
        return gr.update(choices=ids, value=(ids[0] if ids else None))

    manifest_dd.change(fn=_update_samples, inputs=[manifest_dd], outputs=[sample_dd])
    run_qa_btn.click(
        fn=_analyze_single_quality,
        inputs=[upload_img, manifest_dd, sample_dd, blur_slider, min_b_slider, max_b_slider],
        outputs=[img_out, qa_md, qa_table],
    )
    cohort_btn.click(
        fn=_analyze_cohort_quality,
        inputs=[manifest_dd],
        outputs=[cohort_plot, lowest_tbl],
    )
