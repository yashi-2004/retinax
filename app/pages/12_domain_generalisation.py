"""Domain Generalisation, Dataset Shift, Perturbation Stress & OOD Detection page."""

from __future__ import annotations

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from app.components.state import get_available_manifests, get_checkpoint_choices, load_active_manifest
from src.data.caching import FundusDataset
from src.evaluation.generalisation import compare_cohorts_distribution_shift, evaluate_ood_detection
from src.evaluation.robustness import evaluate_perturbation_robustness
from src.models.inference import predict_batch
from src.preprocessing.fundus_processing import preprocess_fundus_image
from src.reporting.figures import plot_domain_shift_distributions
from src.training.checkpoints import load_checkpoint

TAB_TITLE = "12. Domain Generalisation & OOD"


def _run_domain_shift_analysis(
    internal_manifest: str | None,
    external_manifest: str | None,
    ckpt_path: str | None,
    apply_test_time_color_norm: bool,
) -> tuple[str, pd.DataFrame, plt.Figure]:
    int_df = load_active_manifest(internal_manifest)
    ext_df = load_active_manifest(external_manifest)
    if int_df.empty or ext_df.empty or not ckpt_path:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        return "Select internal manifest, external manifest, and trained checkpoint.", pd.DataFrame(), fig

    model, meta = load_checkpoint(ckpt_path, device="cpu")
    img_sz = int(meta.get("image_size", 224))
    temp = float(meta.get("temperature", 1.0))
    preset = "color_norm" if apply_test_time_color_norm else str(meta.get("preprocessing_preset", "fov_clahe_norm"))
    pre_fn = lambda arr: preprocess_fundus_image(arr, preset=preset, image_size=img_sz)

    int_test = int_df[int_df["split"] == "test"]
    if len(int_test) == 0:
        int_test = int_df[~int_df["is_corrupt"].astype(bool)].head(40)
    int_train = int_df[int_df["split"] == "train"].head(60)
    ext_valid = ext_df[~ext_df["is_corrupt"].astype(bool)].head(60)

    int_loader = DataLoader(FundusDataset(int_test, image_size=img_sz, preprocess_fn=pre_fn), batch_size=8)
    ext_loader = DataLoader(FundusDataset(ext_valid, image_size=img_sz, preprocess_fn=pre_fn), batch_size=8)
    tr_loader = DataLoader(FundusDataset(int_train, image_size=img_sz, preprocess_fn=pre_fn), batch_size=8) if len(int_train) >= 2 else None

    int_out = predict_batch(model, int_loader, temperature=temp, extract_embeddings=True)
    ext_out = predict_batch(model, ext_loader, temperature=temp, extract_embeddings=True)
    tr_out = predict_batch(model, tr_loader, temperature=temp, extract_embeddings=True) if tr_loader else None

    shift = compare_cohorts_distribution_shift(
        internal_df=int_test,
        external_df=ext_valid,
        internal_labels=int_out["labels"],
        internal_probs=int_out["probabilities"],
        external_labels=ext_out["labels"],
        external_probs=ext_out["probabilities"],
    )
    ood_res = evaluate_ood_detection(
        id_probs=int_out["probabilities"],
        ood_probs=ext_out["probabilities"],
        id_embeddings=int_out["embeddings"],
        ood_embeddings=ext_out["embeddings"],
        train_embeddings=(tr_out["embeddings"] if tr_out is not None else None),
    )

    fig = plot_domain_shift_distributions(
        shift["internal_feature_table"],
        shift["external_feature_table"],
        title=f"Acquisition Shift: {internal_manifest} vs {external_manifest}",
    )

    rows = [
        {"Metric": k, "Internal": shift["internal_metrics"][k], "External": shift["external_metrics"][k],
         "Delta (Ext - Int)": shift["performance_delta_external_minus_internal"].get(k, "—")}
        for k in shift["internal_metrics"]
    ]
    rows.append({"Metric": "Label JS Divergence", "Internal": 0.0, "External": shift["label_distribution_js_divergence"], "Delta (Ext - Int)": "—"})

    ood_lines = "\n".join(
        f"- **{k}:** AUROC=`{v['auroc']}`, FPR@95TPR=`{v['fpr_at_95_tpr']}`"
        for k, v in ood_res.items()
        if isinstance(v, dict)
    )
    md = (
        f"### Cross-Dataset Generalisation (`preset={preset}`)\n"
        f"**Shift Detection (Internal vs Shifted External):**\n{ood_lines}\n\n"
        f"> *OOD Limitation:* {ood_res['limitation_note']}"
    )
    return md, pd.DataFrame(rows), fig


def _run_stress_test(
    manifest_name: str | None,
    ckpt_path: str | None,
    perturbation: str,
) -> pd.DataFrame:
    df = load_active_manifest(manifest_name)
    if df.empty or not ckpt_path:
        return pd.DataFrame([{"Status": "Select a manifest and trained checkpoint."}])
    model, meta = load_checkpoint(ckpt_path, device="cpu")
    test_df = df[df["split"] == "test"]
    if len(test_df) == 0:
        test_df = df
    res = evaluate_perturbation_robustness(
        model=model,
        manifest_df=test_df,
        perturbation=perturbation,
        preprocess_preset=str(meta.get("preprocessing_preset", "fov_clahe_norm")),
        image_size=int(meta.get("image_size", 224)),
        max_samples=30,
    )
    return pd.DataFrame(res)


def build_tab() -> None:
    manifests = get_available_manifests()
    ckpts = get_checkpoint_choices()
    gr.Markdown(
        "### Cross-Dataset Domain Generalisation, Shift & Robustness Laboratory\n"
        "Evaluates within-dataset vs external cohort performance, colour/brightness/blur shift, "
        "unsupervised test-time colour normalization, and OOD detection."
    )

    with gr.Row():
        int_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Internal Cohort Manifest")
        ext_dd = gr.Dropdown(choices=manifests, value=(manifests[-1] if len(manifests) > 1 else (manifests[0] if manifests else None)), label="External / Shifted Manifest")
        ckpt_dd = gr.Dropdown(choices=ckpts, value=(ckpts[0][1] if ckpts else None), label="Trained Checkpoint")
        refresh_btn = gr.Button("Refresh Checkpoints")

    color_norm_chk = gr.Checkbox(value=False, label="Apply Unsupervised Test-Time Gray-World Colour Normalization (No External Label Tuning)")
    run_shift_btn = gr.Button("Run Cross-Dataset Shift & OOD Analysis", variant="primary")

    shift_md = gr.Markdown("")
    shift_tbl = gr.Dataframe(interactive=False, label="Internal vs External Metrics")
    shift_plot = gr.Plot(label="Image Acquisition Shift Distributions")

    gr.Markdown("#### Controlled Perturbation Stress Test")
    with gr.Row():
        pert_dd = gr.Dropdown(
            choices=["gaussian_blur", "underexposure", "overexposure", "low_contrast", "gaussian_noise", "partial_occlusion"],
            value="gaussian_blur",
            label="Perturbation Type",
        )
        stress_btn = gr.Button("Run Perturbation Stress Test", variant="secondary")
    stress_tbl = gr.Dataframe(interactive=False, label="Performance & Calibration Degradation Across Severities")

    def _refresh_ckpts():
        ch = get_checkpoint_choices()
        return gr.update(choices=ch, value=(ch[0][1] if ch else None))

    refresh_btn.click(fn=_refresh_ckpts, inputs=[], outputs=[ckpt_dd])
    run_shift_btn.click(
        fn=_run_domain_shift_analysis,
        inputs=[int_dd, ext_dd, ckpt_dd, color_norm_chk],
        outputs=[shift_md, shift_tbl, shift_plot],
    )
    stress_btn.click(fn=_run_stress_test, inputs=[int_dd, ckpt_dd, pert_dd], outputs=[stress_tbl])
