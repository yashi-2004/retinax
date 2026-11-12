"""Uncertainty Quantification, Calibration, Conformal Prediction & Selective Abstention page."""

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
from src.config import DR_CLASS_NAMES
from src.data.caching import FundusDataset
from src.evaluation.calibration import TemperatureScaler, apply_temperature_scaling, compute_calibration_metrics
from src.models.inference import predict_batch
from src.preprocessing.fundus_processing import preprocess_fundus_image
from src.reporting.figures import plot_reliability_diagram, plot_risk_coverage_curve
from src.training.checkpoints import load_checkpoint
from src.uncertainty.conformal import SplitConformalPredictor
from src.uncertainty.selective_prediction import compute_risk_coverage_curve

TAB_TITLE = "11. Uncertainty & Selective AI"


def _evaluate_single_uncertainty(
    uploaded_img: np.ndarray | None,
    manifest_name: str | None,
    sample_id: str | None,
    ckpt_path: str | None,
    mc_passes: int,
    min_conf: float,
    max_ent: float,
) -> tuple[str, pd.DataFrame]:
    source = uploaded_img
    if source is None and manifest_name and sample_id:
        df = load_active_manifest(manifest_name)
        hit = df[df["image_id"].astype(str) == str(sample_id)]
        if len(hit) > 0:
            source = str(hit.iloc[0]["abs_path"])

    out = run_full_clinical_research_inference(
        image_arr=source,
        checkpoint_path=ckpt_path,
        mc_passes=int(mc_passes),
        min_confidence=float(min_conf),
        max_entropy=float(max_ent),
    )
    if not out.get("ok"):
        return f"**Notice:** {out.get('error')}", pd.DataFrame()

    mc = out["mc_dropout"]
    ab = out["abstention"]
    cp = out["conformal"]
    qa = out["quality"]

    cp_text = (
        f"Set={cp['prediction_set_labels']} (size={cp['set_size']}, target={cp['target_coverage']*100:.0f}%)"
        if cp
        else "Conformal threshold not calibrated in checkpoint."
    )
    md = (
        f"### Selective Decision: **{ab['status']}**\n\n"
        f"- **Mean MC Probability Prediction:** `{DR_CLASS_NAMES[mc['predicted_class']]}` (`{mc['confidence']*100:.1f}%`)\n"
        f"- **Total Predictive Entropy:** `{mc['predictive_entropy']:.4f}` | "
        f"**Aleatoric (Expected Entropy):** `{mc['expected_entropy']:.4f}` | "
        f"**Epistemic (Mutual Info):** `{mc['mutual_information']:.4f}`\n"
        f"- **Predictive Variance:** `{mc['predictive_variance']:.5f}` (over `{mc['n_passes']}` MC passes, `{mc['active_dropout_layers']}` active dropout layers)\n"
        f"- **Conformal Prediction Set:** {cp_text}\n"
        f"- **Quality Score:** `{qa['quality_score']}/100` (`{qa['quality_category']}`)\n"
        f"- **Abstention Gate Trace:** {'; '.join(ab['reasons'])}"
    )

    mean_p = mc["mean_probabilities"]
    std_p = np.std(mc["pass_samples"], axis=0)
    tbl = pd.DataFrame(
        [
            {
                "Grade": f"{c}: {DR_CLASS_NAMES[c]}",
                "MC Mean Prob": round(float(mean_p[c]), 4),
                "MC Std Dev": round(float(std_p[c]), 4),
            }
            for c in range(5)
        ]
    )
    return md, tbl


def _evaluate_cohort_calibration_and_conformal(
    manifest_name: str | None,
    ckpt_path: str | None,
    alpha: float,
    cp_method: str,
) -> tuple[str, plt.Figure, plt.Figure]:
    df = load_active_manifest(manifest_name)
    if df.empty or not ckpt_path:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.axis("off")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.axis("off")
        return "Select both a valid manifest and trained checkpoint.", fig1, fig2

    model, meta = load_checkpoint(ckpt_path, device="cpu")
    img_sz = int(meta.get("image_size", 224))
    preset = str(meta.get("preprocessing_preset", "fov_clahe_norm"))
    pre_fn = lambda arr: preprocess_fundus_image(arr, preset=preset, image_size=img_sz)

    cal_df = df[df["split"] == "cal"]
    if len(cal_df) < 5:
        cal_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    if len(cal_df) < 5 or len(test_df) < 2:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.axis("off")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.axis("off")
        return "Manifest needs >=5 calibration/validation samples and >=2 test samples.", fig1, fig2

    from torch.utils.data import DataLoader
    cal_loader = DataLoader(FundusDataset(cal_df, image_size=img_sz, preprocess_fn=pre_fn), batch_size=8)
    test_loader = DataLoader(FundusDataset(test_df, image_size=img_sz, preprocess_fn=pre_fn), batch_size=8)

    cal_out = predict_batch(model, cal_loader, temperature=1.0)
    test_out = predict_batch(model, test_loader, temperature=1.0)

    scaler = TemperatureScaler()
    fit_res = scaler.fit(cal_out["logits"], cal_out["labels"])
    opt_t = float(fit_res["temperature"])

    cal_probs_scaled = apply_temperature_scaling(cal_out["logits"], opt_t)
    test_probs_uncal = test_out["probabilities"]
    test_probs_scaled = apply_temperature_scaling(test_out["logits"], opt_t)

    cal_before = compute_calibration_metrics(test_out["labels"], test_probs_uncal)
    cal_after = compute_calibration_metrics(test_out["labels"], test_probs_scaled)

    cp = SplitConformalPredictor(alpha=float(alpha), method=cp_method)
    cp.fit(cal_probs_scaled, cal_out["labels"])
    cp_eval = cp.evaluate(test_probs_scaled, test_out["labels"])

    entropies = -np.sum(np.clip(test_probs_scaled, 1e-8, 1.0) * np.log(np.clip(test_probs_scaled, 1e-8, 1.0)), axis=1)
    rc_data = compute_risk_coverage_curve(test_out["labels"], test_probs_scaled, uncertainty_scores=entropies)

    rel_fig = plot_reliability_diagram(cal_before, cal_after, title=f"Test Reliability Diagram (T={opt_t:.2f})")
    rc_fig = plot_risk_coverage_curve(rc_data, title="Selective Risk vs Coverage (Entropy Ranking)")

    msg = (
        f"### Cohort Calibration & Conformal Results (`test` split, n={len(test_df)})\n\n"
        f"- **Fitted Temperature (from `cal`):** `T = {opt_t:.3f}`\n"
        f"- **ECE (Before -> After):** `{cal_before['ece']}` -> `{cal_after['ece']}` | "
        f"**Brier:** `{cal_before['brier_score']}` -> `{cal_after['brier_score']}`\n"
        f"- **Conformal ({cp_method.upper()}, target={(1-alpha)*100:.0f}%):** "
        f"Empirical Coverage=`{cp_eval['empirical_coverage']*100:.1f}%`, "
        f"Avg Set Size=`{cp_eval['average_set_size']}`, Singleton Rate=`{cp_eval['singleton_rate']*100:.1f}%`\n"
        f"- **Selective Prediction AURC:** `{rc_data['aurc']}` (Full-Coverage Risk: `{rc_data['full_coverage_risk']}`)"
    )
    return msg, rel_fig, rc_fig


def build_tab() -> None:
    manifests = get_available_manifests()
    ckpts = get_checkpoint_choices()
    gr.Markdown("### Uncertainty Quantification, Calibration, Conformal Sets & Selective Prediction")

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### A. Single-Image Uncertainty & Abstention Gate")
            upload_img = gr.Image(type="numpy", label="Upload Fundus Image")
            manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Manifest")
            sample_dd = gr.Dropdown(choices=[], label="Sample ID")
            ckpt_dd = gr.Dropdown(choices=ckpts, value=(ckpts[0][1] if ckpts else None), label="Trained Checkpoint")
            refresh_btn = gr.Button("Refresh Checkpoints")
            mc_sl = gr.Slider(4, 24, value=10, step=2, label="MC Dropout Passes")
            conf_sl = gr.Slider(0.20, 0.90, value=0.45, step=0.05, label="Min Confidence Gate")
            ent_sl = gr.Slider(0.30, 1.60, value=1.25, step=0.05, label="Max Entropy Gate")
            single_btn = gr.Button("Evaluate Single-Image Uncertainty", variant="primary")
            single_md = gr.Markdown("")
            single_tbl = gr.Dataframe(interactive=False, label="Per-Class MC Distribution")

        with gr.Column():
            gr.Markdown("#### B. Cohort Calibration, Conformal Coverage & Risk-Coverage")
            alpha_sl = gr.Slider(0.05, 0.30, value=0.10, step=0.05, label="Conformal Miscoverage Alpha")
            cp_dd = gr.Dropdown(choices=["aps", "lac"], value="aps", label="Conformal Nonconformity Score")
            cohort_btn = gr.Button("Run Cohort Calibration & Conformal Evaluation", variant="primary")
            cohort_md = gr.Markdown("")
            rel_plot = gr.Plot(label="Reliability Diagram")
            rc_plot = gr.Plot(label="Risk-Coverage Curve")

    def _update_samples(m_name: str):
        df = load_active_manifest(m_name)
        ids = df[~df["is_corrupt"].astype(bool)]["image_id"].astype(str).head(50).tolist() if not df.empty else []
        return gr.update(choices=ids, value=(ids[0] if ids else None))

    def _refresh_ckpts():
        ch = get_checkpoint_choices()
        return gr.update(choices=ch, value=(ch[0][1] if ch else None))

    manifest_dd.change(fn=_update_samples, inputs=[manifest_dd], outputs=[sample_dd])
    refresh_btn.click(fn=_refresh_ckpts, inputs=[], outputs=[ckpt_dd])
    single_btn.click(
        fn=_evaluate_single_uncertainty,
        inputs=[upload_img, manifest_dd, sample_dd, ckpt_dd, mc_sl, conf_sl, ent_sl],
        outputs=[single_md, single_tbl],
    )
    cohort_btn.click(
        fn=_evaluate_cohort_calibration_and_conformal,
        inputs=[manifest_dd, ckpt_dd, alpha_sl, cp_dd],
        outputs=[cohort_md, rel_plot, rc_plot],
    )
