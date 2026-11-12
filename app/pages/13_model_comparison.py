"""Model Comparison, Paired Statistical Testing & Fairness/Subgroup Audit page."""

from __future__ import annotations

import json

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.classification import compute_curve_data, compute_qwk
from src.evaluation.statistical_tests import mcnemar_test, paired_bootstrap_comparison
from src.experiments.tracking import ExperimentRegistry
from src.reporting.figures import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_and_pr_curves,
    plot_training_curves,
)

TAB_TITLE = "13. Model Comparison & Fairness"


def _refresh_leaderboard(metric_col: str) -> tuple[pd.DataFrame, plt.Figure, dict, dict, dict]:
    reg = ExperimentRegistry()
    df = reg.comparison_dataframe()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No runs logged yet.", ha="center", va="center")
        ax.axis("off")
        return pd.DataFrame([{"Status": "Run experiments to populate leaderboard."}]), fig, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])

    fig = plot_model_comparison(df, metric_col=metric_col, title=f"Tracked Runs Comparison ({metric_col.upper()})")
    run_ids = df["run_id"].astype(str).tolist()
    cols = [c for c in ["run_id", "experiment_group", "model_name", "dataset_name", "is_synthetic", "accuracy", "balanced_accuracy", "macro_f1", "qwk", "roc_auc", "ece", "brier_score", "total_params", "ms_per_image", "training_seconds"] if c in df.columns]
    return df[cols], fig, gr.update(choices=run_ids, value=run_ids[0]), gr.update(choices=run_ids, value=run_ids[0]), gr.update(choices=run_ids, value=(run_ids[-1] if len(run_ids) > 1 else run_ids[0]))


def _inspect_run_diagnostics(run_id: str | None) -> tuple[plt.Figure, plt.Figure, plt.Figure, pd.DataFrame, str]:
    if not run_id:
        empty = plt.figure(figsize=(5, 3))
        plt.axis("off")
        return empty, empty, empty, pd.DataFrame(), "Select a run ID."

    details = ExperimentRegistry().load_run_details(run_id)
    m = details["metrics"]
    test_m = m["test"]

    cm_fig = plot_confusion_matrix(test_m["confusion_matrix_normalized"], normalized=True, title=f"Normalized Confusion Matrix ({run_id})")
    curves_fig = plot_training_curves(details.get("history", []), title=f"Learning Curves ({run_id})")

    pred_df = details.get("test_predictions_df")
    if pred_df is not None and "prob_0" in pred_df.columns:
        prob_cols = [f"prob_{c}" for c in range(5) if f"prob_{c}" in pred_df.columns]
        c_data = compute_curve_data(pred_df["y_true"].to_numpy(), pred_df[prob_cols].to_numpy())
        roc_fig = plot_roc_and_pr_curves(c_data, title=f"OvR ROC & PR Curves ({run_id})")
    else:
        roc_fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Per-sample predictions unavailable.", ha="center", va="center")
        ax.axis("off")

    per_cls = pd.DataFrame(list(test_m["per_class"].values()))
    summary_md = (
        f"#### Run Diagnostics: `{run_id}`\n"
        f"- **95% Bootstrap QWK CI:** `{m.get('test_qwk_bootstrap_95ci')}`\n"
        f"- **Ordinal Error Breakdown:** `{test_m['ordinal_errors']}`\n"
        f"- **Complexity:** `{m.get('model_complexity')}`"
    )
    return cm_fig, roc_fig, curves_fig, per_cls, summary_md


def _compare_two_runs(run_a: str | None, run_b: str | None) -> str:
    if not run_a or not run_b:
        return "Select two runs to compare statistically."
    reg = ExperimentRegistry()
    da = reg.load_run_details(run_a)
    db = reg.load_run_details(run_b)
    pa = da.get("test_predictions_df")
    pb = db.get("test_predictions_df")
    if pa is None or pb is None:
        return "Both runs must have saved `test_predictions.csv`."

    merged = pa.merge(pb, on="image_id", suffixes=("_a", "_b"))
    if len(merged) < 2:
        return f"Runs `{run_a}` and `{run_b}` do not share matching test image IDs."

    yt = merged["y_true_a"].to_numpy()
    prob_a = merged[[f"prob_{c}_a" for c in range(5)]].to_numpy()
    prob_b = merged[[f"prob_{c}_b" for c in range(5)]].to_numpy()

    mcn = mcnemar_test(yt, merged["y_pred_a"].to_numpy(), merged["y_pred_b"].to_numpy())
    boot = paired_bootstrap_comparison(
        yt, prob_a, prob_b, metric_fn=lambda y, p: compute_qwk(y, np.argmax(p, axis=1)), n_bootstrap=400
    )
    return (
        f"### Paired Comparison (`{run_b}` minus `{run_a}` on n={len(merged)} matched samples)\n"
        f"- **Paired Bootstrap QWK Delta:** `{boot['diff_b_minus_a']}` (95% CI: `[{boot['ci_low']}, {boot['ci_high']}]`, p=`{boot['p_value']}`)\n"
        f"- **McNemar's Accuracy Test:** `{json.dumps(mcn)}`"
    )


def build_tab() -> None:
    gr.Markdown("### Model Comparison, Statistical Validation & Subgroup Fairness Governance")
    with gr.Row():
        metric_dd = gr.Dropdown(choices=["qwk", "macro_f1", "accuracy", "balanced_accuracy", "ece"], value="qwk", label="Leaderboard Sort Metric")
        refresh_btn = gr.Button("Load / Refresh Experiment Registry", variant="primary")

    lb_table = gr.Dataframe(interactive=False, label="Tracked Experiment Leaderboard")
    lb_plot = gr.Plot(label="Benchmark Comparison")

    gr.Markdown("#### Detailed Run Diagnostics")
    with gr.Row():
        run_dd = gr.Dropdown(choices=[], label="Inspect Run ID")
        inspect_btn = gr.Button("Inspect Run Metrics & Curves", variant="secondary")

    run_md = gr.Markdown("")
    per_cls_tbl = gr.Dataframe(interactive=False, label="Per-Class Sensitivity, Specificity, Precision, F1 & AUC")
    with gr.Row():
        cm_plot = gr.Plot(label="Confusion Matrix")
        roc_plot = gr.Plot(label="ROC & PR Curves")
    curves_plot = gr.Plot(label="Training Curves")

    gr.Markdown("#### Paired Statistical Hypothesis Testing")
    with gr.Row():
        run_a_dd = gr.Dropdown(choices=[], label="Baseline Run A")
        run_b_dd = gr.Dropdown(choices=[], label="Candidate Run B")
        cmp_btn = gr.Button("Run Paired Bootstrap & McNemar Test", variant="secondary")
    cmp_md = gr.Markdown("")

    gr.Markdown(
        "#### Fairness & Subgroup Analysis Governance Notice\n"
        "> Public DR datasets (e.g., APTOS 2019 and standard Kaggle EyePACS releases) do **not** provide verified "
        "patient demographic attributes (age, biological sex, ethnicity, systemic comorbidities). Following strict "
        "medical AI ethics, **RetinaX refuses to infer or fabricate demographic labels from retinal pixel values**; "
        "stratified fairness evaluation is instead reported by image-quality stratum and imaging cohort where reliable metadata exists."
    )

    refresh_btn.click(
        fn=_refresh_leaderboard,
        inputs=[metric_dd],
        outputs=[lb_table, lb_plot, run_dd, run_a_dd, run_b_dd],
    )
    inspect_btn.click(
        fn=_inspect_run_diagnostics,
        inputs=[run_dd],
        outputs=[cm_plot, roc_plot, curves_plot, per_cls_tbl, run_md],
    )
    cmp_btn.click(fn=_compare_two_runs, inputs=[run_a_dd, run_b_dd], outputs=[cmp_md])
