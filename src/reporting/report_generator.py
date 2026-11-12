"""Automated reproducible research report generator (Markdown, HTML, PDF)."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from src.config import (
    CENTRAL_RESEARCH_QUESTION,
    CLINICAL_DISCLAIMER,
    RESEARCH_TITLE,
    SUPPORTING_RESEARCH_QUESTIONS,
    get_hardware_summary,
    resolve_path,
)
from src.experiments.tracking import ExperimentRegistry
from src.reporting.figures import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_reliability_diagram,
    plot_risk_coverage_curve,
)


def _format_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No experiment runs recorded yet._\n"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.itertuples(index=False):
        vals = [str(getattr(row, c) if getattr(row, c) is not None else "N/A") for c in cols]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows]) + "\n"


def _sanitize_ascii(text: str) -> str:
    """Replace typographic dashes/quotes cleanly for core Helvetica PDF font compatibility."""
    return (
        text.replace("\u2014", "--")
        .replace("\u2013", "-")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .encode("latin-1", "replace")
        .decode("latin-1")
    )


def generate_research_report(
    experiments_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str = "retinax_research_report",
) -> dict[str, Any]:
    """Compile recorded RetinaX experiments into Markdown, HTML, PDF, and figure PNGs."""
    registry = ExperimentRegistry(experiments_dir)
    out_dir = resolve_path(output_dir or "reports/generated")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs_df = registry.comparison_dataframe()
    hw = get_hardware_summary()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    has_synthetic = bool(runs_df["is_synthetic"].any()) if "is_synthetic" in runs_df.columns and len(runs_df) else False

    generated_figs: list[Path] = []
    if len(runs_df) > 0:
        comp_fig_path = fig_dir / "model_comparison_qwk.png"
        fig = plot_model_comparison(runs_df, metric_col="qwk", title="Tracked Runs: Quadratic Weighted Kappa", save_path=comp_fig_path)
        plt.close(fig)
        generated_figs.append(comp_fig_path)

        # Render detailed diagnostic figures for the best QWK run
        valid_qwk = runs_df.dropna(subset=["qwk"])
        if len(valid_qwk) > 0:
            best_run_id = str(valid_qwk.sort_values("qwk", ascending=False).iloc[0]["run_id"])
            best_details = registry.load_run_details(best_run_id)
            m = best_details["metrics"]

            cm_path = fig_dir / f"{best_run_id}_confusion.png"
            cm_fig = plot_confusion_matrix(m["test"]["confusion_matrix_normalized"], normalized=True, title=f"Normalized Confusion Matrix ({best_run_id})", save_path=cm_path)
            plt.close(cm_fig)
            generated_figs.append(cm_path)

            rel_path = fig_dir / f"{best_run_id}_reliability.png"
            rel_fig = plot_reliability_diagram(m["calibration_before"], m.get("calibration_after"), title=f"Calibration Reliability ({best_run_id})", save_path=rel_path)
            plt.close(rel_fig)
            generated_figs.append(rel_path)

            rc_path = fig_dir / f"{best_run_id}_risk_coverage.png"
            rc_fig = plot_risk_coverage_curve(m["selective_prediction"], title=f"Risk-Coverage Curve ({best_run_id})", save_path=rc_path)
            plt.close(rc_fig)
            generated_figs.append(rc_path)

    display_cols = [
        c for c in [
            "run_id", "experiment_group", "model_name", "dataset_name", "is_synthetic",
            "accuracy", "balanced_accuracy", "macro_f1", "qwk", "ece", "brier_score",
            "total_params", "ms_per_image", "training_seconds",
        ]
        if c in runs_df.columns
    ]
    summary_table = runs_df[display_cols].copy() if display_cols else pd.DataFrame()

    synthetic_banner = (
        "> **IMPORTANT DATA PROVENANCE NOTICE**: One or more runs in this report used "
        "**synthetic demonstration phantoms (`is_synthetic=True`)**. Metrics below verify software "
        "behaviour and experimental methodology ONLY and must NOT be interpreted as clinical retinal performance.\n\n"
        if has_synthetic
        else ""
    )

    questions_md = "\n".join(f"{idx}. {q}" for idx, q in enumerate(SUPPORTING_RESEARCH_QUESTIONS, 1))
    figs_md = "\n".join(f"![{p.stem}](figures/{p.name})\n" for p in generated_figs)

    md_content = f"""# {RESEARCH_TITLE}

**Generated:** {now_utc}  
**Framework:** RetinaX v0.1.0 (Local CPU-Only Reproducible Research Platform)

> **Clinical Safety Disclaimer:** {CLINICAL_DISCLAIMER}

{synthetic_banner}## 1. Central & Supporting Research Questions

**Central Question:** {CENTRAL_RESEARCH_QUESTION}

**Supporting Questions:**
{questions_md}

## 2. Hardware & Reproducibility Environment

- **OS & Architecture:** {hw['os']} {hw['os_release']} ({hw['architecture']})
- **CPU Cores:** {hw['cpu_cores']} | **Execution Mode:** {hw['execution_mode']}
- **PyTorch:** {hw['torch_version']} | **Mixed Precision:** {hw['mixed_precision_enabled']}

## 3. Empirical Experiment Summary

{_format_markdown_table(summary_table)}

## 4. Diagnostic Visualisations

{figs_md if figs_md else "_No figures generated yet — run an experiment first._"}

## 5. Methodological Limitations & Ethics

1. **Class Imbalance & Ordinal Severity:** High overall accuracy can mask poor minority-class sensitivity (Mild/Severe/PDR); QWK, Macro-F1, and per-class recall must always be evaluated jointly.
2. **Post-Hoc Explainability Caveat:** Grad-CAM and Integrated Gradients highlight regions correlated with model logits and do not constitute validated clinical lesion segmentation or causal proof.
3. **Calibration & Conformal Assumptions:** Temperature scaling and Split Conformal Prediction rely on held-out calibration exchangeability; external dataset shift can degrade coverage and reliability.
4. **Fairness & Subgroup Metadata:** Public DR benchmarks (e.g., APTOS 2019) generally lack patient demographics; demographics must never be inferred from fundus pixels.
"""

    md_path = out_dir / f"{report_name}.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Standalone HTML export
    table_html = summary_table.to_html(index=False, border=0, classes="rx-table") if not summary_table.empty else "<p>No runs recorded.</p>"
    imgs_html = "".join(f'<div class="fig-card"><img src="figures/{html.escape(p.name)}" alt="{html.escape(p.stem)}"/></div>' for p in generated_figs)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>RetinaX Research Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #0F172A; background: #F8FAFC; margin: 0; padding: 32px; line-height: 1.55; }}
  .container {{ max-width: 1080px; margin: 0 auto; background: #FFFFFF; padding: 36px; border: 1px solid #E2E8F0; border-radius: 10px; }}
  h1, h2 {{ color: #0F172A; border-bottom: 2px solid #E2E8F0; padding-bottom: 6px; }}
  .disclaimer {{ background: #FEF2F2; border-left: 4px solid #DC2626; padding: 12px 16px; margin: 16px 0; color: #7F1D1D; }}
  .syn-banner {{ background: #FFFBEB; border-left: 4px solid #D97706; padding: 12px 16px; margin: 16px 0; color: #78350F; }}
  .rx-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 16px 0; }}
  .rx-table th {{ background: #0F172A; color: #FFFFFF; padding: 8px; text-align: left; }}
  .rx-table td {{ border-bottom: 1px solid #E2E8F0; padding: 8px; }}
  .fig-card {{ margin: 18px 0; text-align: center; }}
  .fig-card img {{ max-width: 100%; border: 1px solid #E2E8F0; border-radius: 6px; }}
</style>
</head>
<body>
<div class="container">
  <h1>{html.escape(RESEARCH_TITLE)}</h1>
  <p><strong>Generated:</strong> {html.escape(now_utc)} | <strong>Environment:</strong> {html.escape(hw['os'])} ({html.escape(hw['architecture'])}), {html.escape(hw['execution_mode'])}, PyTorch {html.escape(hw['torch_version'])}</p>
  <div class="disclaimer"><strong>Clinical Safety Disclaimer:</strong> {html.escape(CLINICAL_DISCLAIMER)}</div>
  {f'<div class="syn-banner"><strong>Synthetic Data Notice:</strong> Recorded runs include procedural synthetic demonstration phantoms (is_synthetic=True).</div>' if has_synthetic else ''}
  <h2>Central Research Question</h2>
  <p>{html.escape(CENTRAL_RESEARCH_QUESTION)}</p>
  <h2>Empirical Experiment Summary</h2>
  {table_html}
  <h2>Diagnostic Visualisations</h2>
  {imgs_html}
</div>
</body>
</html>
"""
    html_path = out_dir / f"{report_name}.html"
    html_path.write_text(html_content, encoding="utf-8")

    # Pure-Python PDF export via fpdf2
    pdf_path = out_dir / f"{report_name}.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(pdf.epw, 7, _sanitize_ascii(RESEARCH_TITLE), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(pdf.epw, 5, _sanitize_ascii(f"Generated: {now_utc} | Mode: {hw['execution_mode']} | PyTorch: {hw['torch_version']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(pdf.epw, 5, _sanitize_ascii(f"DISCLAIMER: {CLINICAL_DISCLAIMER}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if has_synthetic:
        pdf.ln(1)
        pdf.multi_cell(pdf.epw, 5, "NOTICE: Contains synthetic demonstration runs (is_synthetic=True).", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(pdf.epw, 6, "Recorded Experiment Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    if len(summary_table) == 0:
        pdf.cell(pdf.epw, 5, "No experiment runs recorded yet.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        for row in summary_table.to_dict(orient="records"):
            line = (
                f"{row.get('run_id')} | model={row.get('model_name')} | "
                f"acc={row.get('accuracy')} | f1={row.get('macro_f1')} | "
                f"qwk={row.get('qwk')} | ece={row.get('ece')}"
            )
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 4.5, _sanitize_ascii(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for fig_p in generated_figs[:3]:
        pdf.ln(3)
        pdf.set_x(pdf.l_margin)
        pdf.image(str(fig_p), w=175)

    pdf.output(str(pdf_path))

    # Also export clean CSV summary
    csv_path = out_dir / f"{report_name}_runs.csv"
    summary_table.to_csv(csv_path, index=False)

    return {
        "markdown_path": str(md_path),
        "html_path": str(html_path),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "figures": [str(p) for p in generated_figs],
        "num_runs": int(len(runs_df)),
    }
