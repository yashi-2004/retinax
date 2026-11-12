"""Research Reports & TensorBoard Experiment Tracking page."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.config import resolve_path
from src.reporting.report_generator import generate_research_report

TAB_TITLE = "14. Research Reports"


def _build_reports_action(report_name: str) -> tuple[str, str | None, str | None, str | None, str | None]:
    clean_name = (report_name or "retinax_research_report").strip().replace(" ", "_")
    res = generate_research_report(report_name=clean_name)
    msg = (
        f"Generated reproducible research bundle from **{res['num_runs']}** tracked experiment(s).\n\n"
        f"```json\n{json.dumps(res, indent=2)}\n```"
    )
    return (
        msg,
        res["markdown_path"],
        res["html_path"],
        res["pdf_path"],
        res["csv_path"],
    )


def build_tab() -> None:
    tb_path = resolve_path("experiments/tensorboard")
    gr.Markdown(
        "### Reproducible Research Report Generator & TensorBoard Tracking\n"
        "Exports tracked experiment tables, diagnostic plots, hypotheses, limitations, and safety disclaimers "
        "into downloadable Markdown, standalone HTML, PDF, and CSV artifacts."
    )

    with gr.Row():
        name_tb = gr.Textbox(value="retinax_research_report", label="Report Filename Stem")
        gen_btn = gr.Button("Generate Research Report Bundle", variant="primary")

    status_md = gr.Markdown("")
    with gr.Row():
        md_file = gr.File(label="Download Markdown Report (.md)")
        html_file = gr.File(label="Download HTML Report (.html)")
        pdf_file = gr.File(label="Download PDF Report (.pdf)")
        csv_file = gr.File(label="Download Experiment Metrics (.csv)")

    gr.Markdown(
        "#### Local TensorBoard Experiment Tracking\n"
        f"All supervised and SimCLR SSL runs write scalar event logs to `{tb_path}`.\n\n"
        "Launch TensorBoard locally in a separate terminal (zero cost, offline):\n"
        "```bash\n"
        ".venv/bin/tensorboard --logdir experiments/tensorboard --port 6006\n"
        "```"
    )

    gen_btn.click(
        fn=_build_reports_action,
        inputs=[name_tb],
        outputs=[status_md, md_file, html_file, pdf_file, csv_file],
    )
