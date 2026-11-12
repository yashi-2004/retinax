"""Research Overview page: objectives, datasets, checkpoints, hardware, and synthetic-demo generator."""

from __future__ import annotations

import gradio as gr
import pandas as pd

from src.config import (
    CENTRAL_RESEARCH_QUESTION,
    RESEARCH_TITLE,
    SUPPORTING_RESEARCH_QUESTIONS,
    get_hardware_summary,
    load_config,
)
from src.data.ingestion import discover_datasets, load_raw_records
from src.data.manifests import build_manifest, save_manifest
from src.data.splitting import assign_splits
from src.data.synthetic import generate_synthetic_demo_dataset
from src.experiments.tracking import ExperimentRegistry
from src.preprocessing.quality_assessment import assess_fundus_quality
from src.training.checkpoints import list_checkpoints

TAB_TITLE = "1. Research Overview"


def _collect_overview_state() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    ds_status = discover_datasets()
    ds_rows = [
        {
            "Dataset": v["name"],
            "Title": v["title"],
            "Available": v["available"],
            "Images": v["image_count"],
            "Patient IDs": v["patient_id_available"],
            "Synthetic": v["is_synthetic"],
            "Status Note": v["note"],
        }
        for v in ds_status.values()
    ]
    ds_df = pd.DataFrame(ds_rows)

    ckpts = list_checkpoints()
    ckpt_df = pd.DataFrame(
        [
            {
                "Checkpoint": c.get("name"),
                "Architecture": c.get("model_name"),
                "Dataset": c.get("dataset_name"),
                "Synthetic": c.get("is_synthetic"),
                "Epoch": c.get("epoch"),
                "Val QWK": c.get("metrics", {}).get("val_qwk"),
                "Size (MB)": c.get("size_mb"),
            }
            for c in ckpts
        ]
    ) if ckpts else pd.DataFrame([{"Status": "No trained checkpoints found yet. Train a model or run scripts/train.py."}])

    reg_df = ExperimentRegistry().comparison_dataframe()
    if reg_df.empty:
        reg_df = pd.DataFrame([{"Status": "No tracked experiment runs yet."}])

    hw = get_hardware_summary()
    hw_md = (
        f"**Execution Mode:** `{hw['execution_mode']}` | "
        f"**OS:** `{hw['os']} ({hw['architecture']})` | "
        f"**CPU Cores:** `{hw['cpu_cores']}` | "
        f"**PyTorch:** `{hw['torch_version']}` | "
        f"**Mixed Precision:** `{hw['mixed_precision_enabled']}`"
    )
    return ds_df, ckpt_df, reg_df, hw_md


def _generate_demo_action() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    cfg = load_config()
    generate_synthetic_demo_dataset(num_primary=80, num_external=40, image_size=224, seed=42)

    for ds_key in ("synthetic_primary", "synthetic_external"):
        raw = load_raw_records(ds_key, seed=42)
        manifest_df, summary = build_manifest(raw, processing_config=cfg.get("preprocessing", {}), quality_fn=assess_fundus_quality)
        if ds_key == "synthetic_external":
            manifest_df["split"] = "external_val"
        else:
            manifest_df, audit = assign_splits(manifest_df, seed=42)
            summary["split_audit"] = audit
        save_manifest(manifest_df, name=ds_key, summary=summary)

    ds_df, ckpt_df, reg_df, hw_md = _collect_overview_state()
    msg = (
        "Generated non-clinical synthetic demonstration cohorts (`synthetic_primary`: 80 phantoms, "
        "`synthetic_external`: 40 domain-shifted phantoms) and built leakage-free manifests."
    )
    return ds_df, ckpt_df, reg_df, hw_md, msg


def build_tab() -> None:
    questions_md = "\n".join(f"{i}. {q}" for i, q in enumerate(SUPPORTING_RESEARCH_QUESTIONS, 1))
    gr.Markdown(
        f"### {RESEARCH_TITLE}\n\n"
        f"**Central Research Question:** {CENTRAL_RESEARCH_QUESTION}\n\n"
        f"**Supporting Research Questions:**\n{questions_md}"
    )

    with gr.Row():
        refresh_btn = gr.Button("Refresh Platform Status", variant="secondary")
        gen_demo_btn = gr.Button("Initialize Synthetic Demo Datasets (Non-Clinical)", variant="primary")
    status_out = gr.Markdown("")
    hw_out = gr.Markdown("")

    gr.Markdown("#### Dataset Governance & Availability")
    ds_table = gr.Dataframe(interactive=False, wrap=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Available Trained Checkpoints")
            ckpt_table = gr.Dataframe(interactive=False, wrap=True)
        with gr.Column():
            gr.Markdown("#### Latest Tracked Experiments")
            exp_table = gr.Dataframe(interactive=False, wrap=True)

    ds_init, ck_init, exp_init, hw_init = _collect_overview_state()
    ds_table.value = ds_init
    ckpt_table.value = ck_init
    exp_table.value = exp_init
    hw_out.value = hw_init

    refresh_btn.click(
        fn=_collect_overview_state,
        inputs=[],
        outputs=[ds_table, ckpt_table, exp_table, hw_out],
    )
    gen_demo_btn.click(
        fn=_generate_demo_action,
        inputs=[],
        outputs=[ds_table, ckpt_table, exp_table, hw_out, status_out],
    )
