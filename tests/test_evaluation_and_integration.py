"""Integration tests for evaluation, domain shift, reports, and full Gradio inference workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.components.state import run_full_clinical_research_inference
from app.main import create_app
from src.data.synthetic import render_synthetic_fundus_phantom
from src.evaluation.classification import compute_classification_metrics, compute_curve_data, compute_qwk
from src.evaluation.generalisation import evaluate_ood_detection
from src.evaluation.robustness import evaluate_perturbation_robustness, evaluate_quality_stratified
from src.evaluation.statistical_tests import mcnemar_test, paired_bootstrap_ci, paired_bootstrap_comparison
from src.experiments.ablation import run_supervised_experiment
from src.models.custom_cnn import CustomFundusCNN
from src.reporting.report_generator import generate_research_report
from src.training.checkpoints import save_checkpoint


def test_classification_metrics_and_statistical_tests() -> None:
    y_true = np.array([0, 0, 1, 2, 2, 3, 4, 0, 1, 2])
    probs_a = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.7, 0.2, 0.05, 0.03, 0.02],
        [0.2, 0.6, 0.1, 0.05, 0.05],
        [0.1, 0.1, 0.6, 0.1, 0.1],
        [0.1, 0.2, 0.5, 0.1, 0.1],
        [0.05, 0.05, 0.2, 0.6, 0.1],
        [0.05, 0.05, 0.1, 0.2, 0.6],
        [0.4, 0.45, 0.05, 0.05, 0.05],
        [0.1, 0.5, 0.2, 0.1, 0.1],
        [0.05, 0.15, 0.6, 0.1, 0.1],
    ])
    probs_b = np.clip(probs_a + 0.01, 0, 1)
    probs_b /= probs_b.sum(axis=1, keepdims=True)

    cls_m = compute_classification_metrics(y_true, probs_a)
    assert 0.5 <= cls_m["accuracy"] <= 1.0
    assert 0.5 <= cls_m["quadratic_weighted_kappa"] <= 1.0
    assert cls_m["ordinal_errors"]["exact_match_rate"] + cls_m["ordinal_errors"]["one_grade_error_rate"] + cls_m["ordinal_errors"]["severe_multigrade_error_rate"] == pytest.approx(1.0, abs=1e-3)

    curves = compute_curve_data(y_true, probs_a)
    assert "roc" in curves and "pr" in curves

    # Single-class edge case does not crash
    single_m = compute_classification_metrics(np.array([0, 0, 0]), probs_a[:3])
    assert single_m["macro_roc_auc_ovr"] is None

    ci = paired_bootstrap_ci(y_true, probs_a, metric_fn=lambda y, p: compute_qwk(y, np.argmax(p, axis=1)), n_bootstrap=50)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]
    cmp_res = paired_bootstrap_comparison(y_true, probs_a, probs_b, metric_fn=lambda y, p: compute_qwk(y, np.argmax(p, axis=1)), n_bootstrap=50)
    assert "p_value" in cmp_res
    mcn = mcnemar_test(y_true, np.argmax(probs_a, axis=1), np.argmax(probs_b, axis=1))
    assert 0.0 <= mcn["p_value"] <= 1.0

    # Quality-stratified & OOD evaluation
    q_strat = evaluate_quality_stratified(y_true, probs_a, ["Good"] * 6 + ["Poor / Review Recommended"] * 4)
    assert "filtering_impact" in q_strat and q_strat["filtering_impact"] is not None

    ood = evaluate_ood_detection(
        id_probs=probs_a,
        ood_probs=np.full((5, 5), 0.2),
        train_embeddings=np.random.randn(15, 8),
        id_embeddings=np.random.randn(10, 8),
        ood_embeddings=np.random.randn(5, 8) + 3.0,
    )
    assert "max_softmax_inv" in ood and "mahalanobis_distance" in ood


def test_end_to_end_experiment_pipeline_perturbation_and_report(tmp_path: Path, synthetic_workspace: dict) -> None:
    cfg = dict(synthetic_workspace["cfg"])
    cfg["paths"] = {
        **cfg["paths"],
        "checkpoints_dir": str(tmp_path / "ckpts"),
        "experiments_dir": str(tmp_path / "exps"),
        "reports_dir": str(tmp_path / "reports"),
    }
    manifest_p = synthetic_workspace["manifest_primary"]
    manifest_e = synthetic_workspace["manifest_external"]

    res = run_supervised_experiment(
        manifest_df=manifest_p,
        model_name="custom_cnn",
        run_id="e2e_pytest_cnn",
        experiment_group="Integration_Test",
        cfg=cfg,
        external_manifest_df=manifest_e,
        epochs=2,
        batch_size=8,
        image_size=128,
        seed=42,
    )
    assert Path(res["checkpoint_path"]).is_file()
    assert res["metrics"]["external_generalisation"] is not None

    # Controlled perturbation robustness check
    from src.training.checkpoints import load_checkpoint
    model, _ = load_checkpoint(res["checkpoint_path"])
    pert_res = evaluate_perturbation_robustness(
        model, manifest_p[manifest_p["split"] == "test"], perturbation="gaussian_blur", severities=(0.0, 0.5), image_size=128
    )
    assert len(pert_res) == 2

    # Full MD/HTML/PDF/CSV report generation
    rep = generate_research_report(experiments_dir=tmp_path / "exps", output_dir=tmp_path / "reports")
    assert Path(rep["markdown_path"]).is_file()
    assert Path(rep["html_path"]).is_file()
    assert Path(rep["pdf_path"]).is_file()
    assert Path(rep["csv_path"]).is_file()

    # Full 10-step Gradio inference workflow check (valid + missing checkpoint + invalid image)
    phantom = render_synthetic_fundus_phantom(grade=3, seed=9, size=128)
    wf_ok = run_full_clinical_research_inference(phantom, checkpoint_path=res["checkpoint_path"], mc_passes=4)
    assert wf_ok["ok"] is True
    assert wf_ok["overlay_rgb"].shape == (128, 128, 3)

    wf_no_ckpt = run_full_clinical_research_inference(phantom, checkpoint_path=None)
    assert wf_no_ckpt["ok"] is False and "No trained model checkpoint" in wf_no_ckpt["error"]

    wf_tiny = run_full_clinical_research_inference(np.zeros((10, 10, 3), dtype=np.uint8), checkpoint_path=res["checkpoint_path"])
    assert wf_tiny["ok"] is False


def test_gradio_blocks_app_constructs_cleanly() -> None:
    import asyncio
    import gradio as gr
    import httpx

    app = create_app()
    assert app is not None
    # Verify all 14 tabs rendered into Gradio Blocks config
    blocks_cfg = app.get_config_file()
    assert len(blocks_cfg.get("components", [])) > 20

    # Verify live FastAPI/Gradio HTTP serving routes (including Pydantic bool schema compat)
    async def _check_routes() -> None:
        fastapi_app = gr.routes.App.create_app(app)
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:7860") as client:
            root_resp = await client.get("/")
            cfg_resp = await client.get("/config")
            assert root_resp.status_code == 200 and "RetinaX" in root_resp.text
            assert cfg_resp.status_code == 200 and len(cfg_resp.json().get("components", [])) > 20

    asyncio.run(_check_routes())
