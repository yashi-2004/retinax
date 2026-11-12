"""Unit tests for Grad-CAM, Captum IG/Occlusion, MC Dropout, Ensembles, Calibration, Conformal, and Selective AI."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.data.synthetic import render_synthetic_fundus_phantom
from src.evaluation.calibration import TemperatureScaler, apply_temperature_scaling, compute_calibration_metrics
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.integrated_gradients import compute_integrated_gradients, compute_occlusion_sensitivity
from src.explainability.visualisation import build_explanation_figure, compute_fov_attribution_ratio, overlay_heatmap
from src.models.classifier import build_model
from src.models.custom_cnn import CustomFundusCNN
from src.uncertainty.conformal import SplitConformalPredictor
from src.uncertainty.ensembles import predict_ensemble_single
from src.uncertainty.mc_dropout import enable_dropout_only, mc_dropout_predict_single
from src.uncertainty.selective_prediction import compute_risk_coverage_curve, evaluate_selective_decision


def test_gradcam_gradcampp_vit_and_captum_xai() -> None:
    cnn = CustomFundusCNN(num_classes=5, num_blocks=3, base_channels=8)
    cnn.eval()
    x = torch.randn(1, 3, 128, 128)
    phantom = render_synthetic_fundus_phantom(grade=2, seed=4, size=128)

    explainer = GradCAMExplainer(cnn)
    gc = explainer.generate(x, method="gradcam")
    gcpp = explainer.generate(x, method="gradcam_pp")
    assert gc["heatmap"].shape == (128, 128)
    assert 0.0 <= float(gc["heatmap"].min()) <= float(gc["heatmap"].max()) <= 1.0
    assert gcpp["heatmap"].shape == (128, 128)

    # Test frozen-backbone Grad-CAM still produces valid gradients!
    resnet_frozen = build_model("resnet18", mode_override="frozen", pretrained_override=False)
    gc_frozen = GradCAMExplainer(resnet_frozen).generate(x, method="gradcam")
    assert gc_frozen["heatmap"].shape == (128, 128) and float(gc_frozen["heatmap"].max()) >= 0.0

    # Test ViT token Grad-CAM reshaping
    vit = build_model("vit_tiny", mode_override="frozen", pretrained_override=False)
    gc_vit = GradCAMExplainer(vit).generate(torch.randn(1, 3, 224, 224), method="gradcam")
    assert gc_vit["heatmap"].shape == (224, 224)

    # Captum Integrated Gradients & Occlusion Sensitivity
    ig_res = compute_integrated_gradients(cnn, x, n_steps=6)
    assert ig_res["heatmap"].shape == (128, 128)
    occ_res = compute_occlusion_sensitivity(cnn, x, patch_size=32, stride=32)
    assert occ_res["heatmap"].shape == (128, 128)

    hm_rgb, overlay = overlay_heatmap(phantom, gc["heatmap"])
    assert hm_rgb.shape == (128, 128, 3) and overlay.shape == (128, 128, 3)
    fov_ratio = compute_fov_attribution_ratio(phantom, gc["heatmap"])
    assert 0.0 <= fov_ratio <= 1.0
    fig = build_explanation_figure(phantom, gc["heatmap"], gc["probabilities"], gc["predicted_class"])
    assert fig is not None


def test_mc_dropout_and_deep_ensemble() -> None:
    cnn_a = CustomFundusCNN(num_classes=5, num_blocks=3, base_channels=8, dropout=0.35)
    cnn_b = CustomFundusCNN(num_classes=5, num_blocks=3, base_channels=8, dropout=0.35)

    n_drop = enable_dropout_only(cnn_a)
    assert n_drop >= 1
    for m in cnn_a.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert m.training is False
        elif isinstance(m, nn.Dropout):
            assert m.training is True

    x = torch.randn(1, 3, 128, 128)
    mc_out = mc_dropout_predict_single(cnn_a, x, n_passes=6)
    assert mc_out["predictive_entropy"] >= 0.0
    assert mc_out["mutual_information"] >= 0.0
    assert mc_out["predictive_variance"] >= 0.0

    ens_out = predict_ensemble_single([cnn_a, cnn_b], x)
    assert ens_out["n_members"] == 2
    assert 0.0 <= ens_out["member_agreement"] <= 1.0


def test_calibration_temperature_scaling_conformal_and_selective() -> None:
    rng = np.random.default_rng(123)
    # Overconfident logits (multiplied by 4.0) so temperature scaling improves NLL/ECE
    y = rng.integers(0, 5, size=80)
    logits = rng.normal(0.0, 1.0, size=(80, 5))
    logits[np.arange(80), y] += 1.6
    logits *= 3.5

    raw_probs = apply_temperature_scaling(logits, 1.0)
    before = compute_calibration_metrics(y, raw_probs)

    scaler = TemperatureScaler()
    fit_info = scaler.fit(logits[:40], y[:40])
    assert fit_info["temperature"] > 1.0

    scaled_probs = apply_temperature_scaling(logits[40:], fit_info["temperature"])
    after = compute_calibration_metrics(y[40:], scaled_probs)
    assert after["nll"] <= compute_calibration_metrics(y[40:], raw_probs[40:])["nll"] + 1e-4

    # Conformal LAC & APS
    for method in ("lac", "aps"):
        cp = SplitConformalPredictor(alpha=0.10, method=method)
        cp.fit(scaled_probs[:20], y[40:60])
        eval_res = cp.evaluate(scaled_probs[20:], y[60:])
        assert 0.0 <= eval_res["empirical_coverage"] <= 1.0
        assert eval_res["average_set_size"] >= 1.0
        single = cp.predict_single_set(scaled_probs[20])
        assert single["set_size"] >= 1

    # Selective prediction & abstention gate
    abstain_low = evaluate_selective_decision(np.array([0.25, 0.25, 0.20, 0.15, 0.15]), min_confidence=0.45)
    assert abstain_low["abstain"] is True
    accept_hi = evaluate_selective_decision(np.array([0.92, 0.04, 0.02, 0.01, 0.01]), predictive_entropy=0.35, min_confidence=0.45, max_entropy=1.25)
    assert accept_hi["abstain"] is False

    rc = compute_risk_coverage_curve(y[40:], scaled_probs)
    assert 0.0 <= rc["aurc"] <= 1.0 and len(rc["points"]) > 0
