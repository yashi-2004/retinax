"""Unit tests for architectures, losses, training loop, checkpoints, SSL, and foundation models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.caching import FundusDataset
from src.data.synthetic import render_synthetic_fundus_phantom
from src.foundation.biomedical_clip import check_biomedclip_availability
from src.foundation.dinov2 import DINOv2FeatureExtractor, train_embedding_probe
from src.foundation.ssl_simclr import SimCLREncoderModel, train_simclr_ssl
from src.models.classifier import LinearProbeClassifier, MLPProbeClassifier, build_model, count_parameters
from src.models.custom_cnn import CustomFundusCNN
from src.models.inference import predict_single_image
from src.models.segmentation_unet import LesionUNetWrapper, compute_segmentation_metrics, soft_dice_bce_loss
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.losses import NTXentLoss, build_loss_function, compute_class_weights
from src.training.trainer import Trainer


def test_custom_cnn_forward_receptive_field_and_features() -> None:
    cnn = CustomFundusCNN(num_classes=5, num_blocks=4, base_channels=8, dropout=0.2)
    x = torch.randn(2, 3, 128, 128)
    logits = cnn(x)
    assert logits.shape == (2, 5)
    feats = cnn.forward_features(x)
    assert len(feats) == 4
    rf = cnn.compute_receptive_field(128)
    assert len(rf) == 4 and rf[-1]["receptive_field_px"] > rf[0]["receptive_field_px"]
    assert cnn.get_target_layer() is not None


def test_transfer_models_and_vit_offline_shapes_and_freeze_modes() -> None:
    x = torch.randn(2, 3, 128, 128)

    r_frozen = build_model("resnet18", mode_override="frozen", pretrained_override=False)
    assert r_frozen(x).shape == (2, 5)
    assert all(not p.requires_grad for p in r_frozen.backbone.parameters())
    assert all(p.requires_grad for p in r_frozen.head.parameters())

    e_partial = build_model("efficientnet_b0", mode_override="partial_finetune", pretrained_override=False)
    assert e_partial(x).shape == (2, 5)

    d_frozen = build_model("densenet121", mode_override="frozen", pretrained_override=False)
    assert d_frozen(x).shape == (2, 5)

    vit = build_model("vit_tiny", mode_override="frozen", pretrained_override=False)
    # ViT aligns 128x128 input cleanly to 224x224 without crashing
    assert vit(x).shape == (2, 5)
    sim_grid = vit.extract_cls_patch_similarity(x)
    assert sim_grid.shape == (2, 14, 14)

    with pytest.raises(ValueError, match="Unknown model_name"):
        build_model("nonexistent_arch")


def test_losses_and_segmentation_unet() -> None:
    labels = [0, 0, 0, 1, 2, 3, 4]
    w = compute_class_weights(labels, num_classes=5)
    assert w.shape == (5,) and torch.isfinite(w).all()

    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 4])
    for name in ("ce", "weighted_ce", "focal", "label_smoothing"):
        loss_fn = build_loss_function(name, class_weights=w)
        val = loss_fn(logits, targets)
        assert val.ndim == 0 and torch.isfinite(val)

    ntx = NTXentLoss(temperature=0.5)
    z1, z2 = torch.randn(4, 32, requires_grad=True), torch.randn(4, 32, requires_grad=True)
    assert torch.isfinite(ntx(z1, z2))

    unet = LesionUNetWrapper()
    seg_x = torch.randn(2, 3, 64, 64)
    seg_y = (torch.rand(2, 1, 64, 64) > 0.8).float()
    seg_logits = unet(seg_x)
    assert seg_logits.shape == (2, 1, 64, 64)
    assert torch.isfinite(soft_dice_bce_loss(seg_logits, seg_y))
    seg_m = compute_segmentation_metrics(seg_logits, seg_y)
    assert 0.0 <= seg_m["dice"] <= 1.0


def test_training_loop_checkpoint_and_inference(tmp_path: Path, synthetic_workspace: dict) -> None:
    df = synthetic_workspace["manifest_primary"]
    cfg = synthetic_workspace["cfg"]

    train_ds = FundusDataset(df[df["split"] == "train"], image_size=128)
    val_ds = FundusDataset(df[df["split"] == "val"], image_size=128)
    model = build_model("custom_cnn", cfg=cfg)

    ckpt_path = tmp_path / "test_cnn.pt"
    trainer = Trainer(
        model=model,
        model_name="custom_cnn",
        cfg=cfg,
        run_id="pytest_cnn",
        checkpoint_path=ckpt_path,
        tb_log_dir=tmp_path / "tb",
    )
    res = trainer.fit(train_ds, val_ds, epochs=2, batch_size=8, image_size=128, is_synthetic=True)
    assert ckpt_path.is_file()
    assert len(res["history"]) == 2

    reloaded, meta = load_checkpoint(ckpt_path)
    assert meta["model_name"] == "custom_cnn"
    assert count_parameters(reloaded)["total_params"] == count_parameters(model)["total_params"]

    phantom = render_synthetic_fundus_phantom(grade=2, seed=5, size=128)
    pred = predict_single_image(reloaded, phantom, image_size=128, temperature=1.0)
    assert np.isclose(pred["probabilities"].sum(), 1.0, atol=1e-4)
    assert pred["predicted_class"] in {0, 1, 2, 3, 4}

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "missing_ckpt.pt")


def test_simclr_ssl_and_foundation_probes(tmp_path: Path, synthetic_workspace: dict) -> None:
    df = synthetic_workspace["manifest_primary"]
    train_df = df[df["split"] == "train"]

    ssl_model = SimCLREncoderModel(encoder_type="custom_cnn", projection_dim=32)
    ssl_res = train_simclr_ssl(
        ssl_model=ssl_model,
        unlabelled_df=train_df,
        epochs=2,
        batch_size=8,
        image_size=128,
        checkpoint_path=tmp_path / "ssl.pt",
        tb_dir=tmp_path / "tb_ssl",
    )
    assert Path(ssl_res["checkpoint_path"]).is_file()
    assert np.isfinite(ssl_res["final_nt_xent_loss"])

    # DINOv2 architecture offline check (pretrained=False for offline test)
    dino = DINOv2FeatureExtractor(img_size=224, pretrained=False)
    feats = dino(torch.randn(2, 3, 128, 128))
    assert feats.shape == (2, dino.embedding_dim)

    # Probe tests
    emb_tr = np.random.randn(20, 32).astype(np.float32)
    y_tr = np.array([0, 1, 2, 3, 4] * 4)
    emb_te = np.random.randn(6, 32).astype(np.float32)
    res_log = train_embedding_probe(emb_tr, y_tr, emb_te, probe_type="logistic")
    assert res_log["eval_probabilities"].shape == (6, 5)
    res_mlp = train_embedding_probe(emb_tr, y_tr, emb_te, probe_type="mlp", epochs=5)
    assert res_mlp["eval_probabilities"].shape == (6, 5)

    assert LinearProbeClassifier(32)(torch.randn(3, 32)).shape == (3, 5)
    assert MLPProbeClassifier(32)(torch.randn(3, 32)).shape == (3, 5)

    # Optional BiomedCLIP availability check never crashes
    bclip = check_biomedclip_availability()
    assert isinstance(bclip["available"], bool)
