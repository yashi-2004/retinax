"""CPU-compatible, reproducible training loop with early stopping, checkpointing, and TensorBoard."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.config import resolve_device_and_threads, resolve_path
from src.evaluation.classification import compute_classification_metrics
from src.training.checkpoints import save_checkpoint
from src.training.losses import build_loss_function, compute_class_weights
from src.training.optimizers import build_optimizer
from src.training.scheduler import build_scheduler


def make_weighted_sampler(labels: np.ndarray | list[int]) -> WeightedRandomSampler:
    """Create a class-balanced WeightedRandomSampler for imbalanced training sets."""
    arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(arr)
    class_w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    sample_weights = torch.tensor(class_w[arr], dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(arr), replacement=True)


class Trainer:
    """Unified PyTorch classification trainer designed for memory-constrained CPU execution."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        cfg: dict[str, Any] | None = None,
        run_id: str = "run_default",
        checkpoint_path: str | Path | None = None,
        tb_log_dir: str | Path | None = None,
    ) -> None:
        self.cfg = cfg or {}
        hw = resolve_device_and_threads(self.cfg)
        self.device: torch.device = hw["device"]
        self.amp_enabled: bool = bool(hw["amp_supported"] and self.cfg.get("training", {}).get("mixed_precision", False))

        self.model = model.to(self.device)
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.run_id = run_id

        ckpt_dir = resolve_path(self.cfg.get("paths", {}).get("checkpoints_dir", "checkpoints"))
        self.checkpoint_path = resolve_path(checkpoint_path) if checkpoint_path else (ckpt_dir / f"{run_id}.pt")
        tb_base = resolve_path(self.cfg.get("paths", {}).get("experiments_dir", "experiments")) / "tensorboard"
        self.tb_dir = resolve_path(tb_log_dir) if tb_log_dir else (tb_base / run_id)

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        optimizer_name: str | None = None,
        scheduler_name: str | None = None,
        loss_name: str | None = None,
        use_weighted_sampler: bool | None = None,
        early_stopping_patience: int | None = None,
        gradient_clip_norm: float | None = None,
        seed: int = 42,
        dataset_name: str = "unknown",
        preprocessing_preset: str = "fov_clahe_norm",
        image_size: int = 224,
        is_synthetic: bool = False,
        monitor: str = "val_loss",
    ) -> dict[str, Any]:
        """Run full training loop with validation monitoring, early stopping, and best checkpointing."""
        t_cfg = self.cfg.get("training", {})
        epochs = int(epochs if epochs is not None else t_cfg.get("epochs", 10))
        batch_size = int(batch_size if batch_size is not None else t_cfg.get("batch_size", 8))
        lr = float(lr if lr is not None else t_cfg.get("learning_rate", 3e-4))
        weight_decay = float(weight_decay if weight_decay is not None else t_cfg.get("weight_decay", 1e-4))
        optimizer_name = str(optimizer_name or t_cfg.get("optimizer", "adamw"))
        scheduler_name = str(scheduler_name or t_cfg.get("scheduler", "cosine"))
        loss_name = str(loss_name or t_cfg.get("loss", "weighted_ce"))
        use_sampler = bool(use_weighted_sampler if use_weighted_sampler is not None else t_cfg.get("use_weighted_sampler", False))
        patience = int(early_stopping_patience if early_stopping_patience is not None else t_cfg.get("early_stopping_patience", 4))
        clip_norm = float(gradient_clip_norm if gradient_clip_norm is not None else t_cfg.get("gradient_clip_norm", 1.0))

        # Extract labels cleanly for class-weight / weighted-sampler setup
        if hasattr(train_dataset, "df") and "original_label" in train_dataset.df.columns:
            train_labels = train_dataset.df["original_label"].astype(int).to_numpy()
        else:
            train_labels = np.array([int(train_dataset[i][1]) for i in range(len(train_dataset))], dtype=np.int64)

        sampler = make_weighted_sampler(train_labels) if use_sampler else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, max(len(train_dataset), 1)),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(batch_size, max(len(val_dataset), 1)),
            shuffle=False,
            num_workers=0,
        )

        class_weights = compute_class_weights(train_labels, num_classes=5)
        criterion = build_loss_function(
            loss_name=loss_name,
            class_weights=class_weights,
            focal_gamma=float(t_cfg.get("focal_gamma", 2.0)),
            label_smoothing=float(t_cfg.get("label_smoothing", 0.05)),
            device=self.device,
        )
        optimizer = build_optimizer(self.model, name=optimizer_name, lr=lr, weight_decay=weight_decay)
        scheduler = build_scheduler(optimizer, name=scheduler_name, epochs=epochs)

        self.tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(self.tb_dir))

        history: list[dict[str, Any]] = []
        best_score = float("inf") if monitor == "val_loss" else -float("inf")
        best_epoch = 0
        best_val_metrics: dict[str, Any] = {}
        epochs_without_improvement = 0
        start_wall = time.perf_counter()

        try:
            for epoch in range(1, epochs + 1):
                t_ep = time.perf_counter()
                self.model.train()
                train_loss_sum = 0.0
                train_correct = 0
                train_count = 0

                for batch in train_loader:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    if clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                    optimizer.step()

                    bs = y.size(0)
                    train_loss_sum += float(loss.item()) * bs
                    train_correct += int((logits.argmax(dim=1) == y).sum().item())
                    train_count += bs

                train_loss = train_loss_sum / max(train_count, 1)
                train_acc = train_correct / max(train_count, 1)

                # Validation pass
                self.model.eval()
                val_loss_sum = 0.0
                val_count = 0
                val_probs_list: list[np.ndarray] = []
                val_labels_list: list[np.ndarray] = []
                with torch.inference_mode():
                    for batch in val_loader:
                        x, y = batch[0].to(self.device), batch[1].to(self.device)
                        logits = self.model(x)
                        loss = criterion(logits, y)
                        bs = y.size(0)
                        val_loss_sum += float(loss.item()) * bs
                        val_count += bs
                        val_probs_list.append(F.softmax(logits, dim=-1).cpu().numpy())
                        val_labels_list.append(y.cpu().numpy())

                val_loss = val_loss_sum / max(val_count, 1)
                val_probs = np.concatenate(val_probs_list, axis=0)
                val_labels = np.concatenate(val_labels_list, axis=0)
                val_cls = compute_classification_metrics(val_labels, val_probs)

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                current_lr = float(optimizer.param_groups[0]["lr"])
                ep_sec = time.perf_counter() - t_ep

                row = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": val_cls["accuracy"],
                    "val_macro_f1": val_cls["macro_f1"],
                    "val_qwk": val_cls["quadratic_weighted_kappa"],
                    "lr": round(current_lr, 7),
                    "epoch_seconds": round(ep_sec, 2),
                }
                history.append(row)

                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Accuracy/train", train_acc, epoch)
                writer.add_scalar("Accuracy/val", val_cls["accuracy"], epoch)
                writer.add_scalar("Metrics/val_macro_f1", val_cls["macro_f1"], epoch)
                writer.add_scalar("Metrics/val_qwk", val_cls["quadratic_weighted_kappa"], epoch)
                writer.add_scalar("LR", current_lr, epoch)

                metric_val = val_loss if monitor == "val_loss" else val_cls["quadratic_weighted_kappa"]
                improved = (metric_val < best_score - 1e-4) if monitor == "val_loss" else (metric_val > best_score + 1e-4)

                if improved or epoch == 1:
                    best_score = metric_val
                    best_epoch = epoch
                    best_val_metrics = {
                        "val_loss": round(val_loss, 4),
                        "val_accuracy": val_cls["accuracy"],
                        "val_macro_f1": val_cls["macro_f1"],
                        "val_qwk": val_cls["quadratic_weighted_kappa"],
                    }
                    epochs_without_improvement = 0
                    save_checkpoint(
                        model=self.model,
                        checkpoint_path=self.checkpoint_path,
                        model_name=self.model_name,
                        model_kwargs=self.model_kwargs,
                        epoch=epoch,
                        metrics=best_val_metrics,
                        config=self.cfg,
                        seed=seed,
                        preprocessing_preset=preprocessing_preset,
                        image_size=image_size,
                        dataset_name=dataset_name,
                        is_synthetic=is_synthetic,
                    )
                else:
                    epochs_without_improvement += 1
                    if patience > 0 and epochs_without_improvement >= patience:
                        break
        finally:
            writer.flush()
            writer.close()

        total_duration = time.perf_counter() - start_wall
        # Reload best weights into model before returning
        if self.checkpoint_path.is_file():
            payload = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(payload["model_state_dict"])

        return {
            "run_id": self.run_id,
            "checkpoint_path": str(self.checkpoint_path),
            "tensorboard_dir": str(self.tb_dir),
            "best_epoch": int(best_epoch),
            "epochs_ran": len(history),
            "best_val_metrics": best_val_metrics,
            "training_seconds": round(float(total_duration), 2),
            "history": history,
        }
