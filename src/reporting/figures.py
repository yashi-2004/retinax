"""Clean medical-grade Matplotlib visualizations for RetinaX experiments and Gradio UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DR_CLASS_NAMES

NAVY = "#0F172A"
SLATE = "#475569"
TEAL = "#0D9488"
TEAL_DARK = "#0F766E"
AMBER = "#D97706"
CRIMSON = "#DC2626"
BLUE = "#2563EB"
PALETTE = [TEAL, BLUE, AMBER, "#7C3AED", CRIMSON]


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors=SLATE, labelsize=8.5)
    ax.grid(linestyle="--", alpha=0.35, color="#CBD5E1")


def plot_class_distribution(
    manifest_df: pd.DataFrame,
    title: str = "Diabetic Retinopathy Severity Distribution",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot bar chart of class frequencies highlighting dataset imbalance."""
    valid = manifest_df[~manifest_df["is_corrupt"].astype(bool)] if "is_corrupt" in manifest_df.columns else manifest_df
    counts = [int((valid["original_label"] == c).sum()) for c in range(5)]
    labels = [f"{c}: {DR_CLASS_NAMES[c]}" for c in range(5)]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    _style_axis(ax)

    bars = ax.bar(labels, counts, color=TEAL, edgecolor=TEAL_DARK, width=0.58)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + max(counts, default=1) * 0.02, str(val),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=NAVY)

    ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, pad=10)
    ax.set_ylabel("Image Count", fontsize=9, color=NAVY)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    history: list[dict[str, Any]],
    title: str = "Training & Validation Dynamics",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot epoch-by-epoch train/val loss and validation QWK / accuracy."""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")

    for ax in axes:
        _style_axis(ax)

    if len(df) > 0:
        axes[0].plot(df["epoch"], df["train_loss"], marker="o", color=BLUE, label="Train Loss", linewidth=1.8)
        axes[0].plot(df["epoch"], df["val_loss"], marker="s", color=CRIMSON, label="Val Loss", linewidth=1.8)
        axes[0].set_title("Loss Convergence", fontsize=10, fontweight="bold", color=NAVY)
        axes[0].set_xlabel("Epoch", fontsize=9, color=NAVY)
        axes[0].set_ylabel("Cross-Entropy / Focal Loss", fontsize=9, color=NAVY)
        axes[0].legend(frameon=False, fontsize=8.5)

        axes[1].plot(df["epoch"], df["train_acc"], marker="o", color=BLUE, label="Train Acc", linewidth=1.8)
        axes[1].plot(df["epoch"], df["val_acc"], marker="s", color=TEAL, label="Val Acc", linewidth=1.8)
        if "val_qwk" in df.columns:
            axes[1].plot(df["epoch"], df["val_qwk"], marker="^", color=AMBER, label="Val QWK", linewidth=1.8)
        axes[1].set_title("Predictive Performance", fontsize=10, fontweight="bold", color=NAVY)
        axes[1].set_xlabel("Epoch", fontsize=9, color=NAVY)
        axes[1].set_ylabel("Score", fontsize=9, color=NAVY)
        axes[1].set_ylim(-0.1, 1.05)
        axes[1].legend(frameon=False, fontsize=8.5)

    fig.suptitle(title, fontsize=11, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray | list[list[float]],
    normalized: bool = True,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot annotated 5x5 confusion matrix (counts or row-normalized)."""
    mat = np.asarray(cm, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")

    im = ax.imshow(mat, interpolation="nearest", cmap="GnBu", vmin=0.0, vmax=(1.0 if normalized else None))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    names = [DR_CLASS_NAMES[i] for i in range(mat.shape[0])]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8.5, color=NAVY)
    ax.set_yticklabels(names, fontsize=8.5, color=NAVY)

    thresh = mat.max() * 0.55 if mat.size else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = f"{mat[i, j]:.2f}" if normalized else f"{int(mat[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color=("white" if mat[i, j] > thresh else NAVY))

    ax.set_xlabel("Predicted Severity Grade", fontsize=9, fontweight="bold", color=NAVY)
    ax.set_ylabel("Reference Severity Grade", fontsize=9, fontweight="bold", color=NAVY)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY, pad=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_and_pr_curves(
    curve_data: dict[str, Any],
    title: str = "One-vs-Rest ROC & Precision-Recall Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot multi-class OvR ROC and PR curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in axes:
        _style_axis(ax)

    roc_dict = curve_data.get("roc", {})
    for idx, (cls_name, pts) in enumerate(roc_dict.items()):
        axes[0].plot(pts["fpr"], pts["tpr"], color=PALETTE[idx % len(PALETTE)],
                     linewidth=1.7, label=f"{cls_name} (AUC={pts['auc']:.2f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.0)
    axes[0].set_title("One-vs-Rest ROC Curves", fontsize=10, fontweight="bold", color=NAVY)
    axes[0].set_xlabel("False Positive Rate", fontsize=9, color=NAVY)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=9, color=NAVY)
    if roc_dict:
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    pr_dict = curve_data.get("pr", {})
    for idx, (cls_name, pts) in enumerate(pr_dict.items()):
        axes[1].plot(pts["recall"], pts["precision"], color=PALETTE[idx % len(PALETTE)],
                     linewidth=1.7, label=f"{cls_name} (AP={pts['ap']:.2f})")
    axes[1].set_title("Precision-Recall Curves", fontsize=10, fontweight="bold", color=NAVY)
    axes[1].set_xlabel("Recall", fontsize=9, color=NAVY)
    axes[1].set_ylabel("Precision", fontsize=9, color=NAVY)
    if pr_dict:
        axes[1].legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle(title, fontsize=11, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reliability_diagram(
    cal_before: dict[str, Any],
    cal_after: dict[str, Any] | None = None,
    title: str = "Reliability Diagram (Before vs After Temperature Scaling)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot calibration reliability curve comparing predicted confidence vs empirical accuracy."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    _style_axis(ax)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Perfect Calibration")

    def _extract_xy(cal_dict: dict[str, Any]) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for b in cal_dict.get("bins", []):
            if b.get("accuracy") is not None and b.get("confidence") is not None:
                xs.append(float(b["confidence"]))
                ys.append(float(b["accuracy"]))
        return xs, ys

    x0, y0 = _extract_xy(cal_before)
    if x0:
        ax.plot(x0, y0, marker="o", linewidth=2.0, color=CRIMSON,
                label=f"Uncalibrated (ECE={cal_before.get('ece', 0):.3f})")

    if cal_after is not None:
        x1, y1 = _extract_xy(cal_after)
        if x1:
            ax.plot(x1, y1, marker="s", linewidth=2.0, color=TEAL,
                    label=f"Temp-Scaled (ECE={cal_after.get('ece', 0):.3f})")

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Predicted Confidence", fontsize=9, color=NAVY)
    ax.set_ylabel("Empirical Accuracy", fontsize=9, color=NAVY)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_risk_coverage_curve(
    rc_data: dict[str, Any],
    title: str = "Selective Prediction: Risk vs Coverage",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot selective error rate (risk) as a function of retained dataset coverage."""
    pts = rc_data.get("points", [])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    _style_axis(ax)

    if pts:
        covs = [p["coverage"] * 100.0 for p in pts]
        risks = [p["selective_risk"] * 100.0 for p in pts]
        ax.plot(covs, risks, marker="o", linewidth=2.0, color=TEAL,
                label=f"AURC = {rc_data.get('aurc', 0):.3f}")
        ax.fill_between(covs, risks, alpha=0.12, color=TEAL)

    ax.set_xlabel("Coverage (% of Samples Accepted)", fontsize=9, color=NAVY)
    ax.set_ylabel("Selective Risk (% Error Rate)", fontsize=9, color=NAVY)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_domain_shift_distributions(
    internal_stats: pd.DataFrame,
    external_stats: pd.DataFrame,
    title: str = "Cross-Dataset Acquisition Shift (Brightness & Blur)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compare internal vs external cohort image brightness and Laplacian sharpness distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in axes:
        _style_axis(ax)

    if len(internal_stats) and len(external_stats):
        axes[0].hist(internal_stats["brightness"], bins=15, alpha=0.55, color=TEAL, label="Internal", density=True)
        axes[0].hist(external_stats["brightness"], bins=15, alpha=0.55, color=AMBER, label="External", density=True)
        axes[0].set_title("Mean FOV Brightness Shift", fontsize=10, fontweight="bold", color=NAVY)
        axes[0].set_xlabel("Mean Luminance [0-255]", fontsize=9, color=NAVY)
        axes[0].legend(frameon=False, fontsize=8.5)

        axes[1].hist(internal_stats["blur_laplacian"], bins=15, alpha=0.55, color=TEAL, label="Internal", density=True)
        axes[1].hist(external_stats["blur_laplacian"], bins=15, alpha=0.55, color=AMBER, label="External", density=True)
        axes[1].set_title("Focus Sharpness (Laplacian Var) Shift", fontsize=10, fontweight="bold", color=NAVY)
        axes[1].set_xlabel("Laplacian Variance", fontsize=9, color=NAVY)
        axes[1].legend(frameon=False, fontsize=8.5)

    fig.suptitle(title, fontsize=11, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    registry_df: pd.DataFrame,
    metric_col: str = "qwk",
    title: str = "Model Benchmark Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar comparison across tracked experiment runs."""
    fig, ax = plt.subplots(figsize=(8.0, max(3.6, 0.45 * max(len(registry_df), 1) + 1.5)), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    _style_axis(ax)

    if len(registry_df) > 0 and metric_col in registry_df.columns:
        sub = registry_df.dropna(subset=[metric_col]).sort_values(metric_col, ascending=True)
        y_pos = np.arange(len(sub))
        vals = sub[metric_col].astype(float).tolist()
        labels = sub["run_id"].astype(str).tolist()
        bars = ax.barh(y_pos, vals, color=TEAL, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5, color=NAVY)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2.0, f"{v:.3f}",
                    va="center", fontsize=8, color=NAVY)

    ax.set_xlabel(metric_col.upper(), fontsize=9, fontweight="bold", color=NAVY)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_embedding_projection(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "2D Representation Space Projection",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of 2D PCA/t-SNE embeddings coloured by DR severity grade."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    _style_axis(ax)

    y = np.asarray(labels, dtype=np.int64)
    for c in range(5):
        mask = y == c
        if mask.any():
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                s=28,
                alpha=0.78,
                color=PALETTE[c % len(PALETTE)],
                label=f"{c}: {DR_CLASS_NAMES[c]}",
            )

    ax.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY)
    ax.set_xlabel("Dimension 1", fontsize=9, color=NAVY)
    ax.set_ylabel("Dimension 2", fontsize=9, color=NAVY)
    ax.legend(frameon=False, fontsize=8, loc="best")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
