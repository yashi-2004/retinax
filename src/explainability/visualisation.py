"""Heatmap overlay rendering, FOV sanity metrics, and side-by-side explanation figures."""

from __future__ import annotations

from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import DR_CLASS_NAMES
from src.preprocessing.fundus_processing import extract_fov_mask

EXPLAINABILITY_DISCLAIMER = (
    "Post-hoc attribution maps (Grad-CAM, Integrated Gradients, Occlusion) highlight regions "
    "influencing model logits, NOT validated clinical pathology segmentations. Diffuse or border-focused "
    "attribution may indicate shortcut learning or poor calibration rather than diagnostic evidence."
)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> tuple[np.ndarray, np.ndarray]:
    """Render false-colour heatmap and alpha-blended overlay on an RGB retinal image."""
    img = np.asarray(image_rgb, dtype=np.uint8)
    h, w = img.shape[:2]
    hm = cv2.resize(np.clip(heatmap, 0.0, 1.0).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    hm_u8 = np.uint8(255 * hm)
    colored_bgr = cv2.applyColorMap(hm_u8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1.0 - float(alpha), colored_rgb, float(alpha), 0.0)
    return colored_rgb, overlay


def compute_fov_attribution_ratio(image_rgb: np.ndarray, heatmap: np.ndarray) -> float:
    """Compute fraction of total heatmap saliency mass concentrated inside the retinal FOV."""
    fov = extract_fov_mask(image_rgb)
    h, w = image_rgb.shape[:2]
    hm = cv2.resize(np.clip(heatmap, 0.0, 1.0).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    total_mass = float(hm.sum())
    if total_mass <= 1e-8:
        return 0.0
    fov_mass = float(hm[fov].sum())
    return round(fov_mass / total_mass, 4)


def build_explanation_figure(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    probabilities: np.ndarray,
    predicted_class: int,
    actual_class: int | None = None,
    method_label: str = "Grad-CAM",
    save_path: str | None = None,
) -> plt.Figure:
    """Create clean 4-panel biomedical explainability figure (Original, Heatmap, Overlay, Probs)."""
    colored_hm, overlay = overlay_heatmap(image_rgb, heatmap, alpha=0.45)
    fov_ratio = compute_fov_attribution_ratio(image_rgb, heatmap)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=140)
    fig.patch.set_facecolor("#FFFFFF")

    pred_name = DR_CLASS_NAMES.get(predicted_class, str(predicted_class))
    actual_txt = f" | Actual: {DR_CLASS_NAMES.get(actual_class, actual_class)}" if actual_class is not None else ""
    fig.suptitle(
        f"{method_label} | Predicted: {pred_name} ({probabilities[predicted_class]*100:.1f}%){actual_txt} | FOV Mass: {fov_ratio*100:.1f}%",
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Retinal Image", fontsize=9.5, color="#1E293B")
    axes[0].axis("off")

    axes[1].imshow(colored_hm)
    axes[1].set_title(f"{method_label} Heatmap", fontsize=9.5, color="#1E293B")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Attribution Overlay", fontsize=9.5, color="#1E293B")
    axes[2].axis("off")

    classes = [DR_CLASS_NAMES[i] for i in range(len(probabilities))]
    colors = ["#0D9488" if i == predicted_class else "#94A3B8" for i in range(len(probabilities))]
    y_pos = np.arange(len(classes))
    axes[3].barh(y_pos, probabilities * 100.0, color=colors)
    axes[3].set_yticks(y_pos)
    axes[3].set_yticklabels(classes, fontsize=8.5, color="#1E293B")
    axes[3].invert_yaxis()
    axes[3].set_xlim(0, 105)
    axes[3].set_xlabel("Probability (%)", fontsize=8.5, color="#1E293B")
    axes[3].set_title("Severity Distribution", fontsize=9.5, color="#1E293B")
    axes[3].grid(axis="x", linestyle="--", alpha=0.35)
    axes[3].spines["top"].set_visible(False)
    axes[3].spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
