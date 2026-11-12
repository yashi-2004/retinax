"""Post-hoc visual explainability: Grad-CAM, Grad-CAM++, Integrated Gradients, and Occlusion."""

from src.explainability.gradcam import GradCAMExplainer
from src.explainability.integrated_gradients import (
    compute_integrated_gradients,
    compute_occlusion_sensitivity,
)
from src.explainability.visualisation import (
    EXPLAINABILITY_DISCLAIMER,
    build_explanation_figure,
    compute_fov_attribution_ratio,
    overlay_heatmap,
)

__all__ = [
    "EXPLAINABILITY_DISCLAIMER",
    "GradCAMExplainer",
    "build_explanation_figure",
    "compute_fov_attribution_ratio",
    "compute_integrated_gradients",
    "compute_occlusion_sensitivity",
    "overlay_heatmap",
]
