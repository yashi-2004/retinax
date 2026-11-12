"""Publication-grade clinical figures and reproducible research report generation."""

from src.reporting.figures import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_domain_shift_distributions,
    plot_embedding_projection,
    plot_model_comparison,
    plot_reliability_diagram,
    plot_risk_coverage_curve,
    plot_roc_and_pr_curves,
    plot_training_curves,
)
from src.reporting.report_generator import generate_research_report

__all__ = [
    "generate_research_report",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_domain_shift_distributions",
    "plot_embedding_projection",
    "plot_model_comparison",
    "plot_reliability_diagram",
    "plot_risk_coverage_curve",
    "plot_roc_and_pr_curves",
    "plot_training_curves",
]
