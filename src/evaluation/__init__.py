"""Comprehensive medical classification evaluation, calibration, robustness, domain shift, and statistical tests."""

from src.evaluation.calibration import (
    TemperatureScaler,
    apply_temperature_scaling,
    compute_calibration_metrics,
)
from src.evaluation.classification import (
    compute_classification_metrics,
    compute_curve_data,
    compute_ordinal_error_breakdown,
    compute_qwk,
)
from src.evaluation.generalisation import (
    compare_cohorts_distribution_shift,
    evaluate_ood_detection,
    fit_feature_ood_detector,
)
from src.evaluation.robustness import apply_controlled_perturbation, evaluate_quality_stratified
from src.evaluation.statistical_tests import (
    mcnemar_test,
    paired_bootstrap_ci,
    paired_bootstrap_comparison,
)

__all__ = [
    "TemperatureScaler",
    "apply_controlled_perturbation",
    "apply_temperature_scaling",
    "compare_cohorts_distribution_shift",
    "compute_calibration_metrics",
    "compute_classification_metrics",
    "compute_curve_data",
    "compute_ordinal_error_breakdown",
    "compute_qwk",
    "evaluate_ood_detection",
    "evaluate_quality_stratified",
    "fit_feature_ood_detector",
    "mcnemar_test",
    "paired_bootstrap_ci",
    "paired_bootstrap_comparison",
]
