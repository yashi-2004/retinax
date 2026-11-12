"""Uncertainty quantification: MC Dropout, Deep Ensembles, Conformal Prediction, and Selective Abstention."""

from src.uncertainty.conformal import SplitConformalPredictor
from src.uncertainty.ensembles import predict_ensemble_batch, predict_ensemble_single
from src.uncertainty.mc_dropout import enable_dropout_only, mc_dropout_predict_batch, mc_dropout_predict_single
from src.uncertainty.selective_prediction import compute_risk_coverage_curve, evaluate_selective_decision

__all__ = [
    "SplitConformalPredictor",
    "compute_risk_coverage_curve",
    "enable_dropout_only",
    "evaluate_selective_decision",
    "mc_dropout_predict_batch",
    "mc_dropout_predict_single",
    "predict_ensemble_batch",
    "predict_ensemble_single",
]
