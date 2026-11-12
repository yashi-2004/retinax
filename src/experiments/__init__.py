"""Experiment reproducibility, TensorBoard-backed registry tracking, and ablation runners."""

from src.experiments.ablation import run_experiment_suite, run_supervised_experiment
from src.experiments.reproducibility import compute_config_hash, get_git_revision, set_global_seed
from src.experiments.tracking import ExperimentRegistry

__all__ = [
    "ExperimentRegistry",
    "compute_config_hash",
    "get_git_revision",
    "run_experiment_suite",
    "run_supervised_experiment",
    "set_global_seed",
]
