"""Local experiment registry, artifact storage, and run comparison helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config import resolve_path
from src.experiments.reproducibility import compute_config_hash, get_git_revision


class ExperimentRegistry:
    """Persists traceable experiment runs in JSON/YAML/CSV alongside TensorBoard logs."""

    def __init__(self, experiments_dir: str | Path | None = None) -> None:
        self.base_dir = resolve_path(experiments_dir or "experiments")
        self.runs_dir = self.base_dir / "runs"
        self.registry_path = self.base_dir / "registry.json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.registry_path.is_file():
            return []
        try:
            with self.registry_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_index(self, entries: list[dict[str, Any]]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2)

    def log_run(
        self,
        run_id: str,
        experiment_group: str,
        model_name: str,
        dataset_name: str,
        is_synthetic: bool,
        config: dict[str, Any],
        metrics: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        predictions_df: pd.DataFrame | None = None,
        checkpoint_path: str | None = None,
        tensorboard_dir: str | None = None,
        notes: str = "",
    ) -> Path:
        """Persist complete reproducible experiment run folder and register summary."""
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        payload = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_group": experiment_group,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "is_synthetic": bool(is_synthetic),
            "config_hash": compute_config_hash(config),
            "git_revision": get_git_revision(),
            "checkpoint_path": checkpoint_path,
            "tensorboard_dir": tensorboard_dir,
            "notes": notes,
            "metrics": metrics,
        }
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        if history:
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if predictions_df is not None:
            predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

        # Update summary index (replace existing entry if re-run with same run_id)
        entries = [e for e in self._load_index() if e.get("run_id") != run_id]
        test_m = metrics.get("test", {})
        cal_after = metrics.get("calibration_after", metrics.get("calibration_before", {}))
        param_m = metrics.get("model_complexity", {})
        entries.append(
            {
                "run_id": run_id,
                "timestamp_utc": payload["timestamp_utc"],
                "experiment_group": experiment_group,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "is_synthetic": bool(is_synthetic),
                "accuracy": test_m.get("accuracy"),
                "balanced_accuracy": test_m.get("balanced_accuracy"),
                "macro_f1": test_m.get("macro_f1"),
                "qwk": test_m.get("quadratic_weighted_kappa"),
                "roc_auc": test_m.get("macro_roc_auc_ovr"),
                "ece": cal_after.get("ece"),
                "brier_score": cal_after.get("brier_score"),
                "total_params": param_m.get("total_params"),
                "ms_per_image": metrics.get("inference_ms_per_image"),
                "training_seconds": metrics.get("training_seconds"),
                "checkpoint_path": checkpoint_path,
                "run_dir": str(run_dir),
            }
        )
        self._save_index(entries)
        return run_dir

    def list_runs(self) -> list[dict[str, Any]]:
        return self._load_index()

    def comparison_dataframe(self) -> pd.DataFrame:
        entries = self._load_index()
        if not entries:
            return pd.DataFrame()
        return pd.DataFrame(entries)

    def load_run_details(self, run_id: str) -> dict[str, Any]:
        run_dir = self.runs_dir / run_id
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.is_file():
            raise FileNotFoundError(f"Run '{run_id}' not found at {metrics_file}")
        with metrics_file.open("r", encoding="utf-8") as handle:
            details = json.load(handle)
        hist_file = run_dir / "history.csv"
        if hist_file.is_file():
            details["history"] = pd.read_csv(hist_file).to_dict(orient="records")
        pred_file = run_dir / "test_predictions.csv"
        if pred_file.is_file():
            details["test_predictions_df"] = pd.read_csv(pred_file)
        return details
