"""Dataset manifest generation, persistence, and summary inspection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import PROJECT_ROOT, resolve_path
from src.data.validation import find_near_duplicates, validate_image_record

MANIFEST_COLUMNS = [
    "image_id",
    "dataset_name",
    "relative_path",
    "abs_path",
    "original_label",
    "label_name",
    "patient_id",
    "width",
    "height",
    "channels",
    "aspect_ratio",
    "processed_width",
    "processed_height",
    "sha256",
    "phash",
    "duplicate_cluster_id",
    "is_duplicate_group",
    "is_corrupt",
    "validation_error",
    "quality_score",
    "quality_category",
    "quality_issues",
    "split",
    "is_synthetic",
    "processing_config",
]


def build_manifest(
    records_df: pd.DataFrame,
    processing_config: dict[str, Any] | None = None,
    quality_fn: Callable[[str | Path], dict[str, Any]] | None = None,
    max_hamming: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate records, detect duplicates, assess quality, and return a manifest DataFrame."""
    cfg_json = json.dumps(processing_config or {}, sort_keys=True)
    target_size = int((processing_config or {}).get("image_size", 224))

    enriched_rows: list[dict[str, Any]] = []
    for row_dict in records_df.to_dict(orient="records"):
        val_info = validate_image_record(row_dict)
        qa_info = {"quality_score": None, "quality_category": "Not Assessed", "quality_issues": ""}
        if not val_info["is_corrupt"] and quality_fn is not None:
            try:
                qa_res = quality_fn(row_dict.get("abs_path") or row_dict.get("relative_path"))
                qa_info = {
                    "quality_score": float(qa_res.get("quality_score", 0.0)),
                    "quality_category": str(qa_res.get("quality_category", "Unknown")),
                    "quality_issues": "; ".join(qa_res.get("issues", [])),
                }
            except Exception as exc:  # pragma: no cover - defensive guard
                qa_info = {
                    "quality_score": None,
                    "quality_category": "QA Error",
                    "quality_issues": str(exc),
                }

        enriched_rows.append(
            {
                "image_id": str(row_dict["image_id"]),
                "dataset_name": str(row_dict["dataset_name"]),
                "relative_path": str(row_dict.get("relative_path", "")),
                "abs_path": str(row_dict.get("abs_path", "")),
                "original_label": int(row_dict["original_label"]),
                "label_name": str(row_dict.get("label_name", "")),
                "patient_id": row_dict.get("patient_id"),
                "width": val_info["width"],
                "height": val_info["height"],
                "channels": val_info["channels"],
                "aspect_ratio": val_info["aspect_ratio"],
                "processed_width": target_size if not val_info["is_corrupt"] else 0,
                "processed_height": target_size if not val_info["is_corrupt"] else 0,
                "sha256": val_info["sha256"],
                "phash": val_info["phash"],
                "is_corrupt": val_info["is_corrupt"],
                "validation_error": val_info["validation_error"],
                "quality_score": qa_info["quality_score"],
                "quality_category": qa_info["quality_category"],
                "quality_issues": qa_info["quality_issues"],
                "split": str(row_dict.get("split", "unassigned")),
                "is_synthetic": bool(row_dict.get("is_synthetic", False)),
                "processing_config": cfg_json,
            }
        )

    manifest_df = pd.DataFrame(enriched_rows)
    if len(manifest_df) > 0:
        manifest_df, dup_pairs_df = find_near_duplicates(manifest_df, max_hamming=max_hamming)
    else:
        manifest_df = pd.DataFrame(columns=MANIFEST_COLUMNS)
        dup_pairs_df = pd.DataFrame(columns=["image_a", "image_b", "match_type", "hamming_distance"])

    for col in MANIFEST_COLUMNS:
        if col not in manifest_df.columns:
            manifest_df[col] = None
    manifest_df = manifest_df[MANIFEST_COLUMNS]

    valid_df = manifest_df[~manifest_df["is_corrupt"]]
    class_counts = {str(k): int(v) for k, v in valid_df["original_label"].value_counts().sort_index().items()}
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": int(len(manifest_df)),
        "valid_records": int(len(valid_df)),
        "corrupt_records": int(manifest_df["is_corrupt"].sum()) if len(manifest_df) else 0,
        "duplicate_pairs_detected": int(len(dup_pairs_df)),
        "class_distribution": class_counts,
        "patient_id_present": bool(manifest_df["patient_id"].notna().any()) if len(manifest_df) else False,
        "is_synthetic": bool(manifest_df["is_synthetic"].any()) if len(manifest_df) else False,
        "processing_config": processing_config or {},
    }
    return manifest_df, summary


def save_manifest(
    manifest_df: pd.DataFrame,
    name: str,
    manifests_dir: str | Path | None = None,
    summary: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save manifest CSV and metadata JSON into `data/manifests/`."""
    out_dir = resolve_path(manifests_dir or "data/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_manifest.csv"
    meta_path = out_dir / f"{name}_manifest_meta.json"

    manifest_df.to_csv(csv_path, index=False)
    meta_payload = summary or {"total_records": int(len(manifest_df))}
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2)
    return csv_path, meta_path


def load_manifest(path_or_name: str | Path, manifests_dir: str | Path | None = None) -> pd.DataFrame:
    """Load an existing manifest CSV by full path or dataset manifest name."""
    cand = Path(path_or_name)
    if not cand.is_file():
        m_dir = resolve_path(manifests_dir or "data/manifests")
        cand = m_dir / f"{path_or_name}_manifest.csv"
    if not cand.is_file():
        raise FileNotFoundError(f"Manifest file not found: {cand}")

    df = pd.read_csv(cand)
    # Rehydrate absolute paths cleanly relative to current project root if moved
    if "relative_path" in df.columns:
        abs_paths = []
        for row in df.itertuples(index=False):
            rel = str(getattr(row, "relative_path", "") or "")
            raw_abs = str(getattr(row, "abs_path", "") or "")
            if rel and (PROJECT_ROOT / rel).exists():
                abs_paths.append(str(PROJECT_ROOT / rel))
            else:
                abs_paths.append(raw_abs)
        df["abs_path"] = abs_paths
    return df
