"""Data ingestion, validation, manifests, splitting, caching, and synthetic demo utilities."""

from src.data.caching import FundusDataset, ensure_processed_cache, load_embeddings, save_embeddings
from src.data.ingestion import DATASET_DOCS, discover_datasets, load_raw_records
from src.data.manifests import build_manifest, load_manifest, save_manifest
from src.data.splitting import assign_splits, create_label_efficiency_subsets, verify_no_split_leakage
from src.data.synthetic import generate_synthetic_demo_dataset
from src.data.validation import compute_dhash, compute_sha256, find_near_duplicates, validate_image_record

__all__ = [
    "DATASET_DOCS",
    "FundusDataset",
    "assign_splits",
    "build_manifest",
    "compute_dhash",
    "compute_sha256",
    "create_label_efficiency_subsets",
    "discover_datasets",
    "ensure_processed_cache",
    "find_near_duplicates",
    "generate_synthetic_demo_dataset",
    "load_embeddings",
    "load_manifest",
    "load_raw_records",
    "save_embeddings",
    "save_manifest",
    "validate_image_record",
    "verify_no_split_leakage",
]
