"""Stratified, group-aware dataset splitting and leakage verification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _build_atomic_groups(df: pd.DataFrame) -> list[str]:
    """Combine patient_id and duplicate_cluster_id via union-find so neither leaks across splits."""
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Group by patient_id when available
    if "patient_id" in df.columns:
        pt_map: dict[str, int] = {}
        for idx, pt in enumerate(df["patient_id"]):
            if pd.notna(pt) and str(pt).strip() != "":
                key = str(pt)
                if key in pt_map:
                    union(pt_map[key], idx)
                else:
                    pt_map[key] = idx

    # Group by perceptual/exact duplicate cluster
    if "duplicate_cluster_id" in df.columns:
        dup_map: dict[str, int] = {}
        for idx, cl in enumerate(df["duplicate_cluster_id"]):
            if pd.notna(cl) and str(cl).strip() != "":
                key = str(cl)
                if key in dup_map:
                    union(dup_map[key], idx)
                else:
                    dup_map[key] = idx

    return [f"grp_{find(i)}" for i in range(n)]


def assign_splits(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.65,
    val_ratio: float = 0.125,
    cal_ratio: float = 0.125,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign leakage-free stratified `train`, `val`, `cal`, and `test` splits.

    Corrupt records are assigned `split="excluded_corrupt"`.
    Patients and exact/near-duplicate clusters are kept strictly atomic within a single split.
    """
    ratios = {"train": train_ratio, "val": val_ratio, "cal": cal_ratio, "test": test_ratio}
    total_r = sum(ratios.values())
    if not np.isclose(total_r, 1.0, atol=1e-3):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_r:.4f}")

    out = manifest_df.copy().reset_index(drop=True)
    out["split"] = "excluded_corrupt"

    valid_mask = ~out["is_corrupt"].astype(bool)
    valid_df = out.loc[valid_mask].copy().reset_index()
    if len(valid_df) == 0:
        return out, {"warning": "No valid images available to split."}

    valid_df["_atomic_group"] = _build_atomic_groups(valid_df)
    rng = np.random.default_rng(seed)

    # Aggregate groups by size & primary (rarest) class so rare classes get distributed cleanly
    global_counts = valid_df["original_label"].value_counts().to_dict()
    groups: list[dict[str, Any]] = []
    for grp_id, grp_rows in valid_df.groupby("_atomic_group"):
        lbls = grp_rows["original_label"].astype(int).tolist()
        rarest_cls = min(lbls, key=lambda c: global_counts.get(c, 999999))
        groups.append(
            {
                "group_id": grp_id,
                "orig_indices": grp_rows["index"].tolist(),
                "labels": lbls,
                "rarest_cls": rarest_cls,
                "size": len(lbls),
                "tie_break": float(rng.random()),
            }
        )

    # Sort groups: rarest class first, larger group first, deterministic random tiebreak
    groups.sort(key=lambda g: (global_counts.get(g["rarest_cls"], 0), -g["size"], g["tie_break"]))

    split_names = ["train", "val", "cal", "test"]
    total_valid = len(valid_df)
    target_totals = {s: ratios[s] * total_valid for s in split_names}
    target_class_counts = {
        s: {c: ratios[s] * count for c, count in global_counts.items()} for s in split_names
    }

    current_totals = {s: 0 for s in split_names}
    current_class_counts = {s: {c: 0 for c in global_counts} for s in split_names}

    for grp_idx, grp in enumerate(groups):
        empty_splits = [s for s in split_names if current_totals[s] == 0]
        remaining_groups = len(groups) - grp_idx
        candidates = empty_splits if (empty_splits and remaining_groups <= len(empty_splits)) else split_names

        best_split = candidates[0]
        best_score = float("inf")
        for s in candidates:
            # Penalty for exceeding target class counts + overall split size balance
            cls_deficit_score = 0.0
            for lbl in grp["labels"]:
                after = current_class_counts[s].get(lbl, 0) + 1
                target = max(target_class_counts[s].get(lbl, 0.5), 0.25)
                cls_deficit_score += (after - target) / target
            size_score = ((current_totals[s] + grp["size"]) - target_totals[s]) / max(target_totals[s], 1.0)
            empty_bonus = -3.0 if current_totals[s] == 0 else 0.0
            score = 2.0 * cls_deficit_score + size_score + empty_bonus
            if score < best_score:
                best_score = score
                best_split = s

        for orig_idx in grp["orig_indices"]:
            out.loc[orig_idx, "split"] = best_split
        current_totals[best_split] += grp["size"]
        for lbl in grp["labels"]:
            current_class_counts[best_split][lbl] = current_class_counts[best_split].get(lbl, 0) + 1

    audit = verify_no_split_leakage(out)
    return out, audit


def verify_no_split_leakage(manifest_df: pd.DataFrame) -> dict[str, Any]:
    """Audit splits for patient, SHA-256, or perceptual near-duplicate leakage."""
    active_splits = ["train", "val", "cal", "test"]
    sub = manifest_df[manifest_df["split"].isin(active_splits)].copy()

    def _cross_split_overlaps(col: str) -> list[str]:
        if col not in sub.columns:
            return []
        seen: dict[str, set[str]] = {}
        for row in sub.itertuples(index=False):
            val = getattr(row, col)
            if pd.isna(val) or str(val).strip() == "":
                continue
            seen.setdefault(str(val), set()).add(str(row.split))
        return [k for k, splits in seen.items() if len(splits) > 1]

    sha_overlaps = _cross_split_overlaps("sha256")
    dup_overlaps = _cross_split_overlaps("duplicate_cluster_id")
    patient_overlaps = _cross_split_overlaps("patient_id")

    has_patient_ids = bool(sub["patient_id"].notna().any()) if "patient_id" in sub.columns and len(sub) else False
    split_counts = {s: int((sub["split"] == s).sum()) for s in active_splits}

    return {
        "leakage_free": len(sha_overlaps) == 0 and len(dup_overlaps) == 0 and len(patient_overlaps) == 0,
        "sha256_overlaps": len(sha_overlaps),
        "duplicate_cluster_overlaps": len(dup_overlaps),
        "patient_id_overlaps": len(patient_overlaps),
        "patient_id_available": has_patient_ids,
        "limitation_note": (
            "Patient identifiers available and grouped across splits."
            if has_patient_ids
            else "Patient identifiers unavailable in dataset; near-duplicate perceptual hashing used to prevent image-pair leakage, though bilateral eye correlation cannot be verified without patient IDs."
        ),
        "split_counts": split_counts,
    }


def create_label_efficiency_subsets(
    manifest_df: pd.DataFrame,
    fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
    seed: int = 42,
) -> dict[float, pd.DataFrame]:
    """Create nested stratified subsets of the `train` split ONLY (leaving val/cal/test fixed)."""
    train_df = manifest_df[manifest_df["split"] == "train"].copy().reset_index(drop=True)
    if len(train_df) == 0:
        raise ValueError("Manifest contains no rows assigned to `train` split.")

    rng = np.random.default_rng(seed)
    # Assign a uniform random rank per class so subsets are cleanly nested: 10% subset <= 25% <= 50% <= 100%
    ranks = np.zeros(len(train_df), dtype=np.float64)
    for _, cls_indices in train_df.groupby("original_label").groups.items():
        idx_arr = np.array(list(cls_indices))
        perm = rng.permutation(len(idx_arr))
        # Normalized position inside class in (0, 1]
        ranks[idx_arr] = (perm + 1) / len(idx_arr)

    subsets: dict[float, pd.DataFrame] = {}
    for frac in sorted(fractions):
        if frac >= 1.0:
            subsets[float(frac)] = train_df.copy()
        else:
            keep = ranks <= frac
            # Ensure at least 1 sample per present class if possible
            for _, cls_indices in train_df.groupby("original_label").groups.items():
                idx_arr = np.array(list(cls_indices))
                if not keep[idx_arr].any():
                    best = idx_arr[np.argmin(ranks[idx_arr])]
                    keep[best] = True
            subsets[float(frac)] = train_df.loc[keep].reset_index(drop=True)
    return subsets
