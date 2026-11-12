"""Image integrity validation, label verification, and duplicate/near-duplicate detection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def compute_sha256(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Compute exact SHA-256 hex digest for an image file."""
    hasher = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image_source: str | Path | Image.Image | np.ndarray, hash_size: int = 8) -> str:
    """Compute a 64-bit fundus-aware perceptual difference hash (dHash) as a 16-char hex string.

    Crops inside the central retinal region [20%..80%] and stretches local contrast before
    combining horizontal and vertical gradient bits so circular black borders do not dominate
    and cause unrelated fundus photographs to collide.
    """
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as img:
            pil_gray = ImageOps.exif_transpose(img).convert("L")
    elif isinstance(image_source, Image.Image):
        pil_gray = ImageOps.exif_transpose(image_source).convert("L")
    else:
        arr = np.asarray(image_source)
        if arr.ndim == 3:
            pil_gray = Image.fromarray(arr.astype(np.uint8)).convert("L")
        else:
            pil_gray = Image.fromarray(arr.astype(np.uint8), mode="L")

    w, h = pil_gray.size
    if w >= 24 and h >= 24:
        pil_gray = pil_gray.crop((int(0.20 * w), int(0.20 * h), int(0.80 * w), int(0.80 * h)))
    pil_gray = ImageOps.autocontrast(pil_gray, cutoff=1)

    if hash_size == 8:
        arr_h = np.asarray(pil_gray.resize((9, 4), Image.Resampling.LANCZOS), dtype=np.int16)
        arr_v = np.asarray(pil_gray.resize((4, 9), Image.Resampling.LANCZOS), dtype=np.int16)
        bits = np.concatenate([(arr_h[:, 1:] > arr_h[:, :-1]).flatten(), (arr_v[1:, :] > arr_v[:-1, :]).flatten()])
    else:
        pixels = np.asarray(pil_gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS), dtype=np.int16)
        bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    hex_len = (hash_size * hash_size) // 4
    return f"{value:0{hex_len}x}"


def hamming_distance_hex(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two hexadecimal perceptual hashes."""
    if not hash_a or not hash_b:
        return 64
    xor_val = int(hash_a, 16) ^ int(hash_b, 16)
    return xor_val.bit_count()


def validate_image_file(
    file_path: str | Path,
    label: int | None = None,
    min_dim: int = 32,
) -> dict[str, Any]:
    """Safely validate a single retinal image file and optional class label.

    Never raises on corrupt images; reports `is_corrupt=True` and `validation_error`.
    """
    path = Path(file_path)
    result: dict[str, Any] = {
        "is_corrupt": False,
        "validation_error": "",
        "width": 0,
        "height": 0,
        "channels": 0,
        "aspect_ratio": 0.0,
        "sha256": "",
        "phash": "",
    }

    if not str(file_path) or not path.is_file():
        result["is_corrupt"] = True
        result["validation_error"] = f"Missing file: {file_path}"
        return result

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["is_corrupt"] = True
        result["validation_error"] = f"Unsupported file extension '{path.suffix}'"
        return result

    if path.stat().st_size == 0:
        result["is_corrupt"] = True
        result["validation_error"] = "Zero-byte image file"
        return result

    if label is not None and label not in {0, 1, 2, 3, 4}:
        result["is_corrupt"] = True
        result["validation_error"] = f"Invalid DR severity label {label}; expected integer in 0..4"
        return result

    try:
        # Step 1: verify structural integrity without full pixel decode
        with Image.open(path) as img_verify:
            img_verify.verify()
        # Step 2: reopen and decode to RGB to catch truncated pixel streams
        with Image.open(path) as img_read:
            img_rgb = ImageOps.exif_transpose(img_read).convert("RGB")
            width, height = img_rgb.size
            if width < min_dim or height < min_dim:
                result["is_corrupt"] = True
                result["validation_error"] = f"Image dimensions {width}x{height} below minimum {min_dim}px"
                result["width"] = width
                result["height"] = height
                return result
            arr = np.asarray(img_rgb)
            if arr.ndim != 3 or arr.shape[2] != 3:
                result["is_corrupt"] = True
                result["validation_error"] = f"Unexpected tensor shape {arr.shape}"
                return result
            result["width"] = int(width)
            result["height"] = int(height)
            result["channels"] = 3
            result["aspect_ratio"] = round(float(width) / float(height), 4)
            result["phash"] = compute_dhash(img_rgb)
        result["sha256"] = compute_sha256(path)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        result["is_corrupt"] = True
        result["validation_error"] = f"Corrupt or unreadable image: {exc}"

    return result


def validate_image_record(row: dict[str, Any], min_dim: int = 32) -> dict[str, Any]:
    """Validate a dataset record dict containing `abs_path` and optional `original_label`."""
    path = row.get("abs_path") or row.get("relative_path") or ""
    label = row.get("original_label")
    label_int = int(label) if label is not None and not pd.isna(label) else None
    return validate_image_file(path, label=label_int, min_dim=min_dim)


def find_near_duplicates(
    df: pd.DataFrame,
    max_hamming: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect exact (SHA-256) and perceptual (dHash) duplicates and assign cluster IDs.

    Returns:
        - Updated DataFrame with `duplicate_cluster_id` and `is_duplicate` columns.
        - DataFrame of flagged duplicate/near-duplicate pairs.
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
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

    pairs: list[dict[str, Any]] = []
    valid_idx = [i for i in range(n) if not bool(out.loc[i, "is_corrupt"]) and str(out.loc[i, "phash"]) != ""]

    # Exact SHA-256 index
    sha_map: dict[str, list[int]] = {}
    for i in valid_idx:
        sha = str(out.loc[i, "sha256"])
        if sha:
            sha_map.setdefault(sha, []).append(i)
    for sha, idxs in sha_map.items():
        if len(idxs) > 1:
            base = idxs[0]
            for other in idxs[1:]:
                union(base, other)
                pairs.append(
                    {
                        "image_a": out.loc[base, "image_id"],
                        "image_b": out.loc[other, "image_id"],
                        "match_type": "exact_sha256",
                        "hamming_distance": 0,
                    }
                )

    # Pairwise perceptual dHash check (fast for thousands of images using integer XOR)
    hashes = [(i, int(str(out.loc[i, "phash"]), 16)) for i in valid_idx]
    for pos_a in range(len(hashes)):
        idx_a, val_a = hashes[pos_a]
        for pos_b in range(pos_a + 1, len(hashes)):
            idx_b, val_b = hashes[pos_b]
            dist = (val_a ^ val_b).bit_count()
            if dist <= max_hamming:
                # Skip duplicate logging if already identical sha256 pair
                if find(idx_a) != find(idx_b):
                    union(idx_a, idx_b)
                    pairs.append(
                        {
                            "image_a": out.loc[idx_a, "image_id"],
                            "image_b": out.loc[idx_b, "image_id"],
                            "match_type": "perceptual_dhash",
                            "hamming_distance": int(dist),
                        }
                    )

    cluster_sizes: dict[int, int] = {}
    roots = [find(i) for i in range(n)]
    for r in roots:
        cluster_sizes[r] = cluster_sizes.get(r, 0) + 1

    out["duplicate_cluster_id"] = [f"dup_{r}" for r in roots]
    out["is_duplicate_group"] = [cluster_sizes[r] > 1 for r in roots]
    pairs_df = pd.DataFrame(pairs, columns=["image_a", "image_b", "match_type", "hamming_distance"])
    return out, pairs_df
