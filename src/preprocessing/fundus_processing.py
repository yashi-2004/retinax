"""Fundus-specific inspection, circular FOV extraction, illumination correction, and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.preprocessing.enhancement import apply_clahe, extract_green_channel


def load_rgb_array(image_source: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Load an image path, PIL Image, or array into RGB uint8 NumPy array."""
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as img:
            return np.asarray(ImageOps.exif_transpose(img).convert("RGB"), dtype=np.uint8)
    if isinstance(image_source, Image.Image):
        return np.asarray(ImageOps.exif_transpose(image_source).convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image_source)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def extract_fov_mask(image_rgb: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Detect binary retinal field-of-view mask separating foreground from dark border."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > threshold).astype(np.uint8)
    if mask.sum() < 0.05 * mask.size:
        # Fallback when image is very dark
        mask = (gray > max(2, threshold // 3)).astype(np.uint8)
    if mask.sum() == 0:
        return np.ones_like(gray, dtype=bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        mask = (labels == largest_label).astype(np.uint8)

    return mask.astype(bool)


def crop_to_fov(
    image_rgb: np.ndarray,
    threshold: int = 10,
    pad_to_square: bool = True,
) -> np.ndarray:
    """Remove outer black scanner margins by cropping tightly to the retinal FOV."""
    mask = extract_fov_mask(image_rgb, threshold=threshold)
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        return image_rgb

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = image_rgb[y0:y1, x0:x1]

    if not pad_to_square:
        return cropped

    h, w = cropped.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top : top + h, left : left + w] = cropped
    return canvas


def apply_circular_fov_mask(image_rgb: np.ndarray, shrink_ratio: float = 0.96) -> np.ndarray:
    """Zero out pixels outside a centred circle to eliminate border edge artifacts."""
    h, w = image_rgb.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    radius = min(h, w) * 0.5 * shrink_ratio
    yy, xx = np.ogrid[:h, :w]
    circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius**2)
    out = image_rgb.copy()
    out[~circle] = 0
    return out


def ben_graham_preprocess(
    image_rgb: np.ndarray,
    sigma_ratio: float = 0.05,
    apply_circle_mask: bool = True,
) -> np.ndarray:
    """Ben Graham local illumination subtraction for retinal fundus standardization.

    Computes ``clip(4 * I - 4 * GaussianBlur(I, sigma) + 128, 0, 255)`` inside FOV.
    """
    h, w = image_rgb.shape[:2]
    sigma = max(2.0, min(h, w) * sigma_ratio)
    img_f = image_rgb.astype(np.float32)
    local_avg = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)
    enhanced = np.clip(4.0 * img_f - 4.0 * local_avg + 128.0, 0, 255).astype(np.uint8)
    if apply_circle_mask:
        enhanced = apply_circular_fov_mask(enhanced, shrink_ratio=0.92)
    return enhanced


def gray_world_color_norm(image_rgb: np.ndarray, fov_mask: np.ndarray | None = None) -> np.ndarray:
    """Normalize RGB channel balance inside the FOV toward reference retinal balance."""
    mask = fov_mask if fov_mask is not None else extract_fov_mask(image_rgb)
    if not mask.any():
        return image_rgb

    img_f = image_rgb.astype(np.float32)
    means = img_f[mask].mean(axis=0)
    # Use mean luminance inside FOV as neutral target scaled modestly to preserve retinal hue warmth
    target = np.array([1.25, 0.95, 0.72], dtype=np.float32)
    avg_lum = float(means.mean())
    target = target * (avg_lum / max(float(target.mean()), 1e-4))
    gains = np.clip(target / np.clip(means, 5.0, None), 0.65, 1.55)
    normed = np.clip(img_f * gains[None, None, :], 0, 255).astype(np.uint8)
    normed[~mask] = 0
    return normed


def resize_fundus(image_rgb: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Resize RGB image efficiently with area/bilinear interpolation."""
    h, w = image_rgb.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > image_size else cv2.INTER_LINEAR
    return cv2.resize(image_rgb, (int(image_size), int(image_size)), interpolation=interp)


def inspect_fundus_image(image_source: str | Path | Image.Image | np.ndarray) -> dict[str, Any]:
    """Compute classical pixel, channel, histogram, and geometry diagnostics."""
    img = load_rgb_array(image_source)
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fov = extract_fov_mask(img)
    fg = img[fov] if fov.any() else img.reshape(-1, 3)
    fg_gray = gray[fov] if fov.any() else gray.ravel()

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hist_r = np.histogram(img[:, :, 0], bins=64, range=(0, 256))[0].tolist()
    hist_g = np.histogram(img[:, :, 1], bins=64, range=(0, 256))[0].tolist()
    hist_b = np.histogram(img[:, :, 2], bins=64, range=(0, 256))[0].tolist()
    hist_l = np.histogram(gray, bins=64, range=(0, 256))[0].tolist()

    return {
        "width": int(w),
        "height": int(h),
        "channels": int(c),
        "aspect_ratio": round(float(w) / max(float(h), 1.0), 4),
        "fov_coverage": round(float(fov.mean()), 4),
        "brightness_mean": round(float(fg_gray.mean()), 2),
        "contrast_rms": round(float(fg_gray.std()), 2),
        "blur_laplacian": round(lap_var, 2),
        "mean_rgb": [round(float(x), 2) for x in fg.mean(axis=0)],
        "std_rgb": [round(float(x), 2) for x in fg.std(axis=0)],
        "green_channel_mean": round(float(fg[:, 1].mean()), 2),
        "green_channel_std": round(float(fg[:, 1].std()), 2),
        "histograms": {"r": hist_r, "g": hist_g, "b": hist_b, "luminance": hist_l},
    }


def preprocess_fundus_image(
    image_source: str | Path | Image.Image | np.ndarray,
    preset: str = "fov_clahe_norm",
    image_size: int = 224,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
    apply_fov_crop_override: bool | None = None,
    apply_clahe_override: bool | None = None,
    apply_illumination_override: bool | None = None,
    apply_color_norm_override: bool | None = None,
    apply_circular_mask_override: bool | None = None,
) -> np.ndarray:
    """Run a configurable, reproducible fundus preprocessing pipeline.

    Supported presets:
      - ``raw_resize``: Direct resize without border cropping or contrast alteration.
      - ``fov_crop``: FOV border crop + square pad + resize.
      - ``fov_clahe_norm``: FOV crop + resize + LAB CLAHE + circular mask.
      - ``ben_graham``: FOV crop + resize + local illumination subtraction + circular mask.
      - ``green_channel``: FOV crop + resize + CLAHE-enhanced green channel.
      - ``color_norm``: FOV crop + resize + gray-world FOV colour normalization + circular mask.
    """
    img = load_rgb_array(image_source)

    preset_flags: dict[str, dict[str, bool]] = {
        "raw_resize": {"crop": False, "clahe": False, "ben_graham": False, "color_norm": False, "green": False, "circle": False},
        "fov_crop": {"crop": True, "clahe": False, "ben_graham": False, "color_norm": False, "green": False, "circle": False},
        "fov_clahe_norm": {"crop": True, "clahe": True, "ben_graham": False, "color_norm": False, "green": False, "circle": True},
        "ben_graham": {"crop": True, "clahe": False, "ben_graham": True, "color_norm": False, "green": False, "circle": True},
        "green_channel": {"crop": True, "clahe": False, "ben_graham": False, "color_norm": False, "green": True, "circle": True},
        "color_norm": {"crop": True, "clahe": False, "ben_graham": False, "color_norm": True, "green": False, "circle": True},
    }
    if preset not in preset_flags:
        raise ValueError(f"Unknown preprocessing preset '{preset}'. Valid: {list(preset_flags.keys())}")

    flags = dict(preset_flags[preset])
    if apply_fov_crop_override is not None:
        flags["crop"] = apply_fov_crop_override
    if apply_clahe_override is not None:
        flags["clahe"] = apply_clahe_override
    if apply_illumination_override is not None:
        flags["ben_graham"] = apply_illumination_override
    if apply_color_norm_override is not None:
        flags["color_norm"] = apply_color_norm_override
    if apply_circular_mask_override is not None:
        flags["circle"] = apply_circular_mask_override

    if flags["crop"]:
        img = crop_to_fov(img, pad_to_square=True)

    img = resize_fundus(img, image_size=image_size)

    if flags["color_norm"]:
        img = gray_world_color_norm(img)
    if flags["clahe"]:
        img = apply_clahe(img, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid, channel_mode="lab_l")
    if flags["ben_graham"]:
        img = ben_graham_preprocess(img, apply_circle_mask=flags["circle"])
    if flags["green"]:
        img = extract_green_channel(img, apply_clahe_boost=True)
    if flags["circle"] and not flags["ben_graham"]:
        img = apply_circular_fov_mask(img)

    return img
