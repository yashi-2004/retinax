"""Classical computer vision image enhancement and colour-space transformations."""

from __future__ import annotations

import cv2
import numpy as np


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def apply_histogram_equalization(image_rgb: np.ndarray) -> np.ndarray:
    """Apply global luminance histogram equalization in YCrCb colour space."""
    img = _ensure_uint8(image_rgb)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    channel_mode: str = "lab_l",
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        channel_mode: ``'lab_l'`` enhances lightness in CIE-LAB; ``'green'`` enhances
            the green channel directly and retains RGB; ``'rgb_each'`` applies per channel.
    """
    img = _ensure_uint8(image_rgb)
    grid = (max(2, int(tile_grid_size)), max(2, int(tile_grid_size)))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=grid)

    if channel_mode == "lab_l":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = clahe.apply(l_ch)
        merged = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    if channel_mode == "green":
        out = img.copy()
        out[:, :, 1] = clahe.apply(out[:, :, 1])
        return out
    if channel_mode == "rgb_each":
        channels = [clahe.apply(img[:, :, c]) for c in range(3)]
        return np.stack(channels, axis=-1)
    raise ValueError(f"Unsupported CLAHE channel_mode '{channel_mode}'.")


def apply_gaussian_blur(image_rgb: np.ndarray, kernel_size: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Apply Gaussian smoothing with odd kernel size."""
    img = _ensure_uint8(image_rgb)
    k = max(3, int(kernel_size) | 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma))


def apply_median_filter(image_rgb: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply non-linear median filtering (suppresses salt-and-pepper sensor artifacts)."""
    img = _ensure_uint8(image_rgb)
    k = max(3, int(kernel_size) | 1)
    return cv2.medianBlur(img, k)


def apply_unsharp_mask(image_rgb: np.ndarray, sigma: float = 1.5, strength: float = 1.2) -> np.ndarray:
    """Sharpen fine vascular and microaneurysm edges via unsharp masking."""
    img = _ensure_uint8(image_rgb).astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma))
    sharpened = img + float(strength) * (img - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def contrast_stretch(
    image_rgb: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> np.ndarray:
    """Linear percentile contrast stretching across RGB intensities."""
    img = _ensure_uint8(image_rgb).astype(np.float32)
    lo = float(np.percentile(img, lower_pct))
    hi = float(np.percentile(img, upper_pct))
    if hi <= lo + 1e-5:
        return img.astype(np.uint8)
    stretched = (img - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def extract_green_channel(image_rgb: np.ndarray, apply_clahe_boost: bool = True) -> np.ndarray:
    """Extract clinical green channel (high hemoglobin/vessel contrast) as 3-channel RGB visualization."""
    img = _ensure_uint8(image_rgb)
    green = img[:, :, 1]
    if apply_clahe_boost:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green = clahe.apply(green)
    return np.stack([green, green, green], axis=-1)


def convert_color_space(image_rgb: np.ndarray, target_space: str = "RGB") -> np.ndarray:
    """Convert RGB image into visualizable 3-channel representation in another colour space."""
    img = _ensure_uint8(image_rgb)
    space = target_space.upper()
    if space == "RGB":
        return img
    if space == "GRAY":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return np.stack([gray, gray, gray], axis=-1)
    if space == "LAB":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    if space == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if space == "GREEN":
        return extract_green_channel(img, apply_clahe_boost=False)
    if space == "GREEN_CLAHE":
        return extract_green_channel(img, apply_clahe_boost=True)
    raise ValueError(f"Unsupported colour space '{target_space}'.")
