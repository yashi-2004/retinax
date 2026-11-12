"""Procedural synthetic fundus-like demonstration dataset generator.

IMPORTANT DISCLAIMER:
These images are synthetic geometric phantoms generated strictly for pipeline verification,
unit testing, and software demonstration when Kaggle APTOS 2019 / EyePACS datasets are not
present locally. They are explicitly marked ``is_synthetic=True`` and must NEVER be
interpreted as real clinical retinal photography or clinical model benchmarks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from src.config import DR_CLASS_NAMES, resolve_path


def _draw_branching_vessel(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    angle: float,
    length: float,
    width: int,
    depth: int,
    rng: np.random.Generator,
    color: tuple[int, int, int],
) -> None:
    """Recursively draw smooth bifurcating retinal-like vessel segments."""
    if depth <= 0 or length < 4.0:
        return
    steps = 4
    pts = [start]
    cur_x, cur_y = start
    cur_angle = angle
    seg_len = length / steps
    for _ in range(steps):
        cur_angle += float(rng.normal(0.0, 0.14))
        cur_x += math.cos(cur_angle) * seg_len
        cur_y += math.sin(cur_angle) * seg_len
        pts.append((cur_x, cur_y))

    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=color, width=max(1, width))

    if depth > 1:
        spread = float(rng.uniform(0.28, 0.55))
        _draw_branching_vessel(
            draw, (cur_x, cur_y), cur_angle - spread, length * 0.72, max(1, width - 1), depth - 1, rng, color
        )
        _draw_branching_vessel(
            draw, (cur_x, cur_y), cur_angle + spread, length * 0.68, max(1, width - 1), depth - 1, rng, color
        )


def render_synthetic_fundus_phantom(
    grade: int,
    seed: int,
    size: int = 256,
    domain_shift: bool = False,
) -> np.ndarray:
    """Render a deterministic synthetic fundus-like phantom image (RGB uint8).

    Args:
        grade: Synthetic DR severity grade in {0, 1, 2, 3, 4}.
        seed: Random seed controlling geometry, eye laterality, and synthetic artifacts.
        size: Output square image dimension.
        domain_shift: If True, simulates an external acquisition device with altered
            colour temperature, illumination gradient, contrast, and mild focus blur.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = size / 2.0 + float(rng.uniform(-0.02, 0.02) * size)
    cy = size / 2.0 + float(rng.uniform(-0.02, 0.02) * size)
    radius = size * float(rng.uniform(0.43, 0.47))

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    fov_mask = dist <= radius
    norm_r = np.clip(dist / max(radius, 1.0), 0.0, 1.0)

    # Base retinal pigment & illumination profile (distinct warm palette vs shifted cohort)
    if domain_shift:
        base_r = float(rng.uniform(165, 195))
        base_g = float(rng.uniform(95, 125))
        base_b = float(rng.uniform(55, 82))
        vignette = 1.0 - 0.52 * (norm_r**2)
    else:
        base_r = float(rng.uniform(185, 220))
        base_g = float(rng.uniform(75, 105))
        base_b = float(rng.uniform(35, 55))
        vignette = 1.0 - 0.36 * (norm_r**2)

    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0] = base_r * vignette
    img[..., 1] = base_g * vignette
    img[..., 2] = base_b * vignette

    # Subtle choroidal texture
    texture = rng.normal(0.0, 3.5, size=(size, size)).astype(np.float32)
    img[..., 0] += texture
    img[..., 1] += texture * 0.6

    # Macula (darker central-temporal oval)
    side = -1.0 if rng.random() < 0.5 else 1.0
    disc_x = cx + side * radius * 0.46
    disc_y = cy + float(rng.uniform(-0.08, 0.08) * radius)
    mac_x = cx - side * radius * 0.14
    mac_y = cy + float(rng.uniform(-0.04, 0.04) * radius)

    mac_dist = np.sqrt((xx - mac_x) ** 2 + (yy - mac_y) ** 2)
    mac_dark = np.exp(-0.5 * (mac_dist / (0.14 * radius)) ** 2)
    img[..., 0] -= 32.0 * mac_dark
    img[..., 1] -= 22.0 * mac_dark
    img[..., 2] -= 10.0 * mac_dark

    # Optic disc (bright yellowish ellipse)
    disc_dist = np.sqrt(((xx - disc_x) / 0.92) ** 2 + (yy - disc_y) ** 2)
    disc_glow = np.exp(-0.5 * (disc_dist / (0.12 * radius)) ** 2)
    img[..., 0] += 70.0 * disc_glow
    img[..., 1] += 110.0 * disc_glow
    img[..., 2] += 80.0 * disc_glow

    pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    # Major vascular arcades radiating from optic disc
    vessel_color = (
        int(max(40, base_r * 0.52)),
        int(max(18, base_g * 0.36)),
        int(max(12, base_b * 0.36)),
    )
    base_dir = math.pi if side > 0 else 0.0
    for offset in [-0.75, -0.35, 0.35, 0.75]:
        _draw_branching_vessel(
            draw=draw,
            start=(disc_x, disc_y),
            angle=base_dir + offset + float(rng.uniform(-0.1, 0.1)),
            length=radius * float(rng.uniform(0.55, 0.75)),
            width=max(2, size // 85),
            depth=3,
            rng=rng,
            color=vessel_color,
        )

    # Helper to sample a point cleanly inside FOV
    def sample_fov_point(max_frac: float = 0.78) -> tuple[float, float]:
        r = radius * math.sqrt(float(rng.uniform(0.05, max_frac**2)))
        theta = float(rng.uniform(0, 2 * math.pi))
        return cx + r * math.cos(theta), cy + r * math.sin(theta)

    # Grade-specific synthetic markers
    # 1: Microaneurysm-like tiny dark-red dots
    n_ma = {0: 0, 1: 6, 2: 14, 3: 24, 4: 30}.get(grade, 0)
    for _ in range(n_ma):
        px, py = sample_fov_point()
        rad = max(1.2, size / 140.0)
        draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(95, 18, 16))

    # 2+: Blot hemorrhages & hard yellow exudates
    n_hem = {0: 0, 1: 0, 2: 6, 3: 15, 4: 22}.get(grade, 0)
    for _ in range(n_hem):
        px, py = sample_fov_point()
        rx = float(rng.uniform(size / 70.0, size / 35.0))
        ry = float(rng.uniform(size / 75.0, size / 40.0))
        draw.ellipse([px - rx, py - ry, px + rx, py + ry], fill=(82, 14, 14))

    n_exudates = {0: 0, 1: 0, 2: 7, 3: 16, 4: 24}.get(grade, 0)
    for _ in range(n_exudates):
        px, py = sample_fov_point(0.68)
        rad = float(rng.uniform(size / 110.0, size / 55.0))
        draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(242, 224, 108))

    # 3+: Cotton-wool soft patches
    n_cws = {0: 0, 1: 0, 2: 0, 3: 5, 4: 8}.get(grade, 0)
    for _ in range(n_cws):
        px, py = sample_fov_point(0.65)
        rx = float(rng.uniform(size / 42.0, size / 28.0))
        draw.ellipse([px - rx, py - rx * 0.75, px + rx, py + rx * 0.75], fill=(228, 225, 212))

    # 4: Proliferative fine neovascular tufts near disc/arcades
    if grade >= 4:
        for _ in range(6):
            tx = disc_x + float(rng.uniform(-0.35, 0.35) * radius)
            ty = disc_y + float(rng.uniform(-0.35, 0.35) * radius)
            for _ in range(5):
                ang = float(rng.uniform(0, 2 * math.pi))
                seg = radius * 0.12
                draw.line([tx, ty, tx + seg * math.cos(ang), ty + seg * math.sin(ang)], fill=(125, 22, 20), width=1)

    if domain_shift:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.8))

    arr = np.asarray(pil_img, dtype=np.float32)

    # Apply realistic dark circular border outside FOV
    arr[~fov_mask] = rng.uniform(0, 6, size=arr[~fov_mask].shape)
    if domain_shift:
        # Mild brightness / contrast compression under external device shift
        arr = np.clip(arr * 0.88 + 10.0, 0, 255)
    return np.clip(arr, 0, 255).astype(np.uint8)


def generate_synthetic_demo_dataset(
    output_dir: str | Path | None = None,
    num_primary: int = 100,
    num_external: int = 50,
    image_size: int = 256,
    seed: int = 42,
    include_quality_variants: bool = True,
) -> dict[str, Any]:
    """Generate both primary and domain-shifted synthetic demonstration cohorts.

    Explicitly marks all generated data as synthetic (`is_synthetic=True`).
    """
    base_dir = resolve_path(output_dir or "data/raw/synthetic_demo")
    primary_img_dir = base_dir / "primary_images"
    external_img_dir = base_dir / "external_images"
    primary_img_dir.mkdir(parents=True, exist_ok=True)
    external_img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    # Realistic imbalanced DR distribution (majority class 0, fewer severe/PDR)
    class_probs = np.array([0.42, 0.14, 0.24, 0.11, 0.09], dtype=np.float64)
    class_probs /= class_probs.sum()

    def _build_cohort(
        count: int,
        img_dir: Path,
        domain_shift: bool,
        cohort_tag: str,
        start_seed: int,
    ) -> pd.DataFrame:
        # Ensure every class 0..4 gets at least 2 samples so stratified splitting works cleanly
        min_per_class = 2 if count >= 10 else 1
        labels = [c for c in range(5) for _ in range(min_per_class)]
        remaining = max(0, count - len(labels))
        if remaining > 0:
            labels.extend(rng.choice(5, size=remaining, p=class_probs).tolist())
        labels = labels[:count]
        rng.shuffle(labels)

        rows: list[dict[str, Any]] = []
        for idx, grade in enumerate(labels):
            patient_num = (idx // 2) + 1
            eye = "left" if idx % 2 == 0 else "right"
            image_id = f"{cohort_tag}_{patient_num:04d}_{eye}"
            img_arr = render_synthetic_fundus_phantom(
                grade=int(grade),
                seed=start_seed + idx,
                size=image_size,
                domain_shift=domain_shift,
            )

            # Optionally inject mild quality degradations on ~15% of samples for QA research
            quality_note = "nominal"
            if include_quality_variants and idx % 7 == 6:
                pil_tmp = Image.fromarray(img_arr)
                if idx % 14 == 6:
                    pil_tmp = pil_tmp.filter(ImageFilter.GaussianBlur(radius=2.6))
                    quality_note = "synthetic_blur"
                else:
                    img_arr = np.clip(img_arr.astype(np.float32) * 0.42, 0, 255).astype(np.uint8)
                    pil_tmp = Image.fromarray(img_arr)
                    quality_note = "synthetic_underexposed"
                img_arr = np.asarray(pil_tmp, dtype=np.uint8)

            out_path = img_dir / f"{image_id}.png"
            Image.fromarray(img_arr).save(out_path, format="PNG")
            rows.append(
                {
                    "id_code": image_id,
                    "diagnosis": int(grade),
                    "label_name": DR_CLASS_NAMES[int(grade)],
                    "patient_id": f"{cohort_tag}_pt_{patient_num:04d}",
                    "eye": eye,
                    "is_synthetic": True,
                    "synthetic_quality_note": quality_note,
                    "filename": f"{image_id}.png",
                }
            )
        return pd.DataFrame(rows)

    primary_df = _build_cohort(num_primary, primary_img_dir, domain_shift=False, cohort_tag="syn_aptos", start_seed=seed * 100)
    external_df = _build_cohort(num_external, external_img_dir, domain_shift=True, cohort_tag="syn_eyepacs", start_seed=seed * 100 + 10000)

    primary_csv = base_dir / "primary_labels.csv"
    external_csv = base_dir / "external_labels.csv"
    primary_df.to_csv(primary_csv, index=False)
    external_df.to_csv(external_csv, index=False)

    meta = {
        "is_synthetic": True,
        "disclaimer": (
            "SYNTHETIC DEMO DATA ONLY — NOT REAL CLINICAL IMAGERY. "
            "Generated procedurally for software verification and educational walkthroughs."
        ),
        "primary_count": int(len(primary_df)),
        "external_count": int(len(external_df)),
        "image_size": int(image_size),
        "seed": int(seed),
    }
    with (base_dir / "synthetic_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    return {
        "base_dir": str(base_dir),
        "primary_csv": str(primary_csv),
        "external_csv": str(external_csv),
        "primary_count": len(primary_df),
        "external_count": len(external_df),
        "is_synthetic": True,
    }
