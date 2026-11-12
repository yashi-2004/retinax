"""Preprocessing & Clinical Augmentation Laboratory page."""

from __future__ import annotations

import gradio as gr
import numpy as np

from app.components.state import get_available_manifests, load_active_manifest
from src.preprocessing.augmentation import generate_augmentation_grid
from src.preprocessing.enhancement import apply_clahe, apply_gaussian_blur, apply_histogram_equalization, apply_median_filter, apply_unsharp_mask
from src.preprocessing.fundus_processing import load_rgb_array, preprocess_fundus_image

TAB_TITLE = "4. Preprocessing Lab"


def _run_preprocessing_lab(
    uploaded_img: np.ndarray | None,
    manifest_name: str | None,
    sample_id: str | None,
    image_size: int,
    clahe_clip: float,
    aug_policy: str,
    classical_filter: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[np.ndarray]]:
    rgb = None
    if uploaded_img is not None:
        rgb = load_rgb_array(uploaded_img)
    elif manifest_name and sample_id:
        df = load_active_manifest(manifest_name)
        hit = df[df["image_id"].astype(str) == str(sample_id)]
        if len(hit) > 0:
            rgb = load_rgb_array(hit.iloc[0]["abs_path"])

    if rgb is None:
        return None, None, None, None, None, []

    sz = int(image_size)
    raw_res = preprocess_fundus_image(rgb, preset="raw_resize", image_size=sz)
    fov_crop = preprocess_fundus_image(rgb, preset="fov_crop", image_size=sz)
    clahe_img = preprocess_fundus_image(rgb, preset="fov_clahe_norm", image_size=sz, clahe_clip_limit=float(clahe_clip))
    ben_graham = preprocess_fundus_image(rgb, preset="ben_graham", image_size=sz)

    if classical_filter == "Histogram Equalization":
        extra = apply_histogram_equalization(fov_crop)
    elif classical_filter == "Gaussian Blur (k=5)":
        extra = apply_gaussian_blur(fov_crop, kernel_size=5)
    elif classical_filter == "Median Filter (k=5)":
        extra = apply_median_filter(fov_crop, kernel_size=5)
    elif classical_filter == "Unsharp Mask":
        extra = apply_unsharp_mask(fov_crop)
    elif classical_filter == "Gray-World Color Norm":
        extra = preprocess_fundus_image(rgb, preset="color_norm", image_size=sz)
    else:
        extra = preprocess_fundus_image(rgb, preset="green_channel", image_size=sz)

    aug_list = generate_augmentation_grid(clahe_img, policy=aug_policy, num_samples=4, image_size=sz, seed=42)
    return raw_res, fov_crop, clahe_img, ben_graham, extra, aug_list


def build_tab() -> None:
    manifests = get_available_manifests()
    gr.Markdown(
        "### Fundus Preprocessing & Augmentation Laboratory\n"
        "Treat preprocessing as an experimental hypothesis: compare whether enhancement sharpens retinal lesions "
        "or alters subtle clinical pigmentation."
    )

    with gr.Row():
        upload_img = gr.Image(type="numpy", label="Upload Retinal Image")
        with gr.Column():
            manifest_dd = gr.Dropdown(choices=manifests, value=(manifests[0] if manifests else None), label="Or Manifest")
            sample_dd = gr.Dropdown(choices=[], label="Sample ID")
            size_dd = gr.Dropdown(choices=[128, 224, 320], value=224, label="Target Resolution (px)")
            clip_slider = gr.Slider(1.0, 5.0, value=2.0, step=0.5, label="CLAHE Clip Limit")
            filter_dd = gr.Dropdown(
                choices=[
                    "Green Channel + CLAHE",
                    "Gray-World Color Norm",
                    "Histogram Equalization",
                    "Gaussian Blur (k=5)",
                    "Median Filter (k=5)",
                    "Unsharp Mask",
                ],
                value="Green Channel + CLAHE",
                label="Classical Enhancement / Colour Variant",
            )
            aug_dd = gr.Dropdown(
                choices=["none", "light_clinical", "moderate_clinical", "domain_shift_robust"],
                value="moderate_clinical",
                label="Clinical Augmentation Policy",
            )
            run_btn = gr.Button("Compare Preprocessing & Augmentations", variant="primary")

    with gr.Row():
        out_raw = gr.Image(type="numpy", label="1. Raw Resized")
        out_fov = gr.Image(type="numpy", label="2. FOV Border Crop")
        out_clahe = gr.Image(type="numpy", label="3. FOV + LAB CLAHE")
        out_bg = gr.Image(type="numpy", label="4. Ben Graham Illumination")
        out_extra = gr.Image(type="numpy", label="5. Selected Classical Variant")

    aug_gallery = gr.Gallery(label="Clinical Augmentation Previews (4 Samples)", columns=4, height="auto")

    def _update_samples(m_name: str):
        df = load_active_manifest(m_name)
        ids = df[~df["is_corrupt"].astype(bool)]["image_id"].astype(str).head(50).tolist() if not df.empty else []
        return gr.update(choices=ids, value=(ids[0] if ids else None))

    manifest_dd.change(fn=_update_samples, inputs=[manifest_dd], outputs=[sample_dd])
    run_btn.click(
        fn=_run_preprocessing_lab,
        inputs=[upload_img, manifest_dd, sample_dd, size_dd, clip_slider, aug_dd, filter_dd],
        outputs=[out_raw, out_fov, out_clahe, out_bg, out_extra, aug_gallery],
    )
