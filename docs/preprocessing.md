# Fundus Image Processing & Quality Assessment

## 1. Preprocessing as an Experimental Variable

Medical preprocessing pipelines are never neutral: aggressive contrast enhancement can amplify microaneurysms and hard exudates, or conversely accentuate sensor noise and erase subtle pigment cues. RetinaX treats preprocessing as an explicit experimental parameter (`Experiment E`) rather than hardcoding every filter globally.

### Supported Presets (`src/preprocessing/fundus_processing.py`)
- **`raw_resize`**: Direct area/bilinear resizing to target dimensions (`128x128` or `224x224`).
- **`fov_crop`**: Connected-component luminance thresholding isolates the circular retinal field of view, crops outer black margins, and square-pads to preserve optic disc aspect ratio.
- **`fov_clahe_norm`**: FOV crop + resize + CIE-LAB lightness-channel CLAHE (`clipLimit=2.0`, `8x8` tile grid) + circular FOV mask.
- **`ben_graham`**: Local average illumination subtraction:
  $$I_{\text{BG}}(x,y) = \text{clip}\!\left(4 I(x,y) - 4 (G_{\sigma} * I)(x,y) + 128,\ 0,\ 255\right)$$
- **`green_channel`**: Extracts the green RGB channel (highest hemoglobin absorption contrast for vessels and micro-lesions) enhanced via CLAHE.
- **`color_norm`**: FOV-masked gray-world channel gain stabilization for cross-device illumination harmonization.

## 2. Clinically Plausible Augmentation (`src/preprocessing/augmentation.py`)

Using `albumentations`, policies (`none`, `light_clinical`, `moderate_clinical`, `domain_shift_robust`) include horizontal flips (OD/OS laterality), bounded rotations ($\pm 15^\circ$ to $\pm 25^\circ$), mild random resized crops ($0.86\text{--}1.0$), bounded brightness/contrast shifts, and light blur/noise—while strictly avoiding severe spatial warps or inverted hues that could synthesize artificial lesions.

## 3. Image Quality Assessment (`src/preprocessing/quality_assessment.py`)

Classical statistical proxies computed inside the retinal FOV:
- **Sharpness / Focus:** Laplacian variance and Tenengrad gradient energy.
- **Exposure:** Mean luminance, dark clipping fraction ($<18$), glare/bright clipping fraction ($>240$).
- **Contrast:** Root-mean-square intensity standard deviation.
- **FOV & Occlusion:** Foreground coverage ratio and quadrant luminance standard deviation.

> **Important Limitation:** Classical image statistics detect gross defocus, underexposure, glare, and border clipping, but they do **not** guarantee clinical retinal gradability (such as macula clarity or optic disc visibility).
