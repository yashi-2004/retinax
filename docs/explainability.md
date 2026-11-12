# Visual Explainability & Post-Hoc Attribution Diagnostics

## 1. Motivation & Epistemic Boundaries

Deep retinal classifiers can achieve deceptive validation performance by exploiting **spurious shortcuts** — black FOV vignetting corners, camera bezel reflections, dust artifacts, or clinic brightness signatures — rather than pathological microvascular lesions. Visual attribution helps audit whether predicted features align spatially with retinal structures inside the field of view (FOV).

> **Clinical & Scientific Caveat (`EXPLAINABILITY_DISCLAIMER`):** Post-hoc attribution maps highlight spatial regions influencing model logits. They are **NOT** validated lesion segmentations, and diffuse or border-focused saliency can signal shortcut learning rather than diagnostic evidence.

## 2. Implemented Attribution Methods (`src/explainability/`)

### 2.1 Hook-Based Grad-CAM & Grad-CAM++ (`src/explainability/gradcam.py`)

Given target activation tensor $A \in \mathbb{R}^{K \times H' \times W'}$ and target class logit $y^c$:

- **Grad-CAM:** Computes spatial global average pooled channel importance weights:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{ij}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$
- **Grad-CAM++:** Replaces uniform global pooling with pixel-wise second-order weighted coefficients so small, scattered retinal lesions (microaneurysms, exudate clusters) are less dominated by single large structures.
- **Vision Transformer Token Reshaping (`_reshape_vit_tokens`):** When targeting ViT/DINOv2 normalized transformer blocks producing 3D sequences `[B, N_tokens, D]`, strips prefix tokens (`CLS`, optional register tokens) and reshapes patch tokens back into 2D grids `[B, D, H', W']` prior to CAM computation.
- **Frozen-Backbone Compatibility:** Registers backward hooks directly on the detached forward activation tensor (`requires_grad_(True)` on input) rather than weight parameters, enabling Grad-CAM even when backbones run in frozen feature-extraction mode.

### 2.2 Integrated Gradients (`src/explainability/integrated_gradients.py`)

Backed by **Captum** (`IntegratedGradients`):
$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i}\,d\alpha$$
using a zero baseline $x'$, Gauss-Legendre path quadrature (`n_steps=24`), Completeness Theorem convergence delta logging (`convergence_delta`), and mild Gaussian smoothing ($\sigma = 1.5$) for visual legibility.

### 2.3 Occlusion Sensitivity (`src/explainability/integrated_gradients.py`)

Backed by **Captum** (`Occlusion`), systematically sliding an occluding patch across the retinal image (`patch_size=24`, `stride=16` scaled for CPU runtime) and measuring target class logit drop $\Delta y^c$ as a model-agnostic sanity check independent of gradient noise.

## 3. Quantitative Sanity Audit (`src/explainability/visualisation.py`)

`compute_fov_attribution_ratio` quantifies what fraction of total normalized heatmap energy falls inside the segmented circular retinal fundus (`extract_fov_mask`) versus outside the black background border:
$$\text{FOV Mass Ratio} = \frac{\sum_{(u,v) \in \text{FOV}} H(u,v)}{\sum_{\text{all }(u,v)} H(u,v) + \epsilon}$$
Substantial attribution falling outside the FOV (low ratio) provides quantitative evidence of border-artifact reliance even when classification accuracy appears high.

`build_explanation_figure` renders four aligned panels: **Original Retinal Image**, **False-Colour Heatmap**, **Blended Overlay**, and **Class Probability Bar Chart** with predicted class, confidence, actual grade (when known), and FOV Mass Ratio.
