"""Clinical biomedical theme and safety disclaimer components for Gradio."""

from __future__ import annotations

import gradio as gr

from src.config import CLINICAL_DISCLAIMER

RETINAX_CSS = """
.gradio-container {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif !important;
    max-width: 1380px !important;
    margin: 0 auto !important;
}
.rx-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #0D9488;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 12px;
}
.rx-header h1 {
    margin: 0 0 4px 0;
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.01em;
}
.rx-header p {
    margin: 0;
    font-size: 13.5px;
    color: #475569;
}
.rx-disclaimer {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid #DC2626;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 14px;
    font-size: 13px;
    color: #7F1D1D;
}
.rx-syn-notice {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    border-radius: 6px;
    padding: 9px 14px;
    margin-bottom: 12px;
    font-size: 13px;
    color: #78350F;
}
.rx-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 16px;
}
"""


def build_clinical_theme() -> gr.themes.Base:
    """Construct a restrained, high-contrast medical research theme."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#F8FAFC",
        block_background_fill="#FFFFFF",
        block_border_color="#E2E8F0",
        block_title_text_color="#0F172A",
        body_text_color="#0F172A",
        button_primary_background_fill="#0F766E",
        button_primary_background_fill_hover="#0D9488",
        button_primary_text_color="#FFFFFF",
    )


def render_disclaimer_banner() -> str:
    """Return HTML safety disclaimer banner required across all views."""
    return (
        '<div class="rx-header">'
        '<h1>RetinaX — Retinal Image Intelligence Research Laboratory</h1>'
        '<p>Trustworthy Biomedical Computer Vision: Classical Processing &bull; CNNs &bull; Transfer Learning &bull; ViT &bull; SSL &bull; Foundation Models &bull; XAI &bull; Uncertainty &bull; Domain Shift</p>'
        '</div>'
        f'<div class="rx-disclaimer"><strong>Research Prototype Disclaimer:</strong> {CLINICAL_DISCLAIMER}</div>'
    )
