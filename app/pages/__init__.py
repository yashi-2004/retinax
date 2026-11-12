"""Modular Gradio page builders dynamically imported by filename order."""

from __future__ import annotations

import importlib
from typing import Callable

PAGE_MODULES = [
    "app.pages.01_overview",
    "app.pages.02_image_explorer",
    "app.pages.03_image_quality",
    "app.pages.04_preprocessing_lab",
    "app.pages.05_cnn_research",
    "app.pages.06_transfer_learning",
    "app.pages.07_transformer_lab",
    "app.pages.08_self_supervised_learning",
    "app.pages.09_foundation_models",
    "app.pages.10_explainability",
    "app.pages.11_uncertainty",
    "app.pages.12_domain_generalisation",
    "app.pages.13_model_comparison",
    "app.pages.14_research_reports",
]


def load_page_builders() -> list[tuple[str, Callable[[], None]]]:
    """Import all 14 modular page builders cleanly."""
    builders: list[tuple[str, Callable[[], None]]] = []
    for mod_name in PAGE_MODULES:
        mod = importlib.import_module(mod_name)
        builders.append((getattr(mod, "TAB_TITLE", mod_name), mod.build_tab))
    return builders
