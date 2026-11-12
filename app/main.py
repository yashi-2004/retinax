"""Main entry point for the RetinaX Gradio Biomedical Research Dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src  # noqa: F401 - ensures offline/sandbox-safe environment variables
import gradio as gr
import gradio_client.utils as gc_utils

from app.components.theme import RETINAX_CSS, build_clinical_theme, render_disclaimer_banner
from app.pages import load_page_builders


def _patch_gradio_json_schema_bool_compat() -> None:
    """Guard gradio_client against Pydantic>=2.11 boolean JSON schemas (additionalProperties: True)."""
    if getattr(gc_utils, "_retinax_bool_schema_patched", False):
        return
    orig_get_type = gc_utils.get_type
    orig_inner = gc_utils._json_schema_to_python_type
    orig_outer = gc_utils.json_schema_to_python_type

    def _safe_get_type(schema: Any) -> Any:
        if isinstance(schema, bool):
            return {}
        return orig_get_type(schema)

    def _safe_inner(schema: Any, defs: Any) -> str:
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return orig_inner(schema, defs)

    def _safe_outer(schema: Any) -> str:
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return orig_outer(schema)

    gc_utils.get_type = _safe_get_type
    gc_utils._json_schema_to_python_type = _safe_inner
    gc_utils.json_schema_to_python_type = _safe_outer
    gc_utils._retinax_bool_schema_patched = True


def create_app() -> gr.Blocks:
    """Construct the modular 14-laboratory RetinaX Gradio application."""
    _patch_gradio_json_schema_bool_compat()
    theme = build_clinical_theme()
    builders = load_page_builders()

    with gr.Blocks(title="RetinaX — Retinal Image Intelligence Research Lab", theme=theme, css=RETINAX_CSS) as demo:
        gr.HTML(render_disclaimer_banner())
        with gr.Tabs():
            for title, builder_fn in builders:
                with gr.Tab(title):
                    builder_fn()

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch RetinaX Local Research Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = create_app()
    app.launch(server_name=args.host, server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()
