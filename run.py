"""Root Gradio entrypoint for Hugging Face Spaces, Docker, and cloud CPU hosting."""

from __future__ import annotations

import os

from app.main import create_app

demo = create_app()

if __name__ == "__main__":
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=host, server_port=port, share=False, show_error=True)
