"""CLI script to compile RetinaX experiment results into Markdown, HTML, and PDF reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reporting.report_generator import generate_research_report


def main() -> None:
    res = generate_research_report()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
