"""Verify local CPU hardware environment, package versions, and dataset availability."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_hardware_summary
from src.data.ingestion import discover_datasets
from src.foundation.biomedical_clip import check_biomedclip_availability
from src.training.checkpoints import list_checkpoints


def main() -> None:
    report = {
        "hardware": get_hardware_summary(),
        "datasets": discover_datasets(),
        "biomedclip": check_biomedclip_availability(),
        "checkpoints": list_checkpoints(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
