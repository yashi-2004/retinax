"""Optional IDRiD lesion segmentation extension using MONAI 2D U-Net (CPU-compatible)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve_path
from src.data.caching import normalize_rgb_tensor
from src.models.segmentation_unet import LesionUNetWrapper, compute_segmentation_metrics, soft_dice_bce_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional IDRiD Lesion Segmentation (MONAI U-Net)")
    parser.add_argument("--idrid-dir", type=str, default="data/raw/idrid")
    parser.add_argument("--lesion", type=str, default="Hard Exudates")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Verify MONAI U-Net forward/loss/Dice on synthetic mask phantom.")
    args = parser.parse_args()

    idrid_path = resolve_path(args.idrid_dir)
    if args.smoke_test:
        model = LesionUNetWrapper()
        x = torch.randn(2, 3, args.image_size, args.image_size)
        y = (torch.rand(2, 1, args.image_size, args.image_size) > 0.9).float()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        logits = model(x)
        loss = soft_dice_bce_loss(logits, y)
        loss.backward()
        opt.step()
        metrics = compute_segmentation_metrics(logits, y)
        print(json.dumps({"mode": "smoke_test", "loss": round(float(loss.item()), 4), **metrics}, indent=2))
        return

    imgs = sorted([p for p in idrid_path.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".tif"}])
    if not imgs:
        print(
            f"IDRiD optional segmentation dataset not found under {idrid_path}.\n"
            "Download from https://idrid.grand-challenge.org/ (CC BY 4.0) or run with --smoke-test."
        )
        return

    print(f"Found {len(imgs)} IDRiD files under {idrid_path}. Ready for paired mask training.")


if __name__ == "__main__":
    main()
