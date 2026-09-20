"""Compute real per-channel mean/std over the TRAIN split only, at the
resolution the pipeline will actually use (post resize-and-pad, pre any
random augmentation or Normalize). Not training — just data statistics,
needed so `Normalize()` isn't fed placeholder or ImageNet numbers for the
from-scratch model.

Usage:
    python scripts/compute_normalization_stats.py --manifest data/manifest.csv --image-size 224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covidcxr.data.dataset import CXRDataset  # noqa: E402
from covidcxr.data.transforms import ResizeAndPad  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args(argv)

    transform = T.Compose([ResizeAndPad(args.image_size), T.ToTensor()])
    ds = CXRDataset(args.manifest, split="train", transform=transform)

    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
    for image, _ in ds:
        total_sum += image.sum().item()
        total_sq_sum += (image**2).sum().item()
        total_pixels += image.numel()

    mean = total_sum / total_pixels
    variance = (total_sq_sum / total_pixels) - mean**2
    std = variance**0.5

    print(f"Computed over {len(ds)} TRAIN-split images at {args.image_size}x{args.image_size}, 1 channel:")
    print(f"  mean = {mean:.4f}")
    print(f"  std  = {std:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
