"""Deterministic-by-default image transforms for chest X-rays.

Two design decisions worth calling out (see data/README.md §5 for the full
rationale):

1. **No horizontal flip**, anywhere, ever — chest X-rays have real
   anatomical laterality (heart/aorta position), and mirroring an X-ray is
   not a label-preserving augmentation the way it is for natural images.
2. **Aspect-ratio-preserving resize + pad**, not a naive stretch-to-square
   resize — `ResizeAndPad` resizes the longer edge to `size` and pads the
   shorter edge with zeros (letterboxing), so lung geometry isn't distorted.
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class ResizeAndPad:
    """Resize preserving aspect ratio (longer edge -> `size`), then pad the
    shorter edge with `fill` to produce a `size` x `size` image.
    """

    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = TF.resize(img, [new_h, new_w])

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        # left, top, right, bottom — split padding as evenly as possible.
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        return TF.pad(img, padding, fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, fill={self.fill})"


class ReplicateChannels:
    """Repeat a single-channel tensor to `num_channels` channels.

    Used only when a model expects 3-channel input (e.g. an
    ImageNet-pretrained backbone in the transfer-learning experiments).
    The from-scratch baseline CNN is intended to consume the native
    1-channel tensor — see data/README.md §5 for why this replication,
    rather than any color-izing scheme, is the right (and standard)
    approach for grayscale medical images fed to RGB-pretrained networks.
    """

    def __init__(self, num_channels: int = 3):
        self.num_channels = num_channels

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] == self.num_channels:
            return tensor
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected a 1-channel tensor, got shape {tuple(tensor.shape)}")
        return tensor.repeat(self.num_channels, 1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_channels={self.num_channels})"


def build_transforms(
    image_size: int,
    train: bool,
    mean: list[float],
    std: list[float],
    num_channels: int = 1,
) -> T.Compose:
    """Build the preprocessing pipeline.

    `train=True` adds mild, laterality-safe augmentation (rotation, a small
    zoom via affine scale, brightness/contrast jitter). `train=False`
    (used for both validation and test) is **strictly deterministic** — no
    random operations at all, so the same image always produces the same
    tensor.

    `mean`/`std` must have `num_channels` entries each (e.g. a single
    dataset-computed value for `num_channels=1`, or that same value
    repeated three times — not ImageNet stats — when replicating to 3
    channels for a pretrained backbone; see data/README.md §5).
    """
    if len(mean) != num_channels or len(std) != num_channels:
        raise ValueError(
            f"mean/std must have {num_channels} entries each, got {len(mean)}/{len(std)}"
        )

    steps: list = [ResizeAndPad(image_size)]

    if train:
        steps += [
            T.RandomRotation(degrees=10, fill=0),
            T.RandomAffine(degrees=0, scale=(0.9, 1.1), fill=0),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]

    steps += [T.ToTensor()]

    if num_channels != 1:
        steps += [ReplicateChannels(num_channels)]

    steps += [T.Normalize(mean=mean, std=std)]

    return T.Compose(steps)
