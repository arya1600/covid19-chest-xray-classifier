"""Unit tests for the transform pipeline: aspect-ratio preservation, strict
train/eval separation of randomness, no horizontal flip anywhere, and
correct grayscale-to-N-channel handling.
"""

from __future__ import annotations

import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from covidcxr.data.transforms import ReplicateChannels, ResizeAndPad, build_transforms


def test_resize_and_pad_preserves_aspect_ratio():
    # 100x50 (2:1) image resized to a 64x64 square: the longer edge (width)
    # should hit exactly 64, and the shorter edge should be scaled by the
    # same factor (32) then padded to 64 — not stretched to 64.
    img = Image.new("L", (100, 50), color=128)
    out = ResizeAndPad(64)(img)
    assert out.size == (64, 64)


def test_resize_and_pad_does_not_distort_content_region():
    img = Image.new("L", (100, 50), color=200)  # opaque content
    out = ResizeAndPad(64, fill=0)(img)
    # Content should occupy the middle 64x32 rows (scale factor 0.64), with
    # black padding above/below, not stretched to fill the full 64x64.
    px = out.load()
    assert px[32, 32] == 200  # center: content
    assert px[32, 2] == 0     # near top edge: padding
    assert px[32, 61] == 0    # near bottom edge: padding


def test_eval_transform_is_deterministic():
    img = Image.new("L", (80, 120), color=100)
    transform = build_transforms(image_size=64, train=False, mean=[0.5], std=[0.5])
    out1 = transform(img)
    out2 = transform(img)
    assert torch.equal(out1, out2)


def test_train_transform_pipeline_includes_stochastic_ops():
    transform = build_transforms(image_size=64, train=True, mean=[0.5], std=[0.5])
    op_types = [type(step) for step in transform.transforms]
    assert T.RandomRotation in op_types
    assert T.RandomAffine in op_types
    assert T.ColorJitter in op_types


def test_eval_transform_pipeline_has_no_stochastic_ops():
    transform = build_transforms(image_size=64, train=False, mean=[0.5], std=[0.5])
    op_types = [type(step) for step in transform.transforms]
    assert T.RandomRotation not in op_types
    assert T.RandomAffine not in op_types
    assert T.ColorJitter not in op_types


@pytest.mark.parametrize("train", [True, False])
def test_no_horizontal_flip_in_either_pipeline(train):
    transform = build_transforms(image_size=64, train=train, mean=[0.5], std=[0.5])
    op_types = [type(step) for step in transform.transforms]
    assert T.RandomHorizontalFlip not in op_types


def test_grayscale_image_produces_single_channel_tensor():
    img = Image.new("L", (64, 64), color=100)
    transform = build_transforms(image_size=64, train=False, mean=[0.5], std=[0.5], num_channels=1)
    out = transform(img)
    assert out.shape == (1, 64, 64)


def test_replicate_channels_repeats_grayscale_to_three_channels():
    img = Image.new("L", (64, 64), color=100)
    transform = build_transforms(image_size=64, train=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], num_channels=3)
    out = transform(img)
    assert out.shape == (3, 64, 64)
    assert torch.equal(out[0], out[1])
    assert torch.equal(out[1], out[2])


def test_replicate_channels_rejects_wrong_input_channels():
    bad_tensor = torch.zeros(2, 4, 4)
    with pytest.raises(ValueError):
        ReplicateChannels(3)(bad_tensor)


def test_mean_std_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_transforms(image_size=64, train=False, mean=[0.5, 0.5], std=[0.5], num_channels=1)
