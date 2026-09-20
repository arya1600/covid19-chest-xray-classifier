"""Unit tests for GradCAM: shape/value-range correctness on a tiny
synthetic model, not a claim about what any real model attends to."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from covidcxr.explain.gradcam import GradCAM, overlay_heatmap
from covidcxr.models.baseline_cnn import BaselineCNN


def test_cam_output_shape_and_range():
    model = BaselineCNN(channels=(8, 16))
    cam_tool = GradCAM(model, target_layer=model.features)
    x = torch.randn(1, 1, 64, 64, requires_grad=True)

    cam, target_class = cam_tool.generate(x)

    assert cam.ndim == 2
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6
    assert target_class in (0, 1)


def test_cam_respects_explicit_target_class():
    model = BaselineCNN(channels=(8, 16))
    cam_tool = GradCAM(model, target_layer=model.features)
    x = torch.randn(1, 1, 64, 64)

    _, target_class = cam_tool.generate(x, target_class=0)
    assert target_class == 0


def test_overlay_heatmap_returns_same_size_rgb_image():
    base = Image.new("L", (50, 40), color=100)
    cam = np.random.rand(7, 7).astype(np.float32)

    overlay = overlay_heatmap(base, cam, alpha=0.5)

    assert overlay.size == (50, 40)
    assert overlay.mode == "RGB"
