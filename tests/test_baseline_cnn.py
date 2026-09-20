"""Unit tests for the baseline CNN: shape correctness and parameter budget."""

from __future__ import annotations

import torch

from covidcxr.models.baseline_cnn import BaselineCNN


def test_forward_pass_output_shape():
    model = BaselineCNN(in_channels=1, num_classes=2)
    x = torch.randn(4, 1, 224, 224)
    out = model(x)
    assert out.shape == (4, 2)


def test_accepts_three_channel_input_for_pretrained_style_comparison():
    model = BaselineCNN(in_channels=3, num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)


def test_parameter_count_is_small_relative_to_reference_project():
    # Reference project's Flatten->Dense(64) head alone was ~5.5M params
    # out of ~5.67M total. Global-average-pooling should keep this whole
    # model well under 1M.
    model = BaselineCNN()
    assert model.num_parameters() < 1_000_000


def test_output_is_finite_and_gradients_flow():
    model = BaselineCNN()
    x = torch.randn(2, 1, 224, 224, requires_grad=False)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(out).all()
    assert model.classifier.weight.grad is not None
