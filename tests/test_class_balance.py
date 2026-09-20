"""Unit tests for class-imbalance utilities: weighted loss and sampler."""

from __future__ import annotations

import pytest
import torch

from covidcxr.data.class_balance import compute_class_weights, make_weighted_sampler


def test_balanced_labels_get_equal_weights():
    labels = [0, 1, 0, 1]
    weights = compute_class_weights(labels)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]))


def test_imbalanced_labels_get_inverse_frequency_weights():
    labels = [0] * 90 + [1] * 10  # 90 NORMAL, 10 COVID
    weights = compute_class_weights(labels)
    # class 0: 100/(2*90) = 0.5556 ; class 1: 100/(2*10) = 5.0
    assert weights[0].item() == pytest.approx(100 / 180, rel=1e-4)
    assert weights[1].item() == pytest.approx(5.0, rel=1e-4)
    assert weights[1] > weights[0]  # the rarer class gets the larger weight


def test_missing_class_raises():
    labels = [0, 0, 0]  # no class 1 at all
    with pytest.raises(ValueError):
        compute_class_weights(labels)


def test_weighted_sampler_has_correct_length():
    labels = [0] * 90 + [1] * 10
    sampler = make_weighted_sampler(labels)
    assert len(sampler) == len(labels)


def test_weighted_sampler_oversamples_minority_class_on_average():
    labels = [0] * 90 + [1] * 10
    sampler = make_weighted_sampler(labels)
    torch.manual_seed(0)
    drawn = list(sampler)
    minority_fraction = sum(1 for i in drawn if labels[i] == 1) / len(drawn)
    # With correct inverse-frequency weights, minority draws should be
    # roughly balanced (~50%), nowhere near its true 10% base rate.
    assert minority_fraction > 0.3
