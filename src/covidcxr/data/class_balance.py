"""Configurable class-imbalance handling.

Two independent mechanisms are implemented, and a project using this module
chooses at most one at a time via a plain string ("none" | "loss_weight" |
"sampler") — this file does not decide the strategy, it only provides the
building blocks. Per the Phase 3 brief: no automatic oversampling or image
duplication is implemented anywhere here.
"""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler

NUM_CLASSES = 2  # NORMAL, COVID — see covidcxr.data.dataset.LABEL_TO_INDEX


def compute_class_weights(labels: list[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights, suitable for
    `torch.nn.CrossEntropyLoss(weight=...)`. Class c's weight is
    `N / (num_classes * count[c])`, so a perfectly balanced dataset gets
    weights of 1.0 for every class.
    """
    counts = Counter(labels)
    missing = set(range(num_classes)) - set(counts)
    if missing:
        raise ValueError(f"labels contain no examples of class index(es) {missing}; cannot weight them")

    n = len(labels)
    weights = [n / (num_classes * counts[c]) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: list[int], num_classes: int = NUM_CLASSES) -> WeightedRandomSampler:
    """Per-sample weights (inverse class frequency) for a
    `WeightedRandomSampler`, so minibatches are drawn roughly
    class-balanced without duplicating or discarding any image on disk —
    sampling is with replacement, `num_samples=len(labels)` per epoch.
    """
    class_weights = compute_class_weights(labels, num_classes=num_classes)
    sample_weights = [class_weights[label].item() for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
