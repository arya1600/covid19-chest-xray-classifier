"""Reproducibility helpers: one seed, applied everywhere randomness happens."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_generator(seed: int) -> torch.Generator:
    """A seeded generator for DataLoader(generator=...), so shuffling order
    is reproducible independent of whatever seed_everything() last set
    globally.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker process a distinct, seed-derived numpy
    random state, so multi-worker loading doesn't correlate randomness
    across workers (relevant if workers ever use numpy-based augmentation).
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
