"""Wire manifest -> Dataset -> transforms -> DataLoader, config-driven.

This is the single place that assembles the full data pipeline, so a
training script never hand-builds transforms or DataLoaders itself — it
just calls `build_dataloaders(...)` with a config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from covidcxr.data.class_balance import compute_class_weights, make_weighted_sampler
from covidcxr.data.dataset import CXRDataset
from covidcxr.data.transforms import build_transforms
from covidcxr.utils.seed import make_generator, seed_everything, worker_init_fn

CLASS_BALANCE_STRATEGIES = ("none", "loss_weight", "sampler")


@dataclass
class DataConfig:
    manifest_path: str | Path
    image_size: int = 224
    num_channels: int = 1
    mean: list[float] = field(default_factory=lambda: [0.5])
    std: list[float] = field(default_factory=lambda: [0.5])
    batch_size: int = 32
    seed: int = 42
    num_workers: int = 0
    class_balance: str = "none"  # "none" | "loss_weight" | "sampler"
    root_dir: str | Path | None = None
    label_to_index: dict[str, int] | None = None  # override for non-COVID/NORMAL tasks (e.g. source-confound control)

    def __post_init__(self):
        if self.class_balance not in CLASS_BALANCE_STRATEGIES:
            raise ValueError(f"class_balance must be one of {CLASS_BALANCE_STRATEGIES}, got {self.class_balance!r}")


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_weights: torch.Tensor | None  # non-None only when class_balance == "loss_weight"


def build_dataloaders(config: DataConfig) -> DataBundle:
    seed_everything(config.seed)

    train_transform = build_transforms(
        config.image_size, train=True, mean=config.mean, std=config.std, num_channels=config.num_channels
    )
    eval_transform = build_transforms(
        config.image_size, train=False, mean=config.mean, std=config.std, num_channels=config.num_channels
    )

    train_ds = CXRDataset(config.manifest_path, split="train", transform=train_transform,
                           root_dir=config.root_dir, label_to_index=config.label_to_index)
    val_ds = CXRDataset(config.manifest_path, split="val", transform=eval_transform,
                         root_dir=config.root_dir, label_to_index=config.label_to_index)
    test_ds = CXRDataset(config.manifest_path, split="test", transform=eval_transform,
                          root_dir=config.root_dir, label_to_index=config.label_to_index)

    class_weights = None
    sampler = None
    shuffle_train = True
    if config.class_balance == "loss_weight":
        class_weights = compute_class_weights(train_ds.labels)
    elif config.class_balance == "sampler":
        sampler = make_weighted_sampler(train_ds.labels)
        shuffle_train = False  # PyTorch forbids shuffle=True together with a sampler

    common_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle_train,
        sampler=sampler,
        generator=make_generator(config.seed),
        **common_kwargs,
    )
    # Validation/test are never shuffled and never sampled — deterministic
    # order, every image seen exactly once per epoch.
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

    return DataBundle(train_loader, val_loader, test_loader, class_weights)
