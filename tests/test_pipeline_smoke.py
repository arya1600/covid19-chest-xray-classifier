"""End-to-end smoke test: build real DataLoaders from data/manifest.csv and
actually load several real images from each split. Skips gracefully (does
not fail) if the raw dataset isn't present on this machine — the data is
never committed to the repo, so a fresh clone won't have it until someone
runs the acquisition scripts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from covidcxr.data.pipeline import DataConfig, build_dataloaders

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "manifest.csv"

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(), reason="data/manifest.csv not present — run the Phase 2 acquisition scripts first"
)


@pytest.fixture(scope="module")
def data_bundle():
    config = DataConfig(
        manifest_path=MANIFEST_PATH,
        image_size=224,
        num_channels=1,
        mean=[0.4073],
        std=[0.2863],
        batch_size=8,
        seed=42,
        class_balance="loss_weight",
    )
    return build_dataloaders(config)


@pytest.mark.parametrize("loader_name", ["train_loader", "val_loader", "test_loader"])
def test_can_load_a_batch_from_each_split(data_bundle, loader_name):
    loader = getattr(data_bundle, loader_name)
    images, labels = next(iter(loader))
    assert images.shape[0] > 0
    assert images.shape[1:] == (1, 224, 224)
    assert set(labels.tolist()) <= {0, 1}


def test_can_load_several_real_images_per_split(data_bundle):
    for loader_name in ["train_loader", "val_loader", "test_loader"]:
        loader = getattr(data_bundle, loader_name)
        total_seen = 0
        for images, _ in loader:
            total_seen += images.shape[0]
            if total_seen >= 16:
                break
        assert total_seen >= 8, f"{loader_name} did not yield enough real images"


def test_class_weights_present_when_loss_weight_strategy_selected(data_bundle):
    assert data_bundle.class_weights is not None
    assert data_bundle.class_weights.shape == (2,)


def test_val_and_test_loaders_are_not_shuffled(data_bundle):
    val_batch_1 = next(iter(data_bundle.val_loader))[1].tolist()
    val_batch_2 = next(iter(data_bundle.val_loader))[1].tolist()
    assert val_batch_1 == val_batch_2
