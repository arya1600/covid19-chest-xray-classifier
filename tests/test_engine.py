"""Unit tests for the training engine, against synthetic in-memory tensors
(TensorDataset) so these run fast and don't depend on real image files.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from covidcxr.models.baseline_cnn import BaselineCNN
from covidcxr.training.engine import evaluate, fit, train_one_epoch


def _tiny_loader(n=16, num_classes=2, seed=0):
    torch.manual_seed(seed)
    images = torch.randn(n, 1, 32, 32)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_train_one_epoch_reduces_or_maintains_loss_over_several_calls():
    model = BaselineCNN(channels=(8, 16))
    loader = _tiny_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    first_loss, _ = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    for _ in range(5):
        last_loss, _ = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    assert last_loss < first_loss  # model should fit this tiny, easy synthetic task


def test_evaluate_does_not_update_model_parameters():
    model = BaselineCNN(channels=(8, 16))
    loader = _tiny_loader()
    criterion = nn.CrossEntropyLoss()
    params_before = [p.clone() for p in model.parameters()]

    evaluate(model, loader, criterion, torch.device("cpu"))

    for before, after in zip(params_before, model.parameters()):
        assert torch.equal(before, after)


def test_fit_saves_checkpoint_and_tracks_best_epoch(tmp_path):
    model = BaselineCNN(channels=(8, 16))
    train_loader = _tiny_loader(seed=0)
    val_loader = _tiny_loader(seed=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = tmp_path / "best.pt"

    result = fit(
        model, train_loader, val_loader, optimizer, criterion, torch.device("cpu"),
        max_epochs=5, patience=10, checkpoint_path=checkpoint_path,
    )

    assert len(result.history) == 5
    assert checkpoint_path.exists()
    assert result.best_epoch >= 1


def test_fit_stops_early_when_val_loss_never_improves(tmp_path):
    model = BaselineCNN(channels=(8, 16))
    train_loader = _tiny_loader(seed=0)
    val_loader = _tiny_loader(seed=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # no learning -> val loss won't improve
    criterion = nn.CrossEntropyLoss()

    result = fit(
        model, train_loader, val_loader, optimizer, criterion, torch.device("cpu"),
        max_epochs=20, patience=2, checkpoint_path=tmp_path / "best.pt",
    )

    assert result.stopped_early
    assert len(result.history) < 20
