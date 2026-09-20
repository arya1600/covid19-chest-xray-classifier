"""Training loop primitives: one epoch of training, one epoch of
evaluation, and a `fit()` that ties them together with early stopping and
best-checkpoint saving.

Deliberately minimal at this stage — no metrics beyond loss/accuracy here.
The full metric suite (precision/recall/specificity/F1/ROC-AUC/confusion
matrix + bootstrap CIs) is a separate, later concern
(`covidcxr.evaluation.metrics`), applied once to a finished model's
predictions on the untouched test set, not computed every epoch here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@dataclass
class FitResult:
    history: list[dict] = field(default_factory=list)  # one dict per epoch
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    stopped_early: bool = False


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    max_epochs: int,
    patience: int,
    checkpoint_path: Path | None = None,
) -> FitResult:
    """Train with early stopping on validation loss. Saves the best
    (lowest val-loss) model state_dict to `checkpoint_path`, if given.
    """
    result = FitResult()
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        result.history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        )

        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                result.stopped_early = True
                break

    return result
