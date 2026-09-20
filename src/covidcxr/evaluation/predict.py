"""Collect model predictions over a DataLoader, once, for evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from covidcxr.evaluation.metrics import POSITIVE_LABEL


@torch.no_grad()
def get_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Returns (y_true, y_pred, y_score) as numpy arrays, in loader order.
    y_score is the predicted probability of the positive (COVID) class.
    """
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        y_score.extend(probs[:, POSITIVE_LABEL].cpu().tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_score)
