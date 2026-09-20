"""One evaluation function, run once per finished model on the untouched
test set — never on validation, never per-epoch.

Includes bootstrap confidence intervals because this dataset's test split
is small (280 images, 31 COVID) — a bare point estimate like "94.3%
accuracy" implies more precision than a few dozen images can actually
support, and would be misleading on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_LABEL = 1  # COVID, per covidcxr.data.dataset.LABEL_TO_INDEX


@dataclass
class MetricResult:
    value: float
    ci_low: float
    ci_high: float


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


_METRIC_FUNCTIONS = {
    "accuracy": lambda yt, yp, ys: (yt == yp).mean(),
    "precision": lambda yt, yp, ys: precision_score(yt, yp, pos_label=POSITIVE_LABEL, zero_division=0),
    "sensitivity": lambda yt, yp, ys: recall_score(yt, yp, pos_label=POSITIVE_LABEL, zero_division=0),
    "specificity": lambda yt, yp, ys: _specificity(yt, yp),
    "f1": lambda yt, yp, ys: f1_score(yt, yp, pos_label=POSITIVE_LABEL, zero_division=0),
    "roc_auc": lambda yt, yp, ys: roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan"),
}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Point estimate + a percentile bootstrap CI, resampling test-set
    indices with replacement.
    """
    n = len(y_true)
    point = metric_fn(y_true, y_pred, y_score)

    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        if len(set(yt)) < 2:
            continue  # skip resamples that happen to contain only one class
        boot_values.append(metric_fn(yt, yp, ys))

    if not boot_values:
        return point, float("nan"), float("nan")

    lo = np.percentile(boot_values, 100 * alpha / 2)
    hi = np.percentile(boot_values, 100 * (1 - alpha / 2))
    return point, lo, hi


def compute_metrics(
    y_true, y_pred, y_score, n_bootstrap: int = 1000, seed: int = 42
) -> dict[str, MetricResult]:
    """y_true, y_pred: integer class labels (0=NORMAL, 1=COVID).
    y_score: predicted probability of the positive (COVID) class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    results = {}
    for name, fn in _METRIC_FUNCTIONS.items():
        value, lo, hi = bootstrap_ci(y_true, y_pred, y_score, fn, n_bootstrap=n_bootstrap, seed=seed)
        results[name] = MetricResult(value=value, ci_low=lo, ci_high=hi)
    return results


def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Returns [[TN, FP], [FN, TP]]."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
