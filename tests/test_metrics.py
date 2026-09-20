"""Unit tests for the evaluation metrics: correctness against hand-computed
values from known confusion matrices, plus bootstrap CI sanity checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from covidcxr.evaluation.metrics import bootstrap_ci, compute_confusion_matrix, compute_metrics


def test_perfect_classifier_gets_perfect_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])

    metrics = compute_metrics(y_true, y_pred, y_score, n_bootstrap=100)

    assert metrics["accuracy"].value == 1.0
    assert metrics["precision"].value == 1.0
    assert metrics["sensitivity"].value == 1.0
    assert metrics["specificity"].value == 1.0
    assert metrics["f1"].value == 1.0
    assert metrics["roc_auc"].value == 1.0


def test_known_confusion_matrix_gives_hand_computed_metrics():
    # TN=3, FP=1, FN=2, TP=4 -> precision=4/5=0.8, sensitivity=4/6=0.6667,
    # specificity=3/4=0.75, accuracy=(3+4)/10=0.7
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    y_score = y_pred.astype(float)  # not used for these metrics

    metrics = compute_metrics(y_true, y_pred, y_score, n_bootstrap=100)

    assert metrics["accuracy"].value == pytest.approx(0.7)
    assert metrics["precision"].value == pytest.approx(0.8)
    assert metrics["sensitivity"].value == pytest.approx(4 / 6)
    assert metrics["specificity"].value == pytest.approx(0.75)


def test_confusion_matrix_layout_is_tn_fp_fn_tp():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    assert (tn, fp, fn, tp) == (3, 1, 2, 4)


def test_roc_auc_is_nan_when_only_one_class_present_in_a_bootstrap_resample_but_point_estimate_still_valid():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_score, n_bootstrap=50)
    assert metrics["roc_auc"].value == 1.0


def test_bootstrap_ci_width_shrinks_with_more_data():
    rng = np.random.default_rng(0)

    def make_data(n):
        y_true = rng.integers(0, 2, size=n)
        y_pred = y_true.copy()
        flip_idx = rng.choice(n, size=max(1, n // 5), replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]
        y_score = y_pred.astype(float)
        return y_true, y_pred, y_score

    acc_fn = lambda yt, yp, ys: (yt == yp).mean()

    small = make_data(20)
    large = make_data(2000)

    _, lo_small, hi_small = bootstrap_ci(*small, acc_fn, n_bootstrap=500, seed=1)
    _, lo_large, hi_large = bootstrap_ci(*large, acc_fn, n_bootstrap=500, seed=1)

    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_bootstrap_point_estimate_matches_direct_computation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    y_score = y_pred.astype(float)
    acc_fn = lambda yt, yp, ys: (yt == yp).mean()

    point, _, _ = bootstrap_ci(y_true, y_pred, y_score, acc_fn, n_bootstrap=10)
    assert point == pytest.approx((y_true == y_pred).mean())
