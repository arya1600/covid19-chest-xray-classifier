"""Unit tests for patient-grouped splitting — the direct fix for the
train/val patient leakage found in the reference project.
"""

from __future__ import annotations

import pandas as pd
import pytest

from covidcxr.data.splitting import patient_grouped_split, verify_no_patient_leakage


def _synthetic_manifest(n_patients: int = 60, images_per_patient: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_patients):
        label = "COVID" if i % 2 == 0 else "NORMAL"
        for j in range(images_per_patient):
            rows.append({"patient_id": f"patient-{i}", "label": label, "filename": f"p{i}-{j}.jpg"})
    return pd.DataFrame(rows)


def test_no_patient_appears_in_more_than_one_split():
    df = _synthetic_manifest()
    split_df = patient_grouped_split(df, seed=0)
    verify_no_patient_leakage(split_df)  # must not raise


def test_verify_no_patient_leakage_catches_a_real_leak():
    df = _synthetic_manifest(n_patients=4, images_per_patient=2)
    split_df = patient_grouped_split(df, seed=0)
    # Deliberately corrupt one patient's rows to straddle two splits.
    leaking_patient = split_df["patient_id"].iloc[0]
    idx = split_df.index[split_df["patient_id"] == leaking_patient]
    split_df.loc[idx[0], "split"] = "train"
    split_df.loc[idx[-1], "split"] = "test"

    with pytest.raises(AssertionError):
        verify_no_patient_leakage(split_df)


def test_every_row_assigned_a_split():
    df = _synthetic_manifest()
    split_df = patient_grouped_split(df, seed=0)
    assert (split_df["split"] != "").all()
    assert set(split_df["split"].unique()) <= {"train", "val", "test"}


def test_split_proportions_are_approximately_correct():
    df = _synthetic_manifest(n_patients=200, images_per_patient=1)
    split_df = patient_grouped_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=0)
    fractions = split_df["split"].value_counts(normalize=True)
    assert fractions["train"] == pytest.approx(0.70, abs=0.07)
    assert fractions["val"] == pytest.approx(0.15, abs=0.07)
    assert fractions["test"] == pytest.approx(0.15, abs=0.07)


def test_split_is_reproducible_given_same_seed():
    df = _synthetic_manifest()
    split_a = patient_grouped_split(df, seed=123)
    split_b = patient_grouped_split(df, seed=123)
    pd.testing.assert_series_equal(split_a["split"], split_b["split"])


def test_invalid_fractions_raise():
    df = _synthetic_manifest(n_patients=4)
    with pytest.raises(ValueError):
        patient_grouped_split(df, train_frac=0.5, val_frac=0.3, test_frac=0.3)
