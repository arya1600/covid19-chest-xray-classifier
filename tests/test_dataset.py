"""Unit tests for CXRDataset: label correctness, split membership,
grayscale handling, and clear (never silent) corrupt/missing-file errors.
"""

from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from covidcxr.data.dataset import CXRDataset, CorruptImageError, LABEL_TO_INDEX, load_image
from covidcxr.data.splitting import verify_no_patient_leakage


def _make_image(path, size=(20, 20), mode="L", color=128):
    Image.new(mode, size, color=color).save(path)


def _synthetic_manifest(tmp_path) -> pd.DataFrame:
    rows = []
    # 2 patients per split, 1 image each, mixed labels.
    for split, patient, label in [
        ("train", "p1", "COVID"), ("train", "p2", "NORMAL"),
        ("val", "p3", "COVID"), ("val", "p4", "NORMAL"),
        ("test", "p5", "COVID"), ("test", "p6", "NORMAL"),
    ]:
        filename = f"{patient}.png"
        path = tmp_path / filename
        _make_image(path)
        rows.append({
            "filename": filename, "filepath": str(path), "label": label,
            "patient_id": patient, "patient_id_reliable": True,
            "source_dataset": "synthetic", "original_split": "", "view": "",
            "modality": "X-ray", "date": "", "license": "", "split": split,
        })
    return pd.DataFrame(rows)


def test_dataset_only_contains_requested_split(tmp_path):
    df = _synthetic_manifest(tmp_path)
    train_ds = CXRDataset(df, split="train")
    assert len(train_ds) == 2
    assert set(train_ds.df["split"]) == {"train"}


def test_dataset_labels_are_correctly_encoded(tmp_path):
    df = _synthetic_manifest(tmp_path)
    ds = CXRDataset(df, split="train")
    for i in range(len(ds)):
        _, label = ds[i]
        expected = LABEL_TO_INDEX[ds.df.iloc[i]["label"]]
        assert label == expected
    assert set(ds.labels) <= {0, 1}


def test_no_split_appears_in_dataset_for_a_different_split_value(tmp_path):
    df = _synthetic_manifest(tmp_path)
    val_ds = CXRDataset(df, split="val")
    assert all(df.loc[df["split"] == "val", "patient_id"].isin(val_ds.patient_ids))
    assert not any(pid in val_ds.patient_ids for pid in ["p1", "p2", "p5", "p6"])


def test_full_manifest_has_no_cross_split_patient_leakage(tmp_path):
    df = _synthetic_manifest(tmp_path)
    verify_no_patient_leakage(df)  # must not raise


def test_missing_image_file_raises_corrupt_image_error(tmp_path):
    df = _synthetic_manifest(tmp_path)
    df.loc[df["patient_id"] == "p1", "filepath"] = str(tmp_path / "does_not_exist.png")
    ds = CXRDataset(df, split="train")
    with pytest.raises(CorruptImageError):
        _ = ds[[i for i in range(len(ds)) if ds.df.iloc[i]["patient_id"] == "p1"][0]]


def test_truncated_image_file_raises_corrupt_image_error(tmp_path):
    bad_path = tmp_path / "truncated.png"
    bad_path.write_bytes(b"not a real png")
    with pytest.raises(CorruptImageError):
        load_image(bad_path)


def test_rgb_source_image_is_converted_to_single_channel_grayscale(tmp_path):
    path = tmp_path / "rgb.png"
    _make_image(path, mode="RGB", color=(50, 50, 50))
    img = load_image(path)
    assert img.mode == "L"


def test_unknown_label_in_manifest_raises(tmp_path):
    df = _synthetic_manifest(tmp_path)
    df.loc[0, "label"] = "UNKNOWN"
    with pytest.raises(ValueError):
        CXRDataset(df, split="train")


def test_invalid_split_name_raises(tmp_path):
    df = _synthetic_manifest(tmp_path)
    with pytest.raises(ValueError):
        CXRDataset(df, split="not_a_real_split")
