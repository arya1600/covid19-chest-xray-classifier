"""Unit tests for manifest building and patient-ID recovery.

Uses small synthetic fixtures (a hand-written metadata.csv and a handful
of empty placeholder image files), not the real downloaded datasets, so
these run instantly and don't depend on network access or Kaggle
credentials.
"""

from __future__ import annotations

import csv

import pytest

from covidcxr.data.manifest import build_manifest, parse_cohen_metadata, parse_kermany_directory

COHEN_FIELDS = [
    "patientid", "offset", "sex", "age", "finding", "RT_PCR_positive", "survival",
    "intubated", "intubation_present", "went_icu", "in_icu", "needed_supplemental_O2",
    "extubated", "temperature", "pO2_saturation", "leukocyte_count", "neutrophil_count",
    "lymphocyte_count", "view", "modality", "date", "location", "folder", "filename",
    "doi", "url", "license", "clinical_notes", "other_notes",
]


def _cohen_row(**overrides) -> dict:
    row = {field: "" for field in COHEN_FIELDS}
    row.update(
        patientid="1", finding="Pneumonia/Viral/COVID-19", view="PA",
        modality="X-ray", date="2020-01-01", folder="images",
        filename="patient1-a.jpg", license="CC BY 4.0",
    )
    row.update(overrides)
    return row


def _write_cohen_metadata(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COHEN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_cohen_same_patient_two_images_get_same_patient_id(tmp_path):
    """Regression test for the exact leakage pattern found in the reference
    project: two images of the same patient (like auntminnie-a/-b) must map
    to the same patient_id so a grouped split keeps them together.
    """
    rows = [
        _cohen_row(patientid="7", filename="case7-a.jpg"),
        _cohen_row(patientid="7", filename="case7-b.jpg"),
    ]
    metadata_csv = tmp_path / "metadata.csv"
    _write_cohen_metadata(metadata_csv, rows)

    manifest_rows, stats = parse_cohen_metadata(metadata_csv)

    assert stats.cohen_selected_rows == 2
    patient_ids = {r["patient_id"] for r in manifest_rows}
    assert patient_ids == {"cohen-7"}


def test_cohen_excludes_missing_license(tmp_path):
    rows = [
        _cohen_row(patientid="1", filename="licensed.jpg", license="CC BY 4.0"),
        _cohen_row(patientid="2", filename="unlicensed.jpg", license=""),
    ]
    metadata_csv = tmp_path / "metadata.csv"
    _write_cohen_metadata(metadata_csv, rows)

    manifest_rows, stats = parse_cohen_metadata(metadata_csv)

    assert stats.cohen_total_rows == 2
    assert stats.cohen_selected_rows == 1
    assert stats.cohen_excluded_no_license == 1
    assert manifest_rows[0]["filename"] == "licensed.jpg"


def test_cohen_excludes_non_frontal_and_non_covid(tmp_path):
    rows = [
        _cohen_row(finding="Pneumonia/Viral/COVID-19", view="L"),  # lateral, excluded
        _cohen_row(finding="Pneumonia/Bacterial", view="PA"),  # not COVID, excluded
        _cohen_row(finding="Pneumonia/Viral/COVID-19", view="PA"),  # kept
    ]
    metadata_csv = tmp_path / "metadata.csv"
    _write_cohen_metadata(metadata_csv, rows)

    manifest_rows, stats = parse_cohen_metadata(metadata_csv)

    assert stats.cohen_selected_rows == 1


def _make_kermany_tree(root, files_by_split_class: dict[tuple[str, str], list[str]]) -> None:
    """files_by_split_class: {("train","NORMAL"): ["NORMAL-1-0001.jpeg", ...], ...}"""
    for (split, klass), filenames in files_by_split_class.items():
        class_dir = root / split / klass
        class_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            (class_dir / name).write_bytes(name.encode("utf-8"))  # distinct content per name


def test_kermany_patient_id_extracted_from_real_celldata_filename_pattern(tmp_path):
    """Regression test using the *actual* Mendeley CellData filename
    convention verified against the real downloaded dataset:
    "NORMAL-<patientid>-<imagenum>.jpeg". A prior version of this parser
    assumed a different (Kaggle-mirror) convention that does not match
    these real files at all.
    """
    _make_kermany_tree(tmp_path, {
        ("train", "NORMAL"): ["NORMAL-1064313-0001.jpeg", "NORMAL-1064313-0002.jpeg"],
        ("test", "NORMAL"): ["NORMAL-9999999-0001.jpeg"],
    })

    manifest_rows, stats = parse_kermany_directory(tmp_path)
    manifest_rows.sort(key=lambda r: r["filename"])

    assert stats.kermany_total_files == 3
    assert stats.kermany_unreliable_patient_id == 0
    ids_by_filename = {r["filename"]: r["patient_id"] for r in manifest_rows}
    assert ids_by_filename["NORMAL-1064313-0001.jpeg"] == ids_by_filename["NORMAL-1064313-0002.jpeg"]
    assert ids_by_filename["NORMAL-1064313-0001.jpeg"] == "kermany-1064313"
    assert ids_by_filename["NORMAL-9999999-0001.jpeg"] == "kermany-9999999"


def test_kermany_records_original_split(tmp_path):
    _make_kermany_tree(tmp_path, {
        ("train", "NORMAL"): ["NORMAL-1-0001.jpeg"],
        ("test", "NORMAL"): ["NORMAL-2-0001.jpeg"],
    })

    manifest_rows, _ = parse_kermany_directory(tmp_path)
    split_by_filename = {r["filename"]: r["original_split"] for r in manifest_rows}

    assert split_by_filename["NORMAL-1-0001.jpeg"] == "train"
    assert split_by_filename["NORMAL-2-0001.jpeg"] == "test"


def test_kermany_never_reads_pneumonia_class(tmp_path):
    _make_kermany_tree(tmp_path, {
        ("train", "NORMAL"): ["NORMAL-1-0001.jpeg"],
        ("train", "PNEUMONIA"): ["BACTERIA-2-0001.jpeg"],
    })

    manifest_rows, stats = parse_kermany_directory(tmp_path)

    assert stats.kermany_total_files == 1
    assert all(r["label"] == "NORMAL" for r in manifest_rows)
    assert all("BACTERIA" not in r["filename"] for r in manifest_rows)


def test_kermany_unparseable_filename_flagged_unreliable(tmp_path):
    _make_kermany_tree(tmp_path, {("train", "NORMAL"): ["totally_different_naming.jpeg"]})

    manifest_rows, stats = parse_kermany_directory(tmp_path)

    assert stats.kermany_unreliable_patient_id == 1
    assert manifest_rows[0]["patient_id_reliable"] is False
    assert manifest_rows[0]["patient_id"].startswith("kermany-unknown-")


def test_kermany_detects_duplicate_content(tmp_path):
    _make_kermany_tree(tmp_path, {("train", "NORMAL"): ["NORMAL-1-0001.jpeg"]})
    # Same content, different filename/patient id -> byte-identical duplicate.
    (tmp_path / "train" / "NORMAL" / "NORMAL-1-0002.jpeg").write_bytes(b"NORMAL-1-0001.jpeg")

    _, stats = parse_kermany_directory(tmp_path)

    assert stats.kermany_duplicate_content_files == 1
    assert len(stats.kermany_duplicate_content_pairs) == 1


def test_build_manifest_combines_both_sources(tmp_path):
    metadata_csv = tmp_path / "metadata.csv"
    _write_cohen_metadata(metadata_csv, [_cohen_row(patientid="1", filename="c1.jpg")])
    kermany_dir = tmp_path / "kermany"
    _make_kermany_tree(kermany_dir, {("train", "NORMAL"): ["NORMAL-1-0001.jpeg"]})

    df, stats = build_manifest(metadata_csv, None, kermany_dir)

    assert len(df) == 2
    assert set(df["label"]) == {"COVID", "NORMAL"}
    assert set(df["source_dataset"]) == {"cohen", "kermany"}
