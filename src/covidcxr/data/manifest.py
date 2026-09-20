"""Build a unified image manifest (filename, label, patient_id, source, ...)
from the two upstream sources.

Patient-ID recovery is the load-bearing step here: the reference project's
bundled data made grouped splitting impossible because it dropped the
Cohen metadata.csv entirely. This module recovers a `patient_id` (or
`group_id`, for sources where "patient" isn't quite the right word) for
every image so a later, strictly grouped split (see `splitting.py`) can
guarantee no patient's/group's images appear in more than one of
train/val/test.

The Kermany/Normal-class parser targets the real Mendeley "CellData"
distribution (`chest_xray/{train,test}/{NORMAL,PNEUMONIA}`), whose
filenames follow `NORMAL-<patientid>-<imagenum>.jpeg` — verified against
the actual downloaded files, not assumed. This is a different, and more
informative, filename convention than the repackaged Kaggle mirror's
`IM-<id>-<num>.jpeg`, which is kept as a fallback pattern for robustness
in case a differently-packaged copy is ever used instead.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = [
    "filename",
    "filepath",
    "label",
    "patient_id",
    "patient_id_reliable",
    "source_dataset",
    "original_split",
    "view",
    "modality",
    "date",
    "license",
]

# Chosen acquisition criteria for the Cohen source (see data/README.md §1):
# COVID-19 finding, frontal chest X-ray, and a recorded (non-empty) license.
FRONTAL_VIEWS = {"PA", "AP", "AP Supine"}

# Patient/session-id patterns tried in order, first match wins.
# 1) Real Mendeley CellData convention, verified against the actual
#    downloaded files: "NORMAL-3780378-0001.jpeg" -> patient id 3780378.
# 2) Repackaged-Kaggle-mirror convention (kept as a fallback, not currently
#    the acquisition path used): "IM-0119-0001.jpeg" -> session id 0119.
KERMANY_ID_PATTERNS = [
    re.compile(r"^NORMAL-(\d+)-\d+", re.IGNORECASE),
    re.compile(r"IM-(\d+)"),
]

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}

# Only these class subfolders are ever read from the Kermany source; this
# project uses NORMAL only (per data/README.md §2) and never touches
# PNEUMONIA, and CellData also ships an unrelated OCT dataset that must not
# be swept in by an overly broad directory scan.
KERMANY_SPLITS = ("train", "test")
KERMANY_CLASS_DIR = "NORMAL"


@dataclass
class ManifestBuildStats:
    cohen_total_rows: int = 0
    cohen_selected_rows: int = 0
    cohen_excluded_no_license: int = 0
    kermany_total_files: int = 0
    kermany_unreliable_patient_id: int = 0
    kermany_duplicate_content_files: int = 0
    kermany_duplicate_content_pairs: list[tuple[str, str]] = field(default_factory=list)


def parse_cohen_metadata(metadata_csv: Path, images_dir: Path | None = None) -> tuple[list[dict], ManifestBuildStats]:
    """Read the Cohen covid-chestxray-dataset metadata.csv and return manifest
    rows for the subset this project uses (see module docstring for criteria).

    `images_dir`, if given, is recorded in `filepath`; it is not required to
    exist yet — this function only reads metadata, not image files.
    """
    stats = ManifestBuildStats()
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stats.cohen_total_rows = len(rows)

    out = []
    for row in rows:
        is_covid_frontal = (
            "covid-19" in row["finding"].lower()
            and row["modality"] == "X-ray"
            and row["view"] in FRONTAL_VIEWS
        )
        if not is_covid_frontal:
            continue
        if row["license"].strip() == "":
            stats.cohen_excluded_no_license += 1
            continue

        filename = row["filename"]
        filepath = str(images_dir / filename) if images_dir else filename
        out.append(
            {
                "filename": filename,
                "filepath": filepath,
                "label": "COVID",
                "patient_id": f"cohen-{row['patientid']}",
                "patient_id_reliable": True,
                "source_dataset": "cohen",
                "original_split": "",  # Cohen ships as a flat collection, no train/test split.
                "view": row["view"],
                "modality": row["modality"],
                "date": row["date"],
                "license": row["license"],
            }
        )
    stats.cohen_selected_rows = len(out)
    return out, stats


def _kermany_patient_id(filename: str) -> tuple[str, bool]:
    for pattern in KERMANY_ID_PATTERNS:
        match = pattern.search(filename)
        if match:
            return f"kermany-{match.group(1)}", True
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
    return f"kermany-unknown-{digest}", False


def parse_kermany_directory(chest_xray_root: Path) -> tuple[list[dict], ManifestBuildStats]:
    """Scan the real Mendeley CellData layout for NORMAL-class images only.

    Expects `chest_xray_root` to contain `train/NORMAL` and `test/NORMAL`
    (the actual downloaded structure). PNEUMONIA and any OCT sibling
    directories are never read. Byte-identical duplicate files are
    detected via content hash and reported in stats, not silently dropped
    or silently kept — the caller decides what to do with them.
    """
    stats = ManifestBuildStats()
    out = []
    seen_hashes: dict[str, str] = {}  # content hash -> first filepath seen

    for original_split in KERMANY_SPLITS:
        class_dir = chest_xray_root / original_split / KERMANY_CLASS_DIR
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stats.kermany_total_files += 1

            content_hash = hashlib.md5(path.read_bytes()).hexdigest()
            if content_hash in seen_hashes:
                stats.kermany_duplicate_content_files += 1
                stats.kermany_duplicate_content_pairs.append((seen_hashes[content_hash], str(path)))
            else:
                seen_hashes[content_hash] = str(path)

            patient_id, reliable = _kermany_patient_id(path.name)
            if not reliable:
                stats.kermany_unreliable_patient_id += 1
            out.append(
                {
                    "filename": path.name,
                    "filepath": str(path),
                    "label": "NORMAL",
                    "patient_id": patient_id,
                    "patient_id_reliable": reliable,
                    "source_dataset": "kermany",
                    "original_split": original_split,
                    "view": "",
                    "modality": "X-ray",
                    "date": "",
                    "license": "CC BY 4.0",
                }
            )
    return out, stats


def build_manifest(
    cohen_metadata_csv: Path | None,
    cohen_images_dir: Path | None,
    kermany_chest_xray_dir: Path | None,
) -> tuple[pd.DataFrame, ManifestBuildStats]:
    """Combine both sources into one manifest DataFrame. Either source may be
    omitted (None) to build a partial manifest, e.g. while the other source
    is still being acquired.
    """
    rows: list[dict] = []
    stats = ManifestBuildStats()

    if cohen_metadata_csv is not None:
        cohen_rows, cohen_stats = parse_cohen_metadata(cohen_metadata_csv, cohen_images_dir)
        rows.extend(cohen_rows)
        stats.cohen_total_rows = cohen_stats.cohen_total_rows
        stats.cohen_selected_rows = cohen_stats.cohen_selected_rows
        stats.cohen_excluded_no_license = cohen_stats.cohen_excluded_no_license

    if kermany_chest_xray_dir is not None:
        kermany_rows, kermany_stats = parse_kermany_directory(kermany_chest_xray_dir)
        rows.extend(kermany_rows)
        stats.kermany_total_files = kermany_stats.kermany_total_files
        stats.kermany_unreliable_patient_id = kermany_stats.kermany_unreliable_patient_id
        stats.kermany_duplicate_content_files = kermany_stats.kermany_duplicate_content_files
        stats.kermany_duplicate_content_pairs = kermany_stats.kermany_duplicate_content_pairs

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return df, stats
