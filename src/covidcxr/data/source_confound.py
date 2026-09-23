"""Support code for the source-confound investigation (Experiment D).

Central finding driving this module: in `data/manifest.csv`,
`source_dataset` and `label` are **perfectly collinear** — every COVID
image comes from Cohen, every NORMAL image comes from Kermany, with zero
exceptions (verified: `pd.crosstab(source_dataset, label)` has exactly two
non-zero cells). This is a direct consequence of how the two classes were
acquired, not an artifact of splitting.

That collinearity means the originally-planned "train a classifier to
predict source_dataset instead of label" control is **mathematically
identical** to Experiment A itself — same images, same partition, so
necessarily the same accuracy. Running it as a separate model would not
produce new information, only a second copy of the same number.

The informative version of this control, given what's actually available,
is a **same-source** control: Kermany's own NORMAL-vs-PNEUMONIA split
(both classes from one hospital, one imaging pipeline — no cross-source
confound at all). If a model trained on this genuinely single-source,
genuinely-different-pathology task performs markedly worse than
Experiment A's cross-source COVID-vs-NORMAL task, that's evidence
consistent with (not proof of) Experiment A's near-perfect numbers being
inflated by source shortcuts rather than by task difficulty alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from covidcxr.data.manifest import IMAGE_EXTENSIONS, MANIFEST_COLUMNS

# Kermany filenames for ALL classes (NORMAL, BACTERIA, VIRUS) share the same
# "<LABEL>-<patientid>-<imagenum>.jpeg" convention — verified against every
# file in train/PNEUMONIA and test/PNEUMONIA (0 non-matching).
KERMANY_ANY_CLASS_ID_RE = re.compile(r"^[A-Z]+-(\d+)-\d+", re.IGNORECASE)

KERMANY_CONTROL_SPLITS = ("train", "test")
KERMANY_CONTROL_CLASSES = ("NORMAL", "PNEUMONIA")


def build_kermany_control_manifest(chest_xray_root: Path) -> pd.DataFrame:
    """Both classes, single source (Kermany only) — for the source-confound
    control experiment. Pneumonia sub-types (bacterial/viral) are collapsed
    into one PNEUMONIA label, matching the folder structure.
    """
    rows = []
    for original_split in KERMANY_CONTROL_SPLITS:
        for label in KERMANY_CONTROL_CLASSES:
            class_dir = chest_xray_root / original_split / label
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                match = KERMANY_ANY_CLASS_ID_RE.search(path.name)
                patient_id = f"kermany-{match.group(1)}" if match else f"kermany-unknown-{path.stem}"
                rows.append(
                    {
                        "filename": path.name,
                        "filepath": path.as_posix(),
                        "label": label,
                        "patient_id": patient_id,
                        "patient_id_reliable": match is not None,
                        "source_dataset": "kermany",
                        "original_split": original_split,
                        "view": "",
                        "modality": "X-ray",
                        "date": "",
                        "license": "CC BY 4.0",
                    }
                )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
