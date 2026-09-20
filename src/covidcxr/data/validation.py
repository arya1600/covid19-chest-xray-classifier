"""Preflight validation of every image path in a manifest.

Distinct from `CXRDataset`'s per-item `CorruptImageError` (which stops a
training/eval loop the moment it hits a bad file): this module scans every
row up front and returns a full report, so problems are surfaced clearly
and all at once, before any training run starts — never silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from covidcxr.data.dataset import CorruptImageError, load_image


@dataclass
class ImageValidationReport:
    checked: int = 0
    ok: int = 0
    missing: list[str] = field(default_factory=list)
    corrupt: list[tuple[str, str]] = field(default_factory=list)  # (path, error message)

    @property
    def is_clean(self) -> bool:
        return not self.missing and not self.corrupt

    def summary(self) -> str:
        lines = [f"Checked {self.checked} images: {self.ok} OK, "
                 f"{len(self.missing)} missing, {len(self.corrupt)} corrupt."]
        for path in self.missing:
            lines.append(f"  MISSING: {path}")
        for path, err in self.corrupt:
            lines.append(f"  CORRUPT: {path} ({err})")
        return "\n".join(lines)


def validate_manifest_images(manifest: str | Path | pd.DataFrame, root_dir: str | Path | None = None) -> ImageValidationReport:
    df = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest
    root = Path(root_dir) if root_dir is not None else None

    report = ImageValidationReport()
    for filepath in df["filepath"]:
        path = Path(filepath)
        if root is not None and not path.is_absolute():
            path = root / path
        report.checked += 1

        if not path.exists():
            report.missing.append(str(path))
            continue
        try:
            load_image(path)
            report.ok += 1
        except CorruptImageError as exc:
            report.corrupt.append((str(path), str(exc)))

    return report
