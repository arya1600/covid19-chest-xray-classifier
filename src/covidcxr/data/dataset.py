"""PyTorch Dataset over data/manifest.csv.

Split membership is taken exactly as recorded in the manifest's `split`
column — this module never re-splits, re-shuffles across splits, or
infers a split. That decision was already made once, correctly (patient-
grouped, leakage-checked), by `covidcxr.data.splitting`; re-deriving it
here would risk silently reintroducing the exact leakage bug this project
exists to fix.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Fixed, documented label encoding — used consistently everywhere a label
# needs to be an integer (loss functions, class-weight computation, sampler).
LABEL_TO_INDEX = {"NORMAL": 0, "COVID": 1}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

VALID_SPLITS = ("train", "val", "test")


class CorruptImageError(RuntimeError):
    """Raised when an image file referenced by the manifest cannot be read.

    Deliberately not caught anywhere inside `CXRDataset.__getitem__` — a
    corrupt or missing file should stop you and tell you exactly which
    file and why, not be silently skipped or replaced with a blank image
    (which would quietly change dataset size/composition without anyone
    noticing).
    """


def load_image(path: Path) -> Image.Image:
    try:
        img = Image.open(path)
        img.load()  # force decode now, so a truncated file fails here, not later
    except (OSError, UnidentifiedImageError) as exc:
        raise CorruptImageError(f"Could not read image at '{path}': {exc}") from exc
    # Standardize to single-channel: most source images are already
    # grayscale, but a few (verified: some Cohen case-report scans) are
    # stored as 3-channel JPEGs with identical R=G=B; converting to "L"
    # gives one consistent representation for every image regardless of
    # how the source file happened to be encoded.
    return img.convert("L")


class CXRDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str,
        transform=None,
        root_dir: str | Path | None = None,
        label_to_index: dict[str, int] | None = None,
    ):
        """`label_to_index` defaults to the project's COVID/NORMAL mapping.
        Override it for a different binary task over the same
        infrastructure — e.g. the source-confound control experiment's
        NORMAL-vs-PNEUMONIA task, which is not the project's main label
        space at all.
        """
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

        df = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest
        if "split" not in df.columns:
            raise ValueError("manifest has no 'split' column — build it with patient_grouped_split() first")

        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split={split!r} in the given manifest")

        self.label_to_index = label_to_index if label_to_index is not None else LABEL_TO_INDEX
        unknown_labels = set(self.df["label"]) - set(self.label_to_index)
        if unknown_labels:
            raise ValueError(f"Manifest contains unexpected label(s): {unknown_labels}")

        self.split = split
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, filepath: str) -> Path:
        path = Path(filepath)
        if self.root_dir is not None and not path.is_absolute():
            path = self.root_dir / path
        return path

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = self._resolve_path(row["filepath"])
        image = load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        label = self.label_to_index[row["label"]]
        return image, label

    @property
    def labels(self) -> list[int]:
        """Encoded labels for every row, in dataset order — used for
        class-weight and sampler computation without decoding any images.
        """
        return [self.label_to_index[label] for label in self.df["label"]]

    @property
    def patient_ids(self) -> list[str]:
        return list(self.df["patient_id"])
