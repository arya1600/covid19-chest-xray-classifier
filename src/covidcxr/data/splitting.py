"""Patient/session-grouped train/val/test splitting.

The core guarantee this module provides — and the direct fix for the
leakage bug found in the reference project (same patient's images in both
its Train and Val folders) — is that every row sharing a `patient_id`
lands in exactly one split. Ordinary random or stratified splitting does
not guarantee this; `GroupShuffleSplit` does, at the cost of only
approximate class balance across splits (grouping and stratification
cannot both be satisfied exactly when group sizes vary).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def patient_grouped_split(
    df: pd.DataFrame,
    group_col: str = "patient_id",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a copy of `df` with a new "split" column in {"train","val","test"}.

    Splitting is done in two stages, both grouped on `group_col`:
    1. carve off the test set from everything,
    2. carve val from what remains, leaving train.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    if df[group_col].isna().any():
        raise ValueError(f"{group_col} contains missing values; every row needs a group.")

    df = df.reset_index(drop=True).copy()

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    remaining_idx, test_idx = next(splitter_test.split(df, groups=df[group_col]))

    remaining = df.iloc[remaining_idx]
    relative_val_frac = val_frac / (train_frac + val_frac)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed)
    train_rel_idx, val_rel_idx = next(splitter_val.split(remaining, groups=remaining[group_col]))

    train_idx = remaining.iloc[train_rel_idx].index
    val_idx = remaining.iloc[val_rel_idx].index

    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def verify_no_patient_leakage(df: pd.DataFrame, group_col: str = "patient_id", split_col: str = "split") -> None:
    """Raise AssertionError if any group_col value appears in more than one split."""
    groups_per_split = df.groupby(group_col)[split_col].nunique()
    leaking = groups_per_split[groups_per_split > 1]
    if not leaking.empty:
        raise AssertionError(
            f"{len(leaking)} patient_id value(s) appear in more than one split: "
            f"{list(leaking.index)[:10]}{'...' if len(leaking) > 10 else ''}"
        )
