"""Build data/manifest.csv: join the two sources, recover patient IDs,
assign a patient-grouped train/val/test split, and print a summary.

Usage:
    python scripts/build_manifest.py \
        --cohen-metadata data/raw/covid-chestxray-dataset/metadata.csv \
        --cohen-images data/raw/covid-chestxray-dataset/images \
        --kermany-chest-xray-dir data/raw/CellData/chest_xray \
        --out data/manifest.csv

`--kermany-chest-xray-dir` must point at the real Mendeley CellData
`chest_xray` directory (containing `train/NORMAL`, `train/PNEUMONIA`,
`test/NORMAL`, `test/PNEUMONIA`) — only the NORMAL subfolders are read.

Either --cohen-metadata or --kermany-chest-xray-dir may be omitted to build
a partial manifest (e.g. while still waiting on one source), but the split
and summary are only meaningful once both are present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covidcxr.data.manifest import build_manifest  # noqa: E402
from covidcxr.data.splitting import patient_grouped_split, verify_no_patient_leakage  # noqa: E402


def print_summary(df) -> None:
    print(f"\nTotal images: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")

    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())

    print("\nSource distribution:")
    print(df["source_dataset"].value_counts().to_string())

    if "split" in df.columns:
        print("\nTrain/val/test distribution (rows):")
        print(df["split"].value_counts().to_string())
        print("\nTrain/val/test distribution (unique patients):")
        print(df.groupby("split")["patient_id"].nunique().to_string())
        print("\nClass balance per split:")
        print(pd.crosstab(df["split"], df["label"]).to_string())

    unreliable = df[~df["patient_id_reliable"]]
    print(f"\nImages with an unreliable (fallback) patient_id: {len(unreliable)}")
    if len(unreliable):
        print(unreliable[["filename", "source_dataset", "patient_id"]].head(10).to_string(index=False))

    if "split" in df.columns:
        print("\nUnique patients per split, per class:")
        print(df.groupby(["split", "label"])["patient_id"].nunique().to_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohen-metadata", type=Path, default=None)
    parser.add_argument("--cohen-images", type=Path, default=None)
    parser.add_argument("--kermany-chest-xray-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args(argv)

    df, stats = build_manifest(args.cohen_metadata, args.cohen_images, args.kermany_chest_xray_dir)
    print(f"Cohen: {stats.cohen_total_rows} total rows, {stats.cohen_selected_rows} selected, "
          f"{stats.cohen_excluded_no_license} excluded (no recorded license).")
    print(f"Kermany: {stats.kermany_total_files} NORMAL files found, "
          f"{stats.kermany_unreliable_patient_id} with an unreliable patient_id, "
          f"{stats.kermany_duplicate_content_files} byte-identical duplicate file(s).")
    if stats.kermany_duplicate_content_pairs:
        print("Duplicate content pairs (first file kept as canonical, second is a byte-identical copy):")
        for original, duplicate in stats.kermany_duplicate_content_pairs:
            print(f"  {original}  ==  {duplicate}")

    if df.empty:
        print("Manifest is empty (no source provided or no files found) — nothing to split.")
        return 0

    if df["label"].nunique() < 2:
        print("Only one class present — skipping split until both sources are available.")
        df.to_csv(args.out, index=False)
        print_summary(df)
        return 0

    df = patient_grouped_split(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    verify_no_patient_leakage(df)
    print("\nLeakage check: PASSED — no patient/group id appears in more than one split.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote manifest to {args.out}")
    print_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
