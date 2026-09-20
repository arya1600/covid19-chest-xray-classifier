"""Build the source-confound control manifest: Kermany-only NORMAL vs
PNEUMONIA (single source, no cross-source confound at all), patient-grouped
split, same methodology as the main manifest. See
covidcxr.data.source_confound for why this control — not a literal
"predict source_dataset" classifier — is the informative version of
Experiment D given this project's actual data.

Usage:
    python scripts/build_source_confound_control_manifest.py \
        --kermany-chest-xray-dir data/raw/CellData/chest_xray \
        --out data/manifest_source_confound_control.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covidcxr.data.source_confound import build_kermany_control_manifest  # noqa: E402
from covidcxr.data.splitting import patient_grouped_split, verify_no_patient_leakage  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kermany-chest-xray-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/manifest_source_confound_control.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    df = build_kermany_control_manifest(args.kermany_chest_xray_dir)
    print(f"Built control manifest: {len(df)} images, {df['patient_id'].nunique()} unique patients")
    print(df["label"].value_counts().to_string())

    unreliable = (~df["patient_id_reliable"]).sum()
    print(f"Images with unreliable patient_id: {unreliable}")

    df = patient_grouped_split(df, seed=args.seed)
    verify_no_patient_leakage(df)
    print("\nLeakage check: PASSED")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(df["split"].value_counts().to_string())
    print("\nClass balance per split:")
    print(pd.crosstab(df["split"], df["label"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
