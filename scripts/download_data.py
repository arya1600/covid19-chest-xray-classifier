"""Download the two upstream chest X-ray datasets used by this project.

This script does not bundle or redistribute any images itself — it fetches
them at run time from their original public sources, documented in
`data/README.md`. It is safe to re-run (idempotent): existing files are
skipped unless `--force` is passed.

Usage:
    python scripts/download_data.py --dest data/raw --source all
    python scripts/download_data.py --dest data/raw --source covid
    python scripts/download_data.py --dest data/raw --source normal

Covid-19-positive images (ieee8023/covid-chestxray-dataset) are fetched by
individual raw file, not by cloning the repository: we first download only
`metadata.csv` (~200KB), filter it down to the rows this project actually
uses (frontal chest X-ray, COVID-19 finding, and — importantly — a
non-empty `license` field, since ~43% of candidate rows have no recorded
license and are excluded rather than assumed reusable), and then fetch only
those ~270 image files. This avoids a ~650MB full-repository clone for the
~270 files (a few tens of MB) actually needed.

For the Normal-class images (Kermany et al., "Chest X-Ray Images
(Pneumonia)"), no public per-file metadata API exists, so this project uses
a manually downloaded archive — see the instructions this script prints, or
`data/README.md`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COVID_CHESTXRAY_RAW_BASE = (
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master"
)
COVID_CHESTXRAY_METADATA_URL = f"{COVID_CHESTXRAY_RAW_BASE}/metadata.csv"
FRONTAL_VIEWS = {"PA", "AP", "AP Supine"}

KERMANY_KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
KERMANY_MENDELEY_URL = "https://data.mendeley.com/datasets/rscbjbr9sj/2"


def _is_covid_frontal_licensed(row: dict) -> bool:
    """Selection criteria, in one place so acquisition and documentation agree:
    a COVID-19-labelled frontal chest X-ray with a recorded, non-empty license.
    """
    return (
        "covid-19" in row["finding"].lower()
        and row["modality"] == "X-ray"
        and row["view"] in FRONTAL_VIEWS
        and row["license"].strip() != ""
    )


def download_covid_chestxray(dest: Path, force: bool = False) -> Path:
    """Fetch metadata.csv, filter it, and download only the matching images."""
    target = dest / "covid-chestxray-dataset"
    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = target / "metadata.csv"
    if not metadata_path.exists() or force:
        log.info("Downloading metadata.csv from %s ...", COVID_CHESTXRAY_METADATA_URL)
        urlretrieve(COVID_CHESTXRAY_METADATA_URL, metadata_path)
    else:
        log.info("Found existing %s, skipping re-download (use --force).", metadata_path)

    with open(metadata_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = [r for r in rows if _is_covid_frontal_licensed(r)]
    log.info(
        "metadata.csv: %d total rows -> %d frontal COVID-19 X-rays with a recorded license.",
        len(rows),
        len(selected),
    )

    downloaded, skipped = 0, 0
    for row in selected:
        filename = row["filename"]
        out_path = images_dir / filename
        if out_path.exists() and not force:
            skipped += 1
            continue
        url = f"{COVID_CHESTXRAY_RAW_BASE}/{row['folder']}/{filename}"
        urlretrieve(url, out_path)
        downloaded += 1

    log.info(
        "covid-chestxray-dataset ready at %s (%d downloaded, %d already present).",
        target,
        downloaded,
        skipped,
    )
    return target


def download_kermany(dest: Path, force: bool = False) -> Path:
    """Fetch the Kermany "Chest X-Ray Images (Pneumonia)" dataset via Kaggle,
    falling back to printing manual Mendeley instructions.
    """
    target = dest / "kermany-chest-xray"
    if target.exists() and not force:
        log.info("Found existing %s, skipping (use --force to re-download).", target)
        return target

    if shutil.which("kaggle") is not None:
        target.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s via kaggle CLI ...", KERMANY_KAGGLE_DATASET)
        try:
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    KERMANY_KAGGLE_DATASET,
                    "-p",
                    str(target),
                    "--unzip",
                ],
                check=True,
            )
            log.info("Kermany dataset ready at %s", target)
            return target
        except subprocess.CalledProcessError as exc:
            log.warning("kaggle CLI download failed (%s); falling back to manual instructions.", exc)

    log.warning(
        "\nKaggle CLI not available or not authenticated.\n"
        "Manual download required:\n"
        "  1. Visit %s\n"
        "  2. Download the dataset archive (Mendeley Data, CC BY 4.0).\n"
        "  3. Extract it so that Normal-class images end up under:\n"
        "       %s\n"
        "  4. Re-run this script with --force once the files are in place\n"
        "     (or just proceed directly to scripts/build_manifest.py).\n",
        KERMANY_MENDELEY_URL,
        target,
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw"), help="Destination directory.")
    parser.add_argument(
        "--source",
        choices=["all", "covid", "normal"],
        default="all",
        help="Which dataset(s) to fetch.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if destination exists.")
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)

    if args.source in ("all", "covid"):
        download_covid_chestxray(args.dest, force=args.force)
    if args.source in ("all", "normal"):
        download_kermany(args.dest, force=args.force)

    log.info("Done. Next: python scripts/build_manifest.py --raw-dir %s", args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
