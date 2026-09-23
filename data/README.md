# Data: Provenance, Licensing, and Acquisition

No images are stored in this repository. This document describes where they
come from, under what terms, and how `scripts/download_data.py` /
`scripts/build_manifest.py` reconstruct the working dataset reproducibly.

## 1. COVID-19-positive chest X-rays — `ieee8023/covid-chestxray-dataset`

- **Source**: https://github.com/ieee8023/covid-chestxray-dataset
- **What it is**: a crowd-sourced collection of chest X-ray/CT images
  extracted from published case reports, papers, and clinician-submitted
  cases, with an accompanying `metadata.csv` giving per-image `patientid`,
  `finding`, `view` (PA/AP/lateral), `modality`, and source URL.
- **License**: mixed — the repository itself provides code under Apache-2.0,
  but **each image retains the license of its original source** (case
  report, journal figure, etc.), documented per-row in `metadata.csv`.
  Verified directly against the upstream `metadata.csv` (2026 snapshot,
  951 rows): of 478 rows that are frontal chest X-rays (`view` in
  PA/AP/AP-Supine, `modality=X-ray`) with a COVID-19 `finding`, **206 rows
  (43%) have an empty `license` field** — unspecified provenance. This
  project only acquires the remaining **272 rows (156 unique patients)**
  that carry an explicit license (a mix of CC BY 3.0/4.0, CC BY-NC-SA,
  CC BY-NC-ND, or "listed as authorized for everyone"), and records each
  image's specific license string in `data/manifest.csv` for auditability —
  no blanket "the dataset is licensed" assumption is made.
  **Caveat**: some of the usable licenses are NC (non-commercial — fine for
  this educational/portfolio use) or ND (no-derivatives — worth noting
  since any resizing/preprocessing is technically a derivative; this is
  standard practice in this research area, but stated here rather than
  silently assumed).
- **Required citation**:
  > Cohen, J. P., Morrison, P., & Dao, L. (2020). COVID-19 Image Data
  > Collection. *arXiv:2003.11597*.
  > https://github.com/ieee8023/covid-chestxray-dataset
- **Why fetch metadata.csv directly rather than just the images**: the
  `patientid` field in this metadata is exactly what this project needs
  for a leakage-free, patient-grouped split — reverse-engineering patient
  identity from filenames alone would be unreliable.

## 2. Normal chest X-rays — Kermany et al. "Chest X-Ray Images (Pneumonia)"

- **Source**: originally published alongside Kermany et al. (2018,
  *Cell*), commonly redistributed via Kaggle
  (`paultimothymooney/chest-xray-pneumonia`) and Mendeley Data
  (https://data.mendeley.com/datasets/rscbjbr9sj/2).
- **License**: CC BY 4.0 (Mendeley Data record), which permits reuse and
  redistribution with attribution.
- **Required citation**:
  > Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying
  > Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.
  > *Cell*, 172(5), 1122–1131.e9. https://doi.org/10.1016/j.cell.2018.02.010
- **Patient grouping caveat**: filenames follow an `IM-####-####` (or
  `NORMAL2-IM-####-####`) convention where the first number groups images
  from the same imaging session; this project treats that leading number as
  a best-effort `patient_id` proxy for grouped splitting. This is an
  explicitly documented assumption, not verified against real patient
  records — noted as a limitation in `data/manifest.csv`'s generation log
  and in the README limitations section.
- **Only the Normal-class subset of this dataset is used** — this project
  does not use its pneumonia (non-COVID) images, since the reference project
  and this rebuild both target COVID vs. Normal.

## 3. Access notes

- The `covid-chestxray-dataset` metadata and images are fetched by
  individual file over plain HTTP from `raw.githubusercontent.com` — no
  authentication, and **no full repository clone**. `download_data.py`
  downloads `metadata.csv` (~200KB), filters it to the ~270 rows this
  project uses (see §1), and fetches only those image files (verified: a
  full clone of the upstream repo is ~650MB; the filtered subset is a few
  tens of MB).
- The Kermany dataset has no equivalent per-file metadata API, so this
  project uses a **manually downloaded archive** rather than a scripted
  fetch: download the dataset from Kaggle
  (`paultimothymooney/chest-xray-pneumonia`) or Mendeley Data
  (link above), and place the **Normal-class images only** under
  `data/raw/kermany-chest-xray/` (any internal subfolder layout is fine —
  `build_manifest.py` searches recursively for image files under that
  directory). `download_data.py --source normal` will still attempt the
  `kaggle` CLI first if credentials are configured, and otherwise prints
  these same instructions.

## 4. Reproducing acquisition

```bash
python scripts/download_data.py --dest data/raw --source covid
# Normal-class images: download the Mendeley "CellData" archive manually
# (see §2/§3 above) and extract it so this path exists:
#   data/raw/CellData/chest_xray/{train,test}/{NORMAL,PNEUMONIA}

python scripts/build_manifest.py \
  --cohen-metadata data/raw/covid-chestxray-dataset/metadata.csv \
  --cohen-images data/raw/covid-chestxray-dataset/images \
  --kermany-chest-xray-dir data/raw/CellData/chest_xray \
  --out data/manifest.csv
```

Both scripts are idempotent (safe to re-run) and log per-class/per-source
image counts, license exclusions, and any duplicate/unreliable-ID findings,
so the exact data snapshot used for any experiment can be identified later
from `data/manifest.csv` alone.

## 5. Preprocessing & data pipeline (Phase 3)

Implemented in `src/covidcxr/data/{transforms,dataset,class_balance,
validation,pipeline}.py`, config-driven via `covidcxr.data.pipeline.DataConfig`
(see the `data:` section of any experiment config, e.g. `configs/exp_a.yaml`).

- **Split membership**: taken exactly from `manifest.csv`'s `split` column.
  The pipeline never re-splits or reshuffles across train/val/test —
  that decision was already made once, correctly (patient-grouped,
  leakage-checked), by `covidcxr.data.splitting`.
- **Resizing**: `ResizeAndPad` scales the longer edge to `image_size`
  (default 224) and pads the shorter edge with zeros, preserving aspect
  ratio instead of stretching lung geometry to fit a square.
- **Augmentation (train split only)**: small-angle rotation (±10°), a mild
  zoom via `RandomAffine(scale=(0.9, 1.1))`, and brightness/contrast
  jitter. **No horizontal flip anywhere** — chest X-rays have real
  anatomical laterality (heart/aorta position), so mirroring is not a
  label-preserving augmentation here the way it is for natural images.
  Validation and test transforms are strictly deterministic (verified by
  a unit test asserting no random op appears in that pipeline, and that
  the same input image always produces bit-identical output tensors).
- **Grayscale/channel handling**: every image is standardized to
  single-channel (`convert("L")`) on load, since a few source files
  (verified: some Cohen case-report scans) are stored as 3-channel JPEGs
  with identical R=G=B. The from-scratch baseline model consumes this
  native 1-channel tensor directly. For later transfer-learning
  experiments using ImageNet-pretrained backbones (which expect 3
  channels), `ReplicateChannels` repeats the single channel three times
  — the standard approach for feeding grayscale medical images to
  RGB-pretrained networks — rather than any invented colorization.
- **Normalization**: real, computed statistics, not placeholders and not
  ImageNet stats — `scripts/compute_normalization_stats.py` computed
  mean=0.4073, std=0.2863 over all 1,297 real TRAIN-split images at
  224×224, 1 channel (val/test are never used to compute normalization
  stats, to avoid leaking their distribution into preprocessing).
- **Class imbalance**: configurable via `DataConfig.class_balance` —
  `"none"` (no weighting), `"loss_weight"` (inverse-frequency class
  weights for `CrossEntropyLoss`), or `"sampler"` (`WeightedRandomSampler`,
  draws with replacement so minibatches are roughly class-balanced).
  **No automatic oversampling or image duplication on disk** — the
  sampler only reweights which existing images are drawn more often.
- **Reproducibility**: `covidcxr.utils.seed.seed_everything(seed)` plus a
  seeded `torch.Generator` passed to the train `DataLoader`, so a given
  seed always produces the same shuffle order and augmentation draws.
- **Corrupt/missing files**: `CXRDataset.__getitem__` raises a clear
  `CorruptImageError` naming the exact file and underlying error — it
  never silently skips a bad file or substitutes a blank image, which
  would otherwise quietly change dataset composition. A separate
  preflight utility, `validate_manifest_images()`, scans every path up
  front and returns a full report (missing vs. corrupt vs. OK) before any
  training run starts.
