# COVID-19 Chest X-Ray Classification — A Reproducible Study

**This is a research/educational project. It is not a medical device, has
not been clinically validated, and must not be used for diagnosis.**

## What this project is

An independent, from-scratch reimplementation and critical re-investigation
of a small chest X-ray (CXR) COVID-19/Normal binary classification task,
built as an MSc-application portfolio project. It was inspired by
[`aviralchharia/COVID-19`](https://github.com/aviralchharia/COVID-19) (MIT
license; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)), which was
used only as a learning reference — every component here (data pipeline,
splitting strategy, models, evaluation, explainability, demo) is a new
implementation.

## Why this project exists

An audit of the reference project found several issues common to a lot of
2020-era "COVID-CNN" work: the validation set doubled as the test set
(optimistic metrics), the two classes came from two structurally different
source datasets (a possible shortcut-learning confound), and — concretely —
some images from the *same patient case* were split across its train and
validation folders, a real patient-level leakage bug, not a hypothetical
one. This project rebuilds the pipeline to fix those problems and, more
importantly, treats **"is the model learning real signal or a
dataset-source shortcut?"** as an explicit research question — with a
real answer, below, not just a caveat.

## Headline finding

A baseline CNN trained on this rebuilt, leakage-free pipeline reaches
**98.9% test accuracy and a perfect ROC-AUC (1.0) on COVID vs. Normal**
(`runs/exp_a_baseline/`). **That number should not be trusted at face
value.** A same-architecture model trained on a same-source control task
(Normal vs. Pneumonia, one hospital, no cross-source confound) makes
ordinary, expected mistakes (95.6% accuracy, real false positives and
false negatives) — a same-source-independent version of a comparable
task performs, as expected, less "perfectly." Grad-CAM on the main model
shows attention repeatedly landing on image borders and, in one case, an
embedded text watermark rather than lung fields. An informal live test
with an external, out-of-distribution normal X-ray was misclassified as
COVID at 97.6% confidence, with the same border-focused attention
pattern. Full evidence and real numbers: [`reports/model_comparison.md`](reports/model_comparison.md)
and [`reports/error_analysis.md`](reports/error_analysis.md).

**In short: the model most likely learned to distinguish which dataset an
image came from, not COVID-19 from a chest X-ray.** Demonstrating and
evidencing that — not achieving a high accuracy number — is this
project's actual result.

## Repository structure

```
covid-cxr/
├── data/README.md        # dataset provenance, licenses, citations, download instructions
├── configs/               # one YAML per experiment (hyperparameters, not hardcoded)
├── scripts/               # data acquisition, manifest-building, training, eval, Grad-CAM CLIs
├── src/covidcxr/          # installable package: all real pipeline logic
├── demo/                  # Streamlit prediction demo (with disclaimer)
├── tests/                 # unit tests (splitting, leakage, preprocessing, metrics, Grad-CAM)
├── runs/                  # per-experiment config + real metrics/history (checkpoints excluded)
└── reports/               # model comparison, error analysis, Grad-CAM overlays
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the pipeline

```bash
# 1. Acquire data (see data/README.md for the Normal-class manual download step)
python scripts/download_data.py --dest data/raw --source covid

# 2. Build the patient-grouped manifest + split
python scripts/build_manifest.py \
  --cohen-metadata data/raw/covid-chestxray-dataset/metadata.csv \
  --cohen-images data/raw/covid-chestxray-dataset/images \
  --kermany-chest-xray-dir data/raw/CellData/chest_xray \
  --out data/manifest.csv

# 3. Train
python scripts/train.py --config configs/exp_a.yaml

# 4. Evaluate on the (untouched-until-now) test set
python scripts/evaluate.py --config runs/exp_a_baseline/config.yaml \
  --checkpoint runs/exp_a_baseline/best.pt --split test

# 5. Grad-CAM
python scripts/run_gradcam.py --config runs/exp_a_baseline/config.yaml \
  --checkpoint runs/exp_a_baseline/best.pt --out-dir reports/gradcam/exp_a_baseline

# 6. Interactive demo
streamlit run demo/app.py
```

Run `pytest tests/ -v` (with `PYTHONPATH=src`) at any point — 63 tests, no
data required for most of them.

## Data

No dataset images are committed to this repository — only the manifest
CSVs (filenames/paths/labels/patient IDs, no pixels). See
[`data/README.md`](data/README.md) for the upstream sources, their
licenses, required citations, and the acquisition process.

## License

Code in this repository is released under the MIT License (see
[`LICENSE`](LICENSE)). Third-party code inspiration, datasets, and
pretrained weights retain their own licenses — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Grad-CAM overlays
published under `reports/gradcam/` are excluded automatically when their
source image's license doesn't permit derivatives (see
`scripts/run_gradcam.py`).

## Limitations

- Both experiments use small datasets (hundreds to low thousands of
  images); all reported metrics carry bootstrap 95% confidence intervals,
  not bare point estimates.
- The main task's two classes are drawn from different source datasets —
  a confound this project measures directly (see "Headline finding"
  above) rather than assuming away.
- Grad-CAM is a coarse, sometimes-misleading localization method; it is
  one piece of corroborating evidence here, not proof.
- Not clinically validated. Not a diagnostic tool, under any
  circumstance.
