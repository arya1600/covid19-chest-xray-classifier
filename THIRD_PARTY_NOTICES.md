# Third-Party Notices

## Datasets

This project does not bundle any dataset images. Data is downloaded at build
time by `scripts/download_data.py` from the original public sources, which
retain their own licenses and citation requirements independent of this
repository's license. See [`data/README.md`](data/README.md) for full
citations, license terms, and provenance notes for:

- the `ieee8023/covid-chestxray-dataset` collection (COVID-19-positive chest
  X-rays), and
- the Kermany et al. "Chest X-Ray Images (Pneumonia)" dataset (Normal-class
  chest X-rays).

Users of this repository are responsible for reviewing and complying with
those datasets' own license terms before use.

## Pretrained model weights

Transfer-learning experiments use ImageNet-pretrained backbone weights
distributed via `torchvision` (e.g. ResNet18, EfficientNet-B0), each under
their own respective license as published by `torchvision`/PyTorch. No
pretrained weights are redistributed by this repository.
