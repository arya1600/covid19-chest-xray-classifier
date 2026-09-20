# Third-Party Notices

## Reference project

This project was inspired by and built using
[`aviralchharia/COVID-19`](https://github.com/aviralchharia/COVID-19) as a
**learning and reference starting point**. No code, notebook cells, or bundled
images from that repository are reused here — this project is an independent
reimplementation with a different architecture, training pipeline, dataset
provenance/splitting strategy, and evaluation methodology (see `README.md` for
what was changed and why). The original repository's license and copyright
notice are reproduced below, as required by its MIT license terms.

> MIT License
>
> Copyright (c) 2020 Aviral Chharia
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

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
