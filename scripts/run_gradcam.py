"""Generate Grad-CAM overlays for a curated sample of real test-set
predictions from a trained model: a few correct-high-confidence,
correct-low-confidence, and incorrect examples per class (as available).

Usage:
    python scripts/run_gradcam.py \
        --config runs/exp_a_baseline/config.yaml \
        --checkpoint runs/exp_a_baseline/best.pt \
        --out-dir reports/gradcam/exp_a_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covidcxr.data.dataset import CXRDataset, load_image  # noqa: E402
from covidcxr.data.transforms import ResizeAndPad, build_transforms  # noqa: E402
from covidcxr.explain.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from covidcxr.models.baseline_cnn import BaselineCNN  # noqa: E402
from covidcxr.training.config import load_experiment_config  # noqa: E402

ARCHITECTURES = {"baseline_cnn": BaselineCNN}


def select_examples(y_true, y_pred, y_score, n_per_category: int = 3) -> list[tuple[int, str]]:
    """Returns [(row_index, category_name), ...]."""
    selected = []
    n = len(y_true)

    def top_n(indices, key, n):
        return sorted(indices, key=key, reverse=True)[:n]

    correct = [i for i in range(n) if y_true[i] == y_pred[i]]
    incorrect = [i for i in range(n) if y_true[i] != y_pred[i]]

    correct_1 = [i for i in correct if y_true[i] == 1]
    correct_0 = [i for i in correct if y_true[i] == 0]
    incorrect_fn = [i for i in incorrect if y_true[i] == 1]  # predicted 0, actually 1
    incorrect_fp = [i for i in incorrect if y_true[i] == 0]  # predicted 1, actually 0

    for i in top_n(correct_1, lambda i: y_score[i], n_per_category):
        selected.append((i, "correct_positive_high_conf"))
    for i in top_n(correct_0, lambda i: -y_score[i], n_per_category):
        selected.append((i, "correct_negative_high_conf"))
    for i in top_n(correct, lambda i: -abs(y_score[i] - 0.5), n_per_category):
        selected.append((i, "correct_low_confidence"))
    for i in incorrect_fn[:n_per_category]:
        selected.append((i, "false_negative"))
    for i in incorrect_fp[:n_per_category]:
        selected.append((i, "false_positive"))

    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-per-category", type=int, default=3)
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    device = torch.device("cpu")

    model_cls = ARCHITECTURES[config.model.architecture]
    model = model_cls(
        in_channels=config.model.in_channels,
        channels=tuple(config.model.channels),
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_ds = CXRDataset(
        config.data.manifest_path, split="test", transform=None,
        root_dir=config.data.root_dir, label_to_index=config.data.label_to_index,
    )
    model_transform = build_transforms(
        config.data.image_size, train=False, mean=config.data.mean, std=config.data.std,
        num_channels=config.data.num_channels,
    )

    # First pass: collect predictions for every test image (no grad needed).
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            row = test_ds.df.iloc[i]
            img = load_image(Path(row["filepath"]))
            tensor = model_transform(img).unsqueeze(0)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            y_true.append(test_ds.label_to_index[row["label"]])
            y_pred.append(int(logits.argmax(dim=1).item()))
            y_score.append(probs[0, 1].item())

    selected = select_examples(y_true, y_pred, y_score, n_per_category=args.n_per_category)

    # A Grad-CAM overlay is a derivative of its source image. Skip any
    # source licensed "No-Derivatives" (ND) — safe to publish otherwise
    # (CC BY, CC BY-NC-SA, etc. all permit derivatives).
    filtered = []
    for idx, category in selected:
        license_str = str(test_ds.df.iloc[idx].get("license", ""))
        if "ND" in license_str.upper():
            print(f"  Skipping idx{idx} ({category}): license {license_str!r} does not permit derivatives.")
            continue
        filtered.append((idx, category))
    selected = filtered
    print(f"Selected {len(selected)} examples across categories "
          f"(some categories may have fewer than {args.n_per_category} available).")

    cam_tool = GradCAM(model, target_layer=model.features)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx, category in selected:
        row = test_ds.df.iloc[idx]
        img = load_image(Path(row["filepath"]))
        display_img = ResizeAndPad(config.data.image_size)(img)
        tensor = model_transform(img).unsqueeze(0)
        tensor.requires_grad_(True)

        cam, target_class = cam_tool.generate(tensor, target_class=y_pred[idx])
        overlay = overlay_heatmap(display_img, cam)

        filename = (f"{category}_idx{idx}_true{y_true[idx]}_pred{y_pred[idx]}"
                    f"_score{y_score[idx]:.2f}.png")
        overlay.save(args.out_dir / filename)
        print(f"  {category}: true={y_true[idx]} pred={y_pred[idx]} score={y_score[idx]:.3f} -> {filename}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
