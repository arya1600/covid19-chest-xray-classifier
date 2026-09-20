"""Run the full metric suite ONCE on the TEST split for a trained model.

Usage:
    python scripts/evaluate.py --config runs/exp_a_baseline/config.yaml \
        --checkpoint runs/exp_a_baseline/best.pt \
        --split test

`--split` defaults to "test" but accepts "val" too, only for sanity-
checking (e.g. comparing against training's own val numbers) — the test
split should only be evaluated this way once a model is finalized, not
repeatedly while iterating.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covidcxr.data.pipeline import build_dataloaders  # noqa: E402
from covidcxr.evaluation.metrics import compute_confusion_matrix, compute_metrics  # noqa: E402
from covidcxr.evaluation.predict import get_predictions  # noqa: E402
from covidcxr.models.baseline_cnn import BaselineCNN  # noqa: E402
from covidcxr.training.config import load_experiment_config  # noqa: E402

ARCHITECTURES = {"baseline_cnn": BaselineCNN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write metrics as JSON")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    bundle = build_dataloaders(config.data)
    loader = bundle.val_loader if args.split == "val" else bundle.test_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = ARCHITECTURES[config.model.architecture]
    model = model_cls(
        in_channels=config.model.in_channels,
        channels=tuple(config.model.channels),
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    y_true, y_pred, y_score = get_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, y_score, n_bootstrap=args.n_bootstrap)
    cm = compute_confusion_matrix(y_true, y_pred)

    print(f"Evaluation on '{args.split}' split ({len(y_true)} images), "
          f"checkpoint={args.checkpoint}:\n")
    for name, result in metrics.items():
        print(f"  {name:12s} = {result.value:.4f}  (95% CI: {result.ci_low:.4f} - {result.ci_high:.4f})")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    if args.out is not None:
        payload = {
            "split": args.split,
            "n_images": len(y_true),
            "checkpoint": str(args.checkpoint),
            "metrics": {name: vars(result) for name, result in metrics.items()},
            "confusion_matrix": cm.tolist(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote metrics to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
