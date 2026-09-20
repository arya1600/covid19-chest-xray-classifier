"""Train one experiment from a config YAML.

Usage:
    python scripts/train.py --config configs/exp_a.yaml

Writes to runs/<name>/: a copy of the resolved config, best.pt (best
val-loss checkpoint), history.csv (per-epoch train/val loss+acc), and
prints a final summary. Does not touch the test set — that's a separate,
later evaluation step (covidcxr.evaluation), run once per finished model.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covidcxr.data.pipeline import build_dataloaders  # noqa: E402
from covidcxr.models.baseline_cnn import BaselineCNN  # noqa: E402
from covidcxr.training.config import load_experiment_config  # noqa: E402
from covidcxr.training.engine import fit  # noqa: E402
from covidcxr.utils.seed import seed_everything  # noqa: E402

ARCHITECTURES = {"baseline_cnn": BaselineCNN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    seed_everything(config.train.seed)

    run_dir = Path("runs") / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    bundle = build_dataloaders(config.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = ARCHITECTURES[config.model.architecture]
    model = model_cls(
        in_channels=config.model.in_channels,
        channels=tuple(config.model.channels),
        dropout=config.model.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=bundle.class_weights.to(device) if bundle.class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    print(f"Experiment '{config.name}': {model.num_parameters():,} parameters, device={device}")

    result = fit(
        model,
        bundle.train_loader,
        bundle.val_loader,
        optimizer,
        criterion,
        device,
        max_epochs=config.train.max_epochs,
        patience=config.train.patience,
        checkpoint_path=run_dir / "best.pt",
    )

    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(result.history)

    print(f"Best epoch: {result.best_epoch} (val_loss={result.best_val_loss:.4f}), "
          f"stopped_early={result.stopped_early}, ran {len(result.history)} epoch(s)")
    print(f"Artifacts written to {run_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
