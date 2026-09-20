"""Experiment configuration: one YAML file per experiment, no hardcoded
hyperparameters in training code. Config-driven runs are what make
`runs/<exp_name>/config.yaml` (saved alongside results) a complete record
of exactly what produced a given number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from covidcxr.data.pipeline import DataConfig


@dataclass
class ModelConfig:
    architecture: str = "baseline_cnn"
    in_channels: int = 1
    channels: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.3


@dataclass
class TrainConfig:
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    seed: int = 42


@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_cfg = DataConfig(**raw["data"])
    model_cfg = ModelConfig(**raw.get("model", {}))
    train_cfg = TrainConfig(**raw.get("train", {}))
    return ExperimentConfig(name=raw["name"], data=data_cfg, model=model_cfg, train=train_cfg)
