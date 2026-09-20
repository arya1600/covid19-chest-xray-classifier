"""A small, from-scratch CNN baseline.

Same rough depth as the reference project's model (a handful of conv
blocks), but two deliberate architectural changes:

1. **BatchNorm after every conv**, not just Dropout — standard practice
   the reference project didn't use, and it stabilizes training on a
   small dataset.
2. **Global average pooling instead of Flatten**. The reference project's
   `Flatten()` on a 26x26x128 feature map fed a `Dense(64)` layer with
   ~5.5M parameters — the large majority of its entire parameter count,
   in one layer, for no architectural reason. `AdaptiveAvgPool2d(1)`
   collapses each feature map to a single number, so the classifier head
   is a few hundred parameters instead of millions, which matters more
   the smaller the dataset is (this one has ~1,800 images).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class BaselineCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for c in channels:
            blocks.append(ConvBlock(prev, c))
            prev = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
