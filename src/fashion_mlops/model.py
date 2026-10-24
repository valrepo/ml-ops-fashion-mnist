"""LeNet-5-style CNN for 1x28x28 grayscale images."""

from __future__ import annotations

import torch
from torch import nn


class LeNet5(nn.Module):
    """LeNet-5 adapted to 28x28 inputs (first conv padded to keep the 32x32 geometry).

    conv(1->6, 5x5, pad 2) -> ReLU -> maxpool 2   : 6x14x14
    conv(6->16, 5x5)       -> ReLU -> maxpool 2   : 16x5x5
    fc 400 -> 120 -> 84 -> num_classes
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
