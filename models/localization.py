"""Localization modules."""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder


class BoundingBoxHead(nn.Module):
    """Regression head for normalized center-format boxes."""

    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.pool(x))


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, use_batchnorm: bool = True, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        self.head = BoundingBoxHead(in_channels=self.encoder.output_channels)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized boxes in (x_center, y_center, width, height) format."""
        return self.head(self.encoder(x))
