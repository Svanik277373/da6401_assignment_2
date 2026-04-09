"""Classification components."""

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class ClassificationHead(nn.Module):
    """VGG-style classifier head."""

    def __init__(
        self,
        in_channels: int = 512,
        num_classes: int = 37,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = [nn.Linear(in_channels * 7 * 7, 4096)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(4096))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                CustomDropout(dropout_p),
                nn.Linear(4096, 4096),
            ]
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(4096))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                CustomDropout(dropout_p),
                nn.Linear(4096, num_classes),
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        self.head = ClassificationHead(
            in_channels=self.encoder.output_channels,
            num_classes=num_classes,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return breed classification logits."""
        return self.head(self.encoder(x))
