"""Localization modules."""

import torch
import torch.nn as nn

from .checkpoints import LOCALIZER_CHECKPOINT, read_state_dict
from .vgg11 import VGG11Encoder


class BoundingBoxHead(nn.Module):
    """Regression head for normalized center-format boxes."""

    def __init__(self, in_channels: int = 512):
        super().__init__()
        # FIX: Preserve spatial dimensions (7x7 instead of 1x1) to retain location data
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            # FIX: Input features are now 512 channels * 7 height * 7 width
            nn.Linear(in_channels * 7 * 7, 512),
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

    def __init__(
        self,
        in_channels: int = 3,
        use_batchnorm: bool = True,
        freeze_encoder: bool = False,
        load_checkpoint: bool = True,
        image_space_output: bool = True,
    ):
        super().__init__()
        self.image_space_output = image_space_output
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        self.head = BoundingBoxHead(in_channels=self.encoder.output_channels)
        if load_checkpoint:
            self.load_state_dict(read_state_dict(LOCALIZER_CHECKPOINT), strict=True)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return image-space boxes in (x_center, y_center, width, height) format."""
        boxes = self.head(self.encoder(x))
        if not self.image_space_output:
            return boxes
        scale = boxes.new_tensor([x.shape[-1], x.shape[-2], x.shape[-1], x.shape[-2]])
        return boxes * scale
