"""VGG11 encoder."""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with BatchNorm and skip feature support."""

    def __init__(self, in_channels: int = 3, use_batchnorm: bool = True):
        super().__init__()
        self.output_channels = 512
        self.feature_channels = {
            "block1": 64,
            "block2": 128,
            "block3": 256,
            "block4": 512,
            "block5": 512,
        }

        def conv_block(in_ch: int, out_ch: int, num_convs: int) -> nn.Sequential:
            layers = []
            current_in = in_ch
            for _ in range(num_convs):
                layers.append(nn.Conv2d(current_in, out_ch, kernel_size=3, padding=1))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                current_in = out_ch
            return nn.Sequential(*layers)

        self.block1 = conv_block(in_channels, 64, 1)
        self.block2 = conv_block(64, 128, 1)
        self.block3 = conv_block(128, 256, 2)
        self.block4 = conv_block(256, 512, 2)
        self.block5 = conv_block(512, 512, 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run the encoder and optionally return skip tensors."""
        features: Dict[str, torch.Tensor] = {}

        x = self.block1(x)
        features["block1"] = x
        x = self.pool(x)

        x = self.block2(x)
        features["block2"] = x
        x = self.pool(x)

        x = self.block3(x)
        features["block3"] = x
        x = self.pool(x)

        x = self.block4(x)
        features["block4"] = x
        x = self.pool(x)

        x = self.block5(x)
        features["block5"] = x
        x = self.pool(x)

        if return_features:
            return x, features
        return x
