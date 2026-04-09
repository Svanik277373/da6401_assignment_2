"""Segmentation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11 import VGG11Encoder


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    """Symmetric decoder for U-Net style segmentation."""

    def __init__(self, num_classes: int = 3, use_batchnorm: bool = True):
        super().__init__()
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(1024, 512, use_batchnorm=use_batchnorm)
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 256, use_batchnorm=use_batchnorm)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 128, use_batchnorm=use_batchnorm)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 64, use_batchnorm=use_batchnorm)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, use_batchnorm=use_batchnorm)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _align(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        diff_y = reference.size(2) - x.size(2)
        diff_x = reference.size(3) - x.size(3)
        if diff_x == 0 and diff_y == 0:
            return x
        return F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

    def forward(self, bottleneck: torch.Tensor, features) -> torch.Tensor:
        x = self.up5(bottleneck)
        x = self._align(x, features["block5"])
        x = self.dec5(torch.cat([x, features["block5"]], dim=1))

        x = self.up4(x)
        x = self._align(x, features["block4"])
        x = self.dec4(torch.cat([x, features["block4"]], dim=1))

        x = self.up3(x)
        x = self._align(x, features["block3"])
        x = self.dec3(torch.cat([x, features["block3"]], dim=1))

        x = self.up2(x)
        x = self._align(x, features["block2"])
        x = self.dec2(torch.cat([x, features["block2"]], dim=1))

        x = self.up1(x)
        x = self._align(x, features["block1"])
        x = self.dec1(torch.cat([x, features["block1"]], dim=1))
        return self.head(x)

class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, use_batchnorm: bool = True):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        self.decoder = UNetDecoder(num_classes=num_classes, use_batchnorm=use_batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return segmentation logits."""
        bottleneck, features = self.encoder(x, return_features=True)
        return self.decoder(bottleneck, features)
