import torch
import torch.nn as nn
from .layers import CustomDropout
from .vgg11 import VGG11Encoder

class ClassificationHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=37, dropout_p=0.5, use_batchnorm=True):
        super().__init__()
        layers = []
        for _ in range(2):
            layers.append(nn.Linear(in_channels * 7 * 7 if not layers else 4096, 4096))
            if use_batchnorm: layers.append(nn.BatchNorm1d(4096))
            layers.append(nn.ReLU(inplace=True))
            layers.append(CustomDropout(dropout_p))
        layers.append(nn.Linear(4096, num_classes))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(x), 1))

class VGG11Classifier(nn.Module):
    def __init__(self, num_classes=37, dropout_p=0.5, use_batchnorm=True):
        super().__init__()
        self.encoder = VGG11Encoder(use_batchnorm=use_batchnorm)
        self.head = ClassificationHead(num_classes=num_classes, dropout_p=dropout_p, use_batchnorm=use_batchnorm)
    def forward(self, x): return self.head(self.encoder(x))