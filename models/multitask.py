"""Unified multi-task model initialized from saved task checkpoints."""

import os

import gdown
import torch
import torch.nn as nn

from .classification import ClassificationHead
from .localization import BoundingBoxHead
from .segmentation import UNetDecoder
from .vgg11 import VGG11Encoder

CLASSIFIER_CHECKPOINT = "checkpoints/classifier.pth"
LOCALIZER_CHECKPOINT = "checkpoints/localizer.pth"
UNET_CHECKPOINT = "checkpoints/unet.pth"

CHECKPOINT_DOWNLOADS = {
    CLASSIFIER_CHECKPOINT: "16iMeSR3wC1B4y1P7tT_1yVCoKXXzlK0O",
    LOCALIZER_CHECKPOINT: "12XZtccwznbwS4NiPDG4wKGWWzMWe2vZN",
    UNET_CHECKPOINT: "17wdzjIDlZG-o9CtzCEXZgXpC3iYw7Wun",
}


def _download_checkpoint_if_missing(checkpoint_path: str) -> None:
    if os.path.exists(checkpoint_path):
        return
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    gdown.download(id=CHECKPOINT_DOWNLOADS[checkpoint_path], output=checkpoint_path, quiet=False)


def _read_state_dict(checkpoint_path: str, device: torch.device | str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
        load_checkpoints: bool = True,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        self.classification_encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        self.segmentation_encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.output_channels,
            num_classes=num_breeds,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
        )
        self.localization_head = BoundingBoxHead(in_channels=self.encoder.output_channels)
        self.segmentation_head = UNetDecoder(num_classes=seg_classes, use_batchnorm=use_batchnorm)

        if load_checkpoints:
            self.load_task_checkpoints()

    def load_task_checkpoints(
        self,
        classifier_path: str = CLASSIFIER_CHECKPOINT,
        localizer_path: str = LOCALIZER_CHECKPOINT,
        unet_path: str = UNET_CHECKPOINT,
        device: torch.device | str = "cpu",
    ) -> None:
        """Load the shared encoder and task heads from saved relative checkpoints."""
        for checkpoint_path in (classifier_path, localizer_path, unet_path):
            _download_checkpoint_if_missing(checkpoint_path)

        localizer_state = _read_state_dict(localizer_path, device=device)
        encoder_weights = {k[len("encoder.") :]: v for k, v in localizer_state.items() if k.startswith("encoder.")}
        localizer_weights = {k[len("head.") :]: v for k, v in localizer_state.items() if k.startswith("head.")}
        self.encoder.load_state_dict(encoder_weights, strict=True)
        self.localization_head.load_state_dict(localizer_weights, strict=True)

        classifier_state = _read_state_dict(classifier_path, device=device)
        classifier_encoder_weights = {k[len("encoder.") :]: v for k, v in classifier_state.items() if k.startswith("encoder.")}
        classifier_weights = {k[len("head.") :]: v for k, v in classifier_state.items() if k.startswith("head.")}
        self.classification_encoder.load_state_dict(classifier_encoder_weights, strict=True)
        self.classification_head.load_state_dict(classifier_weights, strict=True)

        unet_state = _read_state_dict(unet_path, device=device)
        segmentation_encoder_weights = {k[len("encoder.") :]: v for k, v in unet_state.items() if k.startswith("encoder.")}
        unet_weights = {k[len("decoder.") :]: v for k, v in unet_state.items() if k.startswith("decoder.")}
        self.segmentation_encoder.load_state_dict(segmentation_encoder_weights, strict=True)
        self.segmentation_head.load_state_dict(unet_weights, strict=True)

    def forward(self, x: torch.Tensor):
        """Return outputs for all three tasks in a single forward pass."""
        classification_bottleneck = self.classification_encoder(x)
        localization_bottleneck = self.encoder(x)
        segmentation_bottleneck, segmentation_features = self.segmentation_encoder(x, return_features=True)
        return {
            "classification": self.classification_head(classification_bottleneck),
            "localization": self.localization_head(localization_bottleneck),
            "segmentation": self.segmentation_head(segmentation_bottleneck, segmentation_features),
        }
