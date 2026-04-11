"""Unified multi-task model initialized from saved task checkpoints."""

import torch
import torch.nn as nn

from .checkpoints import CLASSIFIER_CHECKPOINT, LOCALIZER_CHECKPOINT, UNET_CHECKPOINT, read_state_dict
from .classification import ClassificationHead
from .localization import BoundingBoxHead
from .segmentation import UNetDecoder
from .vgg11 import VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
        load_checkpoints: bool = False,
        image_space_output: bool = True,
    ):
        super().__init__()
        self.image_space_output = image_space_output
        
        # ONE Shared Backbone Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        
        # Task Heads
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
        """Load the shared encoder from classifier and task heads from relative checkpoints."""
        # Base the shared encoder on the classification pretrained weights
        classifier_state = read_state_dict(classifier_path, device=device)
        encoder_weights = {k[len("encoder.") :]: v for k, v in classifier_state.items() if k.startswith("encoder.")}
        classifier_weights = {k[len("head.") :]: v for k, v in classifier_state.items() if k.startswith("head.")}
        self.encoder.load_state_dict(encoder_weights, strict=True)
        self.classification_head.load_state_dict(classifier_weights, strict=True)

        # Load localization head
        localizer_state = read_state_dict(localizer_path, device=device)
        localizer_weights = {k[len("head.") :]: v for k, v in localizer_state.items() if k.startswith("head.")}
        self.localization_head.load_state_dict(localizer_weights, strict=True)

        # Load segmentation head
        unet_state = read_state_dict(unet_path, device=device)
        unet_weights = {k[len("decoder.") :]: v for k, v in unet_state.items() if k.startswith("decoder.")}
        self.segmentation_head.load_state_dict(unet_weights, strict=True)

    def forward(self, x: torch.Tensor):
        """Return outputs for all three tasks in a single forward pass."""
        # Single forward pass through the shared backbone
        shared_bottleneck, shared_features = self.encoder(x, return_features=True)
        
        # Localization output
        localization = self.localization_head(shared_bottleneck)
        if self.image_space_output:
            box_scale = localization.new_tensor([x.shape[-1], x.shape[-2], x.shape[-1], x.shape[-2]])
            localization = localization * box_scale
            
        return {
            "classification": self.classification_head(shared_bottleneck),
            "localization": localization,
            "segmentation": self.segmentation_head(shared_bottleneck, shared_features),
        }
