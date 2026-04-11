"""Checkpoint paths and loading utilities for submission models."""

from __future__ import annotations

import os

import gdown
import torch

CLASSIFIER_CHECKPOINT = "checkpoints/classifier.pth"
LOCALIZER_CHECKPOINT = "checkpoints/localizer.pth"
UNET_CHECKPOINT = "checkpoints/unet.pth"

CHECKPOINT_DOWNLOADS = {
    CLASSIFIER_CHECKPOINT: "1gLILfif9GJDCR-eEqiMok4eg6hpC3bEb",
    LOCALIZER_CHECKPOINT: "12XZtccwznbwS4NiPDG4wKGWWzMWe2vZN",
    UNET_CHECKPOINT: "17wdzjIDlZG-o9CtzCEXZgXpC3iYw7Wun",
}


def download_checkpoint_if_missing(checkpoint_path: str) -> None:
    if os.path.exists(checkpoint_path):
        return
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    gdown.download(id=CHECKPOINT_DOWNLOADS[checkpoint_path], output=checkpoint_path, quiet=False)


def read_state_dict(checkpoint_path: str, device: torch.device | str = "cpu"):
    download_checkpoint_if_missing(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint
