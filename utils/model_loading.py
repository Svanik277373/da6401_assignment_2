"""Checkpoint loading helpers for assignment workflows."""

from __future__ import annotations

from pathlib import Path

import torch


def _read_state_dict(checkpoint_path: str | Path, device: torch.device | str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device | str = "cpu") -> None:
    """Load a full checkpoint strictly."""
    state_dict = _read_state_dict(checkpoint_path, device=device)
    model.load_state_dict(state_dict, strict=True)


def load_encoder_from_checkpoint(
    encoder: torch.nn.Module,
    checkpoint_path: str | Path,
    key_prefixes: tuple[str, ...] = ("encoder.",),
    device: torch.device | str = "cpu",
) -> None:
    """Load encoder weights from a model checkpoint into a standalone encoder."""
    source_state = _read_state_dict(checkpoint_path, device=device)
    encoder_state = encoder.state_dict()
    filtered = {}

    for key, value in source_state.items():
        matched = False
        for prefix in key_prefixes:
            if key.startswith(prefix):
                stripped = key[len(prefix) :]
                if stripped in encoder_state:
                    filtered[stripped] = value
                matched = True
                break
        if matched:
            continue
        if key in encoder_state:
            filtered[key] = value

    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys while loading {checkpoint_path}: {unexpected}")
    if len(filtered) == 0:
        raise RuntimeError(f"No encoder weights matched in checkpoint: {checkpoint_path}")


def initialize_multitask_from_task_checkpoints(
    model,
    classifier_checkpoint: str | Path | None = None,
    localizer_checkpoint: str | Path | None = None,
    segmentation_checkpoint: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> None:
    """Initialize multitask model from individual task checkpoints."""
    if classifier_checkpoint:
        classifier_state = _read_state_dict(classifier_checkpoint, device=device)
        encoder_weights = {k[len("encoder.") :]: v for k, v in classifier_state.items() if k.startswith("encoder.")}
        head_weights = {k[len("head.") :]: v for k, v in classifier_state.items() if k.startswith("head.")}
        model.encoder.load_state_dict(encoder_weights, strict=False)
        model.classification_head.load_state_dict(head_weights, strict=False)

    if localizer_checkpoint:
        localizer_state = _read_state_dict(localizer_checkpoint, device=device)
        head_weights = {k[len("head.") :]: v for k, v in localizer_state.items() if k.startswith("head.")}
        if head_weights:
            model.localization_head.load_state_dict(head_weights, strict=False)

    if segmentation_checkpoint:
        segmentation_state = _read_state_dict(segmentation_checkpoint, device=device)
        decoder_weights = {k[len("decoder.") :]: v for k, v in segmentation_state.items() if k.startswith("decoder.")}
        if decoder_weights:
            model.segmentation_head.load_state_dict(decoder_weights, strict=False)
