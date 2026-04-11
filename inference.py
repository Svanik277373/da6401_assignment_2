"""Inference and quick inspection utilities with default TTA (Test-Time Augmentation)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models import MultiTaskPerceptionModel, VGG11Classifier, VGG11Localizer, VGG11UNet


def load_image(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    
    # Apply proper ImageNet Normalization to match pets_dataset.py
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    return transform(image).unsqueeze(0)


def build_model(task: str):
    if task == "classification":
        return VGG11Classifier()
    if task == "localization":
        return VGG11Localizer()
    if task == "segmentation":
        return VGG11UNet()
    return MultiTaskPerceptionModel()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--task", choices=["classification", "localization", "segmentation", "multitask"], default="multitask")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    return parser.parse_args()


def apply_tta(model: torch.nn.Module, image: torch.Tensor, task: str, image_size: int):
    """Applies Horizontal Flip Test-Time Augmentation (TTA)."""
    # Create the horizontally flipped version of the input image
    flipped_image = torch.flip(image, dims=[3])
    
    with torch.no_grad():
        out_orig = model(image)
        out_flip = model(flipped_image)
        
    if task == "classification":
        # Average the logits
        return (out_orig + out_flip) / 2.0
        
    elif task == "localization":
        # Bounding box format is [x_center, y_center, width, height] in pixel space
        bbox_flip_unflipped = out_flip.clone()
        # Un-flip the x_center coordinate
        bbox_flip_unflipped[:, 0] = image_size - out_flip[:, 0]
        return (out_orig + bbox_flip_unflipped) / 2.0
        
    elif task == "segmentation":
        # Spatially un-flip the predicted mask logits
        seg_flip_unflipped = torch.flip(out_flip, dims=[3])
        return (out_orig + seg_flip_unflipped) / 2.0
        
    else: # Multi-task
        cls_merged = (out_orig["classification"] + out_flip["classification"]) / 2.0
        
        bbox_flip_unflipped = out_flip["localization"].clone()
        bbox_flip_unflipped[:, 0] = image_size - out_flip["localization"][:, 0]
        loc_merged = (out_orig["localization"] + bbox_flip_unflipped) / 2.0
        
        seg_flip_unflipped = torch.flip(out_flip["segmentation"], dims=[3])
        seg_merged = (out_orig["segmentation"] + seg_flip_unflipped) / 2.0
        
        return {
            "classification": cls_merged,
            "localization": loc_merged,
            "segmentation": seg_merged
        }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.task).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    image = load_image(Path(args.image_path), args.image_size).to(device)
    
    # Run inference with or without TTA
    if not args.no_tta:
        outputs = apply_tta(model, image, args.task, args.image_size)
    else:
        with torch.no_grad():
            outputs = model(image)

    # Print outputs based on task
    if args.task == "classification":
        print({"predicted_breed_label": outputs.argmax(dim=1).item()})
        
    elif args.task == "localization":
        print({"predicted_bbox_xywh": outputs.squeeze(0).cpu().tolist()})
        
    elif args.task == "segmentation":
        mask = outputs.argmax(dim=1).squeeze(0).cpu()
        print({"predicted_mask_shape": tuple(mask.shape), "unique_labels": torch.unique(mask).tolist()})
        
    else:
        print(
            {
                "predicted_breed_label": outputs["classification"].argmax(dim=1).item(),
                "predicted_bbox_xywh": outputs["localization"].squeeze(0).cpu().tolist(),
                "predicted_mask_shape": tuple(outputs["segmentation"].argmax(dim=1).squeeze(0).cpu().shape),
            }
        )


if __name__ == "__main__":
    main()
