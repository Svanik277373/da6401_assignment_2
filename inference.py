"""Inference and quick inspection utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import MultiTaskPerceptionModel, VGG11Classifier, VGG11Localizer, VGG11UNet


def load_image(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.task).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    image = load_image(Path(args.image_path), args.image_size).to(device)
    with torch.no_grad():
        outputs = model(image)

    if args.task == "classification":
        print({"predicted_breed_label": outputs.argmax(dim=1).item()})
        
    elif args.task == "localization":
        # FIX: Scale the normalized [0, 1] output by the image size (e.g., 224)
        raw_bbox = outputs.squeeze(0).cpu()
        scaled_bbox = (raw_bbox * args.image_size).tolist()
        print({"predicted_bbox_xywh": scaled_bbox})
        
    elif args.task == "segmentation":
        mask = outputs.argmax(dim=1).squeeze(0).cpu()
        print({"predicted_mask_shape": tuple(mask.shape), "unique_labels": torch.unique(mask).tolist()})
        
    else:
        # FIX: Scale the multitask localization output as well
        raw_bbox = outputs["localization"].squeeze(0).cpu()
        scaled_bbox = (raw_bbox * args.image_size).tolist()
        
        print(
            {
                "predicted_breed_label": outputs["classification"].argmax(dim=1).item(),
                "predicted_bbox_xywh": scaled_bbox,
                "predicted_mask_shape": tuple(outputs["segmentation"].argmax(dim=1).squeeze(0).cpu().shape),
            }
        )


if __name__ == "__main__":
    main()
