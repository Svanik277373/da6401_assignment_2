"""Question 2.7: run the final pipeline on novel pet images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import (
    colorize_mask,
    load_model,
    load_single_image,
    make_wandb_image_from_bbox,
    overlay_mask,
    to_display_image,
    wandb,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def apply_tta(model: torch.nn.Module, image: torch.Tensor) -> dict:
    """Applies Horizontal Flip Test-Time Augmentation (TTA) for the multitask model."""
    # Create the horizontally flipped version of the input image
    flipped_image = torch.flip(image, dims=[3])
    
    with torch.no_grad():
        out_orig = model(image)
        out_flip = model(flipped_image)
        
    # Average the classification logits
    cls_merged = (out_orig["classification"] + out_flip["classification"]) / 2.0
    
    # Bounding boxes are normalized [0.0, 1.0]. Un-flip the x_center coordinate
    bbox_flip_unflipped = out_flip["localization"].clone()
    bbox_flip_unflipped[:, 0] = 1.0 - out_flip["localization"][:, 0]
    loc_merged = (out_orig["localization"] + bbox_flip_unflipped) / 2.0
    
    # Spatially un-flip the predicted mask logits
    seg_flip_unflipped = torch.flip(out_flip["segmentation"], dims=[3])
    seg_merged = (out_orig["segmentation"] + seg_flip_unflipped) / 2.0
    
    return {
        "classification": cls_merged,
        "localization": loc_merged,
        "segmentation": seg_merged
    }


def main():
    args = parse_args()
    model = load_model("multitask", args.checkpoint)
    device = next(model.parameters()).device
    image_paths = sorted([p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:3]

    if wandb is not None and not args.disable_wandb:
        run = wandb.init(project=args.project, name="q2_7_pipeline_showcase")
        log_payload = {}
        for image_path in image_paths:
            image = load_single_image(image_path, args.image_size).unsqueeze(0).to(device)
            
            # Apply Test-Time Augmentation (TTA) instead of a single forward pass
            outputs = apply_tta(model, image)
            
            pred_box = outputs["localization"][0].cpu()
            pred_mask = outputs["segmentation"].argmax(dim=1)[0].cpu()
            pred_label = int(outputs["classification"].argmax(dim=1).item())

            log_payload[f"{image_path.stem}_original"] = wandb.Image(
                to_display_image(image[0].cpu()), caption=f"{image_path.name} | original"
            )
            log_payload[f"{image_path.stem}_bbox"] = make_wandb_image_from_bbox(
                image[0].cpu(), pred_box=pred_box, caption=f"{image_path.name} | breed={pred_label}"
            )
            log_payload[f"{image_path.stem}_mask_overlay"] = wandb.Image(
                overlay_mask(image[0].cpu(), pred_mask), caption=f"{image_path.name} | overlay"
            )
            log_payload[f"{image_path.stem}_mask_trimap"] = wandb.Image(
                colorize_mask(pred_mask), caption=f"{image_path.name} | predicted trimap"
            )
            
        wandb.log(log_payload)
        run.finish()


if __name__ == "__main__":
    main()
