"""Question 2.7: run the final pipeline on novel pet images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import load_model, load_single_image, make_wandb_image_from_bbox, overlay_mask, wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


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
            with torch.no_grad():
                outputs = model(image)
            pred_box = outputs["localization"][0].cpu()
            pred_mask = outputs["segmentation"].argmax(dim=1)[0].cpu()
            pred_label = int(outputs["classification"].argmax(dim=1).item())
            log_payload[f"{image_path.stem}_bbox"] = make_wandb_image_from_bbox(image[0].cpu(), pred_box=pred_box, caption=f"{image_path.name} | breed={pred_label}")
            log_payload[f"{image_path.stem}_mask"] = wandb.Image(overlay_mask(image[0].cpu(), pred_mask), caption=image_path.name)
        wandb.log(log_payload)
        run.finish()


if __name__ == "__main__":
    main()
