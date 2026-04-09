"""Question 2.6: log segmentation examples and compare Dice vs pixel accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import build_args, first_batch, load_model, make_dataloaders, overlay_mask, wandb
from train import dice_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="oxford-iiit-pet")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def main():
    args = parse_args()
    model = load_model("segmentation", args.checkpoint)
    device = next(model.parameters()).device
    _, val_loader = make_dataloaders(build_args(data_root=args.data_root, image_size=args.image_size, batch_size=args.batch_size))
    batch = first_batch(val_loader)

    with torch.no_grad():
        logits = model(batch["image"].to(device)).cpu()

    dice = float(dice_score(logits, batch["segmentation_mask"]).item())
    acc = pixel_accuracy(logits, batch["segmentation_mask"])
    preds = logits.argmax(dim=1)

    if wandb is not None and not args.disable_wandb:
        run = wandb.init(project=args.project, name="q2_6_dice_vs_pixel_accuracy")
        images = []
        for idx in range(min(5, preds.size(0))):
            images.append(wandb.Image(overlay_mask(batch["image"][idx], preds[idx]), caption=f"{batch['image_id'][idx]} prediction"))
            images.append(wandb.Image(overlay_mask(batch["image"][idx], batch["segmentation_mask"][idx]), caption=f"{batch['image_id'][idx]} ground_truth"))
        wandb.log({"pixel_accuracy": acc, "dice_score": dice, "segmentation_examples": images})
        run.finish()


if __name__ == "__main__":
    main()
