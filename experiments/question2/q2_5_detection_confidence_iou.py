"""Question 2.5: log predicted boxes, IoU, and a confidence proxy to W&B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import build_args, first_batch, load_model, make_dataloaders, make_wandb_image_from_bbox, wandb
from train import box_iou_mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=["multitask", "localization"], default="multitask")
    parser.add_argument("--data-root", default="D:/oxford-iiit-pet")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.task, args.checkpoint)
    device = next(model.parameters()).device
    _, val_loader = make_dataloaders(build_args(data_root=args.data_root, image_size=args.image_size, batch_size=args.batch_size))
    batch = first_batch(val_loader)

    with torch.no_grad():
        outputs = model(batch["image"].to(device))

    if args.task == "multitask":
        pred_boxes = outputs["localization"].cpu()
        confidence = torch.softmax(outputs["classification"].cpu(), dim=1).max(dim=1).values
    else:
        pred_boxes = outputs.cpu()
        confidence = torch.ones(pred_boxes.size(0))

    target_boxes = batch["bbox"]
    table_rows = []
    for idx in range(min(10, pred_boxes.size(0))):
        iou_value = float(box_iou_mean(pred_boxes[idx : idx + 1], target_boxes[idx : idx + 1]).item())
        image = make_wandb_image_from_bbox(
            batch["image"][idx],
            gt_box=target_boxes[idx],
            pred_box=pred_boxes[idx],
            caption=f"{batch['image_id'][idx]} | conf={confidence[idx]:.3f} | iou={iou_value:.3f}",
        )
        table_rows.append([batch["image_id"][idx], image, float(confidence[idx].item()), iou_value])

    if wandb is not None and not args.disable_wandb:
        run = wandb.init(project=args.project, name="q2_5_detection_confidence_iou")
        table = wandb.Table(columns=["image_id", "overlay", "confidence", "iou"], data=table_rows)
        wandb.log({"detection_examples": table})
        run.finish()


if __name__ == "__main__":
    main()
