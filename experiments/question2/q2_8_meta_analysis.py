"""Question 2.8: aggregate validation metrics for the final multitask checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import build_args, load_model, make_dataloaders, make_device, wandb
from train import build_criteria, train_or_eval_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="oxford-iiit-pet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = make_device()
    model = load_model("multitask", args.checkpoint, device=device)
    _, val_loader = make_dataloaders(build_args(data_root=args.data_root, image_size=args.image_size, batch_size=args.batch_size, task="multitask"))
    criteria = build_criteria("multitask")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = train_or_eval_epoch(model, val_loader, optimizer, criteria, device, "multitask", train=False)

    if wandb is not None and not args.disable_wandb:
        run = wandb.init(project=args.project, name="q2_8_meta_analysis")
        wandb.log({f"summary/{key}": value for key, value in metrics.items()})
        wandb.summary["reflection_prompt"] = (
            "Discuss dropout and BatchNorm placement, encoder freezing strategy, segmentation loss choice, "
            "and any task interference observed in the shared backbone."
        )
        run.finish()


if __name__ == "__main__":
    main()
