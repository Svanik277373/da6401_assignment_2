"""Question 2.3: compare transfer learning strategies for segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import build_args, train_once


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="oxford-iiit-pet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    for strategy in ("strict", "partial", "full"):
        args = build_args(
            task="segmentation",
            data_root=cli.data_root,
            epochs=cli.epochs,
            batch_size=cli.batch_size,
            lr=cli.lr,
            image_size=cli.image_size,
            wandb_project=cli.project,
            wandb_run_name=f"q2_3_{strategy}",
            disable_wandb=cli.disable_wandb,
        )
        train_once(args, freeze_strategy=strategy)


if __name__ == "__main__":
    main()
