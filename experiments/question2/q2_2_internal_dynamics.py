"""Question 2.2: compare dropout settings for classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import build_args, train_once


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="oxford-iiit-pet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    for dropout in (0.0, 0.2, 0.5):
        args = build_args(
            task="classification",
            data_root=cli.data_root,
            epochs=cli.epochs,
            batch_size=cli.batch_size,
            lr=cli.lr,
            image_size=cli.image_size,
            dropout=dropout,
            wandb_project=cli.project,
            wandb_run_name=f"q2_2_dropout_{str(dropout).replace('.', '_')}",
            disable_wandb=cli.disable_wandb,
        )
        train_once(args)


if __name__ == "__main__":
    main()
