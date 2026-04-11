"""Question 2.1: compare BatchNorm vs no BatchNorm and log activations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import build_args, first_batch, make_dataloaders, make_device, named_conv_layers, train_once, wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="oxford-iiit-pet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def capture_third_conv_activations(model, images: torch.Tensor):
    conv_layers = named_conv_layers(model)
    _, layer = conv_layers[2]
    captured = {}

    def hook(_, __, output):
        captured["activations"] = output.detach().cpu()

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(images)
    finally:
        handle.remove()
    return captured["activations"]


def main():
    cli = parse_args()
    base = dict(
        task="classification",
        data_root=cli.data_root,
        epochs=cli.epochs,
        batch_size=cli.batch_size,
        lr=cli.lr,
        image_size=cli.image_size,
        wandb_project=cli.project,
        disable_wandb=cli.disable_wandb,
    )

    args_bn = build_args(**base, wandb_run_name="q2_1_batchnorm_on")
    model_bn, _ = train_once(args_bn)
    args_no_bn = build_args(**base, disable_batchnorm=True, wandb_run_name="q2_1_batchnorm_off")
    model_no_bn, _ = train_once(args_no_bn)

    _, val_loader = make_dataloaders(args_bn)
    batch = first_batch(val_loader)
    images = batch["image"].to(make_device())
    model_bn.eval()
    model_no_bn.eval()

    act_bn = capture_third_conv_activations(model_bn, images)
    act_no_bn = capture_third_conv_activations(model_no_bn, images)

    if wandb is not None and not cli.disable_wandb:
        run = wandb.init(project=cli.project, name="q2_1_activation_analysis")
        wandb.log(
            {
                "batchnorm_on/third_conv_activation_hist": wandb.Histogram(act_bn.flatten().numpy()),
                "batchnorm_off/third_conv_activation_hist": wandb.Histogram(act_no_bn.flatten().numpy()),
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
