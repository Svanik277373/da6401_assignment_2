"""Question 2.4: visualize early and late feature maps for a dog image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import feature_maps_to_images, load_model, load_single_image, named_conv_layers, wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def capture_features(model, image):
    convs = named_conv_layers(model)
    targets = {"first": convs[0][1], "last": convs[-1][1]}
    captured = {}
    handles = []
    for name, layer in targets.items():
        handles.append(layer.register_forward_hook(lambda _, __, output, key=name: captured.setdefault(key, output.detach().cpu())))
    try:
        with torch.no_grad():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def main():
    args = parse_args()
    model = load_model("classification", args.checkpoint)
    device = next(model.parameters()).device
    image = load_single_image(args.image_path, args.image_size).unsqueeze(0).to(device)
    features = capture_features(model, image)

    if wandb is not None and not args.disable_wandb:
        run = wandb.init(project=args.project, name="q2_4_feature_maps")
        wandb.log(
            {
                "first_conv_feature_maps": [wandb.Image(arr) for arr in feature_maps_to_images(features["first"][0])],
                "last_conv_feature_maps": [wandb.Image(arr) for arr in feature_maps_to_images(features["last"][0])],
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
