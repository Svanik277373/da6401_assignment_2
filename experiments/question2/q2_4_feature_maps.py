"""Question 2.4: visualize early and late feature maps for a dog image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import feature_maps_to_images, load_model, load_single_image, named_conv_layers, wandb

dataset_path = Path(r"D:\oxford-iiit-pet")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/classifier.pth")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def resolve_image_path(image_path: str) -> Path:
    if image_path:
        candidate = Path(image_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Image not found: {candidate}")
        return candidate

    candidate_dirs = [
        dataset_path / "images",
        dataset_path / "images" / "images",
    ]

    for images_dir in candidate_dirs:
        if not images_dir.exists():
            continue

        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            matches = sorted(images_dir.glob(pattern))
            if matches:
                return matches[0]

    searched = ", ".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(f"No image files were found under any of: {searched}")


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
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_path = resolve_image_path(args.image_path)
    device = torch.device("cpu")
    model = load_model("classification", args.checkpoint, device=device)
    model = model.to(device)
    model.eval()
    image = load_single_image(image_path, args.image_size).unsqueeze(0).to(device)
    features = capture_features(model, image)

    if wandb is not None and not args.disable_wandb:
        run = wandb.init(project=args.project, name="q2_4_feature_maps")
        image_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_np = image_np * std + mean
        image_np = image_np.clip(0, 1)
        wandb.log(
            {
                "input_image": wandb.Image(image_np, caption=f"Input image: {image_path.name}"),
                "first_conv_feature_maps": [
                    wandb.Image(arr, caption=f"First conv map {idx + 1}")
                    for idx, arr in enumerate(feature_maps_to_images(features["first"][0]))
                ],
                "last_conv_feature_maps": [
                    wandb.Image(arr, caption=f"Last conv map {idx + 1}")
                    for idx, arr in enumerate(feature_maps_to_images(features["last"][0]))
                ],
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
