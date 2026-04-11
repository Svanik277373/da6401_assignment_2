"""Question 2.6: compare Dice vs pixel accuracy across segmentation checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import (
    build_args,
    evaluate_segmentation_checkpoint,
    load_model,
    make_dataloaders,
    overlay_mask,
    pixel_accuracy_from_logits,
    train_segmentation_for_epochs,
    to_display_image,
    wandb,
)
from train import dice_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-checkpoint", required=True, help="Final or best segmentation checkpoint")
    parser.add_argument("--early-checkpoint", help="Optional early-epoch checkpoint, e.g. epoch 5")
    parser.add_argument("--data-root", default="oxford-iiit-pet")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-batchnorm", action="store_true")
    parser.add_argument("--train-early-epochs", type=int, default=5)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()

def make_example_triplets(examples, stage_name: str):
    payload = {}
    for idx, example in enumerate(examples, start=1):
        image_id = example["image_id"]
        prefix = f"{stage_name}/example_{idx}_{image_id}"
        payload[f"{prefix}_original"] = wandb.Image(
            to_display_image(example["image"]),
            caption=f"{stage_name} | {image_id} | original",
        )
        payload[f"{prefix}_ground_truth"] = wandb.Image(
            overlay_mask(example["image"], example["target"]),
            caption=f"{stage_name} | {image_id} | ground truth",
        )
        payload[f"{prefix}_prediction"] = wandb.Image(
            overlay_mask(example["image"], example["pred"]),
            caption=f"{stage_name} | {image_id} | prediction",
        )
    return payload


def main():
    args = parse_args()
    final_checkpoint_path = Path(args.final_checkpoint)
    if not final_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {final_checkpoint_path}")

    device = torch.device("cpu")
    _, val_loader = make_dataloaders(
        build_args(
            task="segmentation",
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )
    )

    run = None
    if wandb is not None and not args.disable_wandb:
        run = wandb.init(project=args.project, name="q2_6_dice_vs_pixel_accuracy")

    results = {}
    if args.early_checkpoint:
        early_checkpoint_path = Path(args.early_checkpoint)
        if not early_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {early_checkpoint_path}")
        early_model = load_model("segmentation", early_checkpoint_path, device=device)
        early_model = early_model.to(device)
        early_model.eval()
        early_eval = evaluate_segmentation_checkpoint(early_model, val_loader, device)
        results["early"] = {
            "dice_score": float(dice_score(early_eval["logits"], early_eval["targets"]).item()),
            "pixel_accuracy": pixel_accuracy_from_logits(early_eval["logits"], early_eval["targets"]),
            "examples": early_eval["examples"],
        }
    elif args.train_early_epochs > 0:
        early_model, early_history = train_segmentation_for_epochs(args, args.train_early_epochs)
        early_model = early_model.to(device)
        early_model.eval()
        early_eval = evaluate_segmentation_checkpoint(early_model, val_loader, device)
        results["early"] = {
            "dice_score": float(dice_score(early_eval["logits"], early_eval["targets"]).item()),
            "pixel_accuracy": pixel_accuracy_from_logits(early_eval["logits"], early_eval["targets"]),
            "examples": early_eval["examples"],
        }
        if run is not None:
            for entry in early_history:
                wandb.log(
                    {
                        "early_train/epoch": entry["epoch"],
                        "early_train/train_loss": entry["train_loss"],
                        "early_train/train_dice": entry["train_dice"],
                        "early_train/val_loss": entry["val_loss"],
                        "early_train/val_dice_score": entry["val_dice"],
                        "early_train/val_pixel_accuracy": entry["val_pixel_accuracy"],
                    }
                )

    final_model = load_model("segmentation", final_checkpoint_path, device=device)
    final_model = final_model.to(device)
    final_model.eval()
    final_eval = evaluate_segmentation_checkpoint(final_model, val_loader, device)
    results["final"] = {
        "dice_score": float(dice_score(final_eval["logits"], final_eval["targets"]).item()),
        "pixel_accuracy": pixel_accuracy_from_logits(final_eval["logits"], final_eval["targets"]),
        "examples": final_eval["examples"],
    }

    if run is not None:
        log_payload = {}
        for stage_name, result in results.items():
            log_payload[f"{stage_name}/pixel_accuracy"] = result["pixel_accuracy"]
            log_payload[f"{stage_name}/dice_score"] = result["dice_score"]
            log_payload.update(make_example_triplets(result["examples"], stage_name))
        if "early" in results:
            log_payload["comparison/pixel_accuracy_drop_to_dice_gap_early"] = (
                results["early"]["pixel_accuracy"] - results["early"]["dice_score"]
            )
        log_payload["comparison/pixel_accuracy_drop_to_dice_gap_final"] = (
            results["final"]["pixel_accuracy"] - results["final"]["dice_score"]
        )
        wandb.log(log_payload)
        run.finish()

    for stage_name, result in results.items():
        print(
            f"{stage_name}: "
            f"pixel_accuracy={result['pixel_accuracy']:.4f}, "
            f"dice_score={result['dice_score']:.4f}"
        )


if __name__ == "__main__":
    main()
