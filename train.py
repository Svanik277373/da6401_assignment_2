"""Training entrypoint for all assignment tasks."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pets_dataset import OxfordIIITPetDataset
from losses import IoULoss
from models import MultiTaskPerceptionModel, VGG11Classifier, VGG11Localizer, VGG11UNet
from utils import initialize_multitask_from_task_checkpoints, load_checkpoint_strict, load_encoder_from_checkpoint


def get_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    preds_fg = (preds == 0).float()
    targets_fg = (targets == 0).float()
    intersection = (preds_fg * targets_fg).sum(dim=(1, 2))
    union = preds_fg.sum(dim=(1, 2)) + targets_fg.sum(dim=(1, 2))
    return ((2.0 * intersection + eps) / (union + eps)).mean()


def box_iou_mean(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_xyxy = torch.cat(
        [pred_boxes[:, :2] - pred_boxes[:, 2:] * 0.5, pred_boxes[:, :2] + pred_boxes[:, 2:] * 0.5],
        dim=1,
    )
    target_xyxy = torch.cat(
        [target_boxes[:, :2] - target_boxes[:, 2:] * 0.5, target_boxes[:, :2] + target_boxes[:, 2:] * 0.5],
        dim=1,
    )
    inter_tl = torch.maximum(pred_xyxy[:, :2], target_xyxy[:, :2])
    inter_br = torch.minimum(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    inter_wh = (inter_br - inter_tl).clamp(min=0.0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0.0) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0.0)
    target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0.0) * (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=0.0)
    union = pred_area + target_area - inter
    return (inter / (union + eps)).mean()


def build_model(args):
    if args.task == "classification":
        return VGG11Classifier(
            num_classes=37,
            dropout_p=args.dropout,
            use_batchnorm=not args.disable_batchnorm,
        )
    if args.task == "localization":
        return VGG11Localizer(
            use_batchnorm=not args.disable_batchnorm,
            freeze_encoder=args.freeze_encoder,
        )
    if args.task == "segmentation":
        return VGG11UNet(num_classes=3, use_batchnorm=not args.disable_batchnorm)
    return MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        dropout_p=args.dropout,
        use_batchnorm=not args.disable_batchnorm,
    )


def maybe_initialize_model(model, args, device: torch.device) -> None:
    if args.init_from:
        load_checkpoint_strict(model, args.init_from, device=device)
        return

    if args.task in {"localization", "segmentation"} and args.encoder_checkpoint:
        load_encoder_from_checkpoint(model.encoder, args.encoder_checkpoint, device=device)
        return

    if args.task == "multitask":
        initialize_multitask_from_task_checkpoints(
            model,
            classifier_checkpoint=args.classifier_checkpoint,
            localizer_checkpoint=args.localizer_checkpoint,
            segmentation_checkpoint=args.segmentation_checkpoint,
            device=device,
        )


def build_criteria(task: str) -> Dict[str, nn.Module]:
    classification_loss = nn.CrossEntropyLoss()
    segmentation_loss = nn.CrossEntropyLoss()
    localization_loss = IoULoss()
    if task == "classification":
        return {"classification": classification_loss}
    if task == "localization":
        return {"localization": localization_loss}
    if task == "segmentation":
        return {"segmentation": segmentation_loss}
    return {
        "classification": classification_loss,
        "localization": localization_loss,
        "segmentation": segmentation_loss,
    }


def compute_losses(outputs, batch, criteria: Dict[str, nn.Module], task: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics: Dict[str, float] = {}
    if task == "classification":
        loss = criteria["classification"](outputs, batch["breed_label"])
        metrics["loss"] = float(loss.item())
        return loss, metrics
    if task == "localization":
        loss = criteria["localization"](outputs, batch["bbox"])
        metrics["loss"] = float(loss.item())
        metrics["iou"] = float(box_iou_mean(outputs.detach(), batch["bbox"]).item())
        return loss, metrics
    if task == "segmentation":
        loss = criteria["segmentation"](outputs, batch["segmentation_mask"])
        metrics["loss"] = float(loss.item())
        metrics["dice"] = float(dice_score(outputs.detach(), batch["segmentation_mask"]).item())
        return loss, metrics

    cls_loss = criteria["classification"](outputs["classification"], batch["breed_label"])
    loc_loss = criteria["localization"](outputs["localization"], batch["bbox"])
    seg_loss = criteria["segmentation"](outputs["segmentation"], batch["segmentation_mask"])
    loss = cls_loss + loc_loss + seg_loss
    metrics["loss"] = float(loss.item())
    metrics["classification_loss"] = float(cls_loss.item())
    metrics["localization_loss"] = float(loc_loss.item())
    metrics["segmentation_loss"] = float(seg_loss.item())
    metrics["iou"] = float(box_iou_mean(outputs["localization"].detach(), batch["bbox"]).item())
    metrics["dice"] = float(dice_score(outputs["segmentation"].detach(), batch["segmentation_mask"]).item())
    return loss, metrics


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def train_or_eval_epoch(model, loader, optimizer, criteria, device, task: str, train: bool, max_batches: int | None = None):
    if train:
        model.train()
    else:
        model.eval()

    running: Dict[str, float] = {}
    all_targets = []
    all_preds = []

    # Wrap the loader with tqdm for a progress bar
    for batch_idx, batch in enumerate(tqdm(loader, desc="Train" if train else "Eval")):
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch["image"])
            loss, metrics = compute_losses(outputs, batch, criteria, task)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for key, value in metrics.items():
            running[key] = running.get(key, 0.0) + value

        if task == "classification":
            preds = outputs.argmax(dim=1)
            all_targets.extend(batch["breed_label"].detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
        elif task == "multitask":
            preds = outputs["classification"].argmax(dim=1)
            all_targets.extend(batch["breed_label"].detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    num_batches = max(batch_idx + 1 if max_batches else len(loader), 1)
    epoch_metrics = {key: value / num_batches for key, value in running.items()}
    if all_targets:
        epoch_metrics["macro_f1"] = float(f1_score(all_targets, all_preds, average="macro"))
    return epoch_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Assignment-2 models.")
    parser.add_argument("--task", choices=["classification", "localization", "segmentation", "multitask"], default="classification")
    parser.add_argument("--data-root", default="oxford-iiit-pet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-batchnorm", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--init-from", default="")
    parser.add_argument("--encoder-checkpoint", default="")
    parser.add_argument("--classifier-checkpoint", default="")
    parser.add_argument("--localizer-checkpoint", default="")
    parser.add_argument("--segmentation-checkpoint", default="")
    parser.add_argument("--wandb-project", default="da6401-assignment-2")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = OxfordIIITPetDataset(root=args.data_root, split="train", image_size=args.image_size, seed=args.seed)
    val_dataset = OxfordIIITPetDataset(root=args.data_root, split="val", image_size=args.image_size, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args).to(device)
    maybe_initialize_model(model, args, device)
    criteria = build_criteria(args.task)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    wandb = get_wandb() if not args.disable_wandb else None
    use_wandb = wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or None, config=vars(args))

    best_score = float("-inf")
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_or_eval_epoch(
            model,
            train_loader,
            optimizer,
            criteria,
            device,
            args.task,
            train=True,
            max_batches=args.max_train_batches or None,
        )
        val_metrics = train_or_eval_epoch(
            model,
            val_loader,
            optimizer,
            criteria,
            device,
            args.task,
            train=False,
            max_batches=args.max_val_batches or None,
        )

        score = val_metrics.get("macro_f1", 0.0) + val_metrics.get("dice", 0.0) + val_metrics.get("iou", 0.0) - val_metrics.get("loss", 0.0)
        if score > best_score:
            best_score = score
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "task": args.task,
                    },
                    checkpoint_path,
                )

        print(f"Epoch {epoch}")
        print(f"  train: {train_metrics}")
        print(f"  val:   {val_metrics}")

        if use_wandb:
            payload = {f"train/{key}": value for key, value in train_metrics.items()}
            payload.update({f"val/{key}": value for key, value in val_metrics.items()})
            payload["epoch"] = epoch
            wandb.log(payload)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
