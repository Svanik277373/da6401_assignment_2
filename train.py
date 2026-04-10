"""Training entrypoint with OneCycleLR, AdamW, and Label Smoothing."""

from __future__ import annotations

import argparse
import random
import os
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
        import wandb
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
        [pred_boxes[:, :2] - pred_boxes[:, 2:] * 0.5, pred_boxes[:, :2] + pred_boxes[:, 2:] * 0.5], dim=1,
    )
    target_xyxy = torch.cat(
        [target_boxes[:, :2] - target_boxes[:, 2:] * 0.5, target_boxes[:, :2] + target_boxes[:, 2:] * 0.5], dim=1,
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
            load_checkpoint=False,
        )
    if args.task == "localization":
        return VGG11Localizer(
            use_batchnorm=not args.disable_batchnorm,
            freeze_encoder=args.freeze_encoder,
            load_checkpoint=False,
            image_space_output=False,
        )
    if args.task == "segmentation":
        return VGG11UNet(num_classes=3, use_batchnorm=not args.disable_batchnorm, load_checkpoint=False)
    return MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        dropout_p=args.dropout,
        use_batchnorm=not args.disable_batchnorm,
        load_checkpoints=False,
        image_space_output=False,
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
    cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    if task == "classification": return {"classification": cls_loss}
    if task == "localization": return {"localization": IoULoss()}
    if task == "segmentation": return {"segmentation": nn.CrossEntropyLoss()}
    return {"classification": cls_loss, "localization": IoULoss(), "segmentation": nn.CrossEntropyLoss()}

def compute_losses(outputs, batch, criteria, task):
    metrics = {}
    if task == "classification":
        loss = criteria["classification"](outputs, batch["breed_label"])
        metrics["loss"] = loss.item()
        return loss, metrics
    if task == "localization":
        loss = criteria["localization"](outputs, batch["bbox"])
        metrics["loss"] = loss.item()
        metrics["iou"] = box_iou_mean(outputs.detach(), batch["bbox"]).item()
        return loss, metrics
    if task == "segmentation":
        loss = criteria["segmentation"](outputs, batch["segmentation_mask"])
        metrics["loss"] = loss.item()
        metrics["dice"] = dice_score(outputs.detach(), batch["segmentation_mask"]).item()
        return loss, metrics

    c_loss = criteria["classification"](outputs["classification"], batch["breed_label"])
    l_loss = criteria["localization"](outputs["localization"], batch["bbox"])
    s_loss = criteria["segmentation"](outputs["segmentation"], batch["segmentation_mask"])
    loss = c_loss + l_loss + s_loss
    metrics.update({"loss": loss.item(), "cls_loss": c_loss.item(), "loc_loss": l_loss.item(), "seg_loss": s_loss.item()})
    metrics["iou"] = box_iou_mean(outputs["localization"].detach(), batch["bbox"]).item()
    metrics["dice"] = dice_score(outputs["segmentation"].detach(), batch["segmentation_mask"]).item()
    return loss, metrics

def train_or_eval_epoch(model, loader, optimizer, criteria, device, task, train, scheduler=None):
    model.train() if train else model.eval()
    running, all_targets, all_preds = {}, [], []
    correct, total = 0, 0

    for batch in tqdm(loader, desc="Train" if train else "Eval"):
        for k, v in batch.items():
            if torch.is_tensor(v): batch[k] = v.to(device)
        
        with torch.set_grad_enabled(train):
            outputs = model(batch["image"])
            loss, metrics = compute_losses(outputs, batch, criteria, task)
            
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        for k, v in metrics.items(): running[k] = running.get(k, 0.0) + v
        
        if task in ["classification", "multitask"]:
            logits = outputs["classification"] if task == "multitask" else outputs
            preds = logits.argmax(dim=1)
            
            correct += (preds == batch["breed_label"]).sum().item()
            total += batch["breed_label"].size(0)
            all_targets.extend(batch["breed_label"].cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = {k: v / len(loader) for k, v in running.items()}
    if total > 0: metrics["accuracy"] = correct / total
    if all_targets: metrics["macro_f1"] = f1_score(all_targets, all_preds, average="macro")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="classification")
    parser.add_argument("--data-root", default=r"oxford-iiit-pet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--checkpoint-path", default=r"checkpoints\vgg11_best.pth")
    parser.add_argument("--encoder-checkpoint", type=str, help="Path to VGG11 classification weights")
    parser.add_argument("--classifier-checkpoint", type=str)
    parser.add_argument("--localizer-checkpoint", type=str)
    parser.add_argument("--segmentation-checkpoint", type=str)
    parser.add_argument("--init-from", type=str, help="Resume full model from checkpoint")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze VGG11 weights")
    parser.add_argument("--disable-batchnorm", action="store_true")
    # New argument for evaluation mode
    parser.add_argument("--eval-only", action="store_true", help="Run only the evaluation loop")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_ds = OxfordIIITPetDataset(root=args.data_root, split="train")
    val_ds = OxfordIIITPetDataset(root=args.data_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Initialize model and criteria
    model = build_model(args).to(device)
    maybe_initialize_model(model, args, device)
    criteria = build_criteria(args.task)

    # Handle evaluation mode
    if args.eval_only:
        print(f"\nRunning evaluation on {args.task} task...")
        val_m = train_or_eval_epoch(model, val_loader, None, criteria, device, args.task, False)
        print(f"Validation results: {val_m}")
        return

    # Regular training loop
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3
    )

    wandb = get_wandb()
    if wandb: wandb.init(project="da6401-assignment-2", dir="D:/wandb_logs", config=vars(args))

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch} | Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        train_m = train_or_eval_epoch(model, train_loader, optimizer, criteria, device, args.task, True, scheduler=scheduler)
        val_m = train_or_eval_epoch(model, val_loader, optimizer, criteria, device, args.task, False)
        
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, args.checkpoint_path)

        print(f"  train: {train_m}\n  val:   {val_m}")
        if wandb:
            log_data = {f"train/{k}": v for k, v in train_m.items()}
            log_data.update({f"val/{k}": v for k, v in val_m.items()})
            log_data["lr"] = optimizer.param_groups[0]['lr']
            wandb.log(log_data)

if __name__ == "__main__":
    main()
