"""
Single-file training script for Oxford-IIIT Pet classification
using pretrained VGG11-BN (ImageNet weights) for comparison baseline.

Usage:
    python train_pretrained_vgg.py --data-root D:\oxford-iiit-pet
    python train_pretrained_vgg.py --data-root D:\oxford-iiit-pet --epochs 30 --lr 1e-3 --unfreeze-epoch 5
"""

from __future__ import annotations

import argparse
import os
import random

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG11_BN_Weights, vgg11_bn
from tqdm import tqdm

# ─────────────────────────────────────────
# Optional W&B
# ─────────────────────────────────────────
def get_wandb():
    try:
        import wandb
        return wandb
    except Exception:
        return None


# ─────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────
# Dataset  (inline — no external dependency)
# ─────────────────────────────────────────
class OxfordPetClassification(Dataset):
    """Minimal Oxford-IIIT Pet loader for classification only."""

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(self, root: str, split: str = "train",
                 image_size: int = 224, train_ratio: float = 0.8, seed: int = 42):
        super().__init__()
        self.root = root

        ann_dir = os.path.join(root, "annotations")
        if os.path.isdir(os.path.join(ann_dir, "annotations")):
            ann_dir = os.path.join(ann_dir, "annotations")

        img_dir = os.path.join(root, "images")
        if os.path.isdir(os.path.join(img_dir, "images")):
            img_dir = os.path.join(img_dir, "images")

        list_path = os.path.join(ann_dir, "list.txt")
        xml_dir   = os.path.join(ann_dir, "xmls")

        # Parse index — only keep samples that have XML (same filter as main repo)
        samples = []
        with open(list_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                img_id, breed_label, *_ = line.split()
                img_path  = os.path.join(img_dir,  f"{img_id}.jpg")
                mask_path = os.path.join(ann_dir, "trimaps", f"{img_id}.png")
                xml_path  = os.path.join(xml_dir, f"{img_id}.xml")
                if os.path.isfile(img_path) and os.path.isfile(mask_path) and os.path.isfile(xml_path):
                    samples.append((img_path, int(breed_label) - 1))

        rng = random.Random(seed)
        rng.shuffle(samples)
        cut = int(len(samples) * train_ratio)

        if split == "train":
            self.samples = samples[:cut]
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32),
                                hole_width_range=(16, 32), p=0.3),
                A.Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ])
        else:
            self.samples = samples[cut:]
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
def build_model(num_classes: int = 37, dropout_p: float = 0.4) -> nn.Module:
    """
    VGG11-BN with pretrained ImageNet weights.

    Architecture change vs. original VGG head:
      Original: Linear(25088→4096) → Linear(4096→4096) → Linear(4096→1000)
      Here:     Linear(25088→1024) → Linear(1024→num_classes)

    Differential LR is handled in build_optimizer() by separating
    encoder (features+avgpool) and head parameter groups.
    """
    backbone = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)

    # Replace the massive classifier head with a compact one
    backbone.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p),
        nn.Linear(1024, num_classes),
    )

    # Kaiming init for the new head only
    for m in backbone.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)

    return backbone


def freeze_encoder(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────
# Optimizer  (differential LR)
# ─────────────────────────────────────────
def build_optimizer(model: nn.Module, head_lr: float, encoder_lr: float,
                    weight_decay: float) -> torch.optim.Optimizer:
    """
    Differential LR:
      encoder (features + avgpool) → encoder_lr (lower, e.g. 1e-4)
      classifier head             → head_lr    (higher, e.g. 1e-3)
    """
    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and not n.startswith("classifier")]
    head_params    = [p for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith("classifier")]
    return torch.optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": head_params,    "lr": head_lr},
    ], weight_decay=weight_decay)


# ─────────────────────────────────────────
# Train / eval loop
# ─────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, device, train: bool,
              scheduler=None):
    model.train() if train else model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    correct = total = 0

    for images, labels in tqdm(loader, desc="Train" if train else "Eval ", leave=False):
        images, labels = images.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    n = len(loader)
    return {
        "loss":     running_loss / n,
        "accuracy": correct / total,
        "macro_f1": f1_score(all_targets, all_preds, average="macro"),
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pretrained VGG11-BN baseline for Oxford-IIIT Pet 37-class classification")
    parser.add_argument("--data-root",       default=r"oxford-iiit-pet")
    parser.add_argument("--epochs",          type=int,   default=80)
    parser.add_argument("--lr",              type=float, default=1e-3,
                        help="Peak LR for the classifier head")
    parser.add_argument("--encoder-lr",      type=float, default=1e-4,
                        help="Peak LR for the pretrained encoder (10× lower than head)")
    parser.add_argument("--weight-decay",    type=float, default=1e-4)
    parser.add_argument("--dropout",         type=float, default=0.5)
    parser.add_argument("--batch-size",      type=int,   default=16)
    parser.add_argument("--unfreeze-epoch",  type=int,   default=0,
                        help="Epoch at which to unfreeze encoder. 0 = train end-to-end from epoch 1.")
    parser.add_argument("--checkpoint-path", default=r"D:\checkpoints\vgg11_pretrained_best.pth")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────
    train_ds = OxfordPetClassification(root=args.data_root, split="train")
    val_ds   = OxfordPetClassification(root=args.data_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ─────────────────────────────
    model = build_model(num_classes=37, dropout_p=args.dropout).to(device)

    # Optional: freeze encoder for first N epochs (faster warm-up for the head)
    if args.unfreeze_epoch > 0:
        freeze_encoder(model)
        print(f"Encoder frozen until epoch {args.unfreeze_epoch}")

    # ── Optimizer + scheduler ─────────────
    # NOTE: rebuild optimizer after unfreeze so new param groups are registered
    optimizer = build_optimizer(model, head_lr=args.lr,
                                encoder_lr=args.encoder_lr,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.encoder_lr, args.lr],   # per-group max LR
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── W&B ──────────────────────────────
    wandb = get_wandb()
    if wandb:
        wandb.init(project="da6401-assignment-2",
                   name="pretrained-vgg11bn",
                   dir=r"D:\wandb_logs",
                   config=vars(args))

    # ── Training loop ────────────────────
    best_val_f1 = 0.0
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):

        # Unfreeze encoder at the configured epoch and rebuild optimizer+scheduler
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            unfreeze_encoder(model)
            remaining = args.epochs - epoch + 1
            optimizer = build_optimizer(model, head_lr=args.lr,
                                        encoder_lr=args.encoder_lr,
                                        weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[args.encoder_lr, args.lr],
                steps_per_epoch=len(train_loader),
                epochs=remaining,
                pct_start=0.05,
                div_factor=10,
                final_div_factor=100,
            )
            print(f"\n>>> Encoder unfrozen at epoch {epoch}, new scheduler for {remaining} epochs")

        head_lr = optimizer.param_groups[-1]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} | head_lr: {head_lr:.2e}")

        train_m = run_epoch(model, train_loader, optimizer, criterion, device,
                            train=True, scheduler=scheduler)
        val_m   = run_epoch(model, val_loader,   optimizer, criterion, device,
                            train=False)

        print(f"  train: {train_m}")
        print(f"  val:   {val_m}")

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_f1": best_val_f1}, args.checkpoint_path)
            print(f"  *** New best val F1: {best_val_f1:.4f} — saved to {args.checkpoint_path} ***")

        if wandb:
            log = {f"train/{k}": v for k, v in train_m.items()}
            log.update({f"val/{k}": v for k, v in val_m.items()})
            log["lr/head"]    = optimizer.param_groups[-1]["lr"]
            log["lr/encoder"] = optimizer.param_groups[0]["lr"]
            wandb.log(log)

    print(f"\nDone. Best val macro F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
