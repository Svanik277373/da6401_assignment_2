"""Shared utilities for Question 2 experiment scripts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

from data.pets_dataset import OxfordIIITPetDataset
from models import MultiTaskPerceptionModel, VGG11Classifier, VGG11Localizer, VGG11UNet
from train import build_criteria, build_model, set_seed, train_or_eval_epoch


DEFAULTS = {
    "task": "classification",
    "data_root": "oxford-iiit-pet",
    "epochs": 50,
    "batch_size": 16,
    "lr": 5e-4,
    "image_size": 224,
    "dropout": 0.5,
    "seed": 42,
    "num_workers": 0,
    "disable_batchnorm": False,
    "freeze_encoder": False,
    "checkpoint_path": "",
    "wandb_project": "da6401-assignment-2",
    "wandb_run_name": "",
    "disable_wandb": False,
}


def build_args(**overrides) -> SimpleNamespace:
    payload = DEFAULTS.copy()
    payload.update(overrides)
    return SimpleNamespace(**payload)


def make_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    train_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="train",
        image_size=args.image_size,
        seed=args.seed,
    )
    val_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="val",
        image_size=args.image_size,
        seed=args.seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_once(args, freeze_strategy: Optional[str] = None):
    set_seed(args.seed)
    device = make_device()
    train_loader, val_loader = make_dataloaders(args)
    model = build_model(args).to(device)

    if freeze_strategy is not None:
        apply_freeze_strategy(model, freeze_strategy)

    criteria = build_criteria(args.task)
    
    # MATCH TRAIN.PY: Use AdamW with weight decay
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2)

    # MATCH TRAIN.PY: Add OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3
    )

    run = None
    if not args.disable_wandb and wandb is not None:
        run = wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name or None, 
            dir="D:/wandb_logs", 
            config=dict(vars(args))
        )

    history = {"train": [], "val": []}
    best_score = float("-inf")
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

    # Global tqdm progress bar for overall progress
    for epoch in tqdm(range(1, args.epochs + 1), desc="Total Training Progress"):
        
        # EXACT MATCH TO TRAIN.PY LOGGING: Print Epoch and LR
        print(f"\nEpoch {epoch} | Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_metrics = train_or_eval_epoch(model, train_loader, optimizer, criteria, device, args.task, train=True, scheduler=scheduler)
        val_metrics = train_or_eval_epoch(model, val_loader, optimizer, criteria, device, args.task, train=False)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        score = val_metrics.get("macro_f1", 0.0) + val_metrics.get("dice", 0.0) + val_metrics.get("iou", 0.0) - val_metrics.get("loss", 0.0)
        if score > best_score and checkpoint_path is not None:
            best_score = score
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "task": args.task}, checkpoint_path)

        # EXACT MATCH TO TRAIN.PY LOGGING: Print Metrics
        print(f"  train: {train_metrics}\n  val:   {val_metrics}")

        if run is not None:
            payload = {f"train/{k}": v for k, v in train_metrics.items()}
            payload.update({f"val/{k}": v for k, v in val_metrics.items()})
            if "macro_f1" in train_metrics:
                payload["train/f1"] = train_metrics["macro_f1"]
            if "macro_f1" in val_metrics:
                payload["val/f1"] = val_metrics["macro_f1"]
            payload["epoch"] = epoch
            payload["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(payload)

    if run is not None:
        run.finish()
    return model, history


def apply_freeze_strategy(model: torch.nn.Module, strategy: str) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    if strategy == "strict":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    elif strategy == "partial":
        for name, parameter in encoder.named_parameters():
            parameter.requires_grad = name.startswith("block4") or name.startswith("block5")
    elif strategy == "full":
        for parameter in encoder.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError("strategy must be one of {'strict', 'partial', 'full'}")


def load_model(task: str, checkpoint_path: str | Path, device: Optional[torch.device] = None):
    device = device or make_device()
    
    # Initialize with load_checkpoint=False to prevent redundant weight downloads
    # Initialize localizers with image_space_output=False for proper wandb rendering
    if task == "classification":
        model = VGG11Classifier(load_checkpoint=False)
    elif task == "localization":
        model = VGG11Localizer(image_space_output=False, load_checkpoint=False)
    elif task == "segmentation":
        model = VGG11UNet(load_checkpoint=False)
    else:
        model = MultiTaskPerceptionModel(image_space_output=False, load_checkpoints=False)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    # Backward-compatibility for older multitask checkpoints that stored a
    # single shared encoder instead of separate classification/segmentation encoders.
    if task == "multitask":
        model_state = model.state_dict()
        patched_state = dict(state_dict)

        if "classification_encoder.block1.0.weight" not in patched_state:
            encoder_weights = {
                key[len("encoder.") :]: value
                for key, value in state_dict.items()
                if key.startswith("encoder.")
            }
            for branch in ("classification_encoder", "segmentation_encoder"):
                for key, value in encoder_weights.items():
                    branch_key = f"{branch}.{key}"
                    if branch_key in model_state and branch_key not in patched_state:
                        patched_state[branch_key] = value

        model.load_state_dict(patched_state, strict=False)
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def load_single_image(image_path: str | Path, image_size: int = 224) -> torch.Tensor:
    # Proper ImageNet normalization applied for downstream tasks
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform(image)


def make_wandb_image_from_bbox(image_tensor: torch.Tensor, gt_box=None, pred_box=None, caption: str = ""):
    if wandb is None:
        return None
    # Un-normalize image for display
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)
    ax.axis("off")
    height, width = image_np.shape[:2]
    if gt_box is not None:
        add_box(ax, gt_box, width, height, "green", "GT")
    if pred_box is not None:
        add_box(ax, pred_box, width, height, "red", "Pred")
    fig.tight_layout()
    image = wandb.Image(fig, caption=caption)
    plt.close(fig)
    return image


def add_box(ax, box_xywh, width: int, height: int, color: str, label: str) -> None:
    x_center, y_center, box_w, box_h = [float(v) for v in box_xywh]
    left = (x_center - box_w * 0.5) * width
    top = (y_center - box_h * 0.5) * height
    rect = patches.Rectangle((left, top), box_w * width, box_h * height, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(left, max(top - 4, 0), label, color=color, fontsize=8, backgroundcolor="black")


def feature_maps_to_images(feature_tensor: torch.Tensor, limit: int = 16) -> list[np.ndarray]:
    feature_tensor = feature_tensor.detach().cpu()
    images = []
    for channel in feature_tensor[:limit]:
        array = channel.numpy()
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        images.append(array)
    return images


def overlay_mask(image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> np.ndarray:
    # Un-normalize image for display
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    mask = mask_tensor.detach().cpu().numpy()
    color_map = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.float32,
    ) / 255.0
    colored = color_map[np.clip(mask, 0, 2)]
    overlay = 0.6 * image + 0.4 * colored
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def colorize_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    color_map = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.float32,
    ) / 255.0
    return (np.clip(color_map[np.clip(mask, 0, 2)], 0, 1) * 255).astype(np.uint8)


def to_display_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def pixel_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def evaluate_segmentation_checkpoint(model, loader: DataLoader, device: torch.device):
    all_logits = []
    all_targets = []
    examples = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images).cpu()
            targets = batch["segmentation_mask"].cpu()

            all_logits.append(logits)
            all_targets.append(targets)

            if len(examples) < 5:
                preds = logits.argmax(dim=1)
                limit = min(5 - len(examples), preds.size(0))
                for idx in range(limit):
                    examples.append(
                        {
                            "image_id": batch["image_id"][idx],
                            "image": batch["image"][idx].cpu(),
                            "target": targets[idx],
                            "pred": preds[idx],
                        }
                    )

    return {
        "logits": torch.cat(all_logits, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "examples": examples,
    }


def train_segmentation_for_epochs(args, epochs: int):
    run_args = build_args(
        task="segmentation",
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        disable_batchnorm=args.disable_batchnorm,
    )
    set_seed(run_args.seed)
    device = make_device()
    train_loader, val_loader = make_dataloaders(run_args)
    model = build_model(run_args).to(device)
    criteria = build_criteria("segmentation")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=run_args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=run_args.lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    history = []
    for epoch in range(1, epochs + 1):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
                module=r"albumentations\.augmentations\.dropout\.functional",
            )
            train_metrics = train_or_eval_epoch(
                model,
                train_loader,
                optimizer,
                criteria,
                device,
                "segmentation",
                train=True,
                scheduler=scheduler,
            )
        val_eval = evaluate_segmentation_checkpoint(model, val_loader, device)
        val_loss = float(criteria["segmentation"](val_eval["logits"], val_eval["targets"]).item())
        val_dice = float(
            (
                2.0
                * (((val_eval["logits"].argmax(dim=1) == 0).float() * (val_eval["targets"] == 0).float()).sum(dim=(1, 2)))
                + 1e-6
            )
            .div(
                (val_eval["logits"].argmax(dim=1) == 0).float().sum(dim=(1, 2))
                + (val_eval["targets"] == 0).float().sum(dim=(1, 2))
                + 1e-6
            )
            .mean()
            .item()
        )
        val_pixel_accuracy = pixel_accuracy_from_logits(val_eval["logits"], val_eval["targets"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_pixel_accuracy": val_pixel_accuracy,
            }
        )

    return model, history


def first_batch(loader: DataLoader):
    return next(iter(loader))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def named_conv_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Conv2d]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
