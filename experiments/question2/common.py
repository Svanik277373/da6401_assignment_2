"""Shared utilities for Question 2 experiment scripts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

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
    "epochs": 5,
    "batch_size": 16,
    "lr": 1e-3,
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
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    run = None
    if not args.disable_wandb and wandb is not None:
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name or None, config=dict(vars(args)))

    history = {"train": [], "val": []}
    best_score = float("-inf")
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_or_eval_epoch(model, train_loader, optimizer, criteria, device, args.task, train=True)
        val_metrics = train_or_eval_epoch(model, val_loader, optimizer, criteria, device, args.task, train=False)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        score = val_metrics.get("macro_f1", 0.0) + val_metrics.get("dice", 0.0) + val_metrics.get("iou", 0.0) - val_metrics.get("loss", 0.0)
        if score > best_score and checkpoint_path is not None:
            best_score = score
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "task": args.task}, checkpoint_path)

        if run is not None:
            payload = {f"train/{k}": v for k, v in train_metrics.items()}
            payload.update({f"val/{k}": v for k, v in val_metrics.items()})
            payload["epoch"] = epoch
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
    if task == "classification":
        model = VGG11Classifier()
    elif task == "localization":
        model = VGG11Localizer()
    elif task == "segmentation":
        model = VGG11UNet()
    else:
        model = MultiTaskPerceptionModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_single_image(image_path: str | Path, image_size: int = 224) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def make_wandb_image_from_bbox(image_tensor: torch.Tensor, gt_box=None, pred_box=None, caption: str = ""):
    if wandb is None:
        return None
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
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
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
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
    return 0.6 * image + 0.4 * colored


def first_batch(loader: DataLoader):
    return next(iter(loader))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def named_conv_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Conv2d]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
