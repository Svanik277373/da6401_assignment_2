"""Dataset utilities for the Oxford-IIIT Pet assignment with Albumentations and Bounding Boxes."""

from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

@dataclass(frozen=True)
class PetSample:
    image_id: str
    breed_label: int
    species_label: int
    image_path: str
    mask_path: str
    xml_path: str

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader with robust path handling and bounding boxes."""

    def __init__(
        self,
        root: str = r"oxford-iiit-pet",
        split: str = "train",
        image_size: int = 224,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.image_size = image_size

        # Flexible path setup to handle nested 'annotations' or 'images' folders
        self.annotations_dir = os.path.join(self.root, "annotations")
        if os.path.exists(os.path.join(self.annotations_dir, "annotations")):
            self.annotations_dir = os.path.join(self.annotations_dir, "annotations")
            
        self.images_dir = os.path.join(self.root, "images")
        if os.path.exists(os.path.join(self.images_dir, "images")):
            self.images_dir = os.path.join(self.images_dir, "images")

        self.trimaps_dir = os.path.join(self.annotations_dir, "trimaps")
        self.xml_dir = os.path.join(self.annotations_dir, "xmls")
        self.list_path = os.path.join(self.annotations_dir, "list.txt")

        if not os.path.exists(self.list_path):
            raise FileNotFoundError(f"Could not find dataset index: {self.list_path}")

        # Data Split Logic
        all_samples = self._read_index()
        rng = random.Random(seed)
        rng.shuffle(all_samples)
        train_cutoff = int(len(all_samples) * train_ratio)

        if split == "train":
            self.samples = all_samples[:train_cutoff]
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, 
                         scale=(0.95, 1.05), rotate=(-15, 15), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])) # ADDED BBOX PARAMS
        else:
            self.samples = all_samples[train_cutoff:] if split == "val" else all_samples
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])) # ADDED BBOX PARAMS

    def _read_index(self) -> List[PetSample]:
        samples: List[PetSample] = []
        with open(self.list_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                image_id, breed_label, species_label, _ = line.split()
                image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
                mask_path = os.path.join(self.trimaps_dir, f"{image_id}.png")
                xml_path = os.path.join(self.xml_dir, f"{image_id}.xml")
                
                if os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(xml_path):
                    samples.append(PetSample(
                        image_id=image_id,
                        breed_label=int(breed_label) - 1,
                        species_label=int(species_label) - 1,
                        image_path=image_path,
                        mask_path=mask_path,
                        xml_path=xml_path,
                    ))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_bbox(self, xml_path: str) -> list:
        """Load and normalize bounding box to [Xcenter, Ycenter, width, height]. Returns list for Albumentations."""
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width, height = float(size.findtext("width")), float(size.findtext("height"))
        bndbox = root.find("./object/bndbox")
        xmin, ymin = float(bndbox.findtext("xmin")), float(bndbox.findtext("ymin"))
        xmax, ymax = float(bndbox.findtext("xmax")), float(bndbox.findtext("ymax"))

        # Clip values to ensure they stay strictly inside [0.0, 1.0] for Albumentations YOLO format
        x_center = np.clip(((xmin + xmax) * 0.5) / width, 0.0, 1.0)
        y_center = np.clip(((ymin + ymax) * 0.5) / height, 0.0, 1.0)
        box_w = np.clip((xmax - xmin) / width, 0.0, 1.0)
        box_h = np.clip((ymax - ymin) / height, 0.0, 1.0)

        return [x_center, y_center, box_w, box_h]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        mask = np.array(Image.open(sample.mask_path))
        mask = np.clip(mask.astype(np.int64) - 1, 0, 2)
        
        # Extract initial bbox
        raw_bbox = self._load_bbox(sample.xml_path)
        
        # Pass bboxes and a dummy class_label to Albumentations
        transformed = self.transform(image=image, mask=mask, bboxes=[raw_bbox], class_labels=[0])
        
        # If the augmentation (like a harsh crop) completely removed the bounding box, 
        # Albumentations returns an empty list. Fallback to raw if this happens.
        aug_bbox = transformed["bboxes"][0] if len(transformed["bboxes"]) > 0 else raw_bbox

        return {
            "image": transformed["image"],
            "breed_label": torch.tensor(sample.breed_label, dtype=torch.long),
            "bbox": torch.tensor(aug_bbox, dtype=torch.float32), 
            "segmentation_mask": transformed["mask"].long(),
            "image_id": sample.image_id,
        }