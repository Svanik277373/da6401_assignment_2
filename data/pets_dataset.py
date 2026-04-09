"""Dataset utilities for the Oxford-IIIT Pet assignment."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PetSample:
    image_id: str
    breed_label: int
    species_label: int
    image_path: Path
    mask_path: Path
    xml_path: Path


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(
        self,
        root: str = "oxford-iiit-pet",
        split: str = "train",
        image_size: int = 224,
        train_ratio: float = 0.8,
        seed: int = 42,
        transform=None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of {'train', 'val', 'all'}")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must lie in (0, 1)")

        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform

        self.images_dir = self.root / "images" / "images"
        self.annotations_dir = self.root / "annotations" / "annotations"
        self.trimaps_dir = self.annotations_dir / "trimaps"
        self.xml_dir = self.annotations_dir / "xmls"
        self.list_path = self.annotations_dir / "list.txt"

        if not self.list_path.exists():
            raise FileNotFoundError(f"Could not find dataset index: {self.list_path}")

        all_samples = self._read_index()
        if not all_samples:
            raise RuntimeError("No complete Oxford-IIIT Pet samples were found. Check the dataset paths and extracted files.")
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        train_cutoff = int(len(all_samples) * train_ratio)
        if split == "train":
            self.samples = all_samples[:train_cutoff]
        elif split == "val":
            self.samples = all_samples[train_cutoff:]
        else:
            self.samples = all_samples

    def _read_index(self) -> List[PetSample]:
        samples: List[PetSample] = []
        with self.list_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                image_id, breed_label, species_label, _ = line.split()
                image_path = self.images_dir / f"{image_id}.jpg"
                mask_path = self.trimaps_dir / f"{image_id}.png"
                xml_path = self.xml_dir / f"{image_id}.xml"
                if not (image_path.exists() and mask_path.exists() and xml_path.exists()):
                    continue
                samples.append(
                    PetSample(
                        image_id=image_id,
                        breed_label=int(breed_label) - 1,
                        species_label=int(species_label) - 1,
                        image_path=image_path,
                        mask_path=mask_path,
                        xml_path=xml_path,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image_array).permute(2, 0, 1)

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        mask = Image.open(mask_path)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask_array = np.asarray(mask, dtype=np.int64) - 1
        mask_array = np.clip(mask_array, 0, 2)
        return torch.from_numpy(mask_array)

    def _load_bbox(self, xml_path: Path) -> torch.Tensor:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = float(size.findtext("width"))
        height = float(size.findtext("height"))
        bndbox = root.find("./object/bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        x_center = ((xmin + xmax) * 0.5) / width
        y_center = ((ymin + ymax) * 0.5) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        return torch.tensor([x_center, y_center, box_width, box_height], dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "breed_label": torch.tensor(sample.breed_label, dtype=torch.long),
            "species_label": torch.tensor(sample.species_label, dtype=torch.long),
            "bbox": self._load_bbox(sample.xml_path),
            "segmentation_mask": self._load_mask(sample.mask_path),
            "image_id": sample.image_id,
        }
