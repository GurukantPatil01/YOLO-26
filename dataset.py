"""
dataset.py — Desert segmentation dataset with augmentation pipeline.
Handles:
  - Raw mask pixel value → contiguous class index remapping
  - Train / validation splitting
  - Albumentations-based augmentation
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a fast lookup array for mask remapping
# (pixel value 0..max_val → class index)
# ─────────────────────────────────────────────────────────────────────────────
def _build_remap_lut(class_map: dict, ignore_index: int = 255) -> np.ndarray:
    """Return a 1-D uint8 LUT of length max(class_map.keys())+1."""
    max_val = max(class_map.keys()) + 1
    lut = np.full(max_val, ignore_index, dtype=np.uint8)
    for raw, idx in class_map.items():
        lut[raw] = idx
    return lut


REMAP_LUT = _build_remap_lut(config.CLASS_MAP, config.IGNORE_INDEX)


def remap_mask(mask_array: np.ndarray) -> np.ndarray:
    """
    Convert a 2-D mask with raw pixel values (100, 200, …, 10000)
    to contiguous class indices (0–9).  Unknown values → IGNORE_INDEX.
    """
    # For very large pixel values (e.g. 10000) we need to handle them specially
    remapped = np.full(mask_array.shape, config.IGNORE_INDEX, dtype=np.uint8)
    for raw_val, class_idx in config.CLASS_MAP.items():
        remapped[mask_array == raw_val] = class_idx
    return remapped


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms() -> A.Compose:
    transforms = []

    if config.USE_HFLIP:
        transforms.append(A.HorizontalFlip(p=0.5))

    if config.USE_RANDOM_ROTATE:
        transforms.append(
            A.Rotate(limit=config.ROTATE_LIMIT, border_mode=0, p=0.4)
        )

    if config.USE_RANDOM_CROP:
        transforms.append(
            A.RandomResizedCrop(
                height=config.IMAGE_HEIGHT,
                width=config.IMAGE_WIDTH,
                scale=(0.6, 1.0),
                ratio=(0.8, 1.2),
                p=0.6,
            )
        )

    # Always resize to target at the end
    transforms.append(
        A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )

    if config.USE_COLOR_JITTER:
        transforms.append(
            A.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.05, p=0.5
            )
        )

    if config.USE_GAUSS_NOISE:
        transforms.append(
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
        )

    transforms += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class DesertSegDataset(Dataset):
    """
    Expects:
        img_dir/  *.png (or *.jpg)
        mask_dir/ *.png   (same stem as image)
    """

    def __init__(self, img_dir: str, mask_dir: str, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform

        # Collect all image filenames
        valid_ext = {".png", ".jpg", ".jpeg"}
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name  = self.images[idx]
        stem      = os.path.splitext(img_name)[0]

        # Load image (RGB)
        img_path  = os.path.join(self.img_dir, img_name)
        image     = np.array(Image.open(img_path).convert("RGB"))

        # Load mask — try same name, then with _mask suffix
        mask_path = os.path.join(self.mask_dir, img_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, stem + "_mask.png")
        mask = np.array(Image.open(mask_path))

        # Remap mask values to class indices
        mask = remap_mask(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].long()

        return image, mask


class TestDataset(Dataset):
    """Inference-only dataset (no masks)."""

    def __init__(self, img_dir: str, transform=None):
        self.img_dir   = img_dir
        self.transform = transform
        valid_ext = {".png", ".jpg", ".jpeg"}
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        image    = np.array(Image.open(
            os.path.join(self.img_dir, img_name)
        ).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, img_name   # return filename for saving predictions


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders():
    """
    Returns (train_loader, val_loader).
    Splits training data into train/val using config.VAL_SPLIT.
    """
    full_dataset = DesertSegDataset(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        transform=None,  # transforms applied per split below
    )

    n_val   = int(len(full_dataset) * config.VAL_SPLIT)
    n_train = len(full_dataset) - n_val

    # Reproducible split
    generator = torch.Generator().manual_seed(config.SEED)
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # Monkeypatch transforms per split
    class TransformWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset    = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            image, mask = self.subset[idx]
            # image/mask are still numpy here (transform=None above)
            augmented = self.transform(image=image, mask=mask)
            return augmented["image"], augmented["mask"].long()

    train_ds = TransformWrapper(train_subset, get_train_transforms())
    val_ds   = TransformWrapper(val_subset,   get_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader
