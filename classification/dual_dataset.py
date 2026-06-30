# -*- coding: utf-8 -*-
"""Dual-input classification dataset 鈥?grayscale + height map pairs.

Supports two naming conventions:
  A) Same base name: 001.bmp + 001.tif
  B) Prefix pattern: GA_001.bmp + HA_001.tif
"""

import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def _find_pairs(project_dir, split, split_map):
    """Find (gray_path, height_path, label_id) pairs for given split."""
    img_dir = Path(project_dir) / "images"
    if not img_dir.exists():
        return []

    # Load class_labels.json
    label_path = Path(project_dir) / "annotations" / "class_labels.json"
    mapping_dict = {}
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        mapping_dict = label_data.get("mapping", {})

    # Collect all files
    all_bmp = {}  # base -> full path
    all_tif = {}  # base -> full path
    ga_bmp = {}   # timestamp -> full path
    ha_tif = {}   # timestamp -> full path

    for entry in sorted(img_dir.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        stem, ext = os.path.splitext(name)
        ext = ext.lower()

        if ext == ".bmp":
            if stem.startswith("GA_"):
                ga_bmp[stem[3:]] = str(entry)
            else:
                all_bmp[stem] = str(entry)
        elif ext in (".tif", ".tiff"):
            if stem.startswith("HA_"):
                ha_tif[stem[3:]] = str(entry)
            else:
                all_tif[stem] = str(entry)

    pairs = []

    # Convention A: same base name
    for base in all_bmp:
        if base in all_tif:
            gray_path = all_bmp[base]
            height_path = all_tif[base]
            label_id = _get_label(base, gray_path, mapping_dict)
            if label_id is not None and _check_split(base, split, split_map):
                pairs.append((gray_path, height_path, label_id))

    # Convention B: GA_ / HA_ prefix
    for ts in ga_bmp:
        if ts in ha_tif:
            gray_path = ga_bmp[ts]
            height_path = ha_tif[ts]
            base = f"GA_{ts}"
            label_id = _get_label(base, gray_path, mapping_dict)
            if label_id is not None and _check_split(base, split, split_map):
                pairs.append((gray_path, height_path, label_id))

    return pairs


def _get_label(base, path, mapping_dict):
    """Get label ID from mapping or ImageFolder structure."""
    rel = os.path.basename(path)
    if rel in mapping_dict:
        return int(mapping_dict[rel])
    if f"images/{rel}" in mapping_dict:
        return int(mapping_dict[f"images/{rel}"])
    # Check ImageFolder: parent folder = class
    parent = os.path.basename(os.path.dirname(path))
    return None  # Will be resolved by caller


def _check_split(base, split, split_map):
    """Check if sample belongs to requested split."""
    if split == "all":
        return True
    file_split = split_map.get(base, "train")
    return file_split == split


class MixedClassificationDataset(Dataset):
    """Dual-input dataset: loads (gray_img, height_img) -> label."""

    def __init__(self, project_dir, split="train", image_size=224, augment=False):
        self.project_dir = Path(project_dir)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size

        # Load split map
        split_path = self.project_dir / "train_test_split.json"
        split_map = {}
        if split_path.exists():
            with open(split_path, "r", encoding="utf-8") as f:
                split_map = json.load(f)

        self.samples = _find_pairs(project_dir, split, split_map)

        # If no pairs found or labels are None, try ImageFolder
        if self.samples and self.samples[0][2] is None:
            self.samples = self._resolve_imagefolder_labels()

        # Determine num_classes
        label_ids = set(s[2] for s in self.samples if s[2] is not None)
        self.num_classes = max(len(label_ids), 2)

        self.augment = augment

    def _resolve_imagefolder_labels(self):
        """Use parent folder names as class labels (ImageFolder convention)."""
        resolved = []
        class_names = []
        for gray_path, height_path, _ in self.samples:
            cls_name = os.path.basename(os.path.dirname(gray_path))
            if cls_name not in class_names:
                class_names.append(cls_name)
            resolved.append((gray_path, height_path, class_names.index(cls_name)))
        return resolved

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gray_path, height_path, label = self.samples[idx]

        # Load grayscale image
        gray_img = Image.open(gray_path).convert("RGB")
        # Load height map
        height_img = Image.open(height_path).convert("L")  # single channel

        # Resize
        gray_img = gray_img.resize(self.image_size, Image.BILINEAR)
        height_img = height_img.resize(self.image_size, Image.BILINEAR)

        # Convert to tensors
        gray_tensor = torch.from_numpy(np.array(gray_img).astype(np.float32) / 255.0)
        gray_tensor = gray_tensor.permute(2, 0, 1)  # HWC -> CHW

        height_tensor = torch.from_numpy(np.array(height_img).astype(np.float32) / 255.0)
        height_tensor = height_tensor.unsqueeze(0)  # H,W -> 1,H,W

        # Normalize grayscale with ImageNet stats
        for c in range(3):
            gray_tensor[c] = (gray_tensor[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        # Normalize height to [0,1] (already done by /255)
        # Optional: standardize height with its own mean/std
        height_mean = 0.5
        height_std = 0.25
        height_tensor = (height_tensor - height_mean) / height_std

        # Basic augmentation
        if self.augment:
            from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
            import random
            p = 0.5
            if random.random() < p:
                gray_tensor = torch.flip(gray_tensor, [-1])
                height_tensor = torch.flip(height_tensor, [-1])
            if random.random() < p:
                gray_tensor = torch.flip(gray_tensor, [-2])
                height_tensor = torch.flip(height_tensor, [-2])

        return (gray_tensor, height_tensor), label


def auto_split(project_dir, val_ratio=0.2):
    """Auto-split pairs into train/val by image count."""
    project_dir = Path(project_dir)
    split_path = project_dir / "train_test_split.json"

    samples = _find_pairs(project_dir, "all", {})
    if len(samples) < 2:
        return

    # Group by base (GA_/HA_ pairs share same timestamp base)
    bases = []
    seen = set()
    for gray_path, _, _ in samples:
        base = os.path.splitext(os.path.basename(gray_path))[0]
        if base.startswith("GA_"):
            base = base[3:]  # use timestamp as key
        if base not in seen:
            seen.add(base)
            bases.append(base)

    import random
    random.seed(42)
    random.shuffle(bases)
    n_val = max(1, int(len(bases) * val_ratio))
    val_bases = set(bases[:n_val])

    split_map = {}
    for gray_path, _, _ in samples:
        fname = os.path.basename(gray_path)
        base = os.path.splitext(fname)[0]
        if base.startswith("GA_"):
            key = base
        else:
            key = base
        base_key = base[3:] if base.startswith("GA_") else base
        split_map[key] = "val" if base_key in val_bases else "train"

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2)
