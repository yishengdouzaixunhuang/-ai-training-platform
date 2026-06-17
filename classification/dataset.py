# -*- coding: utf-8 -*-
"""Classification Dataset — PyTorch Dataset with ImageFolder + JSON dual modes.

Supports:
- ImageFolder mode: images/cat/001.jpg, images/dog/001.jpg (auto-labeled by folder)
- JSON mode: images/ flat + class_labels.json mapping
- Train/val split via train_test_split.json
- Standard augmentations via torchvision.transforms
"""

import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# Default ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _list_images_mapping(project_dir, split, split_map):
    """Walk images/ dir and return [(rel_path, abs_path, label_id), ...] for given split.

    Handles both flat and nested (ImageFolder) layouts.
    """
    img_dir = Path(project_dir) / "images"
    if not img_dir.exists():
        return [], []

    # Load class_labels.json for label mapping
    label_path = Path(project_dir) / "annotations" / "class_labels.json"
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        labels_list = label_data.get("labels", [])
        mapping_dict = label_data.get("mapping", {})
    else:
        labels_list = []
        mapping_dict = {}

    pairs = []
    class_names = list(labels_list)

    # Scan all images
    def _scan_dir(dir_path, rel_prefix):
        nonlocal pairs, class_names, mapping_dict, labels_list
        for entry in sorted(dir_path.iterdir()):
            if entry.is_dir():
                # ImageFolder mode: directory name is the class
                cls_name = entry.name
                if cls_name not in class_names:
                    class_names.append(cls_name)
                _scan_dir(entry, f"{rel_prefix}{cls_name}/")
            elif entry.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"):
                rel = f"{rel_prefix}{entry.name}"
                abs_path = str(entry)
                # Determine label
                label_id = None
                if rel in mapping_dict:
                    label_id = mapping_dict[rel]
                elif f"images/{rel}" in mapping_dict:
                    label_id = mapping_dict[f"images/{rel}"]
                elif rel_prefix and rel_prefix.strip("/").split("/")[-1] in class_names:
                    # ImageFolder mode: parent folder = class
                    cls_name = rel_prefix.rstrip("/").split("/")[-1]
                    if cls_name in class_names:
                        label_id = class_names.index(cls_name)

                if label_id is not None:
                    # Check split
                    base = os.path.splitext(entry.name)[0]
                    rel_key = f"{rel_prefix}{base}" if rel_prefix else base
                    file_split = split_map.get(rel_key, "train")
                    if split == "all" or file_split == split:
                        pairs.append((rel, abs_path, label_id))

    _scan_dir(img_dir, "")

    return pairs, class_names


def _default_split_map(project_dir):
    """Load train_test_split.json or return empty."""
    split_file = Path(project_dir) / "train_test_split.json"
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_split_map(project_dir, split_map):
    """Persist train_test_split.json."""
    split_path = Path(project_dir) / "train_test_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2, ensure_ascii=False)


class ClassificationDataset(Dataset):
    """Image classification PyTorch Dataset.

    Supports:
    - ImageFolder layout: images/cat/xxx.jpg, images/dog/xxx.jpg
    - JSON layout: images/xxx.jpg + class_labels.json mapping
    - train/val split via train_test_split.json (same format as segmentation module)
    """

    def __init__(self, project_dir, split="train", transform=None, image_size=224):
        self.project_dir = Path(project_dir)
        self.split = split
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)

        split_map = _default_split_map(project_dir)
        pairs, class_names = _list_images_mapping(project_dir, split, split_map)

        self._pairs = pairs  # [(rel_path, abs_path, label_id), ...]
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Build default transforms
        if transform is None:
            import torchvision.transforms as T
            if split == "train":
                self.transform = T.Compose([
                    T.Resize(self.image_size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=10),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize(self.image_size),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        rel_path, abs_path, label = self._pairs[idx]
        image = Image.open(abs_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def auto_split(project_dir, val_split=0.2, seed=42):
    """Generate train_test_split.json for classification projects.

    Splits at sample level ensuring per-class ratios are maintained
    via stratified sampling (when possible).
    """
    img_dir = Path(project_dir) / "images"
    if not img_dir.exists():
        return {}

    # Collect all samples
    label_path = Path(project_dir) / "annotations" / "class_labels.json"
    all_samples = {}  # base_name → label_id

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        labels_list = label_data.get("labels", [])
        mapping = label_data.get("mapping", {})
        for rel_path, cls_id in mapping.items():
            base = os.path.splitext(os.path.basename(rel_path))[0]
            all_samples[base] = cls_id
    else:
        # ImageFolder mode
        for entry in sorted(img_dir.iterdir()):
            if not entry.is_dir():
                continue
            cls_name = entry.name
            for f in sorted(entry.iterdir()):
                if f.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"):
                    base = os.path.splitext(f.name)[0]
                    # Assign temporary ID for splitting
                    all_samples[f"images/{cls_name}/{f.name}"] = base

    if len(all_samples) < 2:
        return {}

    # Stratified split by label
    rng = np.random.RandomState(seed)
    keys = sorted(all_samples.keys())
    rng.shuffle(keys)

    split_map = {}
    if label_path.exists():
        # Group by label for stratified split
        from collections import defaultdict
        by_label = defaultdict(list)
        for k in keys:
            by_label[all_samples[k]].append(k)

        for label_id, items in by_label.items():
            n_val = max(1, int(len(items) * val_split))
            for i, k in enumerate(items):
                split_map[k] = "val" if i < n_val else "train"
    else:
        # Simple random split
        n_val = max(1, int(len(keys) * val_split))
        for i, k in enumerate(keys):
            split_map[k] = "val" if i < n_val else "train"

    _save_split_map(project_dir, split_map)
    return split_map


def get_dataset_stats(project_dir):
    """Return statistics for a classification dataset."""
    split_map = _default_split_map(project_dir)
    train_pairs, class_names = _list_images_mapping(project_dir, "train", split_map)
    val_pairs, _ = _list_images_mapping(project_dir, "val", split_map)

    # Count per class in training set
    from collections import Counter
    train_counts = Counter(pid for _, _, pid in train_pairs)

    return {
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "per_class": {class_names[i]: train_counts.get(i, 0) for i in range(len(class_names))},
    }