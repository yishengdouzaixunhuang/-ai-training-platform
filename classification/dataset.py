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
                    # ★★ 2026-10-22：默认值从 "train" 改成 None（未划分 = 不属于任何集）。
                    #    与分割侧 training/dataset.py 口径一致：没在 train_test_split.json
                    #    里登记的图，训练/验证/测试都取不到，宁可报"训练集为空"也不偷偷训。
                    base = os.path.splitext(entry.name)[0]
                    rel_key = f"{rel_prefix}{base}" if rel_prefix else base
                    file_split = split_map.get(rel_key)
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


class _GlobalAugment:
    """全局增强包装：封装增强开关 dict，可被 DataLoader 多进程 pickle。"""

    def __init__(self, flags):
        self.flags = dict(flags or {})

    def __call__(self, image):
        from training.augment_config import apply_image_augment
        return apply_image_augment(image, self.flags)


class KeepAspectResize:
    """等比缩放后居中填边（letterbox）到固定尺寸，保持原始宽高比、不变形。

    与 T.Resize((h, w)) 直接拉伸不同：本变换按 min(tw/w, th/h) 缩放，使整张图
    完整落在目标画布内，短边用 fill 填充。适合长条状缺陷（如极耳 2.5:1）避免
    压成正方形后丢失形状信息。

    Args:
        size: int（正方形）或 (W, H) 元组（目标画布宽、高）。
        fill: 填充色 0~255，默认 0（黑边）。
    """

    def __init__(self, size, fill=0):
        if isinstance(size, int):
            size = (size, size)
        self.size = (int(size[0]), int(size[1]))  # (W, H)
        self.fill = int(fill)

    def __call__(self, img):
        tw, th = self.size  # 目标 (W, H)
        w, h = img.size
        if w == tw and h == th:
            return img
        scale = min(tw / w, th / h)
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (tw, th), (self.fill,) * 3)
        canvas.paste(img, ((tw - nw) // 2, (th - nh) // 2))
        return canvas


class ClassificationDataset(Dataset):
    """Image classification PyTorch Dataset.

    Supports:
    - ImageFolder layout: images/cat/xxx.jpg, images/dog/xxx.jpg
    - JSON layout: images/xxx.jpg + class_labels.json mapping
    - train/val split via train_test_split.json (same format as segmentation module)
    """

    def __init__(self, project_dir, split="train", transform=None, image_size=224, scale_factor=1.0,
                 augment_flags=None, keep_aspect=False):
        self.project_dir = Path(project_dir)
        self.split = split
        self.image_size = image_size if (image_size is None or isinstance(image_size, (tuple, list))) else (image_size, image_size)
        self.scale_factor = scale_factor
        self.keep_aspect = keep_aspect

        split_map = _default_split_map(project_dir)
        pairs, class_names = _list_images_mapping(project_dir, split, split_map)
        # ★ 未划分 = 空集之后，"训练集 0 张"是很容易撞上的状态（新工程、或只在
        #   列表里标了类别没做划分）⇒ 必须在这里就说清楚，不能等 trainer 训到一半
        #   报个 IndexError、或者更糟——静默训 0 张。
        if split == "train" and not pairs:
            raise RuntimeError(
                "训练集为空：没有任何一张图被划分到 train。\n\n"
                "请先打开「工具 → 数据集划分管理…」，把核对无误的标注显式设为训练集，"
                "再开始训练。\n\n（标注过 ≠ 进训练集 —— 未划分的图不属于任何集）")

        self._pairs = pairs  # [(rel_path, abs_path, label_id), ...]
        self._target_size = self._compute_target_size() if scale_factor != 1.0 else None
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Build default transforms
        if transform is None:
            import torchvision.transforms as T
            resize_step = self._build_resize_step()
            steps = []
            if resize_step is not None:
                steps.append(resize_step)
            if split == "train":
                steps += [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=10),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]
            steps += [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform = T.Compose(steps)
        else:
            self.transform = transform

        # 全局增强参数面板
        # - augment_flags=None                     → 用内置增强（翻转 + ±10° 旋转 + 色彩抖动）
        # - {"enabled": False, ...}                → 明确关闭全部增强
        #   （旧实现在这里直接 return，静默回落到内置增强，与 UI 上「无增强」标签不符）
        # - {"enabled": True, ...}                 → 用可配置增强替换内置增强
        self.augment_desc = "关闭" if split != "train" else "内置(翻转+旋转±10°+色彩抖动)"
        if transform is None and split == "train" and augment_flags is not None:
            keep = []
            for s in self.transform.transforms:
                if isinstance(s, (T.RandomHorizontalFlip, T.RandomRotation, T.ColorJitter)):
                    continue
                keep.append(s)
            if augment_flags.get("enabled"):
                self.transform = T.Compose([_GlobalAugment(augment_flags)] + keep)
                on = [k for k, v in augment_flags.items() if k != "enabled" and v]
                self.augment_desc = ("全局[" + ",".join(on) + "]") if on else "全局(无启用项)"
            else:
                self.transform = T.Compose(keep)
                self.augment_desc = "关闭"

    def _build_resize_step(self):
        """返回 resize 变换；keep_aspect=True 时用 letterbox 等比填边，否则 T.Resize 直接拉伸。

        image_size / _target_size 均约定为 (W, H)。返回 None 表示不做 resize（保持原尺寸，
        由 trainer 的 _pad_collate 在 batch 内补齐）。
        """
        import torchvision.transforms as T
        size = None
        if self.image_size is not None:
            size = self.image_size
        elif self._target_size is not None:
            size = self._target_size
        if size is None:
            return None
        w, h = int(size[0]), int(size[1])
        if self.keep_aspect:
            return KeepAspectResize((w, h))
        return T.Resize((h, w))

    def _compute_target_size(self):
        """Compute target size from scale_factor.
        If scale_factor is a tuple (sw, sh), use per-image scaling in __getitem__.
        If float, compute uniform target from max dimensions."""
        if isinstance(self.scale_factor, (tuple, list)):
            sw, sh = self.scale_factor
            max_w, max_h = 0, 0
            for _, abs_path, _ in self._pairs:
                try:
                    with Image.open(abs_path) as img:
                        w, h = img.size
                    max_w = max(max_w, w)
                    max_h = max(max_h, h)
                except Exception:
                    pass
            tw = max(int(max_w * sw), 1)
            th = max(int(max_h * sh), 1)
            return (tw, th)
        max_w, max_h = 0, 0
        for _, abs_path, _ in self._pairs:
            try:
                with Image.open(abs_path) as img:
                    w, h = img.size
                max_w = max(max_w, w)
                max_h = max(max_h, h)
            except Exception:
                pass
        tw = max(int(max_w * self.scale_factor), 1)
        th = max(int(max_h * self.scale_factor), 1)
        return (tw, th)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        rel_path, abs_path, label = self._pairs[idx]
        image = Image.open(abs_path).convert("RGB")
        # Pre-scale 只在「没有固定输入尺寸」时才需要：image_size 已固定时
        # transform 里还会再 resize 一次，先缩一遍只是多一次重采样、白掉画质。
        if self.image_size is None:
            if isinstance(self.scale_factor, (tuple, list)):
                sw, sh = self.scale_factor
                w, h = image.size
                image = image.resize((max(1, int(w * sw)), max(1, int(h * sh))), Image.BILINEAR)
            elif self.scale_factor != 1.0 and self._target_size is not None:
                image = image.resize(self._target_size, Image.BILINEAR)
        if self.transform:
            image = self.transform(image)
        return image, label



def _preserve_test_entries(project_dir, split_map):
    """Keep existing test entries from the old split file."""
    split_path = Path(project_dir) / "train_test_split.json"
    if split_path.exists():
        try:
            with open(split_path, "r", encoding="utf-8") as f:
                old_split = json.load(f)
            for k, v in old_split.items():
                if v in ("test", "Test"):
                    split_map[k] = "test"
        except Exception:
            pass
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
    all_samples = {}  # 查找键 → 分层标签
    stratified = False

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        labels_list = label_data.get("labels", [])
        mapping = label_data.get("mapping", {})
        for rel_path, cls_id in mapping.items():
            base = os.path.splitext(os.path.basename(rel_path))[0]
            all_samples[base] = cls_id
        stratified = len(all_samples) >= 2

    # mapping 为空（或没有 class_labels.json）时，回退到 ImageFolder 模式按文件夹扫描：
    # 否则样本集为空 -> 不写划分 -> val 恒为空 -> 训练报 "Validation set is empty"
    if len(all_samples) < 2:
        all_samples = {}
        for entry in sorted(img_dir.iterdir()):
            if not entry.is_dir():
                continue
            cls_name = entry.name
            for f in sorted(entry.iterdir()):
                if f.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"):
                    base = os.path.splitext(f.name)[0]
                    # key 必须与 _list_images_mapping 的查找键一致："{类别}/{文件名去扩展名}"
                    # value 用类别名，供下面按类别分层划分
                    all_samples[f"{cls_name}/{base}"] = cls_name
        stratified = True

    if len(all_samples) < 2:
        return {}

    # Stratified split by label
    rng = np.random.RandomState(seed)
    keys = sorted(all_samples.keys())
    rng.shuffle(keys)

    split_map = {}
    if stratified:
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

    _preserve_test_entries(project_dir, split_map)
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