# -*- coding: utf-8 -*-
"""Classification Label Manager — class_labels.json I/O + ImageFolder auto-labeling.

Supports:
1. Manual JSON label editing: {"labels": ["cat","dog"], "mapping": {"img1.jpg": 0, ...}}
2. ImageFolder auto-detection: subfolder name → class ID
"""

import json
import os
from pathlib import Path
from collections import Counter


class LabelManager:
    """Read, write and maintain classification label mappings."""

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)

    @property
    def label_path(self):
        return self.project_dir / "annotations" / "class_labels.json"

    def load(self):
        """Load labels from class_labels.json. Returns {"labels": [...], "mapping": {...}}."""
        if self.label_path.exists():
            with open(self.label_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"labels": [], "mapping": {}}

    def save(self, data):
        """Save labels to class_labels.json."""
        self.label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.label_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def auto_label_from_folders(self):
        """Scan images/ subfolders and assign class labels automatically.

        Example: images/cat/001.jpg, images/dog/001.jpg
        → labels=["cat","dog"], mapping={"images/cat/001.jpg": 0, ...}
        """
        img_dir = self.project_dir / "images"
        if not img_dir.exists():
            return {"labels": [], "mapping": {}}

        # Collect subfolders
        subdirs = sorted(
            d.name for d in img_dir.iterdir() if d.is_dir()
        )
        if not subdirs:
            return {"labels": [], "mapping": {}}

        labels = subdirs  # folder name = class name
        mapping = {}
        for cls_id, cls_name in enumerate(labels):
            cls_dir = img_dir / cls_name
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"):
                    rel_path = f"images/{cls_name}/{f.name}"
                    mapping[rel_path] = cls_id

        result = {"labels": labels, "mapping": mapping}
        self.save(result)
        return result

    def get_num_classes(self):
        """Return number of classes from stored labels."""
        data = self.load()
        return len(data.get("labels", []))

    def get_class_names(self):
        """Return list of class names."""
        data = self.load()
        return data.get("labels", [])

    def get_mapping(self):
        """Return image→class_id mapping dict."""
        data = self.load()
        return data.get("mapping", {})

    def add_class(self, class_name):
        """Add a new class label."""
        data = self.load()
        labels = data.get("labels", [])
        if class_name not in labels:
            labels.append(class_name)
            data["labels"] = labels
            self.save(data)
        return data

    def remove_class(self, class_name):
        """Remove a class label and re-index mappings."""
        data = self.load()
        labels = data.get("labels", [])
        if class_name not in labels:
            return data

        old_idx = labels.index(class_name)
        labels.remove(class_name)
        data["labels"] = labels

        # Re-index mappings: shift IDs down for classes above the removed one
        new_mapping = {}
        for img_path, cls_id in data.get("mapping", {}).items():
            if cls_id == old_idx:
                continue  # drop images belonging to removed class
            elif cls_id > old_idx:
                new_mapping[img_path] = cls_id - 1
            else:
                new_mapping[img_path] = cls_id
        data["mapping"] = new_mapping
        self.save(data)
        return data

    def set_image_label(self, image_rel_path, class_name):
        """Assign a single image to a class (auto-adds class if new)."""
        data = self.load()
        labels = data.get("labels", [])
        if class_name not in labels:
            labels.append(class_name)
            data["labels"] = labels
        cls_id = labels.index(class_name)
        data.setdefault("mapping", {})[image_rel_path] = cls_id
        self.save(data)
        return data

    def get_statistics(self):
        """Return per-class sample counts."""
        data = self.load()
        labels = data.get("labels", [])
        mapping = data.get("mapping", {})
        counts = Counter(mapping.values())
        return {
            labels[i]: counts.get(i, 0) for i in range(len(labels))
        }