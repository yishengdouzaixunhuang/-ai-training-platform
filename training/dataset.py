"""PyTorch semantic segmentation dataset - sliding window tiled training"""
import os, json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, project_dir, split="train", transform=None, image_size=(512, 512), augment_fn=None):
        self.project_dir = project_dir
        self.transform = transform
        # Normalize: accept int (square) or tuple (w, h)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.split = split

        with open(os.path.join(project_dir, "project.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        raw_classes = meta.get("classes", ["background"])
        self.classes = [c for c in raw_classes if c == "background" or ("ignore" not in c.lower() and "gnore" not in c.lower())]
        if "background" not in self.classes:
            self.classes.insert(0, "background")
        self.num_classes = max(2, len(self.classes))

        img_dir = os.path.join(project_dir, "images")
        ann_dir = os.path.join(project_dir, "annotations")
        self.augment_fn = augment_fn
        self._image_samples = []

        split_file = os.path.join(project_dir, "train_test_split.json")
        split_map = {}
        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                split_map = json.load(f)

        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            base = os.path.splitext(fn)[0]
            img_path = os.path.join(img_dir, fn)
            json_path = os.path.join(img_dir, base + ".json")
            mask_path = os.path.join(ann_dir, base + "_mask.png")
            if not (os.path.exists(json_path) or os.path.exists(mask_path)):
                continue
            file_split = split_map.get(base, "train")
            if self.split != "all" and file_split != self.split:
                continue
            self._image_samples.append((img_path, json_path, mask_path))

        self._build_tiles()
        # Init lazy cache: None = not loaded yet
        self._mask_cache = [None] * len(self._image_samples)
        self._mask_meta = []  # (json_path, mask_path) for lazy loading
        for _, jp, mp in self._image_samples:
            self._mask_meta.append((jp, mp))

    def _get_mask(self, sample_idx):
        """Lazy-load mask for a given sample index."""
        if self._mask_cache[sample_idx] is not None:
            return self._mask_cache[sample_idx]
        json_path, mask_path = self._mask_meta[sample_idx]
        img_path = self._image_samples[sample_idx][0]
        if os.path.exists(json_path):
            try:
                from annotation.labelme_io import load_labelme_json, shapes_to_mask
                shapes, info = load_labelme_json(json_path)
                if shapes:
                    h = info.get("imageHeight", self.image_size[1])
                    w = info.get("imageWidth", self.image_size[0])
                    mask = shapes_to_mask(shapes, self.classes, (h, w))
                    self._mask_cache[sample_idx] = mask.astype(np.uint8)
                    return self._mask_cache[sample_idx]
            except Exception:
                pass
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path)
            self._mask_cache[sample_idx] = np.array(mask_img).astype(np.uint8)
            return self._mask_cache[sample_idx]
        # No mask found ? create zero mask
        with Image.open(img_path) as im:
            w, h = im.size
        self._mask_cache[sample_idx] = np.zeros((h, w), dtype=np.uint8)
        return self._mask_cache[sample_idx]

    def _build_tiles(self):
        tw, th = self.image_size
        stride = max(1, tw)  # no overlap
        self._tiles = []
        for sample_idx, (img_path, _, _) in enumerate(self._image_samples):
            with Image.open(img_path) as im:
                iw, ih = im.size
            if iw <= tw and ih <= th:
                self._tiles.append((sample_idx, 0, 0, iw, ih))
            else:
                for y1 in range(0, ih, stride):
                    for x1 in range(0, iw, stride):
                        x2 = min(x1 + tw, iw)
                        y2 = min(y1 + th, ih)
                        self._tiles.append((sample_idx, x1, y1, x2, y2))

    @classmethod
    def from_image_subset(cls, dataset, image_indices):
        new_ds = cls.__new__(cls)
        new_ds.project_dir = dataset.project_dir
        new_ds.transform = dataset.transform
        new_ds.image_size = dataset.image_size
        new_ds.augment_fn = dataset.augment_fn
        new_ds.split = dataset.split
        new_ds.classes = dataset.classes
        new_ds.num_classes = dataset.num_classes
        new_ds._image_samples = [dataset._image_samples[i] for i in image_indices]
        new_ds._mask_cache = [None] * len(new_ds._image_samples)
        new_ds._mask_meta = [dataset._mask_meta[i] for i in image_indices]
        new_ds._build_tiles()
        return new_ds

    def __len__(self):
        return len(self._tiles)

    def _load_mask(self, json_path, mask_path):
        if os.path.exists(json_path):
            try:
                from annotation.labelme_io import load_labelme_json, shapes_to_mask
                shapes, info = load_labelme_json(json_path)
                if shapes:
                    h = info.get("imageHeight", self.image_size[1])
                    w = info.get("imageWidth", self.image_size[0])
                    mask = shapes_to_mask(shapes, self.classes, (h, w))
                    return Image.fromarray(mask.astype(np.uint8))
            except Exception:
                pass
        if os.path.exists(mask_path):
            return Image.open(mask_path)
        return None

    def __getitem__(self, idx):
        sample_idx, x1, y1, x2, y2 = self._tiles[idx]
        img_path = self._image_samples[sample_idx][0]

        # Read image from disk, mask lazy-loaded
        image = Image.open(img_path).convert("RGB")
        mask_np_raw = self._get_mask(sample_idx)

        tw, th = self.image_size
        image = image.crop((x1, y1, x2, y2))
        mask_crop = mask_np_raw[y1:y2, x1:x2].astype(np.int64)

        cw, ch = image.size
        if cw < tw or ch < th:
            new_img = Image.new("RGB", (tw, th), (0, 0, 0))
            new_img.paste(image, (0, 0))
            image = new_img
            new_mask = np.zeros((th, tw), dtype=np.int64)
            new_mask[:ch, :cw] = mask_crop
            mask_crop = new_mask

        if self.split == "train":
            if np.random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask_crop = np.fliplr(mask_crop).copy()
            angle = np.random.uniform(-15, 15)
            if abs(angle) > 1:
                image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
                mask_pil = Image.fromarray(mask_crop.astype(np.uint8))
                mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
                mask_crop = np.array(mask_pil).astype(np.int64)

        if self.augment_fn:
            image, mask_crop = self.augment_fn(image, mask_crop)

        image = TF.to_tensor(image)
        mask_crop = np.where(mask_crop == 255, 255, np.clip(mask_crop, 0, self.num_classes - 1))
        mask = torch.from_numpy(mask_crop).long()

        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)

        return image, mask


def get_train_test_split(project_dir):
    split_file = os.path.join(project_dir, "train_test_split.json")
    img_dir = os.path.join(project_dir, "images")
    ann_dir = os.path.join(project_dir, "annotations")
    samples = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        base = os.path.splitext(f)[0]
        json_path = os.path.join(img_dir, base + ".json")
        mask_path = os.path.join(ann_dir, base + "_mask.png")
        if os.path.exists(json_path) or os.path.exists(mask_path):
            samples.append(base)
    if os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            split_map = json.load(f)
    else:
        split_map = {}
    for s in samples:
        if s not in split_map:
            split_map[s] = "train"
    return split_map, samples


def save_train_test_split(project_dir, split_map):
    split_file = os.path.join(project_dir, "train_test_split.json")
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2, ensure_ascii=False)
