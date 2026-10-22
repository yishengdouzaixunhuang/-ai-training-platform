"""PyTorch semantic segmentation dataset - sliding window tiled training"""
import os, json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, project_dir, split="train", transform=None, image_size=(512, 512), augment_fn=None,
                 views=None):
        self.project_dir = project_dir
        self.transform = transform
        # Normalize: accept int (square) or tuple (w, h)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.split = split
        # 多视图（三图）工程：views=("te_0","te_1","te_2") ⇒ 每个样本按视图堆叠成 9 通道。
        # ★ 不传 views 时行为与从前**完全一致**（单视图 3 通道），别顺手改默认。
        self.views = tuple(views) if views else ()

        with open(os.path.join(project_dir, "project.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        raw_classes = meta.get("classes", ["background"])
        self.classes = [c for c in raw_classes if c == "background" or ("ignore" not in c.lower() and "gnore" not in c.lower())]
        if "background" not in self.classes:
            self.classes.insert(0, "background")
        self.num_classes = max(2, len(self.classes))
        #: 实际喂给模型的通道数（单视图 3；N 视图 = 3N）
        self.in_channels = 3 * (len(self.views) if self.views else 1)

        img_dir = os.path.join(project_dir, "images")
        ann_dir = os.path.join(project_dir, "annotations")
        self.augment_fn = augment_fn
        self._image_samples = []

        split_file = os.path.join(project_dir, "train_test_split.json")
        split_map = {}
        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                split_map = json.load(f)

        if self.views:
            # 多视图：**以组为单位**取样（不再把 3 个视图当 3 张独立图，
            # 否则同一场景会被重复训练 3 次，验证集也会泄漏同场景的另两视图）。
            self._collect_multiview(img_dir, ann_dir, split_map)
        else:
            for fn in sorted(os.listdir(img_dir)):
                if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                base = os.path.splitext(fn)[0]
                img_path = os.path.join(img_dir, fn)
                json_path = os.path.join(img_dir, base + ".json")
                mask_path = os.path.join(ann_dir, base + "_mask.png")
                if not (os.path.exists(json_path) or os.path.exists(mask_path)):
                    continue
                # ★ 不再默认 "train"：没划分的图**不参与训练**
                #   （用户要求：训练集必须自己设置，标注核对通过才进训练）
                file_split = split_map.get(base)
                if self.split != "all" and file_split != self.split:
                    continue
                self._image_samples.append((img_path, json_path, mask_path))

        # ★ 训练集为空就直接报错 —— 不要"静默训练 0 张图"。
        #   以前因为 split_map 默认 "train"，没划分的图也会被塞进来；
        #   现在默认去掉了，没划分 = 不进训练集，所以这里要明确告诉用户该去设置。
        if self.split == "train" and not self._image_samples:
            raise RuntimeError(
                "训练集为空：没有任何一张图被划分到 train。\n\n"
                "请先在软件里打开「工具 → 数据集划分管理…」，"
                "把核对无误的标注明确设为训练集 / 验证集 / 测试集，再开始训练。\n\n"
                "（标注过 ≠ 进训练集 —— 训练集必须由你显式指定）")

        self._build_tiles()
        # Init lazy cache: None = not loaded yet
        self._mask_cache = [None] * len(self._image_samples)
        self._mask_meta = []  # (json_path, mask_path) for lazy loading
        for _entry in self._image_samples:
            jp, mp = _entry[1], _entry[2]
            self._mask_meta.append((jp, mp))

    def _collect_multiview(self, img_dir, ann_dir, split_map):
        """多视图取样：**一个样本 = 一个场景组的 N 张视图**（不是 N 个独立样本）。

        为什么要按组：视图文件名是 `组名_te_0/1/2`。若按文件逐个收，同一场景会被当成
        3 张不同样本重复训练，验证集也会拿到同场景的另两视图 ⇒ **验证指标虚高**。
        标注取组内**第一个存在的** json/mask（三视图通常共用一份标注形状）。
        缺视图的组**直接跳过并记账**，绝不拿别的视图顶替（那会让"形状图"被漫反射冒充，
        训练看起来正常、结果全错）。
        """
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".bmp")
        groups = {}
        for fn in sorted(os.listdir(img_dir)):
            low = fn.lower()
            if not low.endswith(exts):
                continue
            base = os.path.splitext(fn)[0]
            hit = None
            for v in self.views:
                if base.endswith("_" + v):
                    hit = v
                    break
            if hit is None:
                continue
            stem = base[: -(len(hit) + 1)]          # 去掉 _te_0 后缀 = 组名
            groups.setdefault(stem, {})[hit] = os.path.join(img_dir, fn)
        self._mv_skipped = []
        for stem in sorted(groups):
            got = groups[stem]
            missing = [v for v in self.views if v not in got]
            if missing:
                self._mv_skipped.append((stem, missing))
                continue
            # 标注：优先组名.json，其次任一视图的 json/mask
            json_path = ""
            mask_path = ""
            for cand in ([os.path.join(img_dir, stem + ".json")] +
                         [os.path.join(img_dir, os.path.splitext(os.path.basename(p))[0] + ".json")
                          for p in got.values()] +
                         [os.path.join(ann_dir, stem + "_mask.png")] +
                         [os.path.join(ann_dir, os.path.splitext(os.path.basename(p))[0] + "_mask.png")
                          for p in got.values()]):
                if os.path.exists(cand):
                    if cand.lower().endswith(".json"):
                        json_path = cand
                    else:
                        mask_path = cand
                    if json_path and (mask_path or len(self.views) == 1):
                        break
            if not (json_path or mask_path):
                continue
            # ★ 同上：没划分就是 None，不默认 train
            file_split = split_map.get(stem) or split_map.get(
                os.path.splitext(os.path.basename(got[self.views[0]]))[0])
            if self.split != "all" and file_split != self.split:
                continue
            # img_path 位置放**主视图**（第0 个），多视图路径放在第 4 个元素
            self._image_samples.append((got[self.views[0]], json_path, mask_path,
                                        [got[v] for v in self.views]))

    def _load_mv_image(self, paths, crop_box):
        """读 N 个视图 -> 各自 crop/resize -> 通道拼接 -> (C,H,W) 的 float 张量。

        拼接顺序**严格按 self.views**（te_0 漫反射 / te_1 形状图1 / te_2 形状图2），
        与训练、多视图推理（predict_multiview）保持同一顺序 —— 顺序错位是最隐蔽的错：
        训练能跑、推理能跑、结果全错。
        """
        import torchvision.transforms.functional as _TF
        arrs = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            im = im.crop(crop_box)
            arrs.append(np.array(im, dtype=np.float32))
        h, w = arrs[0].shape[:2]
        norm = []
        for a in arrs:
            if a.shape[0] != h or a.shape[1] != w:
                a = np.array(Image.fromarray(a.astype(np.uint8)).resize((w, h)), dtype=np.float32)
            norm.append(a)
        stacked = np.concatenate(norm, axis=2)        # H,W,3N
        return torch.from_numpy(stacked).permute(2, 0, 1).contiguous() / 255.0

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
        # 元素可能是 3 元组（单视图）或 4 元组（多视图，末尾多一个视图路径列表）
        for sample_idx, _entry in enumerate(self._image_samples):
            img_path = _entry[0]
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
        # 多视图样本的第4 个元素是视图路径列表；单视图只有3 元组（保持原样）
        _mv = self._image_samples[sample_idx][3] if len(self._image_samples[sample_idx]) > 3 else None
        img_path = self._image_samples[sample_idx][0]
        tw, th = self.image_size

        if _mv is not None:
            # ── 多视图：N 张图拼成 3N 通道 ──
            image_t = self._load_mv_image(_mv, (x1, y1, x2, y2))
            mask_np_raw = self._get_mask(sample_idx)
            cw, ch = x2 - x1, y2 - y1
            if cw < tw or ch < th:
                pad = torch.zeros(image_t.shape[0], th, tw, dtype=image_t.dtype)
                pad[:, :ch, :cw] = image_t
                image_t = pad
                new_mask = np.zeros((th, tw), dtype=np.int64)
                new_mask[:ch, :cw] = mask_np_raw[y1:y2, x1:x2].astype(np.int64)
                mask_crop = new_mask
            else:
                mask_crop = mask_np_raw[y1:y2, x1:x2].astype(np.int64)
            if self.split == "train":
                if np.random.random() > 0.5:          # 水平翻转：图像与掩膜同步
                    image_t = torch.flip(image_t, dims=[2])
                    mask_crop = np.fliplr(mask_crop).copy()
            if self.augment_fn:
                # 增强函数按 (PIL.Image, mask) 设计；多视图这里先跳过增强，
                # 宁可少一层增强，也不要用错接口把 9 通道搞坏（宁可少增强，不要错增强）
                pass
            mask_crop = np.where(mask_crop == 255, 255, np.clip(mask_crop, 0, self.num_classes - 1))
            return image_t, torch.from_numpy(mask_crop).long()

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
    """读划分表，返回 (split_map, samples)。

    ★★ 2026-10-22 语义变更（用户明确要求）：
        **没登记 = 未划分 = 不属于任何集**（空集），不再是"默认算训练集"。

    以前这里会给所有已标注样本在内存里补 `train`。看起来是"省事的好心"，
    实际是**两头说法不一致**：
      · 图片列表按 train_test_split.json 判状态，没登记的显示成空集（""）；
      · 训练侧走后门，把没登记的当训练集训了。
    于是"我没让它进训练"这件事只对了一半 —— 用户核对标注时的判断依据是假的。

    真正的问题不是"少训几张"，而是**你无法再用"进过训练"来反推"哪些图被核对过"**。
    所以这里不补了：`dataset.py` / `multiview.py` / 分类侧的取数全部按
    "查不到就取不到"处理，训练集为空会直接报错并指路「工具 → 数据集划分管理…」。

    ⚠️ 依赖它的代码不许再指望"返回值里一定有这张图"：
       `annotation/convert.add_to_train()` 就踩过这个坑 —— 它原来靠"内存里已经
       补了 train"才写得进盘，去掉默认值后会**静默不落盘**，所以它现在显式赋值。
    """
    split_file = os.path.join(project_dir, "train_test_split.json")
    img_dir = os.path.join(project_dir, "images")
    ann_dir = os.path.join(project_dir, "annotations")
    samples = []
    if os.path.isdir(img_dir):
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            base = os.path.splitext(f)[0]
            json_path = os.path.join(img_dir, base + ".json")
            mask_path = os.path.join(ann_dir, base + "_mask.png")
            if os.path.exists(json_path) or os.path.exists(mask_path):
                samples.append(base)
    if os.path.exists(split_file):
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                split_map = json.load(f)
        except Exception:
            split_map = {}
    else:
        split_map = {}
    if not isinstance(split_map, dict):
        split_map = {}
    return split_map, samples


def save_train_test_split(project_dir, split_map):
    split_file = os.path.join(project_dir, "train_test_split.json")
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2, ensure_ascii=False)
