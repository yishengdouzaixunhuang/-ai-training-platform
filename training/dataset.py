"""PyTorch semantic segmentation dataset - sliding window tiled training"""
import os, json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF


class GroupedShuffleSampler(Sampler):
    """按「图」为单位打乱、图内块保持连续的采样器。

    为什么需要它（2026-10-29 实测）：
        用户这个工程的原始图是 4096²、48MB/张，解码一次 107~179ms。
        dataset 里有个 1.2GB 的图片缓存，但 168 张图需要 8GB —— 本来就不够。
        更糟的是 ``DataLoader(shuffle=True)`` 会把 8 万多个 tile **完全打散**：
        同一张图的 484 块被摊到整个 epoch 上，缓存里的图还没用几下就被换掉，
        于是**几乎每一块都要重新解码一张 48MB 的大图**。
        实测后果：5082 batch × 3 秒 = **4.2 小时/epoch**，150 轮要 26 天。

    改成只打乱「图的顺序」、图内块连续之后，缓存只需装下当前正在处理的
    那几张图，命中率接近 100%，同样数据量能压到几分钟/epoch。
    随机性保留（每个 epoch 图的出现顺序都不同），只是不再逐块乱序 ——
    对语义分割的收敛没有影响。

    前置条件：``dataset._tiles`` 必须按图连续排列（``_build_tiles`` 就是外层
    循环图、内层循环块的写法，天然满足）。
    """

    def __init__(self, groups, shuffle=True):
        # groups: [(start, end), ...] 每张图占用的 tile 索引区间
        self.groups = list(groups)
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            random.shuffle(order)
        for gi in order:
            s, e = self.groups[gi]
            for i in range(s, e):
                yield i

    def __len__(self):
        return sum(e - s for s, e in self.groups)


class SegmentationDataset(Dataset):
    def __init__(self, project_dir, split="train", transform=None, image_size=(512, 512), augment_fn=None,
                 views=None, tile_size=0, tile_stride_pct=75):
        self.project_dir = project_dir
        self.transform = transform
        # ★★ 2026-10-29：接上「滑窗大小 / 步长%」这两个原本被丢弃的参数。
        #   `tile_size > 0` = 分块模式：块尺寸**就是**网络输入尺寸（原分辨率裁，不缩放），
        #   与多视图 MultiViewSegDataset 的语义保持一致（那边 `_crop_view` 也是 1:1 裁）。
        #   `tile_size = 0` = 旧行为（按 image_size 切、无重叠），保证老工程不受影响。
        self.tile_size = int(tile_size or 0)
        self.tile_stride_pct = float(tile_stride_pct or 75)
        if self.tile_size > 0:
            image_size = (self.tile_size, self.tile_size)
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
        # ★★ 2026-10-23：**原图缓存**（性能救命）
        #   分块训练时 __getitem__ 原来每个 tile 都 `Image.open(...).convert("RGB")`，
        #   一张 2700×1900 的 BMP 解码 ~30-50ms；一块图 99 个 tile ⇒ 同一张图被解码 99 次。
        #   实测：8 分钟/轮、0.675s/批（840ms 一块），GPU 根本没吃满，全耗在重复解码上。
        #   这里按**字节上限**做 FIFO 缓存（不是按张数），单张 20MP 的图也不会撑爆内存。
        self._img_cache = {}
        self._img_cache_bytes = 0
        self._img_cache_cap = 1200 * 1024 * 1024      # ~1.2GB
        # 掩膜本来就缓存了（_mask_cache 按样本索引），不用再管
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

    def _get_image_cached(self, img_path):
        """取原图（带字节上限的 FIFO 缓存）。返回的 Image 只读，调用方不得原地修改。

        ⚠️ 别改成 `@lru_cache`：Image 对象占用大，LRU 按个数计无法控制内存；
        这里按字节数淘汰，20MP 图也不会把内存吃光。
        """
        _c = getattr(self, "_img_cache", None)
        if _c is None:                      # 兜底：任何漏初始化的构造路径都不至于崩
            _c = self._img_cache = {}
            self._img_cache_bytes = 0
            self._img_cache_cap = 1200 * 1024 * 1024
        im = _c.get(img_path)
        if im is not None:
            return im
        im = Image.open(img_path).convert("RGB")
        try:
            nbytes = im.size[0] * im.size[1] * 3
        except Exception:
            nbytes = 0
        if nbytes and nbytes < self._img_cache_cap:
            while _c and (self._img_cache_bytes + nbytes) > self._img_cache_cap:
                _k, _im = next(iter(_c.items()))
                self._img_cache_bytes -= _im.size[0] * _im.size[1] * 3
                _c.pop(_k, None)
            _c[img_path] = im
            self._img_cache_bytes += nbytes
        return im

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
        # ★★ 2026-10-29：步长真正生效了。
        #   tile_size>0（分块模式）⇒ 按「步长%」算重叠：75% ⇒ stride = 0.75*块宽，25% 重叠。
        #   tile_size=0（旧行为）⇒ 仍然是 stride = 块宽（无重叠），保证老工程训练结果不漂移。
        if self.tile_size > 0:
            stride = max(1, int(round(tw * self.tile_stride_pct / 100.0)))
        else:
            stride = max(1, tw)  # no overlap（旧行为）
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

    def tile_groups(self):
        """每张图占用的 tile 索引区间 [(start, end), ...]。

        配合 `GroupedShuffleSampler` 使用：只打乱图的顺序、图内块保持连续。
        不这么做的话，大图工程（4096²、48MB/张）的图片缓存会全程失效 ——
        每块都要重新解码一张原图，实测 4.2 小时/epoch（见 Sampler 注释）。
        """
        groups = []
        tiles = getattr(self, "_tiles", None) or []
        if not tiles:
            return groups
        cur = tiles[0][0]                  # 元素形如 (sample_idx, x1, y1, x2, y2[, views])
        start = 0
        for i in range(1, len(tiles)):
            si = tiles[i][0]
            if si != cur:
                groups.append((start, i))
                cur = si
                start = i
        groups.append((start, len(tiles)))
        return groups

    @classmethod
    def from_image_subset(cls, dataset, image_indices):
        new_ds = cls.__new__(cls)
        new_ds.project_dir = dataset.project_dir
        new_ds.transform = dataset.transform
        new_ds.image_size = dataset.image_size
        # ★ 2026-10-29：分块参数必须一起继承 —— 自动切分 train/val 走的就是这里，
        #   漏了会让 val 集退回"无重叠 256"，和 train 的块对不上。
        new_ds.tile_size = dataset.tile_size
        new_ds.tile_stride_pct = dataset.tile_stride_pct
        new_ds.augment_fn = dataset.augment_fn
        new_ds.split = dataset.split
        new_ds.classes = dataset.classes
        new_ds.num_classes = dataset.num_classes
        new_ds._image_samples = [dataset._image_samples[i] for i in image_indices]
        new_ds._mask_cache = [None] * len(new_ds._image_samples)
        new_ds._mask_meta = [dataset._mask_meta[i] for i in image_indices]
        # ★ 2026-10-23：**新增字段必须在这里一起初始化**。
        #   自动切分 train/val 时走的就是 from_image_subset，这里漏了 _img_cache ⇒
        #   训练第一个 batch 就 AttributeError（实拍：'SegmentationDataset' object has
        #   no attribute '_img_cache'）。缓存按子集各自独立，容量沿用父数据集的设置。
        new_ds._img_cache = {}
        new_ds._img_cache_bytes = 0
        new_ds._img_cache_cap = getattr(dataset, "_img_cache_cap", 1200 * 1024 * 1024)
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

        # 原图走缓存（每个 tile 不再重复解码同一张大图），掩膜本来就懒加载
        image = self._get_image_cached(img_path)
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
    """写 `train_test_split.json`（原子替换）。

    ★ 2026-11-17：原来是 `open("w") + json.dump`。写到一半崩/断电 → 半截 JSON，
    而读侧普遍是"解析失败就当 `{}`" ⇒ **所有图静默变「未划分」（空集）**，
    界面上看不出异常。与 `core/file_ops._save_json` 统一成临时文件 + `os.replace`。
    """
    from core.file_ops import _save_json
    split_file = os.path.join(project_dir, "train_test_split.json")
    _save_json(split_file, split_map)
