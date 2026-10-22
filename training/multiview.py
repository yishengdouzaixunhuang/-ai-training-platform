# -*- coding: utf-8 -*-
"""三图（多视图）语义分割 —— 早融合 in_channels = 3视图 × 3通道 = 9

客户场景：xxx_te_0.jpg = 漫反射, xxx_te_1.jpg = 形状图1, xxx_te_2.jpg = 形状图2
三张图**同尺寸、同裁剪区域**，标注只画在对比度最高的那张上、其余同步 —— 
因此可以直接按通道堆叠做"早融合"，只需把分割 backbone 的首层卷积改成 9 通道。

本模块为**纯新增**，不修改平台原有 segmentation 代码；产出 checkpoint 沿用平台约定：
    models/multiview_best.pth = {model_state_dict, classes, num_classes,
                                 in_channels, views, model_name, image_size, metrics}

用法（命令行）:
    python -m training.multiview <project_dir> --epochs 30 --model deeplabv3 --image-size 512
用法（代码）:
    from training.multiview import train_multiview, predict_multiview
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 视图顺序：te_0=漫反射, te_1=形状图1, te_2=形状图2（客户约定）
DEFAULT_VIEWS = ("te_0", "te_1", "te_2")
VIEW_NAMES = {"te_0": "漫反射", "te_1": "形状图1", "te_2": "形状图2"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ────────────────────────── 数据 ──────────────────────────
def _group_key(filename, views=DEFAULT_VIEWS):
    """'xxx_te_0.jpg' -> ('xxx', 'te_0')；非本约定返回 (None, None)。"""
    base = os.path.splitext(filename)[0]
    for v in views:
        if base.endswith("_" + v):
            return base[: -(len(v) + 1)], v
    return None, None


def _noop_log(msg):
    """模块级函数，不能用 lambda —— DataLoader(num_workers>0) 要 pickle 数据集，
    lambda 不可 picklable，会让多进程加载直接崩掉。"""
    return None


def _print_log(msg):
    """同上：默认日志函数必须是模块级可 pickle 的函数，不能是 lambda。
    否则 `--workers > 0` 时会报
    AttributeError: Can't get local object 'train_multiview.<locals>.<lambda>'"""
    print(msg, flush=True)


class MultiViewSegDataset(Dataset):
    """三图分割数据集：每个样本 = 3 视图堆叠(9通道) + 单张掩膜。

    目录约定（与平台一致）：
        <project>/images/            xxx_te_0.jpg / xxx_te_1.jpg / xxx_te_2.jpg (+ 可选 xxx_te_0.json)
        <project>/annotations/       xxx_te_0_mask.png（掩膜；三视图共用，取任一存在者）
        <project>/train_test_split.json    {"xxx_te_0": "train"|"val"|"test", ...}
        <project>/project.json       {"classes": ["background", ...], ...}
    """

    def __init__(self, project_dir, split="train", image_size=512, views=DEFAULT_VIEWS,
                 augment=False, require_all_views=True, log=None, jitter=0,
                 tile_size=0, neg_ratio=1.5, tile_mode="slide", tile_stride_pct=75):
        self.project_dir = Path(project_dir)
        self.img_dir = self.project_dir / "images"
        self.ann_dir = self.project_dir / "annotations"
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self.views = tuple(views)
        self.augment = augment
        self.jitter = int(jitter or 0)
        # **原分辨率切图训练**（tile_size>0 时生效）：
        #   以前是 `整图 resize 到 image_size` —— 3860×1480 缩到 512 宽（7.5 倍），
        #   中位 234px 的 Atd 缩完只剩 ~4px、127px 的 Zw 剩 ~2px，**在数据加载阶段就被抹掉了**。
        #   改成在**原分辨率**上裁 tile，缺陷像素一个不丢。
        self.tile_size = int(tile_size or 0)
        self.neg_ratio = float(neg_ratio if neg_ratio is not None else 1.5)
        # 取样方式（对齐友商 友商平台 的「分块」）：
        #   slide  —— **均匀滑窗**：滑窗大小 × 步长(%)，全图遍历、25% 重叠时不漏任何区域
        #   defect —— 缺陷居中 + 按比例抽纯背景负样本（对小缺陷、类别极不均衡的数据更省算力）
        self.tile_mode = str(tile_mode or "slide")
        self.tile_stride_pct = float(tile_stride_pct or 75)
        self.tiles = []            # [(sample_idx, x, y, 是否有缺陷)]；tile_size<=0 时为空、退回整图缩放模式
        # 日志函数只在构造期用；self.log 固定为模块级函数，
        # 保证数据集对象**永远可 pickle**（DataLoader num_workers>0 需要），
        # 不管调用方传进来的是不是 lambda。
        _log = log or _noop_log
        self.log = _noop_log

        meta_path = self.project_dir / "project.json"
        classes = ["background"]
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                classes = json.load(f).get("classes", ["background"]) or ["background"]
        self.classes = list(classes)
        self.num_classes = max(2, len(self.classes))

        split_map = {}
        sp = self.project_dir / "train_test_split.json"
        if sp.exists():
            with open(sp, "r", encoding="utf-8") as f:
                split_map = json.load(f)

        self.samples = []          # [(key, {view: path}, mask_path)]
        missing = 0
        unassigned = 0             # 划分表里没有这一组（= 未划分 = 空集，不进任何集）
        seen = set()               # 每个 key 只能生成一个样本（否则三视图各算一次，样本数虚增 3 倍）
        if not self.img_dir.exists():
            _log(f"images 目录不存在: {self.img_dir}")
            return
        for fn in sorted(os.listdir(self.img_dir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            key, view = _group_key(fn, self.views)
            if key is None or key in seen:
                continue
            seen.add(key)
            # split 过滤（split_map 以 "key_te_0" 或 "key" 为键）
            # ★★ 2026-10-22：默认值从 "train" 改成 None —— 没划分的组**不属于任何集**。
            #    原来 `split_map.get(key, "train")` 让"没划分"悄悄等于"进训练集"，
            #    于是用户以为只训了自己划的那批，实际把整个 images/ 都训了。
            if split != "all":
                s = split_map.get(f"{key}_{self.views[0]}", split_map.get(key))
                if s != split:
                    if s is None:
                        unassigned += 1
                    continue
            # 找齐三视图
            paths = {}
            for v in self.views:
                p = self.img_dir / f"{key}_{v}.jpg"
                if not p.exists():
                    for ext in (".png", ".jpeg", ".bmp"):
                        q = self.img_dir / f"{key}_{v}{ext}"
                        if q.exists():
                            p = q
                            break
                paths[v] = str(p) if p.exists() else None
            if require_all_views and any(p is None for p in paths.values()):
                missing += 1
                continue
            # 掩膜：三视图共用，取第一个存在的
            mask_path = None
            for v in self.views:
                for ext in (".png", ".bmp"):
                    mp = self.ann_dir / f"{key}_{v}_mask{ext}"
                    if mp.exists():
                        mask_path = str(mp)
                        break
                if mask_path:
                    break
            if mask_path is None:
                # 退而求其次：LabelMe json（放在 images 下）
                for v in self.views:
                    jp = self.img_dir / f"{key}_{v}.json"
                    if jp.exists():
                        mask_path = str(jp)
                        break
            if mask_path is None:
                missing += 1
                continue
            self.samples.append((key, paths, mask_path))
        if missing:
            _log(f"[{split}] 跳过 {missing} 组（缺视图或缺掩膜）")
        if unassigned:
            _log(f"[{split}] 跳过 {unassigned} 组（未划分 —— 训练/验证/测试都不取，"
                 f"要它进训练请到「工具 → 数据集划分管理…」显式设为训练集）")
        _log(f"[{split}] 多视图样本 {len(self.samples)} 组 × {len(self.views)} 图 "
                 f"→ 输入 {len(self.views) * 3} 通道；类别 {self.num_classes}")

        # ── 建切图索引：每张缺陷各出一个"以缺陷为中心"的正样本，
        #    再按 neg_ratio 抽同样多的"完全无缺陷"负样本；全图无缺陷则全是负样本。
        if self.tile_size > 0:
            self._build_tile_index(_log)
            n_pos = sum(1 for t in self.tiles if t[3])
            if self.tile_mode == "slide":
                _log(f"[{split}] 分块训练：滑窗遍历  滑窗={self.tile_size}"
                     f"  步长={int(self.tile_size * self.tile_stride_pct / 100)}"
                     f"  共 {len(self.tiles)} 块（覆盖全图，不预判缺陷块）")
            else:
                _log(f"[{split}] 分块训练：缺陷居中+负样本  滑窗={self.tile_size}"
                     f"  共 {len(self.tiles)} 块（含缺陷 {n_pos} / 纯背景 {len(self.tiles) - n_pos}）")

    def _slide_positions(self, W, H):
        """均匀滑窗位置：按 步长=滑窗×pct% 铺满全图，并保证右/下边缘被覆盖。"""
        t = self.tile_size
        stride = max(1, int(round(t * self.tile_stride_pct / 100.0)))
        xs = list(range(0, max(1, W - t + 1), stride))
        ys = list(range(0, max(1, H - t + 1), stride))
        if W > t and xs[-1] != W - t:
            xs.append(W - t)
        if H > t and ys[-1] != H - t:
            ys.append(H - t)
        return [x for x in xs if 0 <= x <= max(0, W - t)], [y for y in ys if 0 <= y <= max(0, H - t)]

    def _build_tile_index(self, _log):
        """建 tile 列表。slide 模式只需图像尺寸（索引秒建）；defect 模式要读掩膜找缺陷。"""
        t = self.tile_size
        if self.tile_mode == "slide":
            for si, (_key, paths, _mask) in enumerate(self.samples):
                p = next((x for x in paths.values() if x), None)
                if not p:
                    continue
                try:
                    with Image.open(p) as im:
                        W, H = im.size
                except Exception:
                    continue
                xs, ys = self._slide_positions(W, H)
                for y0 in ys:
                    for x0 in xs:
                        self.tiles.append((si, x0, y0, False))   # slide 模式不预先判"有没有缺陷"
            return
        half = t // 2
        rng = np.random.default_rng(1234)
        for si, (_key, _paths, mask_path) in enumerate(self.samples):
            try:
                if str(mask_path).lower().endswith(".json"):
                    from annotation.labelme_io import load_labelme_json, shapes_to_mask
                    shapes, info = load_labelme_json(mask_path)
                    H = int(info.get("imageHeight") or 0); W = int(info.get("imageWidth") or 0)
                    m = shapes_to_mask(shapes, self.classes, (H, W)).astype(np.uint8) if H and W else None
                else:
                    m = np.array(Image.open(mask_path).convert("L"))
                    H, W = m.shape
            except Exception:
                continue
            if m is None or H < 8 or W < 8:
                continue
            defect = ((m != 0) & (m != 255))
            ys, xs = np.nonzero(defect)
            centers = []
            if len(ys):
                # 用 64px 网格把缺陷像素聚类，避免一块大缺陷出一堆重复 tile
                seen = set()
                for y, x in zip(ys[::max(1, len(ys) // 400)], xs[::max(1, len(xs) // 400)]):
                    ck = (int(x) // 64, int(y) // 64)
                    if ck in seen:
                        continue
                    seen.add(ck)
                    centers.append((int(x), int(y)))
            picks = []
            for (cx, cy) in centers:                      # 正样本：缺陷居中
                x0 = max(0, min(W - t, cx - half)) if W > t else 0
                y0 = max(0, min(H - t, cy - half)) if H > t else 0
                picks.append((x0, y0, True))
            n_neg = max(1, int(round(len(picks) * self.neg_ratio))) if picks else 6
            for _ in range(n_neg * 3):                    # 负样本：随机位置且块内无缺陷
                if sum(1 for p in picks if not p[2]) >= n_neg:
                    break
                x0 = int(rng.integers(0, max(1, W - t + 1))) if W > t else 0
                y0 = int(rng.integers(0, max(1, H - t + 1))) if H > t else 0
                sub = defect[y0:y0 + t, x0:x0 + t]
                if sub.any():
                    continue
                picks.append((x0, y0, False))
            for (x0, y0, pos) in picks:
                self.tiles.append((si, x0, y0, pos))

    def __len__(self):
        return len(self.tiles) if self.tiles else len(self.samples)

    def _load_mask_at(self, mask_path, box):
        """按原分辨率裁掩膜（不做任何缩放）。box=(x0,y0,t)。"""
        t = self.tile_size
        x0, y0 = int(box[0]), int(box[1])
        if str(mask_path).lower().endswith(".json"):
            from annotation.labelme_io import load_labelme_json, shapes_to_mask
            shapes, info = load_labelme_json(mask_path)
            H = int(info.get("imageHeight") or t); W = int(info.get("imageWidth") or t)
            full = shapes_to_mask(shapes, self.classes, (H, W)).astype(np.uint8)
        else:
            full = np.array(Image.open(mask_path).convert("L"))
        H, W = full.shape
        out = np.full((t, t), 255, np.uint8)          # 越界区域算"忽略"，不产生错误梯度
        x1, y1 = min(W, x0 + t), min(H, y0 + t)
        if x1 > x0 and y1 > y0:
            out[: y1 - y0, : x1 - x0] = full[y0:y1, x0:x1]
        return out

    def _crop_view(self, path, box):
        """按原分辨率裁某一个视图（不做任何缩放）。"""
        t = self.tile_size
        x0, y0 = int(box[0]), int(box[1])
        im = Image.open(path).convert("RGB")
        W, H = im.size
        x1, y1 = min(W, x0 + t), min(H, y0 + t)
        tile = Image.new("RGB", (t, t), (0, 0, 0))    # 越界补黑（与训练时的 padding 一致）
        if x1 > x0 and y1 > y0:
            tile.paste(im.crop((x0, y0, x1, y1)), (0, 0))
        return np.array(tile, dtype=np.float32) / 255.0

    def _load_mask(self, mask_path):
        h, w = self.image_size[1], self.image_size[0]
        if mask_path.lower().endswith(".json"):
            try:
                from annotation.labelme_io import load_labelme_json, shapes_to_mask
                shapes, info = load_labelme_json(mask_path)
                mh = info.get("imageHeight", h)
                mw = info.get("imageWidth", w)
                m = shapes_to_mask(shapes, self.classes, (mh, mw)).astype(np.uint8)
                return np.array(Image.fromarray(m).resize((w, h), Image.NEAREST))
            except Exception:
                return np.zeros((h, w), np.uint8)
        m = Image.open(mask_path)
        return np.array(m.convert("L").resize((w, h), Image.NEAREST)).astype(np.uint8)

    def __getitem__(self, idx):
        if self.tiles:                                # ── 原分辨率切图模式 ──
            si, x0, y0, _pos = self.tiles[idx]
            key, paths, mask_path = self.samples[si]
            t = self.tile_size
            chans = []
            for v in self.views:
                p = paths[v]
                if p is None:                          # 缺视图 -> 用第一张视图顶替（保证通道数固定）
                    p = next(x for x in paths.values() if x)
                chans.append(self._crop_view(p, (x0, y0, t)))
            x = np.concatenate(chans, axis=2)          # t,t,9
            y = self._load_mask_at(mask_path, (x0, y0, t))
            if self.augment and np.random.rand() < 0.5:   # 水平翻转（三视图+掩膜同步）
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1].copy()
            x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
            mean = torch.tensor(IMAGENET_MEAN * len(self.views)).view(-1, 1, 1)
            std = torch.tensor(IMAGENET_STD * len(self.views)).view(-1, 1, 1)
            x = (x - mean) / std
            return x, torch.from_numpy(y.astype(np.int64))

        key, paths, mask_path = self.samples[idx]
        w, h = self.image_size
        chans = []
        for v in self.views:
            p = paths[v]
            if p is None:                      # 缺视图 -> 用第一张视图顶替（保证通道数固定）
                p = next(x for x in paths.values() if x)
            im = Image.open(p).convert("RGB").resize((w, h), Image.BILINEAR)
            chans.append(np.array(im, dtype=np.float32) / 255.0)
        x = np.concatenate(chans, axis=2)      # H,W,9
        y = self._load_mask(mask_path)         # H,W  (类别索引)

        # 抖动：切图时缺陷被摆在窗口正中，模型会偷懒"只看中心"。
        # 先补边再随机裁回原尺寸，让缺陷落到窗口内任意位置。
        # 新引入的边缘区域没有有效标注 -> 掩膜填 255（忽略），不产生错误梯度。
        if self.augment and self.jitter > 0:
            j = self.jitter
            h0, w0 = x.shape[0], x.shape[1]
            xp = np.pad(x, ((j, j), (j, j), (0, 0)), mode="edge")
            yp = np.pad(y, ((j, j), (j, j)), mode="constant", constant_values=255)
            oy = int(np.random.randint(0, 2 * j + 1))
            ox = int(np.random.randint(0, 2 * j + 1))
            x = xp[oy:oy + h0, ox:ox + w0]
            y = yp[oy:oy + h0, ox:ox + w0]

        if self.augment and np.random.rand() < 0.5:   # 水平翻转（三视图+掩膜同步）
            x = x[:, ::-1, :].copy()
            y = y[:, ::-1].copy()

        x = torch.from_numpy(x).permute(2, 0, 1)      # 9,H,W
        # 每个视图各自用 ImageNet 归一化
        mean = torch.tensor(IMAGENET_MEAN * len(self.views)).view(-1, 1, 1)
        std = torch.tensor(IMAGENET_STD * len(self.views)).view(-1, 1, 1)
        x = (x - mean) / std
        y = torch.from_numpy(y.astype(np.int64))
        return x, y


# ────────────────────────── 模型 ──────────────────────────
def _find_first_conv(model):
    """在 torchvision 分割模型里找到"第一层卷积"（通用兜底，兼容各 backbone 封装）。"""
    bb = getattr(model, "backbone", None)
    if bb is not None:
        if hasattr(bb, "conv1"):                       # DeepLabV3 / FCN (ResNet)
            return bb.conv1
        # LRASPP(MobileNetV3) backbone 是 IntermediateLayerGetter -> 通用搜索
        for m in bb.modules():
            if isinstance(m, nn.Conv2d) and m.in_channels == 3:
                return m
    # 最后兜底：整个模型里第一个 3 通道、非 1x1 的卷积
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3 and tuple(m.kernel_size) != (1, 1):
            return m
    return None


def patch_first_conv(model, in_channels):
    """把 torchvision 分割模型的首层卷积改成 in_channels 通道。

    新通道权重：前 3 通道保留原预训练权重，其余用原 3 通道均值初始化。
    """
    conv = _find_first_conv(model)
    if conv is None or conv.in_channels == in_channels:
        return model
    new = nn.Conv2d(in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w_old = conv.weight.data
        if in_channels % 3 == 0 and in_channels > 3:
            # 三视图膨胀初始化（I3D 式）：每个视图复制同一份 RGB 预训练核并缩放，
            # 保持激活量级不变。三视图是同一场景，比"均值压缩"收敛快得多。
            n_rep = in_channels // 3
            w_new = w_old.repeat(1, n_rep, 1, 1) * (3.0 / in_channels)
        else:
            w_new = w_old.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            if in_channels >= 3:
                w_new[:, :3] = w_old
        new.weight.copy_(w_new)
        if conv.bias is not None:
            new.bias.copy_(conv.bias.data)
    # 用 dotted name 的字符串键替换（对 Sequential / IntermediateLayerGetter 都成立）
    for name, mod in model.named_modules():
        if mod is conv:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, new)
            break
    return model


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout, mid=None):
        super().__init__()
        mid = mid or cout
        self.block = nn.Sequential(
            nn.Conv2d(cin, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class MultiViewUNet(nn.Module):
    """全分辨率跳连 U-Net，专治小缺陷。

    为什么需要它：lraspp/deeplab 类模型输出步长为 8~16，一个 12px 的缺陷
    到深层特征图上只剩 1~2 个像素，信息已经丢了。U-Net 的第一层跳连把
    **原始分辨率**的特征直接送进解码器，小缺陷不会在下采样中消失。

    输出与输入同分辨率，返回 {"out": logits} 以兼容现有训练/推理代码。
    """

    def __init__(self, in_channels=9, num_classes=6, base=16, depth=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]      # [16,32,64,128,256]
        self.chs, self.depth = chs, depth
        self.enc = nn.ModuleList()
        c_prev = in_channels
        for i in range(depth):
            self.enc.append(_DoubleConv(c_prev, chs[i]))
            c_prev = chs[i]
        self.bottleneck = _DoubleConv(c_prev, chs[depth])
        self.dec = nn.ModuleList()
        c_prev = chs[depth]
        for i in reversed(range(depth)):
            self.dec.append(_DoubleConv(c_prev + chs[i], chs[i]))
            c_prev = chs[i]
        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        feats = []
        for e in self.enc:
            x = e(x)
            feats.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for i, d in enumerate(self.dec):
            skip = feats[-1 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = d(torch.cat([x, skip], 1))
        return {"out": self.head(x)}


def _hub_cached(url):
    """torch.hub 缓存里是否已有该权重文件。

    现场外网常常只有几百 KB/s，训练时若贸然触发权重下载会直接卡死
    （实测：COCO 分割权重 160MB 在 0.5MB/s 的链路上能把进程挂住十分钟以上）。
    所以：缓存里没有就跳过这一级，不下载。
    """
    if not url:
        return False
    import urllib.parse
    fn = os.path.basename(urllib.parse.urlparse(url).path)
    return os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", fn))


# 各模型默认的 ImageNet 骨干权重（用于"退一步只加载骨干"）
_BACKBONE_URL = {
    "deeplabv3": lambda tvm: tvm.ResNet50_Weights.IMAGENET1K_V1.url,
    "fcn": lambda tvm: tvm.ResNet50_Weights.IMAGENET1K_V1.url,
    "lraspp": lambda tvm: tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1.url,
}


def create_multiview_model(model_name="deeplabv3", num_classes=2, views=DEFAULT_VIEWS,
                           pretrained=False, log=None, unet_base=16, unet_depth=4):
    """建三图分割模型（in_channels = 3 × 视图数）。

    pretrained=True 时按三级降级，且**只用本地已缓存的权重**，不发起下载：
        1) 完整 COCO 分割权重  2) 仅 ImageNet 骨干  3) 不加载
    model_name="unet" 时不涉及预训练，用 unet_base/unet_depth 控制容量。
    """
    from torchvision import models as tvm
    in_ch = 3 * len(views)
    if model_name == "unet":
        m = MultiViewUNet(in_ch, num_classes, base=unet_base, depth=unet_depth)
        if log:
            n = sum(p.numel() for p in m.parameters()) / 1e6
            log(f"  模型: U-Net base={unet_base} depth={unet_depth} 全分辨率跳连  参数 {n:.2f}M")
        return m
    _SEG = {
        "deeplabv3": (tvm.segmentation.deeplabv3_resnet50,
                      tvm.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT),
        "fcn": (tvm.segmentation.fcn_resnet50,
                tvm.segmentation.FCN_ResNet50_Weights.DEFAULT),
        "lraspp": (tvm.segmentation.lraspp_mobilenet_v3_large,
                   tvm.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT),
    }
    if model_name == "lraspp_mobilenet":
        model_name = "lraspp"
    if model_name not in _SEG:
        raise ValueError(f"不支持的分割模型: {model_name}")
    fn, w_full = _SEG[model_name]
    m = None
    if pretrained:
        cands = []
        if _hub_cached(w_full.url):
            cands.append(("完整分割权重", dict(weights=w_full)))
        if _hub_cached(_BACKBONE_URL[model_name](tvm)):
            cands.append(("ImageNet 骨干（无分割头）", dict(weights=None)))
        cands.append(("不加载预训练", dict(weights=None, weights_backbone=None)))
        for tag, kw in cands:
            try:
                m = fn(**kw)
                if log:
                    log(f"  预训练加载: {tag}")
                break
            except Exception as e:
                if log:
                    log(f"  预训练加载失败（{tag}）: {type(e).__name__}: {str(e)[:120]}")
    if m is None:
        m = fn(weights=None, weights_backbone=None)
        if log:
            log("  预训练加载: 不加载")
    # 分类头改成 num_classes（含 background）
    if model_name in ("deeplabv3", "fcn"):
        m.classifier[-1] = nn.Conv2d(m.classifier[-1].in_channels, num_classes, kernel_size=1)
    else:
        m.classifier.low_classifier = nn.Conv2d(
            m.classifier.low_classifier.in_channels, num_classes, kernel_size=1)
        m.classifier.high_classifier = nn.Conv2d(
            m.classifier.high_classifier.in_channels, num_classes, kernel_size=1)
    m = patch_first_conv(m, in_ch)
    return m


# ────────────────────────── 训练 ──────────────────────────
def _seg_confusion(pred, target, num_classes, ignore_index=255):
    """返回 (tp, fp, fn)，ignore_index 像素在三个量里都不计。"""
    tp = torch.zeros(num_classes); fp = torch.zeros(num_classes); fn = torch.zeros(num_classes)
    valid = target != ignore_index
    p = pred[valid]; t = target[valid]
    for c in range(num_classes):
        pc = (p == c); tc = (t == c)
        tp[c] = (pc & tc).sum().item()
        fp[c] = (pc & ~tc).sum().item()
        fn[c] = (~pc & tc).sum().item()
    return tp, fp, fn


def _dice_loss(logits, target, num_classes, ignore_index=255, eps=1.0, skip_background=True):
    """多类 Soft-Dice。默认只对前景类取平均，否则会被背景淹没。"""
    valid = target != ignore_index
    t = torch.where(valid, target, torch.zeros_like(target))
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    v = valid.unsqueeze(1).to(probs.dtype)
    probs = probs * v
    onehot = onehot * v
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    card = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (card + eps)
    if skip_background and num_classes > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()


def train_multiview(project_dir, model_name="deeplabv3", epochs=30, batch_size=2,
                    image_size=512, lr=1e-4, weight_decay=1e-4, views=DEFAULT_VIEWS,
                    num_workers=0, device=None, use_amp=True, augment=True,
                    class_weighted=False, log_callback=None, save_name="multiview_best.pth",
                    pretrained=False, ignore_index=255, loss="ce_dice", dice_weight=1.0,
                    jitter=None, unet_base=16, unet_depth=4, resume=None,
                    tile_size=0, neg_ratio=1.5, tile_mode="slide", tile_stride_pct=75,
                    progress_callback=None, batch_callback=None, stop_check=None):
    """训练三图分割模型，返回 metrics dict；模型存到 models/<save_name>。"""
    log = log_callback or _print_log
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    project_dir = Path(project_dir)

    _isz = image_size[0] if isinstance(image_size, (tuple, list)) else image_size
    if jitter is None:                     # 默认按窗口的 1/8 抖动，打破"缺陷总在中心"的捷径
        jitter = max(4, int(_isz) // 8) if augment else 0
    tr = MultiViewSegDataset(project_dir, "train", image_size, views, augment=augment,
                             log=log, jitter=jitter, tile_size=tile_size,
                             neg_ratio=neg_ratio, tile_mode=tile_mode,
                             tile_stride_pct=tile_stride_pct)
    # 验证集**不切图**（整图 resize 跑一遍即可，省时间；分块验证的收益很低）
    va = MultiViewSegDataset(project_dir, "val", image_size, views, augment=False, log=log)
    if jitter:
        log(f"增强: 水平翻转 + 抖动 ±{jitter}px")
    if len(tr) == 0:
        raise RuntimeError("训练集为空：请确认 images/ 下有 xxx_te_0/1/2 三图、"
                          "annotations/ 下有对应 _mask.png、且 train_test_split.json 已划分")
    if len(va) == 0:
        log("⚠️ 验证集为空，将从训练集切 15% 作验证")
        # ⚠️ 必须按 **组数** 切，不能按 len(tr)：开了分块后 `len(tr)` 是"块数"（实测 20790），
        #    而 tr.samples 只有 134 组 —— 拿块数当下标去索引组列表 = IndexError。
        #    （已踩：分块训练一启动就崩，报 list index out of range）
        n = len(tr.samples)
        idx = np.random.RandomState(42).permutation(n)
        nv = max(1, int(n * 0.15))
        va.samples = [tr.samples[i] for i in idx[:nv]]
        tr.samples = [tr.samples[i] for i in idx[nv:]]
        # 组的归属变了 -> 切图索引必须按新 samples 重建（否则块还指向旧的组下标）
        if tr.tiles:
            tr.tiles = []
            tr._build_tile_index(log)
        log(f"划分后：训练 {len(tr.samples)} 组 / 验证 {len(va.samples)} 组"
            + (f"（训练块数 {len(tr)}）" if tr.tiles else ""))

    nc = tr.num_classes
    # GPU 场景的三个关键 DataLoader 选项（早期版本漏配，导致 GPU 利用率只有 ~87%，
    # 一直在等数据）：
    #   pin_memory        锁页内存，host->device 拷贝可以异步，实测差很多
    #   persistent_workers 不每轮重建 worker 进程（Windows 是 spawn，代价高）
    #   prefetch_factor   每个 worker 预取几个 batch
    _pin = (device == "cuda")
    _kw = {}
    if num_workers > 0:
        _kw["persistent_workers"] = True
        _kw["prefetch_factor"] = 4
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                           drop_last=len(tr) > batch_size, pin_memory=_pin, **_kw)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=_pin, **_kw)

    model = create_multiview_model(model_name, nc, views, pretrained=pretrained, log=log,
                                   unet_base=unet_base, unet_depth=unet_depth).to(device)
    if resume:
        try:
            rck = torch.load(resume, map_location="cpu", weights_only=False)
            sd = rck.get("model_state_dict", rck)
            miss, unexp = model.load_state_dict(sd, strict=False)
            log(f"  继续训练: 载入 {resume}（缺失 {len(miss)} 项 / 多余 {len(unexp)} 项）")
        except Exception as e:
            log(f"  ⚠️ 继续训练载入失败，改为从头训练: {type(e).__name__}: {e}")

    out_dir = project_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / f"{Path(save_name).stem}_last.pth"
    log(f"模型 {model_name}  in_channels={3*len(views)} ({' + '.join(VIEW_NAMES.get(v,v) for v in views)})  "
        f"类数={nc}  device={device}  train={len(tr)} val={len(va)}")

    weights = None
    if class_weighted:
        cnt = np.zeros(nc)
        for _, _, mp in tr.samples:
            m = tr._load_mask(mp)
            for c in range(nc):
                cnt[c] += int((m == c).sum())          # 255(忽略) 不计入任何类
        freq = cnt / max(cnt.sum(), 1)
        if class_weighted == "sqrt":                   # 小目标更稳：逆平方根
            weights = torch.tensor(1.0 / np.sqrt(np.maximum(freq, 1e-12)), dtype=torch.float32)
        else:
            weights = torch.tensor(1.0 / np.maximum(freq, 1e-6), dtype=torch.float32)
        # 归一化：让"出现最多的类（背景）"权重 = 1，前景按比例适度加成并设上限。
        # 绝不能按"出现类的均值"归一化 —— 那会把背景压到 0.05，
        # 模型对背景几乎不罚，结果就是满屏乱报、过杀爆炸。
        if (freq > 0).any():
            weights = weights / weights[freq > 0].min()
        weights = weights.clamp(1.0, 5.0)
        weights[freq <= 0] = 1.0
        log("类别权重: " + ", ".join(f"{tr.classes[i] if i < len(tr.classes) else i}={weights[i]:.2f}"
                                     for i in range(nc)))
    crit = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None,
                               ignore_index=ignore_index)

    def _compute_loss(out, yb):
        # 注意：这里读的是外层参数 loss（字符串）。训练循环里的批次损失变量
        # 必须叫 batch_loss —— 早期版本写成 loss = _compute_loss(...)，
        # 会把参数覆盖成 Tensor，导致闭包判断恒为假、Dice 项被静默丢弃。
        if loss == "dice":
            return _dice_loss(out, yb, nc, ignore_index)
        l = crit(out, yb)
        if loss == "ce_dice":
            l = l + dice_weight * _dice_loss(out, yb, nc, ignore_index)
        return l
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device == "cuda") else None

    hist, best_iou, best_state, best_ep, best_pack = [], -1.0, None, 0, None
    stopped = False
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        n_batches = max(1, len(tr_loader))
        for bi, (xb, yb) in enumerate(tr_loader, 1):
            # ⚠️ 停止检查必须放在**每个 batch 里**，不能只在每轮末尾：
            #    滑窗模式下 1 轮 = 4419 个 batch（实测日志 batch 3450/4419），
            #    只在轮末检查的话，点了 Stop 还要等十几分钟才生效 —— 用户看到的就是"停不下来"。
            #    每 10 个 batch 查一次，开销可忽略。
            if stop_check and (bi % 10 == 0):
                try:
                    if stop_check():
                        stopped = True
                        _log("⏹ 已收到停止请求：当前 batch 结束后退出（本轮验证仍会跑完，约几秒）")
                        break
                except Exception:
                    pass
            xb, yb = xb.to(device, non_blocking=_pin), yb.to(device, non_blocking=_pin)
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(xb)["out"]
                    batch_loss = _compute_loss(out, yb)
                scaler.scale(batch_loss).backward(); scaler.step(opt); scaler.update()
            else:
                out = model(xb)["out"]
                batch_loss = _compute_loss(out, yb)
                batch_loss.backward(); opt.step()
            tl += batch_loss.item() * xb.size(0)
            # GUI 用的批次进度回调（每 10 个 batch 报一次，避免刷屏）
            if batch_callback and (bi % 10 == 0 or bi == n_batches):
                try:
                    batch_callback(bi, n_batches, tl / max(1, bi * xb.size(0)))
                except Exception:
                    pass
        sched.step()

        # 用户点"停止"：当前轮跑完就退出，避免中断在半个 batch 上
        if stop_check:
            try:
                if stop_check():
                    stopped = True
            except Exception:
                pass

        model.eval()
        tp = torch.zeros(nc); fp = torch.zeros(nc); fn = torch.zeros(nc)
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb_d, yb_d = xb.to(device, non_blocking=_pin), yb.to(device, non_blocking=_pin)
                out = model(xb_d)["out"]
                if out.shape[-2:] != yb.shape[-2:]:
                    out = F.interpolate(out, size=yb.shape[-2:], mode="bilinear", align_corners=False)
                vl += crit(out, yb_d).item() * xb.size(0)
                pred = out.argmax(1).cpu()
                a, b, c_ = _seg_confusion(pred, yb, nc, ignore_index)
                tp += a; fp += b; fn += c_
        union = tp + fp + fn
        ious = [(tp[c] / union[c]).item() if union[c] > 0 else 0.0 for c in range(nc)]
        recalls = [(tp[c] / (tp[c] + fn[c])).item() if (tp[c] + fn[c]) > 0 else 0.0 for c in range(nc)]
        precs = [(tp[c] / (tp[c] + fp[c])).item() if (tp[c] + fp[c]) > 0 else 0.0 for c in range(nc)]
        miou = float(np.mean(ious))
        fg_idx = [c for c in range(1, nc) if union[c] > 0]
        miou_fg = float(np.mean([ious[c] for c in fg_idx])) if fg_idx else 0.0
        hist.append({"epoch": ep, "loss": round(tl / max(1, len(tr)), 4),
                     "val_loss": round(vl / max(1, len(va)), 4),
                     "mIoU": round(miou, 4),
                     "mIoU_fg": round(miou_fg, 4), "ious": [round(x, 4) for x in ious],
                     "recall": [round(x, 4) for x in recalls], "precision": [round(x, 4) for x in precs]})
        log(f"epoch {ep}/{epochs}  loss={hist[-1]['loss']}  前景mIoU={miou_fg:.4f}  " +
            " ".join(f"{tr.classes[c] if c < len(tr.classes) else c}:IoU={ious[c]:.3f}/R={recalls[c]:.3f}"
                     for c in range(nc)))
        sel = miou_fg if fg_idx else miou
        if sel > best_iou:
            best_iou = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
            best_pack = {"mIoU": miou, "mIoU_fg": miou_fg, "ious": ious,
                         "recall": recalls, "precision": precs}

        # 每轮落一次盘：崩溃不丢进度，也支持之后 --resume 延长训练
        try:
            torch.save({"model_state_dict": {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()},
                        "classes": tr.classes, "num_classes": nc, "in_channels": 3 * len(views),
                        "views": list(views), "model_name": model_name,
                        "image_size": list(tr.image_size), "epoch": ep,
                        "ignore_index": ignore_index, "loss": str(loss),
                        "last_metrics": hist[-1], "best_mIoU_fg": round(best_iou, 4),
                        "created": time.strftime("%Y-%m-%d %H:%M:%S")}, last_path)
        except Exception as e:
            log(f"  ⚠️ 落盘失败: {type(e).__name__}: {e}")

        # GUI 用的进度回调 + 用户停止
        if progress_callback:
            try:
                progress_callback(ep, epochs)
            except Exception:
                pass
        if stopped:
            log("已按用户请求停止训练（当前轮已完整跑完，权重已落盘）")
            break

    bp = best_pack or {"mIoU": miou, "mIoU_fg": miou_fg, "ious": ious,
                       "recall": recalls, "precision": precs}
    ckpt_path = out_dir / save_name
    cname = lambda c: tr.classes[c] if c < len(tr.classes) else str(c)
    metrics = {"best_mIoU": round(bp["mIoU"], 4), "best_mIoU_fg": round(bp["mIoU_fg"], 4),
               "best_epoch": best_ep,
               "ious": {cname(c): round(bp["ious"][c], 4) for c in range(nc)},
               "recall": {cname(c): round(bp["recall"][c], 4) for c in range(nc)},
               "precision": {cname(c): round(bp["precision"][c], 4) for c in range(nc)},
               "loss": str(loss), "ignore_index": ignore_index,
               "history": hist}
    torch.save({"model_state_dict": best_state or model.state_dict(),
                "classes": tr.classes, "num_classes": nc, "in_channels": 3 * len(views),
                "views": list(views), "model_name": model_name,
                "image_size": list(tr.image_size), "metrics": metrics,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}, ckpt_path)
    log(f"✅ 训练完成  best epoch={best_ep}  mIoU={bp['mIoU']:.4f}  前景mIoU={bp['mIoU_fg']:.4f}  "
        f"模型: {ckpt_path}")
    return metrics


# ────────────────────────── 推理 ──────────────────────────
@torch.inference_mode()
def predict_multiview(project_dir, key, ckpt_name="multiview_best.pth", device=None,
                      image_size=None, return_probs=False):
    """按 key（不含 _te_N 的公共前缀）跑三图分割，返回 (mask[H,W] uint8, classes)。"""
    project_dir = Path(project_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(project_dir / "models" / ckpt_name, map_location=device, weights_only=False)
    views = tuple(ckpt.get("views", DEFAULT_VIEWS))
    classes = ckpt.get("classes", ["background"])
    sz = image_size or tuple(ckpt.get("image_size", (512, 512)))
    if isinstance(sz, int):
        sz = (sz, sz)

    chans = []
    for v in views:
        p = None
        for ext in (".jpg", ".png", ".jpeg", ".bmp"):
            q = project_dir / "images" / f"{key}_{v}{ext}"
            if q.exists():
                p = q
                break
        if p is None:
            raise FileNotFoundError(f"缺视图: {key}_{v}")
        chans.append(np.array(Image.open(p).convert("RGB").resize(sz, Image.BILINEAR),
                              dtype=np.float32) / 255.0)
    x = torch.from_numpy(np.concatenate(chans, 2)).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN * len(views)).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD * len(views)).view(1, -1, 1, 1)
    x = ((x - mean) / std).to(device)

    model = create_multiview_model(ckpt.get("model_name", "deeplabv3"), ckpt["num_classes"], views)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    out = model(x)["out"]
    if out.shape[-2:] != (sz[1], sz[0]):
        out = F.interpolate(out, size=(sz[1], sz[0]), mode="bilinear", align_corners=False)
    pred = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return (pred, list(classes)) if not return_probs else (pred, out.softmax(1).squeeze(0).cpu().numpy(), list(classes))


# ────────────────────────── CLI ──────────────────────────
def main():
    ap = argparse.ArgumentParser(description="三图(漫反射+形状1+形状2)语义分割训练")
    ap.add_argument("project_dir")
    ap.add_argument("--model", default="deeplabv3", choices=["deeplabv3", "fcn", "lraspp"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--views", default="te_0,te_1,te_2", help="视图顺序: te_0=漫反射,te_1=形状1,te_2=形状2")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--class-weighted", action="store_true", help="按类别像素频率加权(缓解不均衡)")
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    train_multiview(args.project_dir, model_name=args.model, epochs=args.epochs,
                    batch_size=args.batch_size, image_size=args.image_size, lr=args.lr,
                    views=views, num_workers=args.num_workers,
                    class_weighted=args.class_weighted, augment=not args.no_augment)


if __name__ == "__main__":
    main()
