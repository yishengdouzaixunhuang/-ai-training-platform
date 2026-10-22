# -*- coding: utf-8 -*-
"""结果转标注 / 难例闭环 —— 对标 iCogtiveFlow 的 Convert Label / Mask To Label。

## 三种通路（先分清目录，这是本模块最容易搞错的地方）

    images/<base>.json          人工标注（LabelMe shapes）      ← 训练真值
    annotations/<base>_mask.png 人工标注的掩膜（由 JSON 重建）   ← 训练真值
    images/<base>_det.json      人工标注（检测框 COCO）          ← 检测真值
    outputs/<base>_pred.png     **模型输出**（分割预测掩膜）     ← 不是真值！
    outputs/<base>_det.json     **模型输出**（检测预测框）       ← 不是真值！

"_det.json" 在两个目录里都出现，靠**目录**区分真值/预测；写错目录就会把
模型自己的幻觉当成人工标注喂回训练集（自污染，且不会报错）。

## 为什么"过杀 → 负样本"是把标注**扣掉**，而不是标成 __ignore__

Arthur 2026-10-14 口径：**干净的背景，模型识别出了结果 = 过杀**。
既然人工认定那里是背景，模型就该在那儿学"背景"。

- 标成 `__ignore__`(255)：`training/losses_ext.py` 里 CE 用 `ignore_index=255`，
  该处**不产生任何梯度** ⇒ 模型拿不到"这里是背景"的信号 ⇒ 下一轮还会在同一处幻觉。
- 扣成背景(0)：该处产生"背景"的梯度 ⇒ 这才是"转成负样本"的实际含义。

所以 erase_result_from_annotation() 扣的是**与 AI 输出重叠的自有标注**（通常是上一轮
Convert Label 把 AI 结果固化成的），扣完这块就成了背景。多出来的、人工手画的别的缺陷
（与 AI 输出不重叠）一律保留 —— 绝不整图清空。

## 漏杀（escape）为什么"没什么可自动转的"

漏杀的定义是"标注过的前景，模型没抓出" —— 缺陷**已经**在标注里了，缺的是模型。
所以这里的动作是两件实事：
  1. 由 JSON 重建掩膜 PNG（保证这条正样本真的会被训练集取到，不是只有 JSON 没掩膜）；
  2. 把"人工标了但 AI 没检出"的区域算出来（`miss_mask`）并存成图，给现场一个定位图。
真正"补画那个缺陷"是人工动作，本模块**不会**替你发明标注 —— 凭空造标注比不造更危险。
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from annotation import labelme_io
from core import marks as marks_mod

#: 结果来源：(key, 中文名)
RESULT_SOURCES = (("segment", "分割结果"), ("detect", "检测结果"))
SOURCE_LABELS = dict(RESULT_SOURCES)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: 判定"这个标注就是 AI 自己那条结果"的重叠阈值
DEFAULT_TAU = 0.5


# ───────────────────────── 路径 ─────────────────────────

def ann_json_path(project_dir, base):
    return os.path.join(project_dir, "images", base + ".json")


def ann_mask_path(project_dir, base):
    return os.path.join(project_dir, "annotations", base + "_mask.png")


def ann_det_path(project_dir, base):
    return os.path.join(project_dir, "images", base + "_det.json")


def pred_mask_path(project_dir, base):
    return os.path.join(project_dir, "outputs", base + "_pred.png")


def pred_det_path(project_dir, base):
    return os.path.join(project_dir, "outputs", base + "_det.json")


def miss_mask_path(project_dir, base):
    return os.path.join(project_dir, "outputs", base + "_miss.png")


def find_image_path(project_dir, base):
    img_dir = os.path.join(project_dir, "images")
    for ext in IMAGE_EXTS:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            return p
    return ""


def class_names_of(project_dir):
    """工程类别表（含 background 在 0 号位，255=ignore 不占号）。"""
    import json
    p = os.path.join(project_dir, "project.json")
    if not os.path.exists(p):
        return ["background"]
    try:
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return ["background"]
    cs = [c for c in (meta.get("classes") or []) if c]
    if not cs or cs[0] != "background":
        cs = ["background"] + [c for c in cs if c != "background"]
    return cs


# ───────────────────────── 读 ─────────────────────────

def _read_gray(path):
    """读掩膜。

    ⚠️ 不能无条件 convert("L")：调色板(P)模式的掩膜会被映射成**调色板颜色**而不是
    类别号（0/1/2 会变成某个灰阶），后面按类别号比大小就全错。
    P/L/I 三种模式直接取像素值；只有真有多个通道时才降为灰度。
    """
    with Image.open(path) as im:
        if im.mode in ("L", "P", "I", "I;16"):
            arr = np.array(im)
        else:
            arr = np.array(im.convert("L"))
    return np.asarray(arr)


def load_pred_mask(project_dir, base):
    """模型输出的分割预测掩膜；没有返回 None。"""
    p = pred_mask_path(project_dir, base)
    if not os.path.exists(p):
        return None
    try:
        return _read_gray(p)
    except Exception:
        return None


def load_ann_mask(project_dir, base, class_names=None):
    """人工标注掩膜：优先由 JSON 现算（JSON 是真值，PNG 只是缓存），退回 PNG。"""
    jp = ann_json_path(project_dir, base)
    shapes, info = labelme_io.load_labelme_json(jp)
    if shapes:
        h = info.get("imageHeight") or 0
        w = info.get("imageWidth") or 0
        if not (h and w):
            ip = find_image_path(project_dir, base)
            if ip:
                try:
                    with Image.open(ip) as im:
                        w, h = im.size
                except Exception:
                    pass
        if h and w:
            names = class_names or class_names_of(project_dir)
            return labelme_io.shapes_to_mask(shapes, names, (h, w))
    mp = ann_mask_path(project_dir, base)
    if os.path.exists(mp):
        try:
            return _read_gray(mp)
        except Exception:
            return None
    return None


def load_pred_boxes(project_dir, base):
    from detection.coco_io import load_coco_json
    boxes, _ = load_coco_json(pred_det_path(project_dir, base))
    return list(boxes or [])


def load_ann_boxes(project_dir, base):
    from detection.coco_io import load_coco_json
    boxes, _ = load_coco_json(ann_det_path(project_dir, base))
    return list(boxes or [])


# ───────────────────────── 几何 ─────────────────────────

def _binary(mask):
    if mask is None:
        return None
    return ((mask > 0) & (mask != 255)).astype(np.uint8)


def iou_masks(a, b):
    """两个掩膜的 IoU（只算前景；无前景 return 0.0，形状不一致 return None）。"""
    if a is None or b is None:
        return None
    if getattr(a, "shape", None) != getattr(b, "shape", None):
        return None
    ba, bb = _binary(a), _binary(b)
    inter = int((ba & bb).sum())
    union = int((ba | bb).sum())
    if union == 0:
        return 0.0
    return inter / union


def mask_iou_matrix(pred, ann):
    """逐类别 IoU（给统计/验收报告用）。返回 {class_id: iou}。"""
    out = {}
    if pred is None or ann is None or pred.shape != ann.shape:
        return out
    for cid in sorted(set(np.unique(pred).tolist()) | set(np.unique(ann).tolist())):
        if cid in (0, 255):
            continue
        pa = (pred == cid)
        aa = (ann == cid)
        u = int((pa | aa).sum())
        out[int(cid)] = (int((pa & aa).sum()) / u) if u else 0.0
    return out


def overlap_of_shape(shape, pred_mask):
    """单个 shape 与预测掩膜的重叠率 = |shape ∩ pred| / |shape|。"""
    if pred_mask is None:
        return 0.0
    h, w = pred_mask.shape[:2]
    names = ["background", str(shape.get("label") or "x")]
    m = labelme_io.shapes_to_mask([shape], names, (h, w))
    sb = (m > 0)
    n = int(sb.sum())
    if n == 0:
        return 0.0
    pb = _binary(pred_mask).astype(bool)
    return float((sb & pb).sum()) / n


def box_iou(a, b):
    ax, ay, aw, ah = [float(v) for v in a[:4]]
    bx, by, bw, bh = [float(v) for v in b[:4]]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def miss_mask(project_dir, base, class_names=None):
    """漏检区域 = 人工标注前景 ∩ ¬模型输出前景。没有两边都齐时返回 None。"""
    ann = load_ann_mask(project_dir, base, class_names)
    pred = load_pred_mask(project_dir, base)
    if ann is None or pred is None or ann.shape != pred.shape:
        return None
    return (_binary(ann).astype(bool) & ~_binary(pred).astype(bool)).astype(np.uint8) * 255


# ───────────────────────── 写 ─────────────────────────

def _save_boxes(project_dir, base, boxes):
    """把检测框写回 images/<base>_det.json（COCO）。boxes 为空也写 —— 空标注就是真负样本。"""
    from detection.coco_io import save_coco_json
    ip = find_image_path(project_dir, base)
    w = h = 0
    if ip:
        try:
            with Image.open(ip) as im:
                w, h = im.size
        except Exception:
            pass
    if not (w and h):
        w = h = 0
    cats = list(dict.fromkeys([str(b.get("category") or "object") for b in (boxes or [])]))
    if not cats:
        cats = ["object"]
    save_coco_json(ann_det_path(project_dir, base), ip or (base + ".png"), w, h,
                   list(boxes or []), cats, bbox_type="hbb")
    return True


#: 兼容别名（语义更直白）
_save_empty_det = _save_boxes


def _save_shapes(project_dir, base, shapes, image_path=""):
    """把 shapes 写回 JSON 并重建掩膜 PNG（掩膜由 JSON 生成，保证两者一致）。"""
    ip = image_path or find_image_path(project_dir, base)
    w = h = 0
    if ip:
        try:
            with Image.open(ip) as im:
                w, h = im.size
        except Exception:
            pass
    if not (w and h):
        # 图不在（例如只有预测没有原图）就用现有掩膜/预测的尺寸兜底
        ref = load_pred_mask(project_dir, base)
        if ref is not None:
            h, w = ref.shape[:2]
    if not (w and h):
        return False, "拿不到图像尺寸"
    labelme_io.save_labelme_json(ann_json_path(project_dir, base), ip or (base + ".png"),
                                 w, h, shapes, image_set="Train")
    names = class_names_of(project_dir)
    mask = labelme_io.shapes_to_mask(shapes, names, (h, w))
    mp = ann_mask_path(project_dir, base)
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(mp)
    return True, ""


def add_to_train(project_dir, bases):
    """把一批图显式归入训练集（写 train_test_split.json）。返回改动条数。

    验收链路里"转成负样本 → 进入下一轮训练集"靠这一步落地。

    ⚠️ 三个坑：
      1. **必须显式写 `split_map[b] = "train"`**。别指望"取出来的 map 里已经有这张图"：
         `get_train_test_split()` 曾经会在内存里给所有样本补默认 train，那时"计数
         n>0 但一条都没赋值"也能落盘；2026-10-22 去掉默认值（未划分 = 空集）之后，
         少了这一句就是**数着改了 N 条、盘上一字未动**，验收链路的最后一环静默失效。
      2. 判断"是不是已经在 train 里"必须**读盘上的原始文件**，不能用返回值兜底
         （同上，返回值的语义以后还可能变）。
      3. 已经在 val/test 里的图**不许**挪进 train —— 把测试集偷偷挪进训练集，
         新老模型的过杀率对比就成了自欺欺人。
    """
    from training.dataset import get_train_test_split, save_train_test_split
    split_map, _samples = get_train_test_split(project_dir)

    raw = {}
    sp = os.path.join(project_dir, "train_test_split.json")
    if os.path.exists(sp):
        try:
            import json as _json
            with open(sp, "r", encoding="utf-8") as f:
                d = _json.load(f)
            if isinstance(d, dict):
                raw = d
        except Exception:
            raw = {}

    n = 0
    for b in (bases or []):
        b = str(b)
        if not b:
            continue
        on_disk = raw.get(b)
        if on_disk == "train":
            continue
        if on_disk in ("val", "test"):
            continue                      # 保住 val/test，不偷挪
        split_map[b] = "train"            # ★ 显式写盘（见上面坑 1）
        n += 1
    if n:
        save_train_test_split(project_dir, split_map)
    return n


# ───────────────────────── 三个动作 ─────────────────────────

def convert_result_to_annotation(project_dir, base, source="segment",
                                 label=None, class_names=None, tau=DEFAULT_TAU):
    """Convert Label：把模型输出固化成人工标注。

    source="segment"：outputs/<base>_pred.png → 多边形标注（追加到 JSON）
    source="detect" ：outputs/<base>_det.json  → 矩形标注（追加到 _det.json）

    label：覆盖类别名。分割源默认沿用掩膜里的类别号；检测源默认沿用框自带的 category。
    已存在"与预测重叠 ≥ tau"的同名标注时跳过（幂等，重复点不会翻倍）。
    """
    res = {"base": base, "action": "convert_label", "source": source, "ok": False,
           "added": 0, "skipped": 0, "detail": ""}
    names = class_names or class_names_of(project_dir)

    if source == "detect":
        pboxes = load_pred_boxes(project_dir, base)
        if not pboxes:
            res["detail"] = "没有检测结果（outputs/%s_det.json）" % base
            return res
        aboxes = load_ann_boxes(project_dir, base)
        added = []
        for pb in pboxes:
            dup = any(box_iou(pb["bbox"], ab["bbox"]) >= tau for ab in aboxes)
            if dup:
                res["skipped"] += 1
                continue
            added.append({
                "category": label or pb.get("category") or "object",
                "bbox": list(pb["bbox"]),
                "score": pb.get("score"),
                "area": pb.get("area", 0),
            })
        if not added:
            res["ok"] = True
            res["detail"] = "预测框都已有对应标注，无需新增"
            return res
        new_boxes = list(aboxes) + added
        _save_boxes(project_dir, base, new_boxes)
        res.update(ok=True, added=len(added),
                   detail="新增 %d 个框（跳过已有 %d 个）" % (len(added), res["skipped"]))
        return res

    # ---- 分割 ----
    pred = load_pred_mask(project_dir, base)
    if pred is None:
        res["detail"] = "没有分割结果（outputs/%s_pred.png）" % base
        return res
    new_shapes = labelme_io.mask_to_shapes(pred, names, simplify_epsilon=1.0)
    if new_shapes:
        for s in new_shapes:
            if s.get("label", "").lower().startswith("class_"):
                # mask_to_shapes 拿不到名字时会给 class_N，换成"未知类别"更诚实
                s["label"] = label or "未知类别"
            elif label:
                s["label"] = label
    jp = ann_json_path(project_dir, base)
    old_shapes, _info = labelme_io.load_labelme_json(jp)
    old_shapes = list(old_shapes or [])
    h, w = pred.shape[:2]
    old_mask = (labelme_io.shapes_to_mask(old_shapes, names, (h, w))
                if old_shapes else np.zeros((h, w), dtype=np.uint8))
    keep = []
    for s in new_shapes:
        # 幂等：已有标注里能盖住这条预测的，就不再重复加一条
        if old_shapes and overlap_of_shape(s, old_mask) >= tau:
            res["skipped"] += 1
            continue
        keep.append(s)
    if not keep:
        res["ok"] = True
        res["detail"] = "预测区域都已有对应标注，无需新增"
        return res
    ok, err = _save_shapes(project_dir, base, old_shapes + keep)
    if not ok:
        res["detail"] = err
        return res
    res.update(ok=True, added=len(keep),
               detail="新增 %d 个标注（跳过已有 %d 个）" % (len(keep), res["skipped"]))
    return res


def erase_result_from_annotation(project_dir, base, source="segment",
                                 class_names=None, tau=DEFAULT_TAU):
    """过杀 → 负样本：把与模型输出重叠的**自有标注**扣掉，让这块回退成背景。

    只扣"与 AI 输出重叠 ≥ tau"的那几条；人工手画的、与 AI 无关的标注一律保留。
    扣完之后如果一条标注都不剩，这张图就是一张**全背景的负样本**（正是要的效果）。
    """
    res = {"base": base, "action": "erase_to_negative", "source": source, "ok": False,
           "removed": 0, "kept": 0, "detail": ""}
    names = class_names or class_names_of(project_dir)

    if source == "detect":
        pboxes = load_pred_boxes(project_dir, base)
        aboxes = load_ann_boxes(project_dir, base)
        if not pboxes:
            res["detail"] = "没有检测结果可比对"
            return res
        if not aboxes:
            # 本来就没标注：必须把"空检测标注"落成文件，否则这张图在检测训练集里
            # 根本不存在（detection/dataset 只扫 *_det.json），"负样本"就是空话。
            ok = _save_empty_det(project_dir, base, [])
            if ok:
                res.update(ok=True, detail="本来就没有检测标注；已落成空标注（全背景负样本）")
            else:
                res["detail"] = "本来就没有检测标注，且无法落盘空标注"
            return res
        keep, drop = [], 0
        for ab in aboxes:
            if any(box_iou(ab["bbox"], pb["bbox"]) >= tau for pb in pboxes):
                drop += 1
            else:
                keep.append(ab)
        if not drop:
            res.update(ok=True, kept=len(keep), detail="没有标注与预测框重叠，未改动")
            return res
        ok = _save_empty_det(project_dir, base, keep)
        if not ok:
            res["detail"] = "写入检测标注失败"
            return res
        res.update(ok=True, removed=drop, kept=len(keep),
                   detail="扣掉 %d 个与预测重合的框，保留 %d 个" % (drop, len(keep)))
        return res

    pred = load_pred_mask(project_dir, base)
    if pred is None:
        res["detail"] = "没有分割结果可比对"
        return res
    jp = ann_json_path(project_dir, base)
    shapes, _info = labelme_io.load_labelme_json(jp)
    shapes = list(shapes or [])
    if not shapes:
        # 本来就没有标注 —— 这里**必须把"空标注"落成文件**（JSON + 全背景掩膜 PNG）。
        # 因为训练集取数的门票是"有 json 或 mask"（training/dataset.get_train_test_split）：
        # 不落盘的话这张图在下一轮训练里压根不存在，"转成负样本"就成了写在文档里的空话。
        # 这正是自测 E 组要卡的那条链路，实测确认过。
        ok, err = _save_shapes(project_dir, base, [])
        if ok:
            res.update(ok=True, detail="本来就没有标注；已落成全背景负样本文件")
        else:
            res["detail"] = "本来就没有标注，且无法落盘空标注：%s" % err
        return res
    keep, drop = [], 0
    for s in shapes:
        if overlap_of_shape(s, pred) >= tau:
            drop += 1
        else:
            keep.append(s)
    if not drop:
        res.update(ok=True, kept=len(keep), detail="没有标注与预测重叠，未改动")
        return res
    ok, err = _save_shapes(project_dir, base, keep)
    if not ok:
        res["detail"] = err
        return res
    res.update(ok=True, removed=drop, kept=len(keep),
               detail="扣掉 %d 个与预测重合的标注，保留 %d 个%s"
                      % (drop, len(keep), "（该图现为全背景负样本）" if not keep else ""))
    return res


def rebuild_mask_from_annotation(project_dir, base, class_names=None):
    """漏杀 → 确保正样本完整：由 JSON 重建掩膜 PNG，并算漏检区域图。

    掩膜 PNG 是训练的取数依据之一（JSON 优先、PNG 兜底）。只有 JSON 没有 PNG 时，
    在切窗/缓存路径上有可能取不到这条样本 —— 重建一次是最省事的保险。
    """
    res = {"base": base, "action": "rebuild_positive", "ok": False, "removed": 0,
           "kept": 0, "detail": ""}
    jp = ann_json_path(project_dir, base)
    shapes, _info = labelme_io.load_labelme_json(jp)
    if not shapes:
        res["detail"] = "没有标注 JSON，无从重建"
        return res
    ok, err = _save_shapes(project_dir, base, shapes)
    if not ok:
        res["detail"] = err
        return res
    res["kept"] = len(shapes)
    detail = "由 %d 个标注重建掩膜" % len(shapes)
    mm = miss_mask(project_dir, base, class_names)
    if mm is not None and mm.any():
        try:
            mp = miss_mask_path(project_dir, base)
            os.makedirs(os.path.dirname(mp), exist_ok=True)
            Image.fromarray(mm).save(mp)
            detail += "；漏检区域已存 %s" % os.path.basename(mp)
        except Exception:
            pass
    elif mm is not None:
        detail += "；无漏检区域（AI 已覆盖全部标注）"
    else:
        detail += "；没有预测结果，无法比对漏检区域"
    res.update(ok=True, detail=detail)
    return res


# ───────────────────────── 按标记批量（验收链路的一键） ─────────────────────────

def convert_by_mark(project_dir, bases, marks, source="segment",
                    class_names=None, to_train=True, tau=DEFAULT_TAU):
    """按标记批量执行闭环动作。

        overkill -> erase_result_from_annotation   （扣成背景 = 负样本）
        escape   -> rebuild_mask_from_annotation   （重建掩膜 = 确保正样本）
        normal   -> convert_result_to_annotation   （把 AI 的正确检出固化成标注，省人工）
        doubt    -> 跳过（存疑的东西不进训练集）

    to_train=True 时把参与的图显式归入训练集（不动已在 val/test 的）。
    返回 {"results": [每图一条], "summary": {...}, "train_added": n}
    """
    results = []
    for b in (bases or []):
        b = str(b)
        k = marks_mod.normalize_kind(marks_mod.get_mark(marks, b).get("mark"))
        if k == "overkill":
            r = erase_result_from_annotation(project_dir, b, source=source,
                                             class_names=class_names, tau=tau)
        elif k == "escape":
            r = rebuild_mask_from_annotation(project_dir, b, class_names=class_names)
        elif k == "normal":
            r = convert_result_to_annotation(project_dir, b, source=source,
                                             class_names=class_names, tau=tau)
        else:
            r = {"base": b, "action": "skip", "ok": True, "detail":
                 "标记为%s，按设计不进训练集" % (marks_mod.kind_label(k) or "无")}
        results.append(r)

    done = [r for r in results if r.get("ok") and r.get("action") != "skip"]
    skipped = [r for r in results if r.get("action") == "skip"]
    failed = [r for r in results if not r.get("ok")]
    train_added = 0
    train_error = ""
    if to_train and done:
        # ⚠️ 这一步**必须单独兜住**：标注转换已经落盘成功了，不能因为
        # "归入训练集"这步环境问题（training.dataset 顶层 import torchvision，
        # 部署环境缺包会直接抛）就把整批报成失败 —— 那样用户会以为白干了、
        # 重跑一遍，而重跑本身是幂等的，只会浪费时间。
        # 拆开报：哪些转好了 + 训练集这步为什么没成。
        try:
            train_added = add_to_train(project_dir, [r["base"] for r in done])
        except Exception as e:
            train_error = "%s: %s" % (type(e).__name__, e)
    return {
        "results": results,
        "summary": {"total": len(results), "done": len(done),
                    "skipped": len(skipped), "failed": len(failed)},
        "train_added": train_added,
        "train_error": train_error,
        "failures": [{"base": r["base"], "detail": r.get("detail", "")} for r in failed],
    }


def summarize_report(rep):
    """把 convert_by_mark 的返回压成一行给日志/对话框。"""
    s = (rep or {}).get("summary") or {}
    parts = ["共 %d 张" % s.get("total", 0),
             "处理 %d" % s.get("done", 0)]
    if s.get("skipped"):
        parts.append("跳过 %d" % s["skipped"])
    if s.get("failed"):
        parts.append("失败 %d" % s["failed"])
    if rep.get("train_added"):
        parts.append("归入训练集 %d" % rep["train_added"])
    if rep.get("train_error"):
        parts.append("⚠ 归入训练集未完成（%s）" % rep["train_error"][:80])
    return "，".join(parts)
