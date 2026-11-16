# -*- coding: utf-8 -*-
"""labelme 矩形 JSON → 平台检测标注 `<名字>_det.json`（COCO 子集）。

## 为什么需要它

平台的**目标检测**训练只认 `images/<名字>_det.json`（见 `detection/dataset.py`、
`detection/coco_io.py`），而 iSenseFlow / labelme 导出的标注是 `<名字>.json`
（labelme 5.x 格式，`shapes[].shape_type = "rectangle"`）。
中间没有转换，框就"导进去了但训练扫不到"。

（`annotation/labelme_io.py` 那条链是**语义分割**的：JSON → `_mask.png`，
不产出 `_det.json`，所以这里单独一份。）

## 谁在用

- `tools/labelme_rect_to_det.py`：命令行批量转换 / 体检；
- `ui/main_window.py: _rebuild_det_for_imported()`：界面「导入标注」后自动补
  `_det.json`，让"导图片 + 导标注 + 设参数 + 训练"真的不用再手工跑脚本。

**只有这一份实现**，别再往两边各抄一遍。

## 规则

- 只转 `shape_type == "rectangle"`；其它类型（polygon / circle …）跳过并统计，
  由调用方决定怎么提示（多边形框请先在 labelme 里改矩形，或另开 OBB 通道）；
- 类别名取 `shape["label"]`，一批共用 `categories`（按名字排序）⇒ `category_id`
  稳定，且与训练侧 `build_yolo_dataset` 的 `sorted(class_names)` 对齐；
- 缺 `imageWidth/imageHeight` 时用 PIL 读图补；
- 源图 / JSON 有问题就跳过并回报，不抛异常打断整批。
"""
from __future__ import annotations

import json
import os

from detection.coco_io import save_coco_json

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def rect_to_bbox(points):
    """labelme 矩形的两个角点 → [x, y, w, h]（像素，左上角为原点）。"""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def boxes_from_labelme(data):
    """从一个 labelme dict 里取矩形框。

    Returns:
        (boxes, other_types)
        boxes: [(category, [x, y, w, h]), ...]
        other_types: {shape_type: 次数} —— 非矩形 shape 不计入，供调用方提示
    """
    boxes, other = [], {}
    for s in data.get("shapes") or []:
        st = s.get("shape_type") or "polygon"
        if st != "rectangle":
            other[st] = other.get(st, 0) + 1
            continue
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        cat = str(s.get("label") or "").strip()
        if not cat:
            continue
        boxes.append((cat, rect_to_bbox(pts)))
    return boxes, other


def image_size(image_path, data=None):
    """图宽高：优先 labelme 里记的，缺了用 PIL 补，都拿不到返回 (0, 0)。"""
    w = int((data or {}).get("imageWidth") or 0)
    h = int((data or {}).get("imageHeight") or 0)
    if w and h:
        return w, h
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size
    except Exception:                                    # noqa: BLE001
        return 0, 0


def find_image(images_dir, base):
    """按 base 名找图（扩展名不限）。找不到返回 None。"""
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def categories_of(json_paths):
    """一批 JSON 里出现过的类别名，按名字排序（与训练侧一致）。"""
    names = set()
    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:                                # noqa: BLE001
            continue
        for cat, _bb in boxes_from_labelme(d)[0]:
            names.add(cat)
    return sorted(names)


def convert_one(json_path, image_path, categories=None, out_dir=None,
                dry_run=False):
    """转一个文件，写 `<base>_det.json`。不抛异常。

    Returns:
        dict: written(bool) / boxes(int) / other(dict) / size((w,h)) /
              path(str) / skipped(str, 没写的原因)
    """
    rep = {"written": False, "boxes": 0, "other": {}, "size": (0, 0),
           "path": "", "skipped": ""}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:                               # noqa: BLE001
        rep["skipped"] = "JSON 读不了（%s）" % type(e).__name__
        return rep

    boxes, other = boxes_from_labelme(data)
    rep["other"] = other
    if not boxes:
        rep["skipped"] = "没有矩形框"
        return rep

    w, h = image_size(image_path, data)
    rep["size"] = (w, h)
    if not (w and h):
        rep["skipped"] = "拿不到图宽高"
        return rep

    if categories is None:
        categories = sorted({c for c, _ in boxes})

    base = os.path.splitext(os.path.basename(image_path))[0]
    target_dir = out_dir or os.path.dirname(image_path)
    det_path = os.path.join(target_dir, base + "_det.json")
    rep["path"] = det_path
    rep["boxes"] = len(boxes)
    if dry_run:
        return rep

    bboxes = [{"category": cat, "bbox": [round(v, 2) for v in bb]}
              for cat, bb in boxes]
    try:
        os.makedirs(target_dir, exist_ok=True)
        save_coco_json(det_path, image_path, w, h, bboxes, categories)
        rep["written"] = True
    except Exception as e:                               # noqa: BLE001
        rep["skipped"] = "写不了（%s: %s）" % (type(e).__name__, e)
    return rep


def collect(labels_dir, images_dir):
    """扫一个标注目录，返回 (items, report)。

    items: [(json_path, image_path, image_w, image_h, [(category, bbox), ...])]
    """
    items = []
    skipped = {"bad_json": [], "no_image": [], "non_rect": [], "empty": []}
    other_types = {}

    try:
        names = sorted(os.listdir(labels_dir))
    except Exception as e:                               # noqa: BLE001
        return [], {"items": 0,
                    "skipped": {"bad_json": [(labels_dir, str(e))],
                                "no_image": [], "non_rect": [], "empty": []},
                    "other_shape_types": {}}

    for fn in names:
        if not fn.lower().endswith(".json"):
            continue
        jp = os.path.join(labels_dir, fn)
        try:
            with open(jp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:                           # noqa: BLE001
            skipped["bad_json"].append((fn, "%s: %s" % (type(e).__name__, e)))
            continue

        img_name = os.path.basename(d.get("imagePath") or "")
        ip = (os.path.join(images_dir, img_name) if img_name
              else find_image(images_dir, os.path.splitext(fn)[0]))
        if not ip or not os.path.exists(ip):
            skipped["no_image"].append((fn, img_name or os.path.splitext(fn)[0]))
            continue

        boxes, other = boxes_from_labelme(d)
        for k, v in other.items():
            other_types[k] = other_types.get(k, 0) + v
            skipped["non_rect"].append((fn, k, None))
        if not boxes:
            skipped["empty"].append(fn)
            continue

        w, h = image_size(ip, d)
        items.append((jp, ip, w, h, boxes))

    report = {"items": len(items), "skipped": skipped,
              "other_shape_types": other_types}
    return items, report


def convert_items(items, categories, out_dir=None, dry_run=False):
    """批量写 `<base>_det.json`，返回写出条数。"""
    written = 0
    for _jp, ip, w, h, boxes in items:
        base = os.path.splitext(os.path.basename(ip))[0]
        target_dir = out_dir or os.path.dirname(ip)
        det_path = os.path.join(target_dir, base + "_det.json")
        if dry_run:
            written += 1
            continue
        bboxes = [{"category": cat, "bbox": [round(v, 2) for v in bb]}
                  for cat, bb in boxes]
        os.makedirs(target_dir, exist_ok=True)
        save_coco_json(det_path, ip, w, h, bboxes, categories)
        written += 1
    return written
