# -*- coding: utf-8 -*-
"""labelme 矩形 JSON → 平台检测标注 `<名字>_det.json`（COCO 子集）。

## 为什么需要它

平台的**目标检测**训练只认 `images/<名字>_det.json`（见 `detection/dataset.py`、
`detection/coco_io.py`），而 iSenseFlow / labelme 导出的标注是 `<名字>.json`
（labelme 5.x 格式，`shapes[].shape_type = "rectangle"`）。
中间没有转换，框就"导进去了但训练扫不到"——画布检测模式也读不到框。

（`tools/import_farasis_labelme.py` 是**三视图分割**的导入器，生成 `mask.png`，
不产出 `_det.json`，所以这里单独一个。）

## 做什么

    images/<名字>.json  (labelme rectangle)
        → images/<名字>_det.json (COCO 子集，bbox=[x,y,w,h] 像素)

- 只转换 `shape_type == "rectangle"` 的 shape；其它类型跳过并统计；
- 类别名取 `shape["label"]`，类别表按名字排序（数字字符排在大写字母前，
  保证多次运行的 `category_id` 稳定）；
- 缺 `imageWidth/imageHeight` 时用 PIL 读图补；
- 源图不存在 / JSON 坏了 → 跳过并记进报告，不中断。

## 用法

    # 只体检，不写文件
    python -m tools.labelme_rect_to_det --labels <JSON目录> --images <图片目录> --dry-run

    # 真转换（输出到图片目录，与平台约定一致）
    python -m tools.labelme_rect_to_det --labels <JSON目录> --images <图片目录>

    # 输出到别处（例如先拷出一份可导入的数据包）
    python -m tools.labelme_rect_to_det --labels ... --images ... --out <输出目录>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.coco_io import save_coco_json          # noqa: E402


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rect_to_bbox(points):
    """labelme 矩形的两个角点 → [x, y, w, h]（像素，左上角为原点）。"""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def collect(labels_dir, images_dir, log=print):
    """扫描一遍，返回 (items, report)。

    items: [(json_path, image_path, image_w, image_h, [(category, bbox), ...])]
    """
    items = []
    skipped = {"bad_json": [], "no_image": [], "non_rect": [], "empty": []}
    other_types = {}

    for fn in sorted(os.listdir(labels_dir)):
        if not fn.lower().endswith(".json"):
            continue
        jp = os.path.join(labels_dir, fn)
        try:
            d = _load(jp)
        except Exception as e:                       # noqa: BLE001
            skipped["bad_json"].append((fn, "%s: %s" % (type(e).__name__, e)))
            continue

        img_name = os.path.basename(d.get("imagePath") or "")
        if not img_name:
            img_name = os.path.splitext(fn)[0] + ".png"
        ip = os.path.join(images_dir, img_name)
        if not os.path.exists(ip):
            skipped["no_image"].append((fn, img_name))
            continue

        w = int(d.get("imageWidth") or 0)
        h = int(d.get("imageHeight") or 0)
        if not (w and h):
            try:
                from PIL import Image
                with Image.open(ip) as im:
                    w, h = im.size
            except Exception:                        # noqa: BLE001
                w = h = 0

        boxes = []
        for s in d.get("shapes") or []:
            st = s.get("shape_type") or "polygon"
            if st != "rectangle":
                other_types[st] = other_types.get(st, 0) + 1
                skipped["non_rect"].append((fn, st, s.get("label")))
                continue
            pts = s.get("points") or []
            if len(pts) < 2:
                continue
            cat = str(s.get("label") or "").strip()
            if not cat:
                continue
            boxes.append((cat, _rect_to_bbox(pts)))

        if not boxes:
            skipped["empty"].append(fn)
            continue
        items.append((jp, ip, w, h, boxes))

    report = {"items": len(items), "skipped": skipped, "other_shape_types": other_types}
    return items, report


def convert(items, images_dir, out_dir, categories, dry_run=False, log=print):
    """写 `<base>_det.json`。返回写出条数。"""
    written = 0
    for jp, ip, w, h, boxes in items:
        base = os.path.splitext(os.path.basename(ip))[0]
        target_dir = out_dir or os.path.dirname(ip)
        det_path = os.path.join(target_dir, base + "_det.json")
        bboxes = [{"category": cat, "bbox": [round(v, 2) for v in bb]}
                  for cat, bb in boxes]
        if dry_run:
            written += 1
            continue
        os.makedirs(target_dir, exist_ok=True)
        save_coco_json(det_path, ip, w, h, bboxes, categories)
        written += 1
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description="labelme 矩形 → 平台检测标注 _det.json")
    ap.add_argument("--labels", required=True, help="labelme JSON 目录")
    ap.add_argument("--images", required=True, help="对应图片目录")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认写到图片所在目录，与平台约定一致）")
    ap.add_argument("--dry-run", action="store_true", help="只体检，不写文件")
    args = ap.parse_args(argv)

    items, rep = collect(args.labels, args.images)

    # 类别表按名字排序：数字字符 < 大写字母，多次运行 id 稳定
    cats = sorted({cat for _, _, _, _, boxes in items for cat, _ in boxes})

    print("=== 体检 ===")
    print("可转换: %d 张" % rep["items"])
    print("框总数: %d" % sum(len(b) for _, _, _, _, b in items))
    print("类别 %d 个: %s" % (len(cats), ", ".join(cats)))
    if rep["other_shape_types"]:
        print("⚠️ 非矩形 shape（跳过）: %r" % rep["other_shape_types"])
    for key, label in (("no_image", "图找不到"), ("bad_json", "JSON 坏了"),
                       ("empty", "没有可用矩形")):
        rows = rep["skipped"][key]
        if rows:
            print("⚠️ %s: %d 个  %r" % (label, len(rows), rows[:3]))

    if args.dry_run:
        print("\n（--dry-run，未写任何文件）")
        return 0
    if not items:
        print("\n没有可转换的标注，退出")
        return 1

    n = convert(items, args.images, args.out, cats)
    print("\n已写出 %d 个 _det.json → %s" % (n, args.out or "（图片同目录）"))
    print("类别表: %s" % cats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
