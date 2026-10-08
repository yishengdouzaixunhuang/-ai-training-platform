# -*- coding: utf-8 -*-
"""从标注 JSON 重建掩膜 PNG（掩膜的唯一真源 = 图片的标注 JSON）。

什么时候需要它：
    平台的 `images/<base>.json` 才是标注真源，`annotations/<base>_mask.png` 只是派生产物
    （画布 load_mask_from_json 里就是"优先读 JSON、顺便回写 PNG"）。
    但只要有一步用了错的类别表去光栅化，PNG 的像素值语义就会整体错位
    （真实踩过：工程 classes 为空时 shapes_to_mask 给未知标签"临时编号"，
     同一标签在不同图里拿到不同号，14 张掩膜全废）。

本工具做三件事：
    1. 汇总工程里所有标注 JSON，按"组"聚合（多视图 <组名>_te_N 视为同一台料）；
    2. 保证 project.json 的类别表包含 JSON 里出现过的类别（可用 --no-fix-classes 关掉）；
    3. 用**统一的类别表**把每组 te_0 的 JSON 光栅化成 annotations/<组名>_te_0_mask.png。

用法：
    python tools/rebuild_masks_from_json.py <工程目录> --dry-run     # 只看报告，不写文件
    python tools/rebuild_masks_from_json.py <工程目录> --apply       # 真的重建
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from core.class_discovery import discover_labels, merge_classes, read_project_classes
from annotation.labelme_io import shapes_to_mask

VIEW_SUFFIX_RE = re.compile(r"_te_(\d+)$")


def group_of(stem):
    m = VIEW_SUFFIX_RE.search(stem)
    return (stem[:m.start()], int(m.group(1))) if m else (stem, 0)


def collect(project_dir):
    """返回 {组名: {视图号: json路径}} 与 {组名: {视图号: 图片路径}}"""
    img_dir = os.path.join(project_dir, "images")
    js, jp = collections.defaultdict(dict), collections.defaultdict(dict)
    for f in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(f)
        low = ext.lower()
        if low not in (".json", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        if low == ".json" and stem in ("project", "class_labels", "train_test_split"):
            continue
        key, v = group_of(stem)
        (js if low == ".json" else jp)[key][v] = os.path.join(img_dir, f)
    return js, jp


def shapes_of(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("shapes", []) or []
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--apply", action="store_true", help="真的写盘（默认 dry-run）")
    ap.add_argument("--dry-run", action="store_true", help="只报告不写盘（默认）")
    ap.add_argument("--no-fix-classes", action="store_true", help="不自动补类别表")
    ap.add_argument("--keep-old", action="store_true",
                    help="已有掩膜时跳过（默认重建，因为旧掩膜可能就是错的）")
    args = ap.parse_args()
    project = os.path.abspath(args.project)
    ann_dir = os.path.join(project, "annotations")
    apply_ = args.apply and not args.dry_run

    print(f"工程: {project}")
    print(f"模式: {'写入' if apply_ else 'DRY-RUN（不写任何文件）'}")

    # ---- 1. 类别表 ----
    before = read_project_classes(project)
    found = discover_labels(project)
    if args.no_fix_classes:
        classes = before
    else:
        classes, added = merge_classes(project, found, write=apply_)
        if added and not apply_:
            print(f"[类别] 需要新增 {added}（dry-run 未写盘）")
        elif added:
            print(f"[类别] 已写入 project.json，新增 {added}")
    print(f"[类别] JSON 中出现 = {found}")
    print(f"[类别] 使用的类别表 = {classes}   （下标即掩膜像素值，255=ignore，0=background）")

    # ---- 2. 分组 ----
    js, jp = collect(project)
    print(f"[数据] JSON 组 {len(js)} 个 / 图片组 {len(jp)} 个")
    view_counts = collections.Counter(len(v) for v in js.values())
    print(f"[数据] 每组 JSON 视图数分布 = {dict(view_counts)}")

    # 三视图 JSON 是否互为副本（关系到"以哪个视图为准"）
    same = diff = 0
    diff_samples = []
    for key, vs in js.items():
        cnts = {v: collections.Counter(s.get("label") for s in shapes_of(p))
                for v, p in sorted(vs.items())}
        vals = list(cnts.values())
        if len(vals) >= 2 and all(x == vals[0] for x in vals[1:]):
            same += 1
        else:
            diff += 1
            if len(diff_samples) < 3:
                diff_samples.append((key, {v: dict(c) for v, c in cnts.items()}))
    print(f"[数据] 同组各视图 JSON 内容一致 {same} 组 / 不一致 {diff} 组")
    for k, c in diff_samples:
        print(f"       例: {k} -> {c}")

    # ---- 3. 重建 ----
    os.makedirs(ann_dir, exist_ok=True)
    n_done = n_skip = n_nojson = n_empty = 0
    px_total = collections.Counter()
    old_unique = collections.Counter()
    for key in sorted(js):
        views = js[key]
        # te_0 优先（平台约定：标注只标 te_0，三视图共用一张掩膜）
        v0 = 0 if 0 in views else min(views)
        jpath = views[v0]
        mask_path = os.path.join(ann_dir, f"{key}_te_0_mask.png")
        if args.keep_old and os.path.exists(mask_path):
            n_skip += 1
            continue
        shapes = shapes_of(jpath)
        if not shapes:
            n_empty += 1
            continue
        img_path = jp.get(key, {}).get(v0)
        if img_path and os.path.exists(img_path):
            try:
                h, w = np.array(Image.open(img_path)).shape[:2]
            except Exception:
                with open(jpath, encoding="utf-8") as f:
                    d = json.load(f)
                h, w = d.get("imageHeight", 0), d.get("imageWidth", 0)
        else:
            with open(jpath, encoding="utf-8") as f:
                d = json.load(f)
            h, w = d.get("imageHeight", 0), d.get("imageWidth", 0)
        if not (h and w):
            print(f"  [跳过] {key}: 取不到图像尺寸")
            n_nojson += 1
            continue
        unknown = set()
        mask = shapes_to_mask(shapes, classes, (h, w), unknown_out=unknown)
        if unknown:
            print(f"  ⚠️ {key}: 有类别表里没有的标签 {sorted(unknown)}（已按背景处理）")
        if os.path.exists(mask_path):
            try:
                old = np.array(Image.open(mask_path))
                for u in np.unique(old):
                    old_unique[int(u)] += 1
            except Exception:
                pass
        if apply_:
            Image.fromarray(mask.astype(np.uint8)).save(mask_path)
        n_done += 1
        for u, c in zip(*np.unique(mask, return_counts=True)):
            px_total[int(u)] += int(c)

    print()
    print(f"[结果] 重建 {n_done} 组 / 跳过(已有) {n_skip} / 空标注 {n_empty} / 取不到尺寸 {n_nojson}")
    if old_unique:
        print(f"[对比] 旧掩膜里出现过的像素值 -> 出现组数 = {dict(sorted(old_unique.items()))}")
    print("[结果] 新掩膜像素值 -> 总像素数:")
    for u in sorted(px_total):
        name = classes[u] if 0 < u < len(classes) else ("background" if u == 0 else
                                                        ("__ignore__(255)" if u == 255 else f"?? {u}"))
        print(f"        {u:>3} {name:<16} {px_total[u]:,}")
    if not apply_:
        print()
        print("（dry-run：没有写任何文件。确认无误后加 --apply 执行）")


if __name__ == "__main__":
    main()
