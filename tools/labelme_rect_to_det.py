# -*- coding: utf-8 -*-
"""labelme 矩形 JSON → 平台检测标注 `<名字>_det.json`（COCO 子集）。

转换逻辑**只有一份**，在 `detection/labelme_rect.py`；本文件只是命令行外壳
（界面「导入标注」走的是同一份实现）。

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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.labelme_rect import collect, categories_of, convert_items  # noqa: E402


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
    cats = categories_of([jp for jp, _ip, _w, _h, _b in items])

    print("=== 体检 ===")
    print("可转换: %d 张" % rep["items"])
    print("框总数: %d" % sum(len(b) for *_x, b in items))
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

    n = convert_items(items, cats, out_dir=args.out)
    print("\n已写出 %d 个 _det.json → %s" % (n, args.out or "（图片同目录）"))
    print("类别表: %s" % cats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
