# -*- coding: utf-8 -*-
"""贴边拼接修复的 A/B 对比（真实工程、真实模型）—— 手工探针，不进自动门禁。

A 面 = 关补丁（等价于 48a67c2 之前的代码）
B 面 = 开补丁（当前代码）

**同一进程、同一模型、同一张图**，只把 `Predictor._edge_safe_weights` 换成恒等函数，
所以差异只可能来自这一个改动。

用法（在仓库根）：
    python tests/manual/edge_ab_compare.py
    python tests/manual/edge_ab_compare.py --project "<工程目录>" --tile 256 --overlap 128

输出：
    · stdout：整图 A/B、贴边裁剪 A/B、四条边的逐项统计、缺陷多边形面积与找回占比
    · 图片：默认写到 <仓库父目录>/_audit_artifacts/edge_ab/
      - 贴边四格对比.png（原图 / 标注真值 / 旧掩膜 / 新掩膜）
      - 贴边差异图.png（新掩膜 + 新旧差异像素标黄）
      - 贴边左沿放大.png（原图 / 旧 / 新，左沿 80px ×5）

为什么要把缺陷"裁到贴边"：
    真实工程里没有天生贴边的标注（本工程 1690 个标注 / 269 个 Rd / 0 个贴边），
    所以用一个真实标注的真实缺陷，把它的最左列对齐裁图第 0 列来构造临界样本。
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch  # noqa: F401  ⚠️ 先于 PyQt5（本链路会 import PyQt5）
from PIL import Image, ImageDraw

from inference.predictor import Predictor

DEFAULT_PROJECT = (
    r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程\训练工程_20261021\05-TrainingProject"
    r"\海目星_大连大众_大面_低曝-上-熔点定位_包膜机-M5_Flow66_Arthur_20261016"
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(ROOT), "_audit_artifacts", "edge_ab")


def border(a):
    return np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])


def run(pred, img, tile, overlap):
    return pred._predict_tiled_torch(img, tile_size=tile, overlap=overlap), pred.last_prob


def polygon_area(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _edge_band(size, px=1):
    """最外 px 圈的布尔掩膜。"""
    m = np.zeros((size, size), dtype=bool)
    m[:px, :] = m[-px:, :] = m[:, :px] = m[:, -px:] = True
    return m


def pick_image(pred, img_dir, only_image=None, only_bbox=None):
    """挑一个"小缺陷"（20~300px 的 bbox）。

    默认取多边形面积最大的那个；给了 --image / --bbox 就精确锁定某一例，
    这样报告里的数字可以被别人一字不差地复算出来。
    """
    cands = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".bmp", ".jpg", ".png")):
            continue
        if only_image and fn != only_image:
            continue
        base = os.path.splitext(fn)[0]
        js = os.path.join(img_dir, base + ".json")     # labelme 真源就在 images/ 下
        if not os.path.exists(js):
            continue
        with open(js, "r", encoding="utf-8") as f:
            d = json.load(f)
        for s in (d.get("shapes") or []):
            if s.get("label") not in pred.classes[1:]:
                continue
            pts = np.array(s["points"], dtype=float)
            if len(pts) < 3:
                continue
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            if only_bbox:
                got = (x0, y0, x1, y1)
                if max(abs(got[i] - only_bbox[i]) for i in range(4)) > 1.0:
                    continue
            if 20 < (x1 - x0) < 300 and 20 < (y1 - y0) < 300:
                cands.append((polygon_area(pts), base, fn, js, s["label"],
                              (x0, y0, x1, y1), pts))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--tile", type=int, default=0, help="0=读 checkpoint 的 image_size")
    ap.add_argument("--overlap", type=int, default=0, help="0=half tile")
    ap.add_argument("--crop", type=int, default=600)
    ap.add_argument("--image", default="", help="只取这张图，如 包膜_..._430.bmp")
    ap.add_argument("--bbox", default="", help="只取这个缺陷：x0,y0,x1,y1")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    args.bbox = tuple(float(v) for v in args.bbox.split(",")) if args.bbox else None

    os.makedirs(args.out, exist_ok=True)
    pred = Predictor(args.project, args.model)
    tile = args.tile or int(getattr(pred, "_智能块尺寸", 0) or 0) or 200
    overlap = args.overlap or max(8, tile // 2)
    print(f"model: classes={pred.classes} tile={tile} overlap={overlap} device={pred.device}")

    img_dir = os.path.join(args.project, "images")
    picked = pick_image(pred, img_dir, args.image or None, args.bbox)
    if not picked:
        print("没找到匹配的缺陷标注（检查 --image / --bbox）")
        return 1
    area, base, fn, js, label, (x0, y0, x1, y1), pts = picked
    print(f"选中: {fn} 缺陷={label} bbox=({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}) "
          f"多边形面积={area:.0f}px²")

    full = Image.open(os.path.join(img_dir, fn)).convert("RGB")

    # ---------- 整图 A/B ----------
    m_b, _ = run(pred, full, tile, overlap)
    pred._edge_safe_weights = lambda *a: a[0]
    m_a, _ = run(pred, full, tile, overlap)
    del pred._edge_safe_weights
    diff = (m_a != m_b)
    print("\n[整图] %s 差异像素=%d (%.4f%%) 内部差异=%d"
          % (full.size, diff.sum(), 100.0 * diff.sum() / diff.size, diff[1:-1, 1:-1].sum()))
    print("[整图] 边界非零: 旧=%d 新=%d" % ((border(m_a) != 0).sum(), (border(m_b) != 0).sum()))

    # ---------- 贴边裁剪 A/B ----------
    size = args.crop
    cx = max(0, min(int(round(x0)), full.width - size))
    cy = max(0, min(int(round((y0 + y1) / 2 - size / 2)), full.height - size))
    crop = full.crop((cx, cy, cx + size, cy + size))
    print(f"\n[贴边] crop=({cx},{cy},{cx + size},{cy + size}) 缺陷最左列在裁图里 x={x0 - cx:.0f}")

    gt = Image.new("L", (size, size), 0)
    gd = ImageDraw.Draw(gt)
    with open(js, "r", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("shapes") or []):
        if s.get("label") not in pred.classes[1:]:
            continue
        gd.polygon([(p[0] - cx, p[1] - cy) for p in s["points"]], fill=1)
    gt_mask = np.array(gt)
    gtd = gt_mask != 0

    c_b, _ = run(pred, crop, tile, overlap)
    pred._edge_safe_weights = lambda *a: a[0]
    c_a, _ = run(pred, crop, tile, overlap)
    del pred._edge_safe_weights
    cd = (c_a != c_b)
    print("[贴边] 缺陷像素: 旧=%d 新=%d；标注像素(裁图内)=%d"
          % ((c_a != 0).sum(), (c_b != 0).sum(), gtd.sum()))
    for nm, sl in (("left", (slice(None), 0)), ("right", (slice(None), -1)),
                   ("top", (0, slice(None))), ("bottom", (-1, slice(None)))):
        print("  边[%-6s] 旧=%3d 新=%3d 变化=%3d | 标注=%3d 新掩膜命中=%3d"
              % (nm, int((c_a[sl] != 0).sum()), int((c_b[sl] != 0).sum()), int(cd[sl].sum()),
                 int(gtd[sl].sum()), int(((c_b[sl] != 0) & gtd[sl]).sum())))
    print("[贴边] 差异像素=%d 内部差异=%d；占真实缺陷面积 %.1f%%"
          % (cd.sum(), cd[1:-1, 1:-1].sum(), 100.0 * cd.sum() / max(1.0, area)))

    # 边界上"找回的"到底是真缺陷还是模型在边上的额外预测：对标注分开计数
    gt_edge = ((gt_mask != 0) & _edge_band(size))
    new_edge = ((c_b != 0) & _edge_band(size))
    print("[贴边] 边界像素对照标注: 新掩膜=%d 其中落在标注内=%d 标注内但未预测=%d"
          % (new_edge.sum(), (new_edge & gt_edge).sum(), (gt_edge & ~new_edge).sum()))

    # ---------- 出图 ----------
    def color(m):
        pal = np.array([[0, 0, 0], [230, 40, 40], [40, 120, 230], [40, 200, 90],
                        [240, 200, 40], [180, 80, 220], [0, 200, 200]], dtype=np.uint8)
        return pal[np.clip(m.astype(int), 0, len(pal) - 1)]

    def overlay(img, m):
        arr = np.array(img).astype(np.float32)
        col = color(m).astype(np.float32)
        msk = (m != 0)[..., None].astype(np.float32)
        return Image.fromarray((arr * (1 - 0.55 * msk) + col * (0.55 * msk)).astype(np.uint8))

    panel = Image.new("RGB", (size * 2, size * 2), (20, 20, 20))
    panel.paste(crop, (0, 0))
    panel.paste(overlay(crop, gt_mask), (size, 0))
    panel.paste(overlay(crop, c_a), (0, size))
    panel.paste(overlay(crop, c_b), (size, size))
    panel.save(os.path.join(args.out, "贴边四格对比.png"))

    dvis = color(c_b).copy()
    dvis[cd] = np.array([255, 255, 0], dtype=np.uint8)
    Image.fromarray(dvis).save(os.path.join(args.out, "贴边差异图.png"))

    zx, zy = 80, 300
    y0z = int(max(0, y0 - cy - 40))
    zs = (0, y0z, zx, y0z + zy)
    zsrc = crop.crop(zs).resize((zx * 5, zy * 5), Image.NEAREST)
    zold = overlay(crop, c_a).crop(zs).resize((zx * 5, zy * 5), Image.NEAREST)
    znew = overlay(crop, c_b).crop(zs).resize((zx * 5, zy * 5), Image.NEAREST)
    zp = Image.new("RGB", (zx * 5 * 3 + 20, zy * 5), (20, 20, 20))
    zp.paste(zsrc, (0, 0)); zp.paste(zold, (zx * 5 + 10, 0)); zp.paste(znew, (zx * 5 * 2 + 20, 0))
    zp.save(os.path.join(args.out, "贴边左沿放大.png"))

    print("\n出图目录:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
