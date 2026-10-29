# -*- coding: utf-8 -*-
"""极耳层数评估：用**竖直扫描线**数层，对比推理产物与标注。

为什么不用连通域数层：
    极耳是横贯全图的**斜细线**，分块推理在块边界会断、线本身也会相交 ⇒
    一条线被切成长短多段，连通域个数直接翻倍（实测 GT 16.8 层 → 预测 32.9 层，
    误判成"模型废了"）。横向闭运算 1×61 也只能救到 22.5。

正确口径：**沿竖直方向取若干条扫描线，数每条线上"穿过 Cs 的次数"，取中位数**。
    斜线、断点、相交都不影响；实测同一批图 GT 31.0 层 / 预测 31.4 层。

用法：
    python _probe_layer_count.py <工程目录> [--gap 2] [--cols 15] [--worst 8]
输出：总体层数误差、按相机分组、误差最大的若干张。
"""
import glob
import os
import re
import statistics as st
import sys

import numpy as np
from PIL import Image


def count_layers(mask, gap_tol=2, n_cols=15):
    """沿竖直扫描线数"穿过次数"，取各列中位数。

    gap_tol: 允许连续多少行为空仍算同一条线（抗噪点、抗细断口）。
    """
    m = (mask == 1)
    h, w = m.shape
    counts = []
    for x in np.linspace(int(w * 0.1), int(w * 0.9), max(3, n_cols)).astype(int):
        col = m[:, x]
        seg = 0
        run = False
        gap = 0
        for v in col:
            if v:
                if not run:
                    seg += 1
                    run = True
                gap = 0
            else:
                if run:
                    gap += 1
                    if gap > gap_tol:
                        run = False
        counts.append(seg)
    return int(np.median(counts))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    pd = args[0]
    gap = 2
    cols = 15
    worst_n = 8
    for i, a in enumerate(sys.argv):
        if a == "--gap" and i + 1 < len(sys.argv):
            gap = int(sys.argv[i + 1])
        elif a == "--cols" and i + 1 < len(sys.argv):
            cols = int(sys.argv[i + 1])
        elif a == "--worst" and i + 1 < len(sys.argv):
            worst_n = int(sys.argv[i + 1])

    rows = []
    for mp in sorted(glob.glob(os.path.join(pd, "annotations", "*_mask.png"))):
        base = os.path.basename(mp)[:-len("_mask.png")]
        pp = os.path.join(pd, "outputs", base + "_pred.png")
        if not os.path.exists(pp):
            continue
        gt = np.array(Image.open(mp))
        pr = np.array(Image.open(pp))
        if gt.shape != pr.shape:
            continue
        g, p = count_layers(gt, gap, cols), count_layers(pr, gap, cols)
        inter = int(((gt == 1) & (pr == 1)).sum())
        union = int(((gt == 1) | (pr == 1)).sum())
        iou = inter / union if union else float("nan")
        m = re.search(r"相机([A-Z]\d)", base)
        rows.append((base, g, p, iou, m.group(1) if m else "?"))

    if not rows:
        print("没有可比对的样本（需要 annotations/*_mask.png 与 outputs/*_pred.png 成对存在）")
        return 2

    errs = [abs(r[1] - r[2]) for r in rows]
    print("样本 %d 张   gap=%d  cols=%d" % (len(rows), gap, cols))
    print("  层数：GT 均值 %.1f ｜ 预测均值 %.1f ｜ 平均绝对误差 %.2f" % (
        st.mean(r[1] for r in rows), st.mean(r[2] for r in rows), st.mean(errs)))
    for tol in (0, 1, 2, 3):
        n = sum(1 for e in errs if e <= tol)
        print("  误差 ≤%d 层：%d 张 (%.0f%%)" % (tol, n, 100.0 * n / len(errs)))
    ious = [r[3] for r in rows if r[3] == r[3]]
    if ious:
        print("  Cs IoU：平均 %.3f ｜ 中位 %.3f ｜ 最低 %.3f" % (
            st.mean(ious), st.median(ious), min(ious)))

    groups = sorted(set(r[4] for r in rows))
    if len(groups) > 1:
        print("\n按相机：")
        for gname in groups:
            sub = [r for r in rows if r[4] == gname]
            e = [abs(r[1] - r[2]) for r in sub]
            hit = 100.0 * sum(1 for x in e if x <= 1) / len(e)
            print("  %-3s %2d 张  GT %5.1f 层 预测 %5.1f 层  误差 %.2f  ±1层 %3.0f%%  IoU %.3f" % (
                gname, len(sub), st.mean(r[1] for r in sub), st.mean(r[2] for r in sub),
                st.mean(e), hit, st.mean(r[3] for r in sub)))

    print("\n误差最大的 %d 张：" % worst_n)
    for b, g, p, iou, cam in sorted(rows, key=lambda r: -abs(r[1] - r[2]))[:worst_n]:
        print("  GT %2d → 预测 %2d (差 %+d)  IoU %.2f  %s" % (g, p, p - g, iou, b[-46:]))
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    code = main()
    sys.stdout.flush()
    os._exit(code)
