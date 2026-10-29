# -*- coding: utf-8 -*-
"""标注统计表（右栏 Annotation(n)）按「块」出行的自测。

真实反馈（2026-10-22 截图）：一张图上标了 40 个 Hy，图片列表「标注」列 = 40、
底栏「已标注 40」，但 Annotation 表只有 1 行 —— 那一行是**整个类别**的合计
（面积 228936 = 40 块之和，半径 269 是按总面积反推的）。

根因：`_update_ann_stats()` 里 `for cid in sorted(set(np.unique(mask)) - {0,255})`
= 一个**类别**一行。而「已标注 N」数的是 labelme JSON 里的 shape 数 ⇒ 口径不同。

这个探针钉住三件事：
  ① 表格行数 == JSON shape 数（与列表列、底栏同源，永远相等）
  ② 每行是**这一块**的面积（不是同类合计）
  ③ 40 块的大图上不能慢到卡界面（按 bbox 裁切算，不做整幅逐块扫描）

用法：
    "C:/aitp_venv/Scripts/python.exe" -u _probe_ann_table.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5
from PIL import Image
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OK_N = [0]
BAD = []


def ck(name, cond, detail=""):
    if cond:
        OK_N[0] += 1
        print("  PASS %s" % name)
    else:
        BAD.append(name)
        print("  FAIL %s     <<< %s" % (name, detail))


def _rect(x, y, w, h):
    return {"label": "Hy", "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            "shape_type": "polygon"}


def make_project(root, n_shapes=40, size=(1200, 400)):
    """一张图 + n 个互不相碰的 Hy 方块 + 同步的掩膜。"""
    pd = os.path.join(root, "proj")
    os.makedirs(os.path.join(pd, "images"))
    os.makedirs(os.path.join(pd, "annotations"))
    json.dump({"classes": ["background", "Hy"]},
              open(os.path.join(pd, "project.json"), "w", encoding="utf-8"), indent=2)
    W, H = size
    img_path = os.path.join(pd, "images", "cell_te_0.png")
    Image.fromarray(np.full((H, W), 180, np.uint8)).save(img_path)

    shapes = []
    per_row = 10
    for i in range(n_shapes):
        col, row = i % per_row, i // per_row
        shapes.append(_rect(10 + col * 100, 10 + row * 100, 20, 20))
    json.dump({"version": "5.0.1", "imagePath": "cell_te_0.png",
               "imageHeight": H, "imageWidth": W, "shapes": shapes},
              open(os.path.join(pd, "images", "cell_te_0.json"), "w", encoding="utf-8"),
              indent=2)

    from annotation.labelme_io import shapes_to_mask
    mask = shapes_to_mask(shapes, ["background", "Hy"], (H, W))
    Image.fromarray(mask.astype(np.uint8)).save(
        os.path.join(pd, "annotations", "cell_te_0_mask.png"))
    return pd, img_path, mask, shapes


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    root = tempfile.mkdtemp(prefix="aitp_anntable_")
    try:
        pd, img_path, mask, shapes = make_project(root, 40)
        from ui.main_window import MainWindow

        win = MainWindow()
        win._cached_classes = ["background", "Hy"]
        win.canvas.current_image_path = img_path
        win.canvas.image = QImage(img_path)
        win.canvas.mask = mask

        print("\n[1] 40 个 Hy → 表格必须有 40 行")
        t0 = time.perf_counter()
        win._update_ann_stats()
        dt = time.perf_counter() - t0
        rows = win.ann_table.rowCount()
        ck("表格行数 = 40（原来只有 1 行）", rows == 40, "rows=%d" % rows)
        ck("标题计数也变成 40（与表格同源，不会再各说各话）",
           win._block_labels["ann"].text() == "Annotation (40)",
           win._block_labels["ann"].text())
        names = {win.ann_table.item(i, 1).text() for i in range(rows)}
        ck("每行类别都是 Hy", names == {"Hy"}, str(names))
        ck("序号 1..40 连续",
           [win.ann_table.item(i, 0).text() for i in range(rows)]
           == [str(i + 1) for i in range(rows)])

        print("\n[2] 每行是「这一块」，不是同类合计")
        areas = [int(win.ann_table.item(i, 2).text()) for i in range(rows)]
        total = sum(areas)
        mask_total = int((mask == 1).sum())
        ck("40 块面积各不相同但都接近 400px（20x20）",
           all(300 <= a <= 500 for a in areas), "min=%d max=%d" % (min(areas), max(areas)))
        ck("单块面积远小于总面积（原来那行是 40 块之和）",
           max(areas) < mask_total * 0.1, "max=%d total=%d" % (max(areas), mask_total))
        ck("各行面积之和 = 掩膜里该类像素总数（没漏也没重复计）",
           abs(total - mask_total) <= 40, "sum=%d mask=%d" % (total, mask_total))

        print("\n[3] 性能：40 块大图不能卡界面")
        ck("一次刷新 < 2.0s（整幅逐块扫描会到秒级以上）", dt < 2.0, "耗时 %.2fs" % dt)
        print("      实测 %.3fs" % dt)

        # 现场大图（6MP）再压一次：这条防"退回逐块扫全图"
        pd2, img2, mask2, _sh2 = make_project(root + "_big", 40, size=(4000, 1500))
        win2 = win
        win2.canvas.current_image_path = img2
        win2.canvas.image = QImage(img2)
        win2.canvas.mask = mask2
        t1 = time.perf_counter()
        win2._update_ann_stats()
        big_first = time.perf_counter() - t1
        t2 = time.perf_counter()
        win2._update_ann_stats()
        big_warm = time.perf_counter() - t2
        ck("6MP / 40 块：首次 < 1.0s（含灰度缓存）", big_first < 1.0, "%.3fs" % big_first)
        ck("6MP / 40 块：稳态 < 0.25s", big_warm < 0.25, "%.3fs" % big_warm)
        ck("6MP 那张也是 40 行", win2.ann_table.rowCount() == 40,
           "rows=%d" % win2.ann_table.rowCount())
        print("      实测 首次 %.3fs / 稳态 %.3fs" % (big_first, big_warm))
        # 后面的用例换回小图，别把 6MP 的掩膜带过去
        win.canvas.current_image_path = img_path
        win.canvas.image = QImage(img_path)
        win.canvas.mask = mask

        print("\n[4] 没有 JSON 的老工程：退回「一个连通域一行」")
        os.remove(os.path.join(pd, "images", "cell_te_0.json"))
        win._update_ann_stats()
        ck("无 JSON 时仍是 40 行（按连通域拆）", win.ann_table.rowCount() == 40,
           "rows=%d" % win.ann_table.rowCount())

        print("\n[5] 同类两块粘在一起：JSON 有两条就还是两行（计数跟着标注走）")
        shapes2 = [_rect(10, 10, 30, 30), _rect(35, 10, 30, 30)]   # 两块紧邻、可能连通
        from annotation.labelme_io import shapes_to_mask
        m2 = shapes_to_mask(shapes2, ["background", "Hy"], (400, 200))
        json.dump({"version": "5.0.1", "imagePath": "cell_te_0.png",
                   "imageHeight": 200, "imageWidth": 400, "shapes": shapes2},
                  open(os.path.join(pd, "images", "cell_te_0.json"), "w", encoding="utf-8"),
                  indent=2)
        win.canvas.mask = m2
        win._update_ann_stats()
        ck("两条 shape → 两行（连通域可能并成一块，但表要跟标注数一致）",
           win.ann_table.rowCount() == 2, "rows=%d" % win.ann_table.rowCount())

        # ── 20MP 大图：显示图被降采样，region 与原图尺寸必然对不上 ──
        print("\n[6] 20MP 大图：Result 表的「平均灰度」不许整列显示 —")
        import numpy as _np
        big_pd = os.path.join(root, "proj_big")
        os.makedirs(os.path.join(big_pd, "images"))
        os.makedirs(os.path.join(big_pd, "annotations"))
        json.dump({"classes": ["background", "Hy"]},
                  open(os.path.join(big_pd, "project.json"), "w", encoding="utf-8"))
        BW, BH = 5472, 3648                      # 和现场截图同量级（20MP）
        big_img = os.path.join(big_pd, "images", "big.bmp")
        grad = _np.tile(_np.linspace(40, 220, BW, dtype=_np.uint8), (BH, 1))
        Image.fromarray(grad, mode="L").save(big_img)
        # 106 个假"检出块"
        pred = _np.zeros((BH, BW), _np.uint8)
        for i in range(106):
            col, row = i % 10, i // 10
            pred[30 + row * 300: 30 + row * 300 + 120, 40 + col * 500: 40 + col * 500 + 200] = 1
        win.canvas.project_dir = big_pd
        t0 = time.perf_counter()
        win.canvas.load_image(big_img)           # 建立显示图（_cached_gray / _cached_qimage_scale）
        t_load = time.perf_counter() - t0
        win.canvas._prediction_mask = pred
        win.canvas._show_prediction = True
        t1 = time.perf_counter()
        win._update_stats()
        t_res = time.perf_counter() - t1
        rows_n = win.stats_table.rowCount()
        ck("Result 表扫出 106 个块", rows_n == 106, "rows=%d" % rows_n)
        heads = [win.stats_table.horizontalHeaderItem(i).text()
                 for i in range(win.stats_table.columnCount())]
        ci = heads.index("平均灰度")
        cells = [win.stats_table.item(i, ci).text() for i in range(min(20, rows_n))]
        n_dash = sum(1 for v in cells if v.strip() in ("", "—"))
        nums = []
        for v in cells:
            try:
                nums.append(float(v.split("，")[0]))
            except Exception:
                pass
        ck("★「平均灰度」是数字，不是整列 —", n_dash == 0, "前20行里 — 的个数=%d 样例=%s" % (n_dash, cells[:5]))
        ck("灰度数量级合理（40~220 的渐变图上）",
           nums and 30 <= min(nums) and max(nums) <= 230, str(nums[:5]))
        ck("20MP 图 Result 表 < 1.5s", t_res < 1.5, "%.2fs" % t_res)
        print("      大图 load_image %.3fs / Result 表 %.3fs / 前3行灰度 %s"
              % (t_load, t_res, cells[:3]))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 70)
    print("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (OK_N[0], len(BAD), OK_N[0] + len(BAD)))
    for b in BAD:
        print("  FAILED: %s" % b)
    print("=" * 70)
    return 0 if not BAD else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        print("CRASH:\n" + traceback.format_exc())
        code = 2
    sys.stdout.flush()
    os._exit(code)
