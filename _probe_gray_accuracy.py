# -*- coding: utf-8 -*-
"""平均灰度精度自校验：用合成数据集（真值已知）验平台两张表的「平均灰度」。

数据由 `_gen_gray_testset.py` 生成：背景灰度 255，前景 200×200 的平均灰度
从 0 覆盖到 255（含 127.5 / 63.5 这类半像素值），真值写在 `manifest.csv`。

本探针**走平台自己的代码**（不是另写一套算法）：
  · 标注表：`MainWindow._update_ann_stats()`，读「平均灰度」单元格
  · 结果表：把标注掩膜当成"完美预测"喂进去，`MainWindow._update_stats()`
判定标准：两张表显示值与 manifest 真值之差 < 0.05（即显示精度 0.1 内）。

为什么要这么测：平均灰度这一列现场是拿来卡灰度差阈值的（15~20 级），
口径一旦漂移（缩放 / 加权 / 用了缩略图），阈值就废了。这条探针把它钉死。

用法：
    "C:/aitp_venv/Scripts/python.exe" -u _probe_gray_accuracy.py [工程目录]
"""
import csv
import os
import sys
import traceback

import numpy as np
import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_PD = os.path.join(
    r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程\训练工程_20261021\05-TrainingProject",
    "灰度校验_合成_Arthur_20261022")

TOL = 0.05


def main():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    pd = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PD
    man = os.path.join(pd, "manifest.csv")
    if not os.path.exists(man):
        print("找不到 manifest.csv：%s" % man)
        print("先跑：python _gen_gray_testset.py")
        return 2
    with open(man, encoding="utf-8-sig") as f:
        truth = {r["file"]: float(r["target_mean"]) for r in csv.DictReader(f)}

    from ui.main_window import MainWindow
    win = MainWindow()
    win.current_project = {"name": os.path.basename(pd)}
    win.pm.get_project_dir = lambda n=None: pd
    win._cached_classes = ["background", "Fore"]
    canvas = win.canvas
    canvas.project_dir = pd

    rows = []
    for name in sorted(truth):
        ip = os.path.join(pd, "images", name + ".png")
        mp = os.path.join(pd, "annotations", name + "_mask.png")
        if not (os.path.exists(ip) and os.path.exists(mp)):
            continue
        canvas.current_image_path = ip
        canvas.image = Image.open(ip).convert("RGB")
        canvas.mask = np.array(Image.open(mp), np.uint8)
        canvas._prediction_mask = canvas.mask.copy()   # 完美预测
        canvas._pred_prob = None
        canvas._cached_qimage = None
        canvas._cached_gray = None
        canvas._disp_image = None
        canvas._cache_qimage()

        win._update_ann_stats()
        ha = [win.ann_table.horizontalHeaderItem(i).text()
              for i in range(win.ann_table.columnCount())]
        a = (win.ann_table.item(0, ha.index("平均灰度")).text()
             if win.ann_table.rowCount() else "—")

        win._update_stats()
        hr = [win.stats_table.horizontalHeaderItem(i).text()
              for i in range(win.stats_table.columnCount())]
        r = (win.stats_table.item(0, hr.index("平均灰度")).text()
             if win.stats_table.rowCount() else "—")

        t = truth[name]
        fa = abs(float(a.split("，")[0]) - t) if a not in ("—", "-", "") else 999.0
        fr = abs(float(r.split("，")[0]) - t) if r not in ("—", "-", "") else 999.0
        rows.append((name, t, a, r, fa, fr))

    if not rows:
        print("没有可用样本")
        return 2

    print("%-12s %7s %10s %10s %8s %8s" % ("图", "真值", "标注表", "结果表", "误差A", "误差R"))
    for x in rows[::3]:
        print("%-12s %7.1f %10s %10s %8.3f %8.3f" % x)

    wa = max(x[4] for x in rows)
    wr = max(x[5] for x in rows)
    bad = [x for x in rows if x[4] > TOL or x[5] > TOL]
    print("\n%d 张：标注表最大误差 %.4f ｜ 结果表最大误差 %.4f" % (len(rows), wa, wr))
    if bad:
        print("FAIL %d 张超差：" % len(bad))
        for x in bad:
            print("   %-12s 真值 %.1f 标注=%s 结果=%s" % (x[0], x[1], x[2], x[3]))
        return 1
    print("PASS 两张表的平均灰度与真值一致（容差 %.2f）" % TOL)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        print("CRASH:\n" + traceback.format_exc())
        code = 2
    sys.stdout.flush()
    os._exit(code)
