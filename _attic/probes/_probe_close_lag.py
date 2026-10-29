# -*- coding: utf-8 -*-
"""定位"多边形闭合后卡几秒"的真实热点（2026-10-28）。

方法：真实 MainWindow + 真实工程（用户截图里那个 489 张图的），
模拟画一个多边形并闭合，然后**把随后 2 秒内所有排队事件跑完**，
用 cProfile 统计这段里谁最贵。

⚠️ 只读 safeguards：
  · 先把当前图的 JSON / mask.png 各备份一份，跑完原样还原（不污染用户标注）；
  · 不调工程级保存、不写 train_test_split。
"""
import cProfile
import io
import os
import pstats
import shutil
import sys
import time

import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PyQt5.QtCore import QEvent, QPoint, Qt  # noqa: E402
from PyQt5.QtGui import QMouseEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication([])

from ui.main_window import MainWindow  # noqa: E402

PROJ = "验证_极耳层数_Flow66_Tao_20261020"
PDIR = (r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程"
        r"\训练工程_20261021\05-TrainingProject\%s" % PROJ)
NAME = "20260702003804_529125B25B36983_2074423_极耳翻折检测_翻折相机A1-铝内_层数NG_NG"


def pump(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.005)


def main():
    json_p = os.path.join(PDIR, "images", NAME + ".json")
    png_p = os.path.join(PDIR, "annotations", NAME + "_mask.png")
    baks = []
    for p in (json_p, png_p):
        if os.path.exists(p):
            b = p + ".probe_bak"
            shutil.copy2(p, b)
            baks.append((p, b))

    try:
        win = MainWindow()
        win.resize(1600, 980)
        win.show()
        app.processEvents()

        print("打开工程 …")
        t0 = time.perf_counter()
        win._open_project_by_name(PROJ)
        pump(1.5)
        print("  打开耗时 %.1f s，图片数 %d"
              % (time.perf_counter() - t0, len(getattr(win, "_all_images", []))))

        # 选中目标图
        bases = [os.path.splitext(os.path.basename(p))[0] for p in win._all_images]
        if NAME in bases:
            win._load_image_by_index(bases.index(NAME))
        pump(1.0)
        c = win.canvas
        print("  当前图 %s  尺寸 %s  zoom=%.3f"
              % (os.path.basename(c.current_image_path or "")[:36], c.image.size, c.zoom_level))

        # ---- 单独量几个可疑点 ----
        print("\n[单点耗时]")
        for label, fn in (
            ("_filter_images(489张)", lambda: win._filter_images(win.search_input.text())),
            ("_refresh_annotation_views(首次)", win._refresh_annotation_views),
            ("_refresh_annotation_views(二次/缓存)", win._refresh_annotation_views),
            ("_refresh_versions", win._refresh_versions),
            ("_sync_mask_png_for_current", win._sync_mask_png_for_current),
            ("_do_update_stats", win._do_update_stats),
        ):
            pump(0.2)
            t0 = time.perf_counter()
            try:
                fn()
            except Exception as e:
                print("  %-28s 异常 %s: %s" % (label, type(e).__name__, e))
                continue
            dt = (time.perf_counter() - t0) * 1000
            print("  %-28s %8.1f ms" % (label, dt))

        # ---- 完整闭合流程 profile ----
        print("\n[完整闭合 + 2 秒事件] cProfile")
        c.set_mode("polygon")
        c._polygon_points = []
        cx, cy, R = 600, 410, 120
        import math
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        c.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPoint(cx + R, cy),
                                      Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        for i in range(1, 61):
            a = 2 * math.pi * i / 60.0
            c.mouseMoveEvent(QMouseEvent(QEvent.MouseMove,
                                         QPoint(int(cx + R * math.cos(a)),
                                                int(cy + R * math.sin(a))),
                                         Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        closed = not c._lasso_lmb
        pump(2.0)                      # 把 500/600/1500ms 三个定时器全跑完
        pr.disable()
        total = (time.perf_counter() - t0) * 1000
        print("  自动闭合=%s  合计 %.0f ms（含刻意等待的 2000ms 空转）" % (closed, total))

        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
        txt = s.getvalue()
        # 只打有意义的行
        for line in txt.splitlines():
            if any(k in line for k in ("cumtime", "main_window.py", "labelme_io.py",
                                       "mask_editor.py", "mask_overlay.py", "_filter_images",
                                       "_refresh", "_sync", "_auto_save", "paintEvent",
                                       "function calls")):
                print("   " + line.strip()[:150])
    finally:
        for p, b in baks:
            try:
                shutil.move(b, p)
            except Exception as e:
                print("还原失败", p, e)
        print("\n[已还原被改动的标注文件]")
        for p, _ in baks:
            print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
