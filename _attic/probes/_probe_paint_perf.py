# -*- coding: utf-8 -*-
"""标注画布绘制性能探针（只读：不写任何工程文件、不改任何代码）。

对标友商 iCogtiveFlow 状态栏那行「耗时 17ms 平均 15.216ms」。

用法：
    "C:/aitp_venv/Scripts/python.exe" -u _probe_paint_perf.py

测三件事：
  ① 静止一帧（fit / 100% 缩放）花多久
  ② 画笔拖动时**每一帧**花多久（这是"卡不卡"的真身）
  ③ overlay 重建单独花多久（拖动期间的周期性尖峰来自它）

⚠️ 刻意不 show()：`QWidget.render(QImage)` 对未显示的控件照样走 paintEvent，
   既避免弹窗干扰，又保留 windows 平台的字体引擎（offscreen 没字体，文字全空白）。
"""
import os
import statistics
import sys
import time

import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PyQt5.QtCore import QEvent, QPoint, Qt
from PyQt5.QtGui import QImage, QMouseEvent
from PyQt5.QtWidgets import QApplication

app = QApplication([])

from annotation.mask_editor import AnnotationCanvas  # noqa: E402

BASE = (r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程"
        r"\训练工程_20261021\05-TrainingProject")

CASES = [
    ("小图   630x400",  "华屹_江门莱普华生_框架产品_上大面_缺陷检测_IC引线框架检测设备_Flow66_Arthur_20261022",
     "上大面_NG_划痕裂纹磕伤_001.bmp"),
    ("中图  2787x1694", "验证_极耳层数_Flow66_Tao_20261020",
     None),
    ("三视图 1175x3739", "新松_233_赣州孚能_大面-缺陷检测_iSense104_Arthur_20260913",
     "-0908-6号机（压伤）-大面-漏检1_te_0.jpg"),
    ("细长  456x7041",  "联赢激光_惠州欣旺达_焊印定位(灰度图分割)_C20_AI训练平台v0.1.0_Arthur_20261022",
     "0_2026-10-21-15-38-50-801.bmp"),
]

CANVAS_W, CANVAS_H = 1200, 820


def stat(ts, label):
    if not ts:
        return
    ts_s = sorted(ts)
    p95 = ts_s[min(len(ts_s) - 1, int(len(ts_s) * 0.95))]
    print("    %-26s mean %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f ms"
          % (label, statistics.mean(ts), statistics.median(ts), p95, max(ts)))


def measure_frames(canvas, n=25, label="", collect=None):
    """连续渲染 n 帧，返回每帧毫秒。"""
    img = QImage(canvas.width(), canvas.height(), QImage.Format_ARGB32_Premultiplied)
    ts = []
    for _ in range(n):
        img.fill(0)
        t0 = time.perf_counter()
        canvas.render(img)
        ts.append((time.perf_counter() - t0) * 1000.0)
    if collect is not None:
        collect.extend(ts)
    return ts


def run_case(name, proj, img_name):
    pdir = os.path.join(BASE, proj)
    idir = os.path.join(pdir, "images")
    if not os.path.isdir(idir):
        print("  [跳过] 没有 images/ :", proj[:40])
        return
    if img_name is None:
        cand = sorted(f for f in os.listdir(idir)
                      if os.path.splitext(f)[1].lower() in (".bmp", ".jpg", ".png"))
        if not cand:
            print("  [跳过] images/ 为空")
            return
        img_name = cand[0]
    path = os.path.join(idir, img_name)
    if not os.path.exists(path):
        print("  [跳过] 文件不存在:", img_name)
        return

    print("\n" + "=" * 78)
    print("【%s】 %s" % (name, img_name[:56]))

    canvas = AnnotationCanvas()
    canvas.resize(CANVAS_W, CANVAS_H)
    canvas.set_project(pdir)

    t0 = time.perf_counter()
    canvas.load_image(path)
    t_load = (time.perf_counter() - t0) * 1000.0
    iw, ih = canvas.image.size
    print("  图像 %dx%d  fit_zoom=%.4f   载入耗时 %.1f ms"
          % (iw, ih, canvas._fit_zoom, t_load))

    # ① 静止帧（fit）
    canvas.zoom_level = canvas._fit_zoom
    ts = measure_frames(canvas, 25)
    stat(ts, "静止帧 fit")

    # ② 静止帧（100%，大图会走裁切路径）
    canvas.zoom_level = 1.0
    ts = measure_frames(canvas, 25)
    stat(ts, "静止帧 100%")

    # ③ overlay 重建本身（强制 dirty 后渲染一帧 = 重建 + 绘制）
    canvas.zoom_level = canvas._fit_zoom
    measure_frames(canvas, 3)  # 预热
    canvas._overlay_dirty = True
    t0 = time.perf_counter()
    canvas._build_overlay()
    t_ovl = (time.perf_counter() - t0) * 1000.0
    canvas._overlay_dirty = False
    print("    %-26s %6.2f ms" % ("overlay 全量重建", t_ovl))

    # ④ 画笔拖动：每帧一次 mouseMove + 一次完整重绘
    canvas._mode = AnnotationCanvas.MODE_BRUSH
    canvas.current_class_id = 1
    canvas.brush_size = 15
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    press = QMouseEvent(QEvent.MouseButtonPress, QPoint(cx - 200, cy),
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    canvas.mousePressEvent(press)
    drag_ts = []
    img = QImage(canvas.width(), canvas.height(), QImage.Format_ARGB32_Premultiplied)
    for i in range(40):
        px = cx - 200 + i * 10
        mv = QMouseEvent(QEvent.MouseMove, QPoint(px, cy + (i % 2) * 30),
                         Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        t0 = time.perf_counter()
        canvas.mouseMoveEvent(mv)
        img.fill(0)
        canvas.render(img)
        drag_ts.append((time.perf_counter() - t0) * 1000.0)
    rel = QMouseEvent(QEvent.MouseButtonRelease, QPoint(cx + 200, cy),
                      Qt.LeftButton, Qt.NoButton, Qt.NoModifier)
    canvas.mouseReleaseEvent(rel)
    stat(drag_ts, "画笔拖动 每帧")
    slow = [t for t in drag_ts if t > 16.7]
    print("    %-26s %d/40 帧掉出 60fps，%d/40 帧掉出 30fps"
          % ("拖帧分布", len(slow), len([t for t in drag_ts if t > 33.4])))

    # ⑤ 套索拖动（多边形逐点追加）
    canvas._mode = AnnotationCanvas.MODE_POLYGON
    canvas._polygon_points = []
    press = QMouseEvent(QEvent.MouseButtonPress, QPoint(cx - 200, cy - 100),
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    canvas.mousePressEvent(press)
    lasso_ts = []
    for i in range(40):
        ang = i * 0.16
        import math
        px = int(cx + 180 * math.cos(ang))
        py = int(cy + 120 * math.sin(ang))
        mv = QMouseEvent(QEvent.MouseMove, QPoint(px, py),
                         Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        t0 = time.perf_counter()
        canvas.mouseMoveEvent(mv)
        img.fill(0)
        canvas.render(img)
        lasso_ts.append((time.perf_counter() - t0) * 1000.0)
    canvas.mouseReleaseEvent(rel)
    stat(lasso_ts, "套索拖动 每帧")

    canvas.deleteLater()


def main():
    print("画布尺寸 %dx%d（模拟主窗中央区域）" % (CANVAS_W, CANVAS_H))
    for name, proj, img in CASES:
        try:
            run_case(name, proj, img)
        except Exception as e:
            import traceback
            print("  [异常]", e)
            traceback.print_exc()
    print("\n" + "=" * 78)
    print("参照：友商 iCogtiveFlow 状态栏「平均 15.216 ms」≈ 65 fps")
    print("60fps 预算 16.7ms / 30fps 预算 33.3ms")


if __name__ == "__main__":
    main()
