# -*- coding: utf-8 -*-
"""验证「拖动预览层 + 抬笔重建 overlay」是否真的画出来了（数像素，不看几何）。

产出 PNG 到 _ui_probe_shots/stroke_preview/：
  01_before.png     加载后、未画
  02_mid_drag.png   拖动中（预览层应可见，overlay 尚未重建）
  03_after_release 抬笔后（overlay 已重建，预览层已清）
  04_mask_check.png 掩膜叠加可视化（确认 mask 真的被改了）

⚠️ 判据全部用**像素计数**：交接文档地雷 4 —— value / geometry / 信号都会骗人。
"""
import os
import sys
import time

import numpy as np
import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PyQt5.QtCore import QEvent, QPoint, Qt
from PyQt5.QtGui import QImage, QMouseEvent
from PyQt5.QtWidgets import QApplication

app = QApplication([])

from annotation.mask_editor import AnnotationCanvas  # noqa: E402
from core.config import CLASS_COLORS  # noqa: E402

OUT = os.path.join(HERE, "_ui_probe_shots", "stroke_preview")
PDIR = (r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程"
        r"\训练工程_20261021\05-TrainingProject\验证_极耳层数_Flow66_Tao_20261020")
CW, CH = 1200, 820


def snap(canvas, name):
    img = QImage(canvas.width(), canvas.height(), QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    canvas.render(img)
    path = os.path.join(OUT, name)
    img.save(path)
    return img, path


def arr(qimg):
    qimg = qimg.convertToFormat(QImage.Format_RGB32)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    return np.frombuffer(ptr, np.uint8).reshape(h, w, 4)[:, :, :3].copy()


def main():
    os.makedirs(OUT, exist_ok=True)
    idir = os.path.join(PDIR, "images")
    f = [x for x in sorted(os.listdir(idir)) if x.lower().endswith((".bmp", ".jpg", ".png"))][0]

    c = AnnotationCanvas()
    c.resize(CW, CH)
    c.set_project(PDIR)
    c.load_image(os.path.join(idir, f))
    c.zoom_level = c._fit_zoom
    print("图像 %s  mask %s  zoom=%.3f" % (c.image.size, c.mask.shape, c.zoom_level))

    _, p1 = snap(c, "01_before.png")
    before = arr(QImage(p1))
    print("[01] 未画：已保存", os.path.basename(p1))

    # 清空掩膜，从一张白纸开始画（原工程已有标注，会干扰计数）
    c.mask[:] = 0
    c._overlay_dirty = True
    snap(c, "01b_cleared.png")

    # ---- 落笔 + 拖动 ----
    c._mode = AnnotationCanvas.MODE_BRUSH
    c.current_class_id = 1
    c.brush_size = 24
    r = c._brush_radius()
    print("笔刷半径(图像坐标) = %d" % r)

    cx, cy = CW // 2, CH // 2
    press = QMouseEvent(QEvent.MouseButtonPress, QPoint(cx - 220, cy),
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    c.mousePressEvent(press)

    pts = 24
    for i in range(pts):
        px = cx - 220 + int(i * 18)
        py = cy + int(60 * np.sin(i * 0.5))
        mv = QMouseEvent(QEvent.MouseMove, QPoint(px, py),
                         Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        c.mouseMoveEvent(mv)

    t0 = time.perf_counter()
    img_mid, p2 = snap(c, "02_mid_drag.png")
    t_mid = (time.perf_counter() - t0) * 1000
    mid = arr(img_mid)
    color = CLASS_COLORS[1 % len(CLASS_COLORS)]
    # 预览层用该类别色 + alpha 165 画在底图上 ⇒ 数"明显偏该色相"的像素
    print("[02] 拖动中：截图耗时 %.1f ms  _overlay_dirty=%s  _drawing=%s  预览点数=%d"
          % (t_mid, c._overlay_dirty, c._drawing, len(c._stroke_pts)))
    print("     预览层颜色 = %s" % (color,))

    # 预览像素判定：与底图同位置比较，明显变化且朝该类别色偏
    diff = np.abs(mid.astype(np.int16) - before.astype(np.int16)).sum(axis=2)
    changed = int((diff > 30).sum())
    print("     相对未画状态：变化像素 %d（>0 说明预览层真的画出来了）" % changed)

    # ---- 抬笔 ----
    rel = QMouseEvent(QEvent.MouseButtonRelease, QPoint(cx + 200, cy),
                      Qt.LeftButton, Qt.NoButton, Qt.NoModifier)
    c.mouseReleaseEvent(rel)
    print("[03] 抬笔后：_drawing=%s  _stroke_pts=%d  _overlay_dirty=%s"
          % (c._drawing, len(c._stroke_pts), c._overlay_dirty))

    t0 = time.perf_counter()
    img_after, p3 = snap(c, "03_after_release.png")
    t_after = (time.perf_counter() - t0) * 1000
    print("     抬笔后首帧（含 overlay 全量重建）%.1f ms" % t_after)

    # ---- 掩膜是否真的改了 ----
    painted = int((c.mask == 1).sum())
    print("[04] 掩膜中类别1 像素数 = %d（必须 > 0）" % painted)
    ok_mask = painted > 0

    # ---- 抬笔后再画一帧，确认 overlay 已缓存（应很快）----
    t0 = time.perf_counter()
    snap(c, "03b_after_cached.png")
    t_cached = (time.perf_counter() - t0) * 1000
    print("     抬笔后次帧（overlay 命中缓存）%.1f ms  —— 与首帧的差就是重建成本"
          % t_cached)

    after = arr(img_after)
    diff2 = np.abs(after.astype(np.int16) - before.astype(np.int16)).sum(axis=2)
    print("     抬笔后相对未画：变化像素 %d（overlay 着色应覆盖笔迹）" % int((diff2 > 30).sum()))

    print("\n结论：")
    print("  预览层画出   :", "OK" if changed > 500 else "**失败**")
    print("  抬笔清预览   :", "OK" if not c._stroke_pts and not c._drawing else "**失败**")
    print("  掩膜已写入   :", "OK" if ok_mask else "**失败**")
    print("  抬笔后重建   :", "OK" if t_after > t_cached * 2 else "**可疑**")
    print("\n截图目录：", OUT)
    return 0 if (changed > 500 and ok_mask and not c._drawing) else 1


if __name__ == "__main__":
    sys.exit(main())
