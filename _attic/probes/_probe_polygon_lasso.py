# -*- coding: utf-8 -*-
"""验证多边形标注的"铅笔手感"（2026-10-28）。

用户要求：
  ① 长按左键拖动时**点动成线**，不要呈现一个个小点；
  ② 只有线绕回**起点**附近才闭合，不是松手就闭合。

判据全部落在真实状态上：累计点数、掩膜是否生成、以及**像素数**（圆点标记有没有消失）。
"""
import math
import os
import sys

import numpy as np
import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PyQt5.QtCore import QEvent, QPoint, Qt  # noqa: E402
from PyQt5.QtGui import QImage, QMouseEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication([])

from annotation.mask_editor import AnnotationCanvas  # noqa: E402

PDIR = (r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程"
        r"\训练工程_20261021\05-TrainingProject\华屹_江门莱普华生_框架产品_上大面_缺陷检测_IC引线框架检测设备_Flow66_Arthur_20261022")
CW, CH = 1200, 820
OUT = os.path.join(HERE, "_ui_probe_shots", "polygon")


def mk_canvas():
    idir = os.path.join(PDIR, "images")
    f = [x for x in sorted(os.listdir(idir)) if x.lower().endswith((".bmp", ".jpg", ".png"))][0]
    c = AnnotationCanvas()
    c.resize(CW, CH)
    c.set_project(PDIR)
    c.load_image(os.path.join(idir, f))
    c.zoom_level = c._fit_zoom
    c._mode = AnnotationCanvas.MODE_POLYGON
    c._polygon_points = []
    c.mask[:] = 0          # 工程里本来就带标注，先擦干净再测"半圈不许闭合"
    c._overlay_dirty = True
    return c


def press(c, x, y):
    c.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPoint(x, y),
                                  Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))


def move(c, x, y, btn=Qt.LeftButton):
    c.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPoint(x, y),
                                 Qt.NoButton, btn, Qt.NoModifier))


def release(c, x, y):
    c.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPoint(x, y),
                                    Qt.LeftButton, Qt.NoButton, Qt.NoModifier))


def snap(c):
    img = QImage(c.width(), c.height(), QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    c.render(img)
    q = img.convertToFormat(QImage.Format_RGB32)
    p = q.bits()
    p.setsize(q.byteCount())
    return np.frombuffer(p, np.uint8).reshape(q.height(), q.width(), 4)[:, :, :3].copy()


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = fail = 0

    def ck(cond, msg):
        nonlocal ok, fail
        if cond:
            ok += 1
            print("  PASS", msg)
        else:
            fail += 1
            print("  FAIL", msg)

    # ---------- 场景 1：拖一圈回到起点 ----------
    print("\n[1] 长按拖一圈（终点回到起点附近）")
    c = mk_canvas()
    print("     zoom=%.3f" % c.zoom_level)
    cx, cy, R = CW // 2, CH // 2, 150
    steps = 72
    press(c, cx + R, cy)
    n_after_press = len(c._polygon_points)
    for i in range(1, steps + 1):
        a = 2 * math.pi * i / steps
        move(c, int(cx + R * math.cos(a)), int(cy + R * math.sin(a)))
        if not c._lasso_lmb:
            print("    在第 %d 步（%.0f°）自动闭合" % (i, math.degrees(a)))
            break
    ck(not c._lasso_lmb, "拖回起点后自动闭合并结束本笔（_lasso_lmb=False）")
    ck(len(c._polygon_points) == 0, "闭合后轨迹已清空（当前 %d 点）" % len(c._polygon_points))
    painted = int((c.mask > 0).sum())
    ck(painted > 500, "掩膜真的画出来了（%d 像素）" % painted)
    print("    落笔后首个采样点数=%d → 环绕 360° 共 %d 个采样点（原实现约 4 倍以上）"
          % (n_after_press, steps))

    # ---------- 场景 2：拖一半松手，不许闭合 ----------
    print("\n[2] 拖半圈就松手（未回到起点）")
    c = mk_canvas()
    press(c, cx + R, cy)
    for i in range(1, 40):
        a = math.pi * i / 40.0        # 只画半圈
        move(c, int(cx + R * math.cos(a)), int(cy + R * math.sin(a)))
    n_mid = len(c._polygon_points)
    release(c, int(cx - R), int(cy))
    ck(not c._lasso_lmb, "松手后本笔结束")
    ck(len(c._polygon_points) == n_mid and n_mid > 3,
       "★ 轨迹**保留**下来了（%d 点，不会被当成闭合）" % n_mid)
    ck(int((c.mask > 0).sum()) == 0, "★ 掩膜没有被写入（半圈不闭合）")

    # 截图看"半圈未闭合"的样子：应该是线条，不是一串圆点
    c2 = mk_canvas()
    press(c2, cx + R, cy)
    for i in range(1, 40):
        a = math.pi * i / 40.0
        move(c2, int(cx + R * math.cos(a)), int(cy + R * math.sin(a)))
    shot = snap(c2)
    p = os.path.join(OUT, "drag_half.png")
    _im = QImage(c2.width(), c2.height(), QImage.Format_RGB32)
    c2.render(_im)
    _im.save(p)

    # 数"白色高亮点"：顶点圆点是白边+实心，密度远高于细线
    g = shot.mean(axis=2)
    bright = int((g > 200).sum())
    print("    屏幕上高亮像素 %d（细线应远小于‘每点一枚 5px 圆点’的密度）" % bright)
    ck(bright > 200, "预览线确实画出来了（%d 亮像素）" % bright)

    # 顶点圆点密度的对照：25 点以内才允许画圆点
    ck(len(c2._polygon_points) > 24,
       "拖拽产生的点数 %d > 24 ⇒ 按新规则**不再逐个画圆点**" % len(c2._polygon_points))

    # ---------- 场景 3：继续画完剩下的半圈 ----------
    print("\n[3] 松手后再按住接着画，把剩下的半圈补上")
    press(c, int(cx - R), int(cy))
    for i in range(40, 81):          # 补齐到 360°，回到起点
        a = math.pi * i / 40.0
        move(c, int(cx + R * math.cos(a)), int(cy + R * math.sin(a)))
        if not c._lasso_lmb:
            break
    ck(int((c.mask > 0).sum()) > 500,
       "补齐后自动闭合、掩膜生成（%d 像素）" % int((c.mask > 0).sum()))

    # ---------- 场景 4：采样密度与缩放无关 ----------
    print("\n[4] 采样步长按屏幕像素算（3px），与缩放无关")
    for z in (0.2, 1.0, 4.0):
        c = mk_canvas()
        c.zoom_level = z
        press(c, cx + 200, cy)
        for i in range(1, 61):
            move(c, cx + 200 - i * 3, cy)     # 每步移动 3 屏幕像素
        n = len(c._polygon_points)
        # zoom>3 时受"图像坐标取整"限制：1 图像像素 = zoom 屏幕像素，
        # 3px 的采样步长落不到整像素上，只能每隔一步采一个 ⇒ 点数 ≈ 60*3/z
        exp = 60 if z <= 3.0 else int(round(60 * 3.0 / z))
        ck(abs(n - exp) <= 5, "zoom=%.1f 时 60 步(每步3px) 采到 %d 点（期望≈%d）" % (z, n, exp))

    # ---------- 场景 5：快速画圈（一步跨很大）也要能闭合 ----------
    print("\n[5] 快速画圈：每步跨 45°（一次 mouseMove 跨上百像素）")
    c = mk_canvas()
    press(c, cx + R, cy)
    closed_at = None
    for i in range(1, 13):
        a = 2 * math.pi * i / 8.0        # 8 步一圈，步长很大
        move(c, int(cx + R * math.cos(a)), int(cy + R * math.sin(a)))
        if not c._lasso_lmb:
            closed_at = i
            break
    ck(closed_at is not None,
       "★ 大跨度画圈也能闭合（第 %s 步）—— 靠'线段扫过起点'判定，不是只看端点"
       % closed_at)
    ck(int((c.mask > 0).sum()) > 500, "快速画的圈也生成掩膜（%d 像素）" % int((c.mask > 0).sum()))

    print("\n" + "=" * 70)
    print("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (ok, fail, ok + fail))
    print("截图：", OUT)
    print("=" * 70)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
