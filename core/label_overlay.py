# -*- coding: utf-8 -*-
"""标注标签（`IGNORE` / 类别名）的绘制 —— **主画布与多视图参考窗共用这一份**。

为什么必须共用（同 `core/mask_overlay.py` 的理由）：
    参考窗只拿到「仅掩膜」的透明层，标签块得各自画。两边各写一套字号/配色/锚点规则，
    就会出现"同一个标注在窗口 1 和窗口 2 长得不一样"。参考窗**完全没有**标签绘制
    这段逻辑时，现象就是"只有漫反射图有 IGNORE/Ys 字样，形状图1/形状图2 只有色块"。

调用方只需给出自己的「图像坐标 -> 控件坐标」变换 `to_widget(ix, iy)`：
    * 主画布：rect.x() + ix * zoom      （rect 来自 _image_to_widget_rect）
    * 参考窗：vw/2 + (ix - cx_img)*zoom  （把图像点放到面板中心）
其余（取锚点、字号、配色、圆角块、两个开关的归属）全在这里，保证三窗逐像素同款。

两个开关的归属（与掩膜层 `build_overlay_rgba` 保持一致，别各管各的）：
    * Ctrl+Space -> show_classes：缺陷类别的标签
    * Ctrl+Alt   -> show_ignore ：忽略区（__ignore__, 255）的 IGNORE 标签
历史实现里 `_draw_class_labels` 是"一刀切"（`_show_annotation` 一关全不画），
于是会出现"忽略区掩膜已经隐藏、IGNORE 字样还挂在图上"的不一致；这里按掩膜的
同一套归属修正。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QFont, QFontMetrics


def _xy(pt):
    """兼容 QPointF（.x()/.y()）与 tuple —— 两个调用方各传各的。"""
    try:
        return float(pt.x()), float(pt.y())
    except AttributeError:
        return float(pt[0]), float(pt[1])


def is_ignore_label(label):
    """忽略区标签判据（与主画布历史行为一致：ignore / gnore 均可）。"""
    s = (label or "").lower()
    return "ignore" in s or "gnore" in s


def label_anchor(points):
    """标签锚点 = 区域外接框左上角（所有点的 min x / min y）。

    与主画布历史实现一致（不用质心）—— 质心落在细长区域里可能跑到中间，
    左上角始终贴着"这个标注从哪开始"，更好认。
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys)


def paint_shape_labels(painter, shapes, to_widget, zoom,
                       show_classes=True, show_ignore=True,
                       ignore_text="IGNORE"):
    """把每个标注区域的标签块画到 `painter` 上。

    参数
    ----
    shapes      : LabelMe shapes 列表（每项含 label / points）
    to_widget   : 图像坐标 -> 控件坐标 的变换；返回 QPointF 或 (x, y) 均可
    zoom        : 当前缩放（决定字号，与主画布同一公式 -> 三窗字号相等）
    show_classes: 是否画缺陷类别标签（Ctrl+Space）
    show_ignore : 是否画忽略区 IGNORE 标签（Ctrl+Alt）

    单条 shape 出错只跳过它自己（历史实现就是 try/except pass 到 shape 级），
    绝不因为一条坏数据让整张图的标签消失。
    """
    if not shapes:
        return
    for shape in shapes:
        try:
            label = shape.get("label", "") or ""
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            ign = is_ignore_label(label)
            if ign and not show_ignore:
                continue
            if (not ign) and not show_classes:
                continue
            cx, cy = label_anchor(points)
            wx, wy = _xy(to_widget(cx, cy))
            text = ignore_text if ign else label
            font_size = max(8, int(10 * zoom))
            font = QFont("Sans", font_size, QFont.Bold)
            painter.setFont(font)
            fm = QFontMetrics(font)
            tw = fm.horizontalAdvance(text) + 8
            th = fm.height() + 4
            bg_rect = QRectF(wx - 2, wy - th, tw, th)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(80, 80, 80, 180) if ign else QColor(0, 0, 0, 170))
            painter.drawRoundedRect(bg_rect, 3, 3)
            painter.setPen(QColor(200, 200, 200) if ign else QColor(255, 255, 255))
            painter.drawText(bg_rect, Qt.AlignCenter, text)
        except Exception:
            continue
