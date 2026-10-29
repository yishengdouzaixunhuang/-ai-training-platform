# -*- coding: utf-8 -*-
"""把「指针 / 手掌」那对按钮渲染成 PNG，并数像素确认真的画出来了。

产出到 _ui_probe_shots/nav_mode/：
  a_ptr.png   指针选中态
  b_hand.png  手掌选中态
  c_folded.png 工具行折叠时（两枚按钮必须仍在，且明显可见）

⚠️ 交接文档地雷 4：判"控件画出来了没有"要数像素 —— value/geometry/信号都会骗人。
   这里比较两张图同一区域的像素差异：选中态必须有可见高亮，且图标不是空白。
"""
import os
import sys

import numpy as np
import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

app = QApplication([])

from ui.annotation_toolbar import AnnotationToolbar  # noqa: E402

OUT = os.path.join(HERE, "_ui_probe_shots", "nav_mode")


def grab(w, name):
    w.adjustSize()
    app.processEvents()
    img = QImage(w.size(), QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    w.render(img)
    path = os.path.join(OUT, name)
    img.save(path)
    return img, path


def rgb(qimg):
    q = qimg.convertToFormat(QImage.Format_RGB32)
    p = q.bits()
    p.setsize(q.byteCount())
    return np.frombuffer(p, np.uint8).reshape(q.height(), q.width(), 4)[:, :, :3].copy()


def btn_pixels(w, img_rgb):
    """取按钮矩形内的像素（用于数"非背景"像素）。"""
    r = w.geometry()
    return img_rgb[r.y():r.y() + r.height(), r.x():r.x() + r.width()]


def ink(px, bg):
    """与背景色差异明显的像素数 = 真的画了东西。"""
    d = np.abs(px.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    return int((d > 40).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    tb = AnnotationToolbar()
    tb.set_tools_visible(True)
    tb.resize(760, 70)
    tb.show()
    app.processEvents()

    # 背景色：取工具栏右下角一块（那里通常没有控件）
    img, p = grab(tb, "a_ptr.png")
    a = rgb(img)
    bg = a[a.shape[0] - 4, a.shape[1] - 12]
    print("工具栏尺寸 %dx%d  背景色 %s" % (a.shape[1], a.shape[0], tuple(int(v) for v in bg)))

    ptr_ink = ink(btn_pixels(tb._ptr_btn, a), bg)
    hand_ink = ink(btn_pixels(tb._hand_btn, a), bg)
    print("[a] 指针选中：指针钮非背景像素 %d ｜ 手掌钮 %d" % (ptr_ink, hand_ink))

    tb._hand_btn.click()
    app.processEvents()
    img2, p2 = grab(tb, "b_hand.png")
    b = rgb(img2)
    ptr_ink2 = ink(btn_pixels(tb._ptr_btn, b), bg)
    hand_ink2 = ink(btn_pixels(tb._hand_btn, b), bg)
    print("[b] 手掌选中：指针钮非背景像素 %d ｜ 手掌钮 %d" % (ptr_ink2, hand_ink2))

    # 选中态差异：同一按钮两种状态的像素必须不同（否则看不出选中的是哪个）
    r = tb._hand_btn.geometry()
    diff_hand = int((np.abs(
        a[r.y():r.y() + r.height(), r.x():r.x() + r.width()].astype(np.int16)
        - b[r.y():r.y() + r.height(), r.x():r.x() + r.width()].astype(np.int16)) > 25).sum())
    r2 = tb._ptr_btn.geometry()
    diff_ptr = int((np.abs(
        a[r2.y():r2.y() + r2.height(), r2.x():r2.x() + r2.width()].astype(np.int16)
        - b[r2.y():r2.y() + r2.height(), r2.x():r2.x() + r2.width()].astype(np.int16)) > 25).sum())
    print("     两态切换后：手掌钮变了 %d 像素，指针钮变了 %d 像素（都要 > 20）"
          % (diff_hand, diff_ptr))

    # 折叠工具行
    tb.set_tools_visible(False)
    app.processEvents()
    img3, p3 = grab(tb, "c_folded.png")
    c = rgb(img3)
    ptr_fold = ink(btn_pixels(tb._ptr_btn, c), bg)
    hand_fold = ink(btn_pixels(tb._hand_btn, c), bg)
    print("[c] 工具行折叠：指针钮 %d ｜ 手掌钮 %d（都还必须 > 20）"
          % (ptr_fold, hand_fold))

    print("\n结论：")
    print("  两个图标都画出来了 :", "OK" if min(ptr_ink, hand_ink, ptr_fold, hand_fold) > 20 else "**失败**")
    print("  选中态肉眼可分辨   :", "OK" if min(diff_hand, diff_ptr) > 20 else "**失败**")
    print("  折叠后仍可切换     :", "OK" if min(ptr_fold, hand_fold) > 20 else "**失败**")
    print("\n截图：", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
