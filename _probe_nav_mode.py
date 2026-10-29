# -*- coding: utf-8 -*-
"""验证「指针 / 手掌」模式切换与箭头光标（2026-10-28）。

只读：不写工程、不改配置。
判据全部落在**真实状态**上：发出的模式信号、画布实际 _mode、控件的 cursor().shape()。
"""
import os
import sys

import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

app = QApplication([])

from annotation.mask_editor import AnnotationCanvas  # noqa: E402
from ui.annotation_toolbar import AnnotationToolbar  # noqa: E402

_ok = 0
_fail = 0


def ck(cond, msg):
    global _ok, _fail
    if cond:
        _ok += 1
        print("  PASS", msg)
    else:
        _fail += 1
        print("  FAIL", msg)


def main():
    tb = AnnotationToolbar()
    cv = AnnotationCanvas()
    cv.resize(900, 700)

    # 复刻 main_window 的接线（1349 行 + 1339 行）
    tb.mode_selected.connect(cv.set_mode)
    cv.mode_changed.connect(tb.set_mode)

    print("\n[1] 按钮存在且在常驻条（不在折叠的工具行里）")
    ck(tb._ptr_btn is not None and tb._hand_btn is not None, "指针/手掌两枚按钮都建出来了")
    ck(tb._ptr_btn.objectName() == "navPtr", "指针 objectName=navPtr")
    ck(tb._hand_btn.objectName() == "navHand", "手掌 objectName=navHand")
    ck(tb._ptr_btn.parent() is tb and tb.tools_box.isAncestorOf(tb._ptr_btn) is False,
       "★ 两枚按钮不在 tools_box（折叠工具行）里 —— 否则 Ctrl+B 一折叠就没法拖图了")
    ck(tb._ptr_btn.isHidden() is False, "默认可见")
    ck(tb.nav_mode() == "annot", "默认处于标注模式（指针选中）")

    print("\n[2] 「平移」已从工具行移除，但快捷键位仍在")
    ck("pan" not in tb._tool_buttons, "工具行里不再有 pan 按钮（搬去常驻条了）")
    ck(len(tb._tool_buttons) == 9, "工具行剩 9 个工具，实际 %d" % len(tb._tool_buttons))

    print("\n[3] 点「手掌」-> 画布进入平移，光标变张手")
    tb._hand_btn.click()
    ck(cv._mode == AnnotationCanvas.MODE_PAN, "画布 _mode == pan")
    ck(tb.nav_mode() == "pan", "工具栏显示为手掌态")
    ck(tb._hand_btn.isChecked() and not tb._ptr_btn.isChecked(), "互斥：手掌选中、指针弹起")
    ck(cv.cursor().shape() == Qt.OpenHandCursor,
       "画布光标 = OpenHand，实际 %s" % cv.cursor().shape())

    print("\n[4] 点「指针」-> 回到上次的标注工具，光标变箭头")
    tb._ptr_btn.click()
    ck(cv._mode == AnnotationCanvas.MODE_BRUSH, "回到默认标注工具 brush")
    ck(tb.nav_mode() == "annot", "工具栏显示为指针态")
    ck(cv.cursor().shape() == Qt.ArrowCursor,
       "★ 光标 = Arrow（不是十字），实际 %s" % cv.cursor().shape())

    print("\n[5] 换工具后再切手掌/指针，工具不能丢")
    tb._tool_buttons["polygon"].click()
    ck(cv._mode == AnnotationCanvas.MODE_POLYGON, "选了多边形")
    tb._hand_btn.click()
    ck(cv._mode == AnnotationCanvas.MODE_PAN, "切到手掌")
    tb._ptr_btn.click()
    ck(cv._mode == AnnotationCanvas.MODE_POLYGON,
       "★ 点指针回到**多边形**（不是硬跳画笔）")
    ck(tb._tool_buttons["polygon"].isChecked(), "工具行里多边形仍高亮（两处同源）")

    print("\n[6] 每个标注工具的光标都必须是箭头（不许有十字漏网）")
    bad = []
    for t in ("brush", "eraser", "ignore", "polygon", "line", "rect", "circle", "ellipse", "sam"):
        cv.set_mode(t)
        s = cv.cursor().shape()
        if s != Qt.ArrowCursor:
            bad.append("%s->%s" % (t, s))
    ck(not bad, "9 个工具全是箭头" + ("" if not bad else "；漏网：%s" % bad))

    print("\n[7] 快捷键 H 切平移后，常驻条按钮也要跟着亮")
    cv.set_mode("pan")
    ck(tb.nav_mode() == "pan", "canvas.set_mode('pan') 后工具栏同步为手掌态")
    cv.set_mode("brush")
    ck(tb.nav_mode() == "annot", "切回标注后同步为指针态")

    print("\n" + "=" * 70)
    print("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (_ok, _fail, _ok + _fail))
    print("=" * 70)
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
