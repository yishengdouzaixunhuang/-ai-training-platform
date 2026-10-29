# -*- coding: utf-8 -*-
"""验证工具按钮「选中」高亮在真实主题下是否真的看得见（2026-10-28）。

用户反馈："选中某个标注工具后，不易观察是否选中，因为背景也是白的。"

判据用**像素**（交接文档地雷 4）：选中与未选中同一按钮的像素差要够大，
且选中块本身必须是明显的彩色（不是"白图标消失"那种假变化）。

⚠️ 走真实主题链路（apply_theme + sweep_inline_styles），否则测的不是用户看到的样子。
⚠️ 不修改 .ui_settings.json。
"""
import ast
import os
import sys

import numpy as np
import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PyQt5.QtGui import QImage  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication([])

from ui.main_window import MainWindow  # noqa: E402

OUT = os.path.join(HERE, "_ui_probe_shots", "tool_highlight")
UI_SETTINGS = os.path.join(HERE, ".ui_settings.json")


def load_base_qss():
    try:
        src = open(os.path.join(HERE, "main.py"), encoding="utf-8-sig").read()
        tree = ast.parse(src)
    except Exception:
        return ""
    best = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "setStyleSheet":
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                            and len(a.value) > len(best):
                        best = a.value
    return best


def apply_theme(name):
    from ui.themes import apply_theme as at, set_base_qss, base_qss, sweep_inline_styles
    b = load_base_qss()
    if b and not base_qss():
        set_base_qss(b)
    bak = None
    try:
        if os.path.exists(UI_SETTINGS):
            bak = open(UI_SETTINGS, "rb").read()
    except Exception:
        bak = None
    at(app, name)
    sweep_inline_styles(app, restore=False)
    if bak is not None:
        try:
            open(UI_SETTINGS, "wb").write(bak)
        except Exception:
            pass


def grab(w):
    app.processEvents()
    img = QImage(w.size(), QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    w.render(img)
    q = img.convertToFormat(QImage.Format_RGB32)
    p = q.bits()
    p.setsize(q.byteCount())
    return np.frombuffer(p, np.uint8).reshape(q.height(), q.width(), 4)[:, :, :3].copy(), img


def main():
    os.makedirs(OUT, exist_ok=True)
    win = MainWindow()
    win.resize(1600, 980)
    win.show()
    app.processEvents()
    tb = win.annotation_toolbar
    tb.set_tools_visible(True)
    app.processEvents()

    ok = fail = 0

    def ck(cond, msg):
        nonlocal ok, fail
        if cond:
            ok += 1
            print("  PASS", msg)
        else:
            fail += 1
            print("  FAIL", msg)

    for theme in ("light", "dark"):
        apply_theme(theme)
        app.processEvents()
        print("\n[%s]" % theme)

        # 基准：全部取消选中。⚠️ QButtonGroup 互斥时 `setChecked(False)` 对**当前选中项无效**，
        # 必须临时关掉互斥，否则"未选中态"的基准其实还是选中态（实测差异恒为 0）。
        grp = tb._tool_group
        grp.setExclusive(False)
        for b in tb._tool_buttons.values():
            b.setChecked(False)
        app.processEvents()
        base, _ = grab(tb)

        checked_report = []
        for tid in ("brush", "rect", "polygon"):
            btn = tb._tool_buttons[tid]
            for b in tb._tool_buttons.values():
                b.setChecked(False)
            app.processEvents()
            # ⚠️ 直接 grab 按钮自己，不按坐标去父控件里抠 ——
            #    geometry() 是相对 tools_box 的，换算一次错一次（实测截到橙红色）。
            off_img = btn.grab().toImage()
            btn.setChecked(True)
            app.processEvents()
            on_img = btn.grab().toImage()

            def _rgb(qi):
                qi = qi.convertToFormat(QImage.Format_RGB32)
                pp = qi.bits()
                pp.setsize(qi.byteCount())
                # ⚠️ Format_RGB32 在内存里是 **BGRX**（小端 0xffRRGGBB），
                #    直接取前 3 通道拿到的是 BGR —— 不翻转会把 #2563EB 读成 (235,99,37)。
                return np.frombuffer(pp, np.uint8).reshape(qi.height(), qi.width(), 4)[:, :, 2::-1].astype(np.int16)

            o = _rgb(off_img)
            n = _rgb(on_img)
            diff = int((np.abs(n - o).sum(axis=2) > 30).sum())
            mean_on = n.reshape(-1, 3).mean(axis=0)
            is_blue = bool(mean_on[2] > mean_on[0] + 30 and mean_on[2] > 60)
            checked_report.append((tid, diff, tuple(int(v) for v in mean_on), is_blue))
            print("  %-8s 选中/未选中像素差 %5d ｜ 选中块均色 RGB%s ｜ 偏蓝=%s"
                  % (tid, diff, tuple(int(v) for v in mean_on), is_blue))

        for tid, diff, mean, blue in checked_report:
            ck(diff > 150, "%s 选中态与未选中差异明显（%d 像素）" % (tid, diff))
            ck(blue, "%s 选中块是真的彩色底（RGB%s），不是白底白图标" % (tid, mean))

        # 存一张选中 brush 的截图
        for b in tb._tool_buttons.values():
            b.setChecked(False)
        tb._tool_buttons["brush"].setChecked(True)
        app.processEvents()
        grp.setExclusive(True)
        _, img = grab(tb)
        img.save(os.path.join(OUT, "tools_%s_brush.png" % theme))

    print("\n" + "=" * 70)
    print("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (ok, fail, ok + fail))
    print("截图：", OUT)
    print("=" * 70)
    win.close()
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
