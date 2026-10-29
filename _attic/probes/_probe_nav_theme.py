# -*- coding: utf-8 -*-
"""在**真实主题链路**下截图「指针 / 手掌」按钮，确认选中态看得见。

为什么必须走产品路径：`sweep_inline_styles` 会把控件级内联 QSS 的**颜色**全部剥掉
（只留度量），工具栏 `_QSS` 里那句 `QToolButton:checked { background:#0E3A52 ... }`
正好在白名单里 ⇒ 剥完就没了。选中态只剩"图标由青变白"。

在**深色**底上白图标看得见；在**浅色**底上白图标几乎不可见 —— 这就是本探针要抓的。
2026-10-15 的 `#layerSw` 就是同一个坑（详见 ui/themes.py 第 92 行注释）。

产出 _ui_probe_shots/nav_mode/theme_{light,dark}_{ptr,hand}.png

⚠️ 不修改用户的 .ui_settings.json（apply_theme 会回写，用完还原）。
"""
import ast
import json
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

OUT = os.path.join(HERE, "_ui_probe_shots", "nav_mode")
UI_SETTINGS = os.path.join(HERE, ".ui_settings.json")


def load_base_qss():
    """从 main.py AST 里抠出产品基础 QSS（同 probe_mark_ui._load_product_base_qss）。"""
    try:
        src = open(os.path.join(HERE, "main.py"), encoding="utf-8-sig").read()
        tree = ast.parse(src)
    except Exception as e:
        print("  ⚠ 基础 QSS 抠取失败: %s" % e)
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


def apply_product_theme(name):
    from ui.themes import apply_theme, set_base_qss, base_qss, sweep_inline_styles
    b = load_base_qss()
    if b and not base_qss():
        set_base_qss(b)
    bak = None
    try:
        if os.path.exists(UI_SETTINGS):
            bak = open(UI_SETTINGS, "rb").read()
    except Exception:
        bak = None
    apply_theme(app, name)
    sweep_inline_styles(app, restore=False)
    if bak is not None:
        try:
            open(UI_SETTINGS, "wb").write(bak)
        except Exception:
            pass


def grab(w, name):
    app.processEvents()
    img = QImage(w.size(), QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    w.render(img)
    img.save(os.path.join(OUT, name))
    q = img.convertToFormat(QImage.Format_RGB32)
    p = q.bits()
    p.setsize(q.byteCount())
    return np.frombuffer(p, np.uint8).reshape(q.height(), q.width(), 4)[:, :, :3].copy()


def btn_px(w, a):
    r = w.geometry()
    return a[r.y():r.y() + r.height(), r.x():r.x() + r.width()]


def main():
    os.makedirs(OUT, exist_ok=True)
    win = MainWindow()
    win.resize(1600, 980)
    win.show()
    app.processEvents()

    tb = win.annotation_toolbar
    tb.set_tools_visible(True)
    app.processEvents()

    report = {}
    for theme in ("light", "dark"):
        apply_product_theme(theme)
        app.processEvents()

        tb._ptr_btn.click()
        app.processEvents()
        a = grab(tb, "theme_%s_ptr.png" % theme)
        tb._hand_btn.click()
        app.processEvents()
        b = grab(tb, "theme_%s_hand.png" % theme)

        r1 = tb._ptr_btn.geometry()
        r2 = tb._hand_btn.geometry()
        # 同一按钮在"选中/未选中"两态下的像素差 = 选中态到底看不看得出来
        d_ptr = int((np.abs(a[r1.y():r1.y() + r1.height(), r1.x():r1.x() + r1.width()].astype(np.int16)
                          - b[r1.y():r1.y() + r1.height(), r1.x():r1.x() + r1.width()].astype(np.int16)) > 25).sum())
        d_hand = int((np.abs(a[r2.y():r2.y() + r2.height(), r2.x():r2.x() + r2.width()].astype(np.int16)
                           - b[r2.y():r2.y() + r2.height(), r2.x():r2.x() + r2.width()].astype(np.int16)) > 25).sum())
        report[theme] = (d_ptr, d_hand)
        print("[%s] 切换选中态后像素变化：指针钮 %d ｜ 手掌钮 %d" % (theme, d_ptr, d_hand))

        # 图标本身画出来没有（与按钮外背景比）
        bg = a[2, tb.width() - 6]
        for nm, r in (("指针", r1), ("手掌", r2)):
            px = a[r.y():r.y() + r.height(), r.x():r.x() + r.width()]
            ink = int((np.abs(px.astype(np.int16) - bg.astype(np.int16)).sum(axis=2) > 40).sum())
            print("      %s图标非背景像素 %d" % (nm, ink))

    print("\n结论：")
    ok = True
    for theme, (dp, dh) in report.items():
        good = min(dp, dh) > 60
        print("  %-5s 选中态可辨 : %s (%d/%d 像素)" % (theme, "OK" if good else "**太弱**", dp, dh))
        ok = ok and good
    print("\n截图：", OUT)
    win.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
