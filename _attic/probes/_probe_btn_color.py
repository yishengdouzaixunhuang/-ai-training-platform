# -*- coding: utf-8 -*-
"""诊断：工具按钮选中态到底渲染成了什么颜色。

只为了看清"我设的 #2563EB 有没有真的生效"，出 2 张按钮小图 + 打印主色分布。
"""
import ast
import os
import sys

import numpy as np
import torch  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PyQt5.QtGui import QImage  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
from ui.main_window import MainWindow  # noqa: E402

OUT = os.path.join(HERE, "_ui_probe_shots", "tool_highlight")
UI = os.path.join(HERE, ".ui_settings.json")


def apply_light():
    from ui.themes import apply_theme, set_base_qss, base_qss, sweep_inline_styles
    if not base_qss():
        try:
            src = open(os.path.join(HERE, "main.py"), encoding="utf-8-sig").read()
            tree = ast.parse(src)
            best = ""
            for n in ast.walk(tree):
                if isinstance(n, ast.Call):
                    f = n.func
                    if isinstance(f, ast.Attribute) and f.attr == "setStyleSheet":
                        for a in n.args:
                            if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                                    and len(a.value) > len(best):
                                best = a.value
            if best:
                set_base_qss(best)
        except Exception as e:
            print("base qss 失败", e)
    bak = None
    try:
        if os.path.exists(UI):
            bak = open(UI, "rb").read()
    except Exception:
        pass
    apply_theme(app, "light")
    sweep_inline_styles(app, restore=False)
    if bak is not None:
        try:
            open(UI, "wb").write(bak)
        except Exception:
            pass


def rgb(qi):
    qi = qi.convertToFormat(QImage.Format_RGB32)
    p = qi.bits()
    p.setsize(qi.byteCount())
    return np.frombuffer(p, np.uint8).reshape(qi.height(), qi.width(), 4)[:, :, :3]


def main():
    os.makedirs(OUT, exist_ok=True)
    win = MainWindow()
    win.resize(1600, 980)
    win.show()
    tb = win.annotation_toolbar
    tb.set_tools_visible(True)
    apply_light()
    app.processEvents()

    grp = tb._tool_group
    btn = tb._tool_buttons["rect"]
    print("按钮样式表 =", repr(btn.styleSheet()))
    print("主题 light =", True)

    grp.setExclusive(False)
    for b in tb._tool_buttons.values():
        b.setChecked(False)
    app.processEvents()
    btn.grab().save(os.path.join(OUT, "btn_off.png"))
    off = rgb(btn.grab().toImage())

    btn.setChecked(True)
    app.processEvents()
    btn.grab().save(os.path.join(OUT, "btn_on.png"))
    on = rgb(btn.grab().toImage())
    grp.setExclusive(True)

    for tag, a in (("未选中", off), ("选中", on)):
        cols, counts = np.unique(a.reshape(-1, 3), axis=0, return_counts=True)
        top = np.argsort(-counts)[:4]
        print("%s 主色：" % tag,
              ", ".join("RGB%s×%d" % (tuple(int(v) for v in cols[i]), counts[i]) for i in top))
    print("\n按钮尺寸", on.shape, " 图已存:", OUT)


if __name__ == "__main__":
    main()
