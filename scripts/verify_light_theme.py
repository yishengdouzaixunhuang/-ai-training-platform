# -*- coding: utf-8 -*-
"""浅色主题可用性核验：把 dark / light 两套主题各渲染一遍 + 对比度审计。

做法（严格照 main.py 的样式栈，但**不写** .ui_settings.json，避免动用户的设置）：
  1. 从 main.py 用 AST 取出那套基础 QSS；
  2. app.setStyleSheet(base + LIGHT_OVERLAY / DARK_INDICATOR)；
  3. 构造 MainWindow → show(不映射) → sweep_inline_styles（同 main.py 的 singleShot 清扫）；
  4. 用**产品路径** _render_page() 给图片列表灌几行合成数据（空工程下列表是空的，测不到）；
  5. 遍历控件算 WCAG 对比度，列出 < 4.5 的；同时截图供肉眼核验。

用法：
    C:/aitp_venv/Scripts/python.exe scripts/verify_light_theme.py
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_ui_render"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.chdir(ROOT)
import torch  # noqa: F401,E402

sys.path.insert(0, str(ROOT))
_spec = importlib.util.find_spec("PyQt5")
if _spec is not None and _spec.origin:
    os.environ.setdefault("QT_PLUGIN_PATH", str(Path(_spec.origin).parent / "Qt5" / "plugins"))

from PyQt5.QtCore import QEventLoop, Qt, QTimer           # noqa: E402
from PyQt5.QtGui import QColor, QFontDatabase, QPalette    # noqa: E402
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QGroupBox, QLabel,       # noqa: E402
                             QLineEdit, QListWidget, QMenuBar, QPushButton, QRadioButton,
                             QStatusBar, QTabBar, QTableWidget, QToolButton, QTreeWidget,
                             QWidget)

_FONTS = (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
          r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\segoeui.ttf")


def pump(ms):
    loop = QEventLoop()
    QTimer.singleShot(int(ms), loop.quit)
    loop.exec_()


def load_fonts():
    n = 0
    for fp in _FONTS:
        if os.path.exists(fp) and QFontDatabase.addApplicationFont(fp) != -1:
            n += 1
    return n


def extract_base_qss():
    """从 main.py 取 app.setStyleSheet(<长字符串>) 的那段。"""
    src = (ROOT / "main.py").read_text(encoding="utf-8-sig")
    best = ""
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setStyleSheet" and node.args):
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and len(a.value) > len(best):
                best = a.value
    return best


def lum(c: QColor) -> float:
    def f(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c.red()) + 0.7152 * f(c.green()) + 0.0722 * f(c.blue())


def contrast(a: QColor, b: QColor) -> float:
    l1, l2 = lum(a), lum(b)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)


#: 控件 -> (取前景的 palette 角色, 取背景的角色)
ROLE_MAP = [
    ((QTableWidget, QListWidget, QTreeWidget), (QPalette.Text, QPalette.Base)),
    ((QPushButton, QToolButton), (QPalette.ButtonText, QPalette.Button)),
    ((QLineEdit, QComboBox), (QPalette.Text, QPalette.Base)),
]


def roles_for(w):
    for types, roles in ROLE_MAP:
        if isinstance(w, types):
            return roles
    return (QPalette.WindowText, QPalette.Window)


def audit(win, tag):
    print(f"\n{'=' * 92}\n对比度审计 [{tag}]   （WCAG：正文 >=4.5 合格，<3.0 严重）\n{'=' * 92}")
    seen = {}
    bad = []
    wtype = (QLabel, QPushButton, QToolButton, QCheckBox, QRadioButton, QGroupBox,
             QTableWidget, QListWidget, QTreeWidget, QLineEdit, QComboBox, QTabBar, QMenuBar,
             QStatusBar)
    for w in win.findChildren(QWidget):
        if not isinstance(w, wtype) or not w.isVisible():
            continue
        try:
            fg_role, bg_role = roles_for(w)
            pal = w.palette()
            fg = pal.color(fg_role)
            bg = pal.color(bg_role)
            if not fg.isValid() or not bg.isValid():
                continue
            r = contrast(fg, bg)
        except Exception:
            continue
        cls = type(w).__name__
        key = (cls, fg.name(), bg.name())
        if key in seen:
            seen[key][0] += 1
            continue
        seen[key] = [1, w, r, fg, bg]
    rows = sorted(seen.values(), key=lambda x: x[2])
    for cnt, w, r, fg, bg in rows:
        mark = "❌严重" if r < 3.0 else ("⚠️偏低" if r < 4.5 else "  ok ")
        txt = ""
        try:
            if isinstance(w, (QLabel, QPushButton, QToolButton, QCheckBox, QRadioButton)):
                txt = (w.text() or "")[:22].replace("\n", "⏎")
        except Exception:
            pass
        oid = ""
        try:
            oid = w.objectName() or ""
        except Exception:
            pass
        line = (f"  {mark} {r:5.2f}  {type(w).__name__:<14} fg={fg.name()} bg={bg.name()}"
                f"  x{cnt:<3} {('#'+oid) if oid else '':<18} {txt}")
        print(line)
        if r < 4.5:
            bad.append(line)
    print(f"  ---- 低于 4.5 的控件种类：{len(bad)} / {len(rows)}")
    return bad


def seed_image_list(win, n=6):
    """用产品路径 _render_page() 给左栏图片列表灌合成数据（空工程测不到着色逻辑）。"""
    mk = lambda i: {
        "path": f"D:/__vr__/fake_{i}_te_0.jpg",
        "fname": f"17_31_凹点_{i}_te_0.jpg" if i % 2 else f"20260810图片-3号机-大面-划痕{i}_te_0.jpg",
        "view_fnames": None, "height_fname": None,
        "img_w": 1145, "img_h": 3670, "split_status": "[train]",
        "shape_labels": ["划痕"] if i % 2 else [], "shape_count": 3 if i % 2 else 0,
        "cls_text": "OK" if i % 2 else "", "ocr_text": "", "pred_cnt": 0,
        "ocr_score": "", "score_str": f"0.9{i}", "ocr_time": "", "infer_ms": 10 + i,
    }
    data = [mk(i) for i in range(n)]
    win._all_images = list(data)
    win._filtered_data = list(data)
    try:
        win._render_page()
        print(f"  [seed] 图片列表灌入 {win.image_list_widget.rowCount()} 行")
    except Exception as e:
        print(f"  [seed] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()


def build(theme, base):
    from ui.themes import LIGHT_OVERLAY, DARK_INDICATOR, set_base_qss, sweep_inline_styles
    set_base_qss(base)
    app = QApplication.instance()
    app.setStyleSheet(base + (LIGHT_OVERLAY if theme == "light" else DARK_INDICATOR))
    from ui.main_window import MainWindow
    win = MainWindow()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(1680, 1000)
    win.show()
    pump(1200)
    sweep_inline_styles(app, restore=False)     # 同 main.py 的 singleShot(0) 清扫
    pump(400)
    return win


def main():
    OUT.mkdir(exist_ok=True)
    app = QApplication(sys.argv[:1])
    nf = load_fonts()
    print(f"字体 {nf} 个 / QPA={os.environ['QT_QPA_PLATFORM']}")
    base = extract_base_qss()
    print(f"基础 QSS 取出 {len(base)} 字符")
    if len(base) < 500:
        print("!! 没取到基础 QSS，后续结果无意义")
        return 1

    for theme in ("dark", "light"):
        print(f"\n\n########## 主题 = {theme} ##########")
        win = build(theme, base)
        seed_image_list(win)
        pump(300)
        win.grab().save(str(OUT / f"theme_{theme}_full.png"))
        st = getattr(win, "_right_stack", None)
        if st is not None:
            win.show_right_tab("params")
            pump(900)
            st.grab().save(str(OUT / f"theme_{theme}_params.png"))
        audit(win, theme)
        win.close()
        win.deleteLater()
        pump(200)
    print(f"\n已输出到 {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
