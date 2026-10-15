# -*- coding: utf-8 -*-
"""render_left_panel.py —— DL Studio **左栏**重构后的离屏渲染核验

背景（2026-10-14 晚）：左栏从"工程树 + 图片列表上下叠放"改成
"**竖排页签两页**（工程管理 / 图像管理）"，点页签显示、再点收起。

这个脚本要回答三个只有**看图 + 量尺寸**才能回答的问题：
  ① 竖排页签真的画出来了吗（不是两条空按钮）？
  ② 切到"工程管理"时，工程树是不是拿到了**整栏高度**（不再被 150px 钉死）？
  ③ 两个页签都弹起来时，内容真的收起了、而**竖条还在**（不然点不回来）？

用法（**必须**用 aitp_venv 的解释器；torch 在沙箱内会 WinError 1114，需关沙箱）：
    C:/aitp_venv/Scripts/python.exe scripts/render_left_panel.py

输出：`_ui_render/left_<state>.png`（左栏特写）+ `left_<state>_full.png`（整窗），
      并把几何数字打到 stdout。`*.png` 已在 .gitignore 里。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_ui_render"

# offscreen 平台没有字体库 —— 不手动挂字体，抓出来是**空白色块、一个字都没有**
# （见 skill dlstudio-ui-verify 第 2 条）。不要改用 windows 平台：它会真把窗口映射到桌面。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ---- 1) 先 torch、再碰 Qt（严格照 main.py 的顺序）----
# PyQt5 若先加载 Qt DLL，随后的 import torch 必然 WinError 1114 c10.dll。
# 取 Qt 插件路径用 find_spec（只查 spec、不执行 __init__），不会把 Qt DLL 拉进来。
os.chdir(ROOT)
import torch  # noqa: F401,E402

sys.path.insert(0, str(ROOT))

_spec = importlib.util.find_spec("PyQt5")
if _spec is not None and _spec.origin:
    os.environ.setdefault(
        "QT_PLUGIN_PATH", str(Path(_spec.origin).parent / "Qt5" / "plugins")
    )

from PyQt5.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

_FONT_FILES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
)

#: 三种状态：(名字, 要按下的页签列表)
#:   [] = 两个都不按 ⇒ 内容收起（但竖条必须还在）
STATES = [
    ("img", ["img"]),
    ("proj", ["proj"]),
    ("collapsed", []),
]


def pump(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(ms), loop.quit)
    loop.exec_()


def _load_system_fonts() -> int:
    from PyQt5.QtGui import QFontDatabase
    n = 0
    for fp in _FONT_FILES:
        if os.path.exists(fp) and QFontDatabase.addApplicationFont(fp) != -1:
            n += 1
    return n


def _g(w):
    """几何 -> 可读串（已销毁的控件要能看出来，不能假装是 0）。"""
    if w is None:
        return "-"
    try:
        return "%dx%d @%d,%d" % (w.width(), w.height(), w.x(), w.y())
    except RuntimeError:
        return "已销毁"


def apply_state(win, keys, app):
    """把左栏摆成指定状态：先全弹起，再按需要按下一个。"""
    for k, b in win._left_tab_btns.items():
        b.setChecked(False)
    win._left_stack.setVisible(False)
    app.processEvents()
    for k in keys:
        win._left_tab_btns[k].click()      # 走**真实点击**路径，不是直接设状态
    app.processEvents()
    pump(400)


def report(win, tag):
    ls = win._left_stack
    st = win.left_tab_state()
    print("-" * 78)
    print("[%s]  按下的页签=%s  内容可见=%s  当前页索引=%s"
          % (tag, st["checked"] or "（无）", st["content_visible"], st["page"]))
    print("  左栏整体        : %s  (sizeHint %dx%d)"
          % (_g(win._left_panel), win._left_panel.sizeHint().width(),
             win._left_panel.sizeHint().height()))
    for k, b in win._left_tab_btns.items():
        print("  页签 %-5s     : %s  checked=%-5s sizeHint=%dx%d"
              % (k, _g(b), b.isChecked(), b.sizeHint().width(), b.sizeHint().height()))
    print("  页栈 _left_stack: %s  页数=%d  当前=%d"
          % (_g(ls), ls.count(), ls.currentIndex()))
    print("  工程树          : %s  (maxHeight=%d)"
          % (_g(win.project_tree), win.project_tree.maximumHeight()))
    pg = win._list_page
    print("  图片页          : %s  visible=%s" % (_g(pg), pg.isVisible()))
    print("  图片表          : %s  rows=%d  (maxHeight=%d)"
          % (_g(win.image_list_widget), win.image_list_widget.rowCount(),
             win.image_list_widget.maximumHeight()))
    print("  _accordion_stack: %s  页数=%d  当前=%d"
          % (_g(win._accordion_stack), win._accordion_stack.count(),
             win._accordion_stack.currentIndex()))


def main(argv):
    OUT.mkdir(exist_ok=True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv[:1])

    nf = _load_system_fonts()
    print("系统字体挂载 %d 个%s" % (nf, "" if nf else "  ⚠️ 一个都没挂上 → 截图会是空白块！"))

    from PyQt5.QtCore import QT_VERSION_STR
    print("Qt %s / QPA=%s / torch %s" % (QT_VERSION_STR, os.environ["QT_QPA_PLATFORM"], torch.__version__))

    from ui.main_window import MainWindow
    from ui.themes import apply_theme, sweep_inline_styles, set_base_qss, base_qss

    # 走产品路径的主题（含 sweep）：不然"内联样式被剥色后长什么样"看不出真实效果
    try:
        from probe_mark_ui import _load_product_base_qss
        _b = _load_product_base_qss()
        if _b and not base_qss():
            set_base_qss(_b)
    except Exception as e:
        print("  基础表加载失败(不影响布局): %s" % e)
    apply_theme(app, "dark")

    win = MainWindow()
    win.resize(1680, 1000)
    win.show()
    sweep_inline_styles(app, restore=False)
    pump(1200)

    # 左栏加宽到和右栏一样，方便看细节；同时保证"工程管理页"里的长工程名不挤
    try:
        win._main_splitter.setSizes([300, 1000, 380])
        pump(500)
    except Exception as e:
        print("  splitter 设置失败: %s" % e)

    for name, keys in STATES:
        apply_state(win, keys, app)
        report(win, name)
        win._left_panel.grab().save(str(OUT / ("left_%s.png" % name)))
        win.grab().save(str(OUT / ("left_%s_full.png" % name)))

    # 收尾：把状态还原成默认（图像管理）
    apply_state(win, ["img"], app)
    print("-" * 78)
    print("已输出到 %s" % OUT)
    win.close()
    pump(100)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
