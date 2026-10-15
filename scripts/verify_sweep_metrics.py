# -*- coding: utf-8 -*-
"""「主题清扫」回归测试：确认清掉的是**颜色**，不是**度量**。

背景（维护者 实测报的「显示不全」）：
  左栏 `Image List (99)` 按钮在浅色主题下**文字上下被切掉**。
  根因不是布局，是 `ui.themes.sweep_inline_styles` 旧实现把整段内联样式
  `setStyleSheet("")` 清空 —— 于是 `font-size:10px; padding:1px 6px` 跟着颜色一起没了，
  控件回落到应用级 `QPushButton { padding:5px 12px }` + 默认字号：
  按钮 111x22，字体要 19px，扣掉 padding 内容区只剩 10px -> 字形底部被削平。
  （深色主题同样中招：main.py 的清扫是**不分主题**跑的。）

现在 `strip_colors()` 只剥颜色、留度量。本脚本守住这条：
  1. `strip_colors` 单元行为（含**没有选择器的裸声明**，`_make_block_header` 就是那种写法）；
  2. 真实窗口里**每一个**被清扫的控件：新表不得残留颜色、原有度量不得丢；
  3. 用户报的那个按钮：`sizeHint().height() <= height()`（否则文字必被切）；
  4. 浅色主题下被清扫的文字控件不得解析成深色主题的亮青（白底上不可读）。

用法：
    C:/aitp_venv/Scripts/python.exe scripts/verify_sweep_metrics.py
    C:/aitp_venv/Scripts/python.exe scripts/verify_sweep_metrics.py --render   # 额外出图
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_ui_render" / "sweepchk"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.chdir(ROOT)
import torch  # noqa: F401,E402

sys.path.insert(0, str(ROOT))
_spec = importlib.util.find_spec("PyQt5")
if _spec is not None and _spec.origin:
    os.environ.setdefault("QT_PLUGIN_PATH", str(Path(_spec.origin).parent / "Qt5" / "plugins"))

from PyQt5.QtCore import QEventLoop, Qt, QTimer            # noqa: E402
from PyQt5.QtGui import QColor, QFontDatabase, QPalette    # noqa: E402
from PyQt5.QtWidgets import QApplication                   # noqa: E402

PASS, FAIL = [], []


def check(ok, title, detail=""):
    (PASS if ok else FAIL).append(title)
    print(f"  {'PASS' if ok else 'FAIL'}  {title}" + (f"   {detail}" if detail else ""))
    return ok


def pump(ms=300):
    loop = QEventLoop()
    QTimer.singleShot(int(ms), loop.quit)
    loop.exec_()


def load_fonts():
    n = 0
    for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
              r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        if os.path.exists(c) and QFontDatabase.addApplicationFont(c) != -1:
            n += 1
    return n


def load_themes():
    """直接从文件加载 ui/themes.py —— 模块级没有 PyQt5 依赖，不用起 UI。"""
    s = importlib.util.spec_from_file_location("dsh_themes", ROOT / "ui" / "themes.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def extract_base_qss():
    src = (ROOT / "main.py").read_text(encoding="utf-8-sig")
    best = ""
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setStyleSheet" and node.args):
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and len(a.value) > len(best):
                best = a.value
    return best


#: 「度量」属性 —— 清扫后必须还在
METRIC_RE = re.compile(
    r"(?<!-)\b(font-size|font-family|font-weight|padding|margin|"
    r"min-width|max-width|min-height|max-height|border-radius|"
    r"border-width|border-style|border-top-width|border-top-style|"
    r"border-right-width|border-right-style|border-bottom-width|border-bottom-style|"
    r"border-left-width|border-left-style|text-align|spacing)\s*:", re.I)
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}")


def metrics_of(qss):
    return {m.group(1).lower() for m in METRIC_RE.finditer(qss or "")}


# ─────────────────────── 1. strip_colors 单元行为 ───────────────────────

UNIT_CASES = [
    # (说明, 原样式, 期望保留的度量, 期望是否被剥色, 必须原样保留的子串)
    ("无选择器的裸声明（_make_block_header 的写法）",
     "color:#00E5FF; font-size:10px; font-weight:bold; padding:0 2px;",
     {"font-size", "font-weight", "padding"}, True, []),
    ("带选择器 + border 简写（有颜色 -> 拆成长属性）",
     "QPushButton { font-size: 10px; padding: 1px 6px; border: 1px solid #2A3A5C;"
     " border-radius: 2px; color: #D6E4FF; }",
     {"font-size", "padding", "border-width", "border-style", "border-radius"}, True, []),
    ("带强调条 border-left + text-align",
     "QPushButton { text-align:left; font-size:11px; font-weight:bold; color:#00E5FF;"
     " border:none; border-left:3px solid #00E5FF; padding:3px 6px; }",
     {"text-align", "font-size", "font-weight",
      "border-left-width", "border-left-style", "padding"}, True,
     ["border: none", "border-left-width: 3px", "border-left-style: solid"]),
    ("透明边框是布局占位（无颜色 -> 整条原样留下）",
     "QPushButton { font-size: 12px; padding: 2px 0;"
     " border: 1px solid transparent; color: #D6E4FF; }",
     {"font-size", "padding"}, True, ["border: 1px solid transparent"]),
    ("整张表只有颜色 -> 剥完为空",
     "color:#D6E4FF;",
     set(), True, []),
]


def test_unit(th):
    print("\n[1] strip_colors 单元行为")
    for desc, raw, want_metrics, want_changed, must_keep in UNIT_CASES:
        thin, changed = th.strip_colors(raw)
        got = metrics_of(thin)
        missing = want_metrics - got
        gone = [s for s in must_keep if s not in thin]
        check(changed == want_changed and not missing and not gone, desc,
              f"-> {thin!r}"
              + (f"  丢了 {sorted(missing)}" if missing else "")
              + (f"  被改坏 {gone}" if gone else ""))
    # 语义色不进清扫通路
    sem = "QLabel { color: #FF2A6D; font-size: 12px; }"
    check(not th._is_theme_sheet(sem), "语义色（红）整段保留，不进清扫通路")
    # 幂等：剥过一遍的表再剥应无变化
    thin, _ = th.strip_colors(UNIT_CASES[1][1])
    _, changed2 = th.strip_colors(thin)
    check(not changed2 and not HEX_RE.search(thin), "剥色后无残留颜色，且再剥一次不再变化")


# ─────────────────────── 2-4. 真实窗口 ───────────────────────

def build(theme, base):
    from ui.themes import LIGHT_OVERLAY, DARK_INDICATOR, set_base_qss
    set_base_qss(base)
    app = QApplication.instance()
    app.setStyleSheet(base + (LIGHT_OVERLAY if theme == "light" else DARK_INDICATOR))
    from ui.main_window import MainWindow
    win = MainWindow()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(1680, 1000)
    win.show()
    pump(1200)
    from ui.themes import sweep_inline_styles
    sweep_inline_styles(app, restore=False)     # 同 main.py singleShot(0) 那一步
    pump(400)
    return win


def contrast(a: QColor, b: QColor) -> float:
    def f(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    la = 0.2126 * f(a.red()) + 0.7152 * f(a.green()) + 0.0722 * f(a.blue())
    lb = 0.2126 * f(b.red()) + 0.7152 * f(b.green()) + 0.0722 * f(b.blue())
    if la < lb:
        la, lb = lb, la
    return (la + 0.05) / (lb + 0.05)


def test_window(theme, base, render):
    print(f"\n[{'2' if theme == 'dark' else '3'}] 真实窗口（{theme}）：被清扫控件逐个核验")
    app = QApplication.instance()
    win = build(theme, base)

    swept = [w for w in app.allWidgets() if w.property("_dsh_swept_qss")]
    check(len(swept) >= 20, f"{theme}：确有控件被清扫（{len(swept)} 个）")

    bad_color, bad_metric = [], []
    for w in swept:
        saved = w.property("_dsh_swept_qss")
        cur = w.styleSheet()
        if HEX_RE.search(cur):
            bad_color.append((type(w).__name__, cur[:50]))
        lost = metrics_of(saved) - metrics_of(cur)
        # border-left 简写会被拆成 border-left-width/style，属正常
        lost = {p for p in lost if not p.startswith("border")}
        if lost:
            bad_metric.append((type(w).__name__, sorted(lost), saved[:50]))
    check(not bad_color, f"{theme}：清扫后不残留颜色", str(bad_color[:2]))
    check(not bad_metric, f"{theme}：清扫后不丢度量", str(bad_metric[:2]))

    # 用户报的那个按钮
    btn = win._accordion_btns[0]
    btn.setText("Image List (99)")
    pump(300)
    fm_h = btn.fontMetrics().height()
    check(btn.sizeHint().height() <= btn.height(),
          f"{theme}：Image List 按钮放得下（要 {btn.sizeHint().height()}px / 有 {btn.height()}px）",
          f"字号 {btn.font().pixelSize()}px 行高 {fm_h}")

    # 浅色主题下，被清扫的文字控件不能解析成深色主题的亮青
    if theme == "light":
        bg = QColor("#F4F6FA")
        low, skipped = [], 0
        for w in swept:
            if not w.inherits("QAbstractButton") and type(w).__name__ != "QLabel":
                continue
            if not w.isEnabled():
                skipped += 1            # 禁用态本来就是低对比（WCAG 豁免），主题表有意为之
                continue
            role = QPalette.ButtonText if w.inherits("QAbstractButton") else QPalette.WindowText
            fg = w.palette().color(role)
            r = contrast(fg, bg)
            if r < 4.5:
                low.append((type(w).__name__, fg.name(), round(r, 2)))
        check(not low, "浅色：被清扫的文字控件在浅底上都够清晰（对比度 >= 4.5）",
              f"（跳过 {skipped} 个禁用控件）" + str(low[:3]))

    if render:
        OUT.mkdir(parents=True, exist_ok=True)
        win.grab().save(str(OUT / f"window_{theme}.png"))
        btn.grab().save(str(OUT / f"hdrbtn_{theme}.png"))
    win.close()
    win.deleteLater()
    pump(200)


def main():
    render = "--render" in sys.argv
    th = load_themes()
    base = extract_base_qss()
    if len(base) < 500:
        print("!! 没从 main.py 取到基础 QSS，结果无意义")
        return 2
    app = QApplication(sys.argv[:1])
    print(f"字体 {load_fonts()} 个 / QPA={os.environ['QT_QPA_PLATFORM']} / 基础表 {len(base)} 字符")

    test_unit(th)
    for theme in ("dark", "light"):
        test_window(theme, base, render)

    print("\n" + "=" * 78)
    print(f"PASS {len(PASS)}   FAIL {len(FAIL)}")
    if FAIL:
        for t in FAIL:
            print("  FAIL:", t)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
