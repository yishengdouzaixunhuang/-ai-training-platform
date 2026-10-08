# -*- coding: utf-8 -*-
"""界面主题：**深色（默认）/ 浅色**，以及勾选框指示器的可见性修复。

为什么要单独一份：原来全局 QSS 直接写在 main.py 里，而且**没有**给
`QCheckBox::indicator` / `QRadioButton::indicator` 写样式 —— Fusion 风格下指示器
用系统调色板绘制，在深色背景上基本看不见（用户反馈："我都看不到装√的小方框"）。
另外硬编码的深色也不允许用户换主题。

做法：
  · 基础表 = main.py 里那套赛博朋克 QSS（首次调用时抓下来存住，不重复抄一遍）；
  · 深色 = 基础表 + 指示器修复；
  · 浅色 = 基础表 + 浅色覆盖（覆盖主要表面的背景/文字/边框/表格/输入控件）。
QSS 后写的规则优先，所以"覆盖"这种方式对没写到的控件会保留深色 —— 已知副作用，
详见 README 的"主题"一节；以后新增控件尽量用主题变量而不是内联颜色。
"""
from __future__ import annotations

_BASE_QSS = None            # main.py 里那套原始深色表（只抓一次）

# ── 勾选框/单选框指示器：深色下必须自己画，否则看不见 ──
DARK_INDICATOR = """
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px; height: 14px;
    border: 1px solid #5B6B8C;
    background-color: #0E1420;
    border-radius: 3px;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #00E5FF; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #00E5FF;
    border: 1px solid #00E5FF;
    image: none;
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border-color: #2A3A5C; background-color: #10151F;
}
QRadioButton::indicator { border-radius: 7px; }
QCheckBox { spacing: 6px; }

/* 图片列表选中态（原来写死在 main_window 的内联样式里，浅色主题下整行看不见） */
QTableWidget#imageList::item:selected { background-color: #0E3A52; color: #00E5FF; }
QTableWidget#imageList::item:selected:!active { background-color: #0E3A52; color: #00E5FF; }
"""

# ── 浅色主题：覆盖主要表面 ──
LIGHT_OVERLAY = """
QWidget { background-color: #F4F6FA; color: #1A2233; }
QMainWindow, QDialog { background-color: #F4F6FA; }
QLabel { color: #1A2233; background: transparent; }
QLabel:disabled { color: #98A2B3; }

QPushButton {
    background-color: #FFFFFF; color: #1A2233;
    border: 1px solid #C9D3E0; border-radius: 4px; padding: 5px 12px;
}
QPushButton:hover { background-color: #EAF3FB; border-color: #0077B6; color: #00527A; }
QPushButton:pressed { background-color: #DCE9F5; }
QPushButton:checked { background-color: #D6E9F8; border-color: #0077B6; color: #00527A; }
QPushButton:disabled { color: #98A2B3; border-color: #DDE3EC; background-color: #F0F2F6; }

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
    background-color: #FFFFFF; color: #1A2233;
    border: 1px solid #C9D3E0; border-radius: 3px; selection-background-color: #BBDDF5;
}
QComboBox QAbstractItemView { background-color: #FFFFFF; color: #1A2233; selection-background-color: #D6E9F8; }

QTableWidget, QTableView, QListWidget, QTreeWidget, QTableCornerButton::section {
    background-color: #FFFFFF; alternate-background-color: #F7F9FC;
    color: #1A2233; gridline-color: #E3E8F0;
    selection-background-color: #D6E9F8; selection-color: #00527A;
}
QHeaderView::section {
    background-color: #E9EEF6; color: #24405E; border: none;
    border-right: 1px solid #D8E0EA; border-bottom: 1px solid #C9D3E0; padding: 5px 4px;
}

QMenuBar, QMenu { background-color: #FFFFFF; color: #1A2233; }
QMenuBar::item:selected, QMenu::item:selected { background-color: #D6E9F8; color: #00527A; }
QMenu::separator { height: 1px; background: #E3E8F0; margin: 4px 6px; }

QGroupBox { border: 1px solid #D8E0EA; border-radius: 4px; margin-top: 8px; }
QGroupBox::title { color: #24405E; subcontrol-origin: margin; left: 8px; padding: 0 3px; }

QSplitter::handle { background-color: #E3E8F0; }
QScrollBar:vertical, QScrollBar:horizontal { background: #F0F2F6; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #C2CCDA; border-radius: 4px; }
QScrollBar::handle:hover { background: #9FB0C6;		 }
QScrollBar::add-line, QScrollBar::sub-line { background: none; height: 0; width: 0; }
QProgressBar { background-color: #E9EEF6; border: 1px solid #D8E0EA; border-radius: 3px; text-align: center; color: #24405E; }
QProgressBar::chunk { background-color: #4FA3D1; border-radius: 2px; }

QCheckBox::indicator, QRadioButton::indicator {
    width: 14px; height: 14px;
    border: 1px solid #8FA0B8; background-color: #FFFFFF; border-radius: 3px;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #0077B6; border: 1px solid #0077B6;
}
QRadioButton::indicator { border-radius: 7px; }
QCheckBox { spacing: 6px; }

QToolTip { background-color: #FFFFFF; color: #1A2233; border: 1px solid #C9D3E0; }
QStatusBar { background-color: #E9EEF6; color: #1A2233; }

/* 图片列表选中态：浅色主题必须用「浅底深字」——
   否则按类别色着色过的单元格（深色字）压在深底上就整行看不见。
   ⚠️ 这里不能只改 background：内联 setForeground 的单元格**不受 QSS color 影响**，
   所以浅色主题必须给出足够的底/字对比，让深色类别字也能读。 */
QTableWidget#imageList::item:selected { background-color: #CFE6FA; color: #0B2A44; }
QTableWidget#imageList::item:selected:!active { background-color: #CFE6FA; color: #0B2A44; }

/* 工程列表(QTreeWidget) 与各类列表的 hover/选中态：
   浅色主题下原来沿用了深色的 hover 底（深蓝），文字还是深色 -> 悬浮上去整行看不见
   （用户实测反馈）。这里把 hover/选中/非激活选中三种状态都显式给成"浅底深字"。 */
QTreeWidget::item:hover, QListWidget::item:hover {
    background-color: #E3EEF9; color: #0B2A44;
}
QTreeWidget::item:selected, QListWidget::item:selected,
QTreeWidget::item:selected:!active, QListWidget::item:selected:!active {
    background-color: #CFE6FA; color: #0B2A44;
}
QTableWidget::item:hover, QTableView::item:hover { background-color: #EDF4FB; }
QTableWidget::item:selected, QTableView::item:selected,
QTableWidget::item:selected:!active, QTableView::item:selected:!active {
    background-color: #CFE6FA; color: #0B2A44;
}
QMenu::item:selected { background-color: #D6E9F8; color: #00527A; }
QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected {
    background-color: #D6E9F8; color: #00527A;
}
"""

THEME_LABELS = [("深色（默认）", "dark"), ("浅色", "light")]

# ── 「主题色」白名单 ──
# 内联样式里凡是**只用这些颜色**的，都属于"主题皮肤色"，切主题时应当让位给主题表；
# 不在名单里的（红/绿/橙等）是**语义色**，一律保留不动。
#   #0A0E17 #0D1220 #0C1220 #101828 #0F1728 #131B2E #0E1420 #10151F  底色/面色
#   #22304A #2A3A5C #1A2440 #1C2740 #D8E0EA                             边框/网格
#   #E6F0FF #D6E4FF #C8D2E0 #8A93A6 #55607A #24405E                    文字
#   #00E5FF #0E3A52                                                     强调色/选中底
import re as _re

THEME_COLORS = {
    "#0a0e17", "#0d1220", "#0c1220", "#101828", "#0f1728", "#131b2e",
    "#0e1420", "#10151f", "#0e1830", "#1b2740",
    "#22304a", "#2a3a5c", "#1a2440", "#1c2740", "#d8e0ea", "#c9d3e0",
    "#e6f0ff", "#d6e4ff", "#c8d2e0", "#8a93a6", "#55607a", "#24405e",
    "#00e5ff", "#0e3a52", "#282828",
}
_HEX_RE = _re.compile(r"#[0-9a-fA-F]{6}")
_SWEEP_PROP = "_dsh_swept_qss"


def _is_theme_sheet(qss):
    """这段内联样式是否"全是主题皮肤色"（是 -> 切主题时可以让位）。"""
    cols = {c.lower() for c in _HEX_RE.findall(qss or "")}
    if not cols:
        return False
    return cols.issubset(THEME_COLORS)


def sweep_inline_styles(app, restore=False):
    """把"只用了主题皮肤色"的内联样式暂存并清掉，让主题表接管配色。

    为什么要这一步：Qt 里**控件自己的 setStyleSheet 优先级高于应用级主题表**，
    平台里有几十处 `setStyleSheet("color:#8A93A6")` 之类写死的主题色 ——
    不清掉它们，切浅色主题时这些字仍是深色主题的灰/青，永远变不成黑字。

    规则（保守，宁可不改也不改错）：
      · 只清"颜色**全部**落在 THEME_COLORS 白名单里"的；
      · 只要出现一个语义色（红/绿/橙…）就整段保留 —— 那些是"用颜色表达含义"的，
        比如绿色的 Start Training、红色的错误提示，误清会丢信息。
      · 切回深色时按属性还原，原样不差。
    """
    for w in app.allWidgets():
        try:
            cur = w.styleSheet()
        except Exception:
            continue
        if restore:
            saved = w.property(_SWEEP_PROP)
            if saved:
                w.setStyleSheet(saved)
                w.setProperty(_SWEEP_PROP, None)
            continue
        if not cur or w.property(_SWEEP_PROP):
            continue
        if _is_theme_sheet(cur):
            w.setProperty(_SWEEP_PROP, cur)
            w.setStyleSheet("")
            print(f"[theme] swept inline qss on {type(w).__name__}: {cur[:60]!r}", flush=True)


def set_base_qss(qss):
    """把 main.py 里那套基础表存起来（只存第一次）。"""
    global _BASE_QSS
    if _BASE_QSS is None:
        _BASE_QSS = qss or ""


def base_qss():
    return _BASE_QSS or ""


def apply_theme(app, name="dark"):
    """把主题应用到整个 QApplication（app 为 None 时用当前实例）。"""
    from PyQt5.QtWidgets import QApplication
    if app is None:
        app = QApplication.instance()
    if app is None:
        return
    if not base_qss():                       # main.py 没提前登记 -> 用当前表当基础
        set_base_qss(app.styleSheet())
    if name == "light":
        # 先还原上一次清扫（否则深色主题暂存的样式会丢），再按新主题清扫一遍
        sweep_inline_styles(app, restore=True)
        app.setStyleSheet(base_qss() + LIGHT_OVERLAY)
        sweep_inline_styles(app, restore=False)
    else:
        sweep_inline_styles(app, restore=True)
        app.setStyleSheet(base_qss() + DARK_INDICATOR)
    # 记住选择：直接写 .ui_settings.json（不依赖主窗口的 _save_ui_setting 落在哪个文件）
    try:
        import json as _json, os as _os
        p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                          ".ui_settings.json")
        d = {}
        if _os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                d = _json.load(f) or {}
        d["theme"] = name
        with open(p, "w", encoding="utf-8") as f:
            _json.dump(d, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    try:
        app.setProperty("dsh_theme", name)
    except Exception:
        pass
