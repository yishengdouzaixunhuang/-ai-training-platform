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


def _res_uri(name):
    """`resources/` 下资源的绝对路径（QSS 的 url() 用正斜杠，Windows 反斜杠会解析失败）。"""
    import os as _os
    p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "resources", name)
    return p.replace("\\", "/")


# ── 勾选框：勾上显示 ✓（而不是一坨实心方块）──────────────────────────
# 2026-10-22 用户要求："把这些蓝色小方块换成√"。
# 为什么用 SVG 图片而不是 QSS 画：QSS **画不了文字/图形**，`::indicator` 只有
# background / border / image 三种手段。所以给两种主题各配一张 14×14 的勾图，
# 底色沿用原来的高饱和实心块（青 / 蓝），只把"里面是空的"改成"里面有个勾"。
#
# ⚠️ 这两段必须写在 DARK_INDICATOR / LIGHT_OVERLAY 的**后面**：
#    同优先级的选择器，后出现的生效；前面那句 `image: none` 就是靠这个被覆盖的。
# ⚠️ 颜色不要往这里写死主题皮肤色 —— 勾的描边色是要表达"已勾选"的语义色，
#    不属于主题皮肤，`sweep_inline_styles` 也不会碰应用级样式表（它只扫控件级）。
DARK_CHECK = """
QCheckBox::indicator:checked { image: url("%s"); }
QCheckBox::indicator:checked:disabled { image: url("%s"); }
""" % (_res_uri("check_dark.svg"), _res_uri("check_dark.svg"))

LIGHT_CHECK = """
QCheckBox::indicator:checked { image: url("%s"); }
QCheckBox::indicator:checked:disabled { image: url("%s"); }
""" % (_res_uri("check_light.svg"), _res_uri("check_light.svg"))

# ── 滑块（QSlider）：**必须写在主题表里，不能写在控件的内联样式里** ──
# 2026-10-14 踩过，代价是"一个像素都不画"：
# 底栏缩放滑块的内联 QSS 用了 #2A3A5C / #0E3A52 / #00E5FF 三个色，**全在
# THEME_COLORS 白名单里** ⇒ `sweep_inline_styles` 判定"这段只用了主题皮肤色"，
# 把颜色声明整段剥掉（度量留下）。槽和手柄于是**没有背景色**，而主题表当时
# 没有 QSlider 规则可兜底 ⇒ 滑块占着 84px 宽，画出来是一片空槽。
# 功能断言（value / geometry / valueChanged / zoom_level）**全绿**，肉眼全空 ——
# 只有把控件 `grab()` 出来数像素才看得见。
# 结论：内联样式只写度量；颜色写进这里（主题表的规则 sweep 不会碰）。
_DARK_SLIDER = """
QSlider::groove:horizontal { height: 4px; background: #2A3A5C; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #0E3A52; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 9px; margin: -5px 0; border-radius: 5px; background: #00E5FF;
}
QSlider::handle:horizontal:hover { background: #E6F0FF; }
"""

# ── 工具栏里的三个"文字型"小按钮：#toolsToggle / #layerSw / #classPick ──
# 同一条规矩的第二个受害者（2026-10-15 记录）：这几个按钮的内联 QSS 颜色
# （#2A3A5C/#8A93A6/#00E5FF/#0E3A52/#E6F0FF/#C8D2E0）**全在白名单里** ⇒
# `sweep_inline_styles` 剥掉颜色、只留度量 ⇒ 它们退回全局通用 `QToolButton` 规则
# （#D6E4FF 亮字 + 透明底）。不会"空槽"，但**变成了另一种配色**（比作者意图更刺眼、
# 未勾选态看不出是开关）。所以配色在这里落一份，主题表的规则 sweep 不会碰。
# 教训固化成一句话：**新增任何自绘/文字型子控件，都要问"主题表里有它吗"**。
_DARK_TOOLBAR_SW = """
QToolButton#toolsToggle {
    border: 1px solid #2A3A5C; border-radius: 4px;
    padding: 2px 8px; margin: 0px;
    color: #C8D2E0; font-size: 12px;
}
QToolButton#toolsToggle:hover { border-color: #00E5FF; color: #E6F0FF; }
QToolButton#toolsToggle:checked {
    background: #0E3A52; border-color: #00E5FF; color: #00E5FF;
}
QToolButton#layerSw, QToolButton#classPick {
    border: 1px solid #2A3A5C; border-radius: 4px;
    padding: 1px 5px; margin: 0px;
    color: #8A93A6; font-size: 12px;
}
QToolButton#layerSw:hover, QToolButton#classPick:hover { border-color: #00E5FF; }
QToolButton#layerSw:checked {
    background: #0E3A52; border-color: #00E5FF; color: #E6F0FF;
}
QToolButton#classPick { color: #C8D2E0; }

# ── 三个图层开关（2026-10-15 照友商改成真复选框：勾上=实心青块，没勾=空框+灰字）──
# 勾选态用**实心方块**而不是打勾：对 12px 的小方块，实心色块的"开/关"对比度
# 比一个 1px 的对勾高得多，远看也分得出（友商截图就是这么做的）。
QLabel#layerGroupLabel { color: #8A93A6; font-size: 12px; padding-right: 2px; }
QCheckBox#layerSw { color: #8A93A6; spacing: 4px; }
QCheckBox#layerSw:hover { color: #C8D2E0; }
QCheckBox#layerSw:checked { color: #E6F0FF; }
QCheckBox#layerSw::indicator {
    width: 12px; height: 12px;
    border: 1px solid #2A3A5C; border-radius: 2px;
    background: #0E1420;
}
QCheckBox#layerSw::indicator:hover { border-color: #00E5FF; }
QCheckBox#layerSw::indicator:checked {
    background: #00E5FF; border: 1px solid #00E5FF;
}
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

/* 页签（crop_tool / resize_tool / heatmap_analyzer 三个工具窗在用）：
   base 表用的是 `QTabBar::tab{bg:#0C1220; color:#8A93A6}` —— 伪状态选择器，
   `QWidget` 通配**盖不住它**（specificity 规则不同），不显式覆盖就一直是深底浅灰。 */
QTabWidget::pane { border: 1px solid #D8E0EA; }
QTabBar::tab {
    background: #E9EEF6; color: #24405E;
    border: 1px solid #D8E0EA; padding: 6px 14px;
}
QTabBar::tab:selected { background: #FFFFFF; color: #00527A; }

QToolTip { background-color: #FFFFFF; color: #1A2233; border: 1px solid #C9D3E0; }
QStatusBar { background-color: #E9EEF6; color: #1A2233; }

/* 图片列表选中态：浅色主题必须用「浅底深字」——
   否则按类别色着色过的单元格（深色字）压在深底上就整行看不见。
   ⚠️ 这里不能只改 background：内联 setForeground 的单元格**不受 QSS color 影响**，
   所以浅色主题必须给出足够的底/字对比，让深色类别字也能读。 */
QTableWidget#imageList::item:selected { background-color: #CFE6FA; color: #0B2A44; }
QTableWidget#imageList::item:selected:!active { background-color: #CFE6FA; color: #0B2A44; }

/* 图片列表**默认行色**：浅色主题必须给深字（深色主题由 base 表给 #FFFFFF）。
   这条是 main_window._render_page 里 setForeground(白) 退役后，行色的唯一来源 —— 
   正因为它是 QSS，切换主题时才能**立即**生效、不必重刷列表。
   ⚠️ 类别色（_apply_class_color）是内联 setForeground，仍压过这条 —— 那是语义色，
   浅色主题下 CLASS_COLORS 本来就是深色（见 core/config.py），浅底上可读。 */
QTableWidget#imageList { color: #1A2233; }

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

# 浅色主题的滑块配色（与上面的 _DARK_SLIDER 成对；度量必须一致，
# 否则切主题时手柄会跳一下）。色取自本表已用的：边框 #C9D3E0、强调 #0077B6。
_LIGHT_SLIDER = """
QSlider::groove:horizontal { height: 4px; background: #C9D3E0; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #0077B6; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 9px; margin: -5px 0; border-radius: 5px; background: #0077B6;
}
QSlider::handle:horizontal:hover { background: #00527A; }
"""

#: 浅色主题下上面三个小按钮的配色（与 _DARK_TOOLBAR_SW 成对）。
_LIGHT_TOOLBAR_SW = """
QToolButton#toolsToggle {
    border: 1px solid #C9D3E0; border-radius: 4px;
    padding: 2px 8px; margin: 0px;
    color: #1A2233; font-size: 12px;
}
QToolButton#toolsToggle:hover { border-color: #0077B6; color: #00527A; }
QToolButton#toolsToggle:checked {
    background: #D6E9F8; border-color: #0077B6; color: #00527A;
}
QToolButton#layerSw, QToolButton#classPick {
    border: 1px solid #C9D3E0; border-radius: 4px;
    padding: 1px 5px; margin: 0px;
    color: #4A5568; font-size: 12px;
}
QToolButton#layerSw:hover, QToolButton#classPick:hover { border-color: #0077B6; }
QToolButton#layerSw:checked {
    background: #D6E9F8; border-color: #0077B6; color: #00527A;
}
QToolButton#classPick { color: #1A2233; }

QLabel#layerGroupLabel { color: #4A5568; font-size: 12px; padding-right: 2px; }
QCheckBox#layerSw { color: #4A5568; spacing: 4px; }
QCheckBox#layerSw:hover { color: #1A2233; }
QCheckBox#layerSw:checked { color: #1A2233; }
QCheckBox#layerSw::indicator {
    width: 12px; height: 12px;
    border: 1px solid #C9D3E0; border-radius: 2px;
    background: #FFFFFF;
}
QCheckBox#layerSw::indicator:hover { border-color: #0077B6; }
QCheckBox#layerSw::indicator:checked {
    background: #0077B6; border: 1px solid #0077B6;
}
"""


# ── 任务队列（左栏第三页，2026-10-16 对标友商同名模块）──
# 颜色一律写在这里：控件上的内联白名单色会被 sweep_inline_styles 剥掉（滑块"空槽"同型坑）。
_DARK_TASK_Q = """
QWidget#taskRow { border: 1px solid #2A3A5C; border-radius: 3px; }
QLabel#taskKind { color: #00E5FF; font-size: 11px; font-weight: bold; }
QLabel#taskTitle { color: #D6E4FF; font-size: 11px; }
QLabel#taskState { color: #8A93A6; font-size: 11px; }
QLabel#taskStatus { color: #8A93A6; font-size: 11px; }
QLabel#taskDetail { color: #6C7688; font-size: 10px; }
QLabel#taskEmpty { color: #6C7688; font-size: 11px; }
QProgressBar#taskProgress { background: #0E1420; border: none; border-radius: 3px; }
QProgressBar#taskProgress::chunk { background: #00E5FF; border-radius: 3px; }
QPushButton#taskCancelBtn { color: #8A93A6; font-size: 13px; padding: 0px; margin: 0px;
    border: 1px solid #2A3A5C; border-radius: 3px; background: transparent; }
QPushButton#taskCancelBtn:hover { color: #ff4444; border-color: #ff4444; }
QPushButton#taskClearBtn { font-size: 11px; padding: 1px 8px; }
"""
_LIGHT_TASK_Q = """
QWidget#taskRow { border: 1px solid #C9D3E0; border-radius: 3px; }
QLabel#taskKind { color: #0077B6; font-size: 11px; font-weight: bold; }
QLabel#taskTitle { color: #1A2233; font-size: 11px; }
QLabel#taskState { color: #4A5568; font-size: 11px; }
QLabel#taskStatus { color: #4A5568; font-size: 11px; }
QLabel#taskDetail { color: #6B7684; font-size: 10px; }
QLabel#taskEmpty { color: #6B7684; font-size: 11px; }
QProgressBar#taskProgress { background: #E4EAF2; border: none; border-radius: 3px; }
QProgressBar#taskProgress::chunk { background: #0077B6; border-radius: 3px; }
QPushButton#taskCancelBtn { color: #4A5568; font-size: 13px; padding: 0px; margin: 0px;
    border: 1px solid #C9D3E0; border-radius: 3px; background: transparent; }
QPushButton#taskCancelBtn:hover { color: #C0392B; border-color: #C0392B; }
QPushButton#taskClearBtn { font-size: 11px; padding: 1px 8px; }
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

# ── 「剥颜色、留度量」用的分解规则 ──
# 纯颜色属性：整条删掉，配色交回主题表。
_PURE_COLOR_PROPS = {
    "color", "background-color", "border-color",
    "border-top-color", "border-right-color", "border-bottom-color", "border-left-color",
    "selection-color", "selection-background-color", "gridline-color",
    "alternate-background-color", "outline-color",
}
# 边框简写：值里颜色和度量混在一起（如 `border: 1px solid #2A3A5C`），
# 不能整条删，得拆成长属性，否则会留下 `1px solid` 这种缺色简写（Qt 解析不稳）。
_BORDER_SIDES = {
    "border": "", "border-top": "-top", "border-right": "-right",
    "border-bottom": "-bottom", "border-left": "-left",
}
_BORDER_STYLES = {"none", "solid", "dashed", "dotted", "double",
                  "groove", "ridge", "inset", "outset", "hidden"}
_LEN_RE = _re.compile(r"^\d+(?:\.\d+)?(?:px|pt|em|%)?$")
_COMMENT_RE = _re.compile(r"/\*.*?\*/", _re.S)
_RULE_RE = _re.compile(r"([^{}]*)\{([^{}]*)\}")


def _split_decls(body):
    """`font-size: 10px; padding: 1px 6px` -> [(prop, value), ...]"""
    out = []
    for d in (body or "").split(";"):
        if ":" not in d:
            continue
        p, v = d.split(":", 1)
        p, v = p.strip(), v.strip()
        if p:
            out.append((p, v))
    return out


def _decolor_decls(body):
    """把一段声明剥成「只剩度量」的形式。返回 (props, changed)。"""
    changed = False
    props = []
    for prop, val in _split_decls(body):
        lp = prop.lower()
        if lp in _PURE_COLOR_PROPS:                 # 纯颜色 -> 交给主题表
            changed = True
            continue
        had_hex = bool(_HEX_RE.search(val))
        if had_hex:
            val = _HEX_RE.sub("", val).strip()      # 从简写值里摘掉颜色
            changed = True
        if not val:
            continue                                # 这条本来就只是颜色（如 background:#0E1420）
        if lp in _BORDER_SIDES and had_hex:
            # 只有**确实摘掉了颜色**的简写才拆成长属性（避免留下 `1px solid` 这种缺色简写）。
            # 不带颜色的简写一律原样保留：`border: 1px solid transparent` 是**布局占位**
            # （防止 hover 时才出现边框导致尺寸抖动），`border: none` 也有语义 ——
            # 拆成 width/style 会把 `transparent` 丢掉，边框就冒出来了。
            side = _BORDER_SIDES[lp]
            toks = val.split()
            w = [t for t in toks if _LEN_RE.match(t)]
            s = [t for t in toks if t.lower() in _BORDER_STYLES]
            if w:
                props.append((f"border{side}-width", " ".join(w)))
            if s:
                props.append((f"border{side}-style", " ".join(s)))
            if not w and not s:                     # 没认出来的值原样保留，别丢信息
                props.append((prop, val))
            continue
        props.append((prop, val))
    return props, changed


def _emit(props):
    return " ".join(f"{p}: {v};" for p, v in props)


def strip_colors(qss):
    """把样式表里的**颜色**剥掉，只留字号/内边距/边框宽度等度量。

    返回 (剥色后的样式表, 是否真的剥掉了东西)。

    为什么不是把整段内联样式清成空串（旧做法，已修）：
      `QPushButton { font-size: 10px; padding: 1px 6px; border: 1px solid #2A3A5C; }`
      这句话里，颜色该让位给主题表，**度量不该**。整段清掉后控件回落到应用级
      `QPushButton { padding: 5px 12px }` + 默认 13px 字号 —— 对一个
      `setFixedHeight(22)` 的小按钮就是"文字上下被切掉"（维护者 实测报的
      「显示不全」：按钮 111x22，字号需要 19px，扣掉 padding 后内容区只剩 10px）。
      平台上共 **19 个**控件的内联样式会被清扫，都受这个影响。

    ⚠️ 必须支持**没有选择器的裸声明**（如 `color:#00E5FF; font-size:10px;`，
    `_make_block_header` 就是这么写的）：这种形式没有 `{}` 块，只按规则块解析会
    整段漏掉 → 控件不被清扫 → 浅色主题下亮青字压在白底上**几乎看不见**
    （已实测踩到，见 probe 记录）。所以下面的解析要把 `{}` 之外的部分也当成
    "作用于本控件自身"的声明来处理。
    """
    if not qss:
        return qss, False
    qss = _COMMENT_RE.sub("", qss)      # 注释里可能有花括号，先去掉再解析
    changed = False
    parts = []
    pos = 0
    for m in _RULE_RE.finditer(qss):
        head = qss[pos:m.start()].strip()
        if ":" in head:                  # 块之前的裸声明：作用于本控件自身
            props, ch = _decolor_decls(head)
            changed |= ch
            if props:
                parts.append(_emit(props))
        sel = m.group(1).strip()
        props, ch = _decolor_decls(m.group(2))
        changed |= ch
        if sel and props:
            parts.append("%s { %s }" % (sel, _emit(props)))
        pos = m.end()
    tail = qss[pos:].strip()
    if ":" in tail:                      # 整张表就是裸声明（无 {}）的情况
        props, ch = _decolor_decls(tail)
        changed |= ch
        if props:
            parts.append(_emit(props))
    return " ".join(parts), changed


def _is_theme_sheet(qss):
    """这段内联样式是否"全是主题皮肤色"（是 -> 切主题时可以让位）。"""
    cols = {c.lower() for c in _HEX_RE.findall(qss or "")}
    if not cols:
        return False
    return cols.issubset(THEME_COLORS)


def sweep_inline_styles(app, restore=False):
    """把"只用了主题皮肤色"的内联样式**剥掉颜色**，让主题表接管配色。

    为什么要这一步：Qt 里**控件自己的 setStyleSheet 优先级高于应用级主题表**，
    平台里有几十处 `setStyleSheet("color:#8A93A6")` 之类写死的主题色 ——
    不清掉它们，切浅色主题时这些字仍是深色主题的灰/青，永远变不成黑字。

    规则（保守，宁可不改也不改错）：
      · 只动"颜色**全部**落在 THEME_COLORS 白名单里"的；
      · 只剥**颜色声明**，度量（font-size / padding / font-weight / border-width…）
        必须原样留着 —— 见 strip_colors 的说明；
      · 只要出现一个语义色（红/绿/橙…）就整段保留 —— 那些是"用颜色表达含义"的，
        比如绿色的 Start Training、红色的错误提示，误清会丢信息。
      · 切回深色时按属性还原原样式，原样不差。
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
            thin, changed = strip_colors(cur)
            if not changed:
                continue
            w.setProperty(_SWEEP_PROP, cur)      # 原样式留着，切回深色原样还原
            w.setStyleSheet(thin)                # 只少了颜色，度量还在
            print(f"[theme] swept colors in inline qss on {type(w).__name__}: "
                  f"{len(cur)} -> {len(thin)} chars  {cur[:48]!r}", flush=True)



# ── 主题感知的内联前景色 ──
# 为什么需要：QSS **管不到** setForeground —— 内联前景色的优先级高于应用级主题表。
# 凡是「深色主题下写死了浅色字」的地方（列表里的占位文字等），切浅色后必然是浅字浅底。
# 这类着色点必须走这里取色，别再自己写 QColor(...) 常量。
def is_light():
    """当前是否浅色主题（看 apply_theme 打在 QApplication 上的 dsh_theme 属性）。"""
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        return bool(app is not None and app.property("dsh_theme") == "light")
    except Exception:
        return False


def fg_muted():
    """次级/占位文字（如「这一格没有文件」）的前景：深色主题浅灰、浅色主题中灰，两边都可读。"""
    from PyQt5.QtGui import QColor
    return QColor("#6B7688") if is_light() else QColor("#CCCCCC")


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
        app.setStyleSheet(base_qss() + LIGHT_OVERLAY + LIGHT_CHECK
                          + _LIGHT_SLIDER + _LIGHT_TOOLBAR_SW + _LIGHT_TASK_Q)
        sweep_inline_styles(app, restore=False)
    else:
        sweep_inline_styles(app, restore=True)
        app.setStyleSheet(base_qss() + DARK_INDICATOR + DARK_CHECK
                          + _DARK_SLIDER + _DARK_TOOLBAR_SW + _DARK_TASK_Q)
    # 记住选择：统一走 core/ui_prefs（原子写、保留其它 UI 键）
    try:
        from core.ui_prefs import save as _prefs_save
        _prefs_save({"theme": name})
    except Exception:
        pass
    try:
        app.setProperty("dsh_theme", name)
    except Exception:
        pass
