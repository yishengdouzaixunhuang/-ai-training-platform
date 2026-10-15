# -*- coding: utf-8 -*-
"""横向标注工具栏（画布上方）。

按友商 PRD 拆解：宽度微调 | 形状工具 | 自由画笔与 AI | 动作 | 视图导航 | **右侧状态区**。
工具按钮互斥选中并高亮；图标来自 ui.icons（内嵌 SVG，选中态反白）。

右侧状态区是 2026-10-14 "顶栏并进底栏"那次重构加上的：
    `Annot` / `Ignore` / `Pred` 三个图层开关 + 当前类别格，原先只以**纯文本**
    挂在多视图顶栏上（`Pan · Class: Zw · Annot:ON · Ignore:ON · Pred:OFF · Brush: 15`），
    顶栏删掉后它们在全屏**没有第二个显示处**（只能靠快捷键盲切），所以搬到这里并做成可点按钮。
    实测宽度（tmp/measure_toolbar.py）：本工具栏 sizeHint 已有 **773px**，
    而中央列在左栏展开时只有 757px ⇒ 右侧这几个控件必须**紧凑**（见 _SW_QSS），
    且只在多视图模式下显示（单图模式画布上本就叠着状态行，再显示一处就是冗余）。
"""
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QFrame, QHBoxLayout, QLabel, QMenu, QSizePolicy, QSpinBox,
    QToolButton, QVBoxLayout, QWidget,
)

from ui import icons

_QSS = """
QToolButton {
    border: none; border-radius: 4px;
    padding: 3px; margin: 1px;
    background: transparent;
}
QToolButton:hover { background: #131B2E; border: 1px solid #00E5FF; }
QToolButton:pressed { background: #0E1830; }
QToolButton:checked {
    background: #0E3A52;
    border: 1px solid #00E5FF;
}
QLabel { color: #00E5FF; font-weight: bold; }
QSpinBox {
    border: 1px solid #2A3A5C; border-radius: 4px;
    padding: 2px 4px; min-width: 58px;
    background: #0E1420; color: #E6F0FF;
}
QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #2A3A5C; }
"""

# 右侧状态区专用样式。
# ⚠️ 必须用 objectName 选中，**不能**写成全局 `QToolButton { padding:2px 4px }` ——
#    那会把左边那 20 个图标按钮一起压扁（它们是 setFixedSize(30,30)，
#    padding 一变图标就挤没了）。紧凑是有代价的：这里每一项的宽度都要算进
#    工具栏总宽（773px 已经接近中央列可用宽）。
_SW_QSS = """
QToolButton#layerSw {
    border: 1px solid #2A3A5C; border-radius: 4px;
    padding: 1px 4px; margin: 0px;
    color: #8A93A6; font-size: 12px;
}
QToolButton#layerSw:hover { border: 1px solid #00E5FF; }
QToolButton#layerSw:checked {
    background: #0E3A52; border: 1px solid #00E5FF; color: #E6F0FF;
}
QToolButton#classPick {
    border: 1px solid #2A3A5C; border-radius: 4px;
    padding: 1px 5px; margin: 0px;
    color: #C8D2E0; font-size: 12px;
}
QToolButton#classPick:hover { border: 1px solid #00E5FF; }
"""

#: 三个图层开关（2026-10-15 从"带边框方块按钮"改成友商那种**真复选框**）。
#: ⚠️ 这里只写**度量**（indicator 尺寸/间距）；颜色（青块/灰字/勾选底色）全部写进
#:    themes.py 的 `#layerSw` 规则 —— 内联白名单色会被 sweep_inline_styles 剥掉，
#    剥完只剩度量，复选框就成了"一个 12px 的空方块"（同底栏滑块"占位不画"那类坑）。
_SW_CHECK_QSS = """
QCheckBox#layerSw { padding: 0px 2px; margin: 0px; font-size: 12px; }
QCheckBox#layerSw::indicator { width: 12px; height: 12px; }
"""

# (key, 显示文字, tooltip)
# 文字取中文两字：实测英文 `Annot`/`Ignore`/`Pred` 三枚要 159px，中文只要 141px，
# 而右侧可用宽是卡着算的（见文件头）。tooltip 里把快捷键写全 —— 这三个开关
# 本来就只有快捷键，界面文字短一点不影响可发现性。
_LAYERS = [
    ("annot", "标注", "标注图层显示 / 隐藏（Ctrl+Space）"),
    ("ignore", "忽略", "忽略区显示 / 隐藏（Ctrl+Alt）"),
    ("pred", "预测", "预测结果显示 / 隐藏（Space）"),
]


# (id, tooltip, icon_key)
_TOOLS = [
    ("line", "直线 (L)", "line"),
    ("rect", "矩形 (R)", "rect"),
    ("circle", "圆形 (C)", "circle"),
    ("ellipse", "椭圆 (O)", "ellipse"),
    ("polygon", "多边形 (P)", "polygon"),
    ("brush", "自由画笔 (B)", "brush"),
    ("eraser", "橡皮 (E)", "eraser"),
    ("ignore", "忽略区域 (I)", "ignore"),
    ("sam", "AI 智能标注 (A)", "ai"),
    ("pan", "平移视图 (H)", "pan"),
]

_ACTIONS = [
    # (key, tooltip, icon_key, signal_name)
    ("undo", "撤销 (Ctrl+Z)", "undo", "undo_requested"),
    ("redo", "重做 (Ctrl+Y)", "redo", "redo_requested"),
    ("cancel", "取消当前操作 (Esc)", "cancel", "cancel_requested"),
    ("clear", "清空标注", "clear", "clear_requested"),
    ("reload", "刷新视图", "reload", "reload_requested"),
]


def _separator():
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(26)
    return line


class AnnotationToolbar(QWidget):
    mode_selected = pyqtSignal(str)
    width_changed = pyqtSignal(int)
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    save_requested = pyqtSignal()
    prev_requested = pyqtSignal()
    next_requested = pyqtSignal()
    # 右侧状态区：三个图层开关被点 -> ("annot"|"ignore"|"pred", 新状态)
    layer_toggled = pyqtSignal(str, bool)
    # 类别格菜单里选了第 i 个类别
    class_pick_requested = pyqtSignal(int)
    # 工具行展开/折叠（友商那套"点标注显示隐藏工具栏"，2026-10-15）
    tools_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_QSS)
        # ★ 2026-10-15 参照友商顶栏拆成两段（用户给截图："点击标注或者掩膜可以显示隐藏工具栏"）：
        #   常驻条 chrome_bar —— 「✏ 标注」展开开关 + 上一张/下一张 + 保存 + 类别/图层开关
        #   工具行 tools_box —— 宽度 + 10 个形状工具 + 撤销/重做/取消/清空/刷新（**默认折叠**）
        # 为什么这样切分（不是随便分的，是按"丢了会不会迷路"定的）：
        #   · 上一张/下一张 **全应用没有任何快捷键**（只有 Ctrl+A / Shift+X），
        #     藏起来 = 图片导航只剩"点列表行" ⇒ 必须常驻；
        #   · 保存留了 tooltip 里的 Ctrl+S，但仍常驻（高频且不可逆风险高）；
        #   · 形状工具只在"要画"时才需要 ⇒ 折叠起来把 30px 纵向空间还给画布。
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 3, 6, 3)
        outer.setSpacing(2)

        # ────────── 常驻条 ──────────
        chrome = QHBoxLayout(); chrome.setContentsMargins(0, 0, 0, 0); chrome.setSpacing(4)
        outer.addLayout(chrome)

        # 展开/收起开关。用 `toggled` 而不是 `clicked`：它**就是**一个状态按钮，
        # 外部不会来 setChecked 回写（不像左栏竖排页签那种"回写会绕回"的情况），
        # 所以不需要 blockSignals 那套，也不会出现"点了没反应"。
        self.tools_toggle = QToolButton()
        self.tools_toggle.setObjectName("toolsToggle")
        self.tools_toggle.setCheckable(True)
        self.tools_toggle.setChecked(False)          # 默认折叠（Arthur 拍板）
        self.tools_toggle.setText("标注")
        self.tools_toggle.setToolTip("显示 / 隐藏标注工具栏（形状工具 + 撤销重做）\n快捷键 Ctrl+B")
        # ⚠️ QToolButton 默认是 ToolButtonIconOnly ⇒ 光设 setText("标注") **一个字都不会显示**
        #    （实测：折叠态抓图只剩一个斜杠图标，根本看不出这是折叠开关）。
        #    要文字就必须显式 TextBesideIcon。
        self.tools_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.tools_toggle.setIcon(icons.make_icon("brush"))
        self.tools_toggle.setIconSize(QSize(16, 16))
        self.tools_toggle.setCursor(Qt.PointingHandCursor)
        self.tools_toggle.toggled.connect(self._on_tools_toggled)
        # Ctrl+B：全应用只有 Ctrl+A / Shift+X 两个 setShortcut，Ctrl+B 未被占
        # （已 grep 确认），用来在键盘上开关工具行 —— 折叠后鼠标不在画布上时也能展开。
        self.tools_toggle.setShortcut(QKeySequence("Ctrl+B"))
        chrome.addWidget(self.tools_toggle)
        chrome.addWidget(_separator())

        # 视图导航（**无快捷键，只能住在这里**，见上面注释）
        for key, tip, ikey, sig_name in (
            ("prev", "上一张", "prev", "prev_requested"),
            ("next", "下一张", "next", "next_requested"),
            ("save", "保存标注 (Ctrl+S)", "save", "save_requested"),
        ):
            btn = QToolButton()
            # objectName 让"这枚按钮必须在常驻条、不能在折叠区"能被探针直接查 ——
            # 上一张/下一张全应用没有快捷键，一旦被人挪进 tools_box 就是导航失联，
            # 而那种改动**不会报任何错**（只是少了两枚按钮）。
            btn.setObjectName("nav" + key.capitalize())
            btn.setToolTip(tip)
            btn.setIcon(icons.make_icon(ikey))
            btn.setIconSize(QSize(24, 24))
            btn.setFixedSize(30, 30)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda *a, s=getattr(self, sig_name): s.emit())
            setattr(self, "_chrome_" + key, btn)
            chrome.addWidget(btn)
        chrome.addStretch()

        # ---- 常驻条右侧状态区（多视图顶栏拆掉后搬来的，见文件头）----
        # 用 wrapper + 内部 layout 装：可见性只用一个开关（set_layers_visible），
        # 不必逐个控件 setVisible —— 漏一个就是"半边显示半边不显示"这种最难发现的状态。
        self.right_box = QWidget(self)
        self._right_lay = QHBoxLayout(self.right_box)
        self._right_lay.setContentsMargins(0, 0, 0, 0)
        self._right_lay.setSpacing(4)
        chrome.addWidget(self.right_box)

        self._layer_btns = {}
        # ── 「显示」前缀标签（照友商那一行的排法）──
        # 原来三个开关是**带边框的方块按钮**，右边紧跟一个"类别:xx"下拉，
        # 看不出"这三个是图层可见性开关"。友商的做法是：一个「显示」小标题
        # + 一串**真正的复选框**（勾上 = 实心青块，没勾 = 空框 + 灰字）。
        # 实测宽度：友商方案 191px vs 原方块按钮 203px —— **反而更窄**，不占地方。
        self._display_label = QLabel("显示")
        self._display_label.setObjectName("layerGroupLabel")
        self._display_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._right_lay.addWidget(self._display_label)
        for key, text, tip in _LAYERS:
            chk = QCheckBox(text)
            chk.setObjectName("layerSw")
            # 只写度量（间距），颜色在 themes.py 的主题表（见文件头"剥色"那段的教训）
            chk.setStyleSheet(_SW_CHECK_QSS)
            chk.setChecked(True)
            chk.setToolTip(tip)
            chk.setCursor(Qt.PointingHandCursor)
            chk.toggled.connect(lambda on, k=key: self.layer_toggled.emit(k, bool(on)))
            self._layer_btns[key] = chk
            self._right_lay.addWidget(chk)

        # ⚠️ 这里原来还有一格「类别:xx▾」（下拉菜单 = 选类别入口 + 当前类别显示）。
        #    2026-10-15 Arthur 判定冗余，删除。删之前逐条核过"别处还有没有"：
        #      · 选类别：数字键 1-9（`_on_canvas_class_select`）、右栏「标注管理 → 类别」表
        #        （`class_list.setCurrentRow` 本来就被 `_on_canvas_class_select` 同步）——
        #        `_on_canvas_class_select` 里那句 setCurrentRow 正是"两处同源"的保证。
        #      · 当前类别：单图模式画布状态行本来就显示；多视图模式画布叠加是关的，
        #        现在由**右栏类别表的选中行**承担（它才是与 canvas 同步的那一份）。
        #    ⇒ 菜单对象 `class_btn` / `_class_menu` 与 set_class_name/set_classes 一并消失，
        #       不留"看着能用、其实没接上"的死代码。
        self._display_label.setToolTip("图层显示开关（叠加显示的是哪几层）")

        # ────────── 可折叠工具行 ──────────
        self.tools_box = QWidget(self)
        tl = QHBoxLayout(self.tools_box)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(4)

        # ---- 模块 1：宽度 ----
        _wlab = QLabel("宽度")
        # ⚠️ QLabel 默认水平策略是 Preferred：整行余量没人要时，**它会全吞** ——
        #    实测被撑到 605px 宽，文字左对齐 ⇒ 「宽度」和数字框之间裂开一大段空白。
        #    （第一版只把数字框钉成 Fixed，标签照样吞余量 —— 抓图才看出来，
        #      而"标签右边缘到输入框的间距"那种断言算出来是 4px，**照样 PASS**，假绿。）
        _wlab.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        tl.addWidget(_wlab)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 100)
        self.width_spin.setValue(15)
        self.width_spin.setSuffix(" px")
        self.width_spin.setToolTip("画笔/直线粗细（1-100，实时生效）")
        # ⚠️ 必须钉 Fixed：QSpinBox 的默认水平策略会**吞掉整行多余的宽度**，
        #    展开态下数字框被拉成 ~290px，「宽度」标签和它之间裂开一大段空白
        #    （实测截图才发现，label/sizeHint 类断言都看不出来）。
        self.width_spin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.width_spin.valueChanged.connect(self.width_changed)
        tl.addWidget(self.width_spin)
        tl.addWidget(_separator())

        # ---- 模块 2+3：工具（互斥选中） ----
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons = {}
        for tid, tip, ikey in _TOOLS:
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setIcon(icons.make_ai_icon() if ikey == "ai" else icons.make_icon(ikey))
            btn.setIconSize(QSize(24, 24))
            btn.setFixedSize(30, 30)
            btn.setCursor(Qt.PointingHandCursor)
            self._tool_group.addButton(btn)
            self._tool_buttons[tid] = btn
            btn.clicked.connect(lambda checked, m=tid: self._on_tool_clicked(m))
            tl.addWidget(btn)
            if tid in ("polygon", "sam"):
                tl.addWidget(_separator())

        # ---- 模块 4：动作 ----
        tl.addWidget(_separator())
        for key, tip, ikey, sig_name in _ACTIONS:
            btn = QToolButton()
            btn.setToolTip(tip)
            btn.setIcon(icons.make_icon(ikey))
            btn.setIconSize(QSize(24, 24))
            btn.setFixedSize(30, 30)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda *a, s=getattr(self, sig_name): s.emit())
            setattr(self, "_act_" + key, btn)
            tl.addWidget(btn)

        outer.addWidget(self.tools_box)
        self.tools_box.setVisible(False)             # 默认折叠

        # 默认隐藏右侧状态区的理由：单图模式下**画布上本来就叠着状态行**
        # （`set_status_overlay(True)`），这里再显示一处就是"同一个状态同屏两遍"；
        # 多视图模式下画布叠加被关掉、顶栏又已删除，这时它才是唯一显示处
        # ⇒ 由主窗口在进多视图时打开。
        self.set_layers_visible(False)

        self.set_mode("brush")

    # ---- 工具行展开 / 折叠（友商那套"点标注显示隐藏工具栏"）----
    def _on_tools_toggled(self, on):
        self.tools_box.setVisible(bool(on))
        # 折叠态下画布会**真的变大**（工具行是这个布局里的兄弟控件），
        # 所以顺手让画布重排一次 —— 否则缩放值还停在旧尺寸的适配上。
        vp = getattr(self, "viewport_updated", None)
        if callable(vp):
            vp()
        self.tools_toggled.emit(bool(on))

    def set_tools_visible(self, on):
        """外部（快捷键 / 探针 / 以后可能的记忆设置）设置工具行可见性。"""
        on = bool(on)
        if self.tools_toggle.isChecked() != on:
            self.tools_toggle.setChecked(on)      # 走 toggled ⇒ 内部状态自洽
        else:
            self.tools_box.setVisible(on)

    def tools_visible(self):
        """工具行**有没有被折叠**（不是"此刻看不看得见"）。

        ⚠️ 用 `not isHidden()` 而不是 `isVisible()`：`isVisible()` 要求所有祖先可见，
        而本工具栏在**检测工程里整体是隐藏的**（det/annot 两栏互斥）⇒
        展开后 `isVisible()` 仍为 False，API 会误报"没展开"。
        """
        return not self.tools_box.isHidden()

    def toggle_tools(self):
        self.tools_toggle.setChecked(not self.tools_toggle.isChecked())

    # ---- 右侧状态区（原顶栏那行的承载者）----
    def set_layers_visible(self, on):
        box = getattr(self, "right_box", None)
        if box is not None:
            box.setVisible(bool(on))

    def layers_visible(self):
        box = getattr(self, "right_box", None)
        return bool(box is not None and box.isVisible())

    def set_layer_states(self, annot, ignore, pred):
        """外部同步三个开关（**不触发** `layer_toggled`）。

        ⚠️ 不挡住信号就是回环：外部（画布）说"现在是开" → 按钮 setChecked(True)
        → 按钮发 toggled → 主窗口又去切一次画布 → 状态来回翻。
        用控件级的 blockSignals、而不是记一个 `_suppress` 标志：标志只要漏在某条
        路径上就会**真的把状态切反**（静默错），blockSignals 绕不过去。
        """
        for key, val in (("annot", annot), ("ignore", ignore), ("pred", pred)):
            b = self._layer_btns.get(key)
            if b is None:
                continue
            val = bool(val)
            if b.isChecked() != val:
                b.blockSignals(True)
                b.setChecked(val)
                b.blockSignals(False)

    def layer_states(self):
        """三个开关的当前显示状态（探针拿它跟画布对拍，不必去读私有属性）。"""
        return {k: bool(b.isChecked()) for k, b in self._layer_btns.items()}

    def class_name(self):
        """工具栏不再显示/选择类别（2026-10-15 删除那一格后没有数据源）。

        保留这个方法是为了让"谁还在问类别名"变成**一条会立刻炸的错**，
        而不是静默返回 "类别:?" —— 上游若还有漏改的调用点，构造期就会暴露。
        """
        raise AttributeError(
            "AnnotationToolbar 已不再持有类别格（2026-10-15 删除）；"
            "当前类别请读 canvas.current_class_id 或右栏 class_list。")

    def _on_tool_clicked(self, mode):
        self.mode_selected.emit(mode)

    def set_mode(self, mode):
        """外部同步选中态（不触发 mode_selected）。"""
        btn = self._tool_buttons.get(mode)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    def set_width(self, v):
        v = max(1, min(100, int(v)))
        if self.width_spin.value() != v:
            self.width_spin.setValue(v)