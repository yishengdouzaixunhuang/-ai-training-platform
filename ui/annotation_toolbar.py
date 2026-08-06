# -*- coding: utf-8 -*-
"""横向标注工具栏（画布上方）。

按友商 PRD 拆解：宽度微调 | 形状工具 | 自由画笔与 AI | 动作 | 视图导航。
工具按钮互斥选中并高亮；图标来自 ui.icons（内嵌 SVG，选中态反白）。
"""
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QSpinBox, QToolButton, QWidget,
)

from ui import icons

_QSS = """
QToolButton {
    border: none; border-radius: 4px;
    padding: 3px; margin: 1px;
    background: transparent;
}
QToolButton:hover { background: #DFF1F4; }
QToolButton:pressed { background: #BFE9ED; }
QToolButton:checked {
    background: #2E7E8A;
    border: 1px solid #1F6E78;
}
QLabel { color: #14444E; font-weight: bold; }
QSpinBox {
    border: 1px solid #BFDDE2; border-radius: 4px;
    padding: 2px 4px; min-width: 58px;
    background: #FFFFFF; color: #14444E;
}
QFrame[frameShape="4"] { color: #CFE7EA; }
"""

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
    ("save", "保存标注 (Ctrl+S)", "save", "save_requested"),
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_QSS)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(4)

        # ---- 模块 1：宽度 ----
        lay.addWidget(QLabel("宽度"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 100)
        self.width_spin.setValue(15)
        self.width_spin.setSuffix(" px")
        self.width_spin.setToolTip("画笔/直线粗细（1-100，实时生效）")
        self.width_spin.valueChanged.connect(self.width_changed)
        lay.addWidget(self.width_spin)
        lay.addWidget(_separator())

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
            lay.addWidget(btn)
            if tid in ("polygon", "sam"):
                lay.addWidget(_separator())

        # ---- 模块 4：动作 ----
        lay.addWidget(_separator())
        for key, tip, ikey, sig_name in _ACTIONS:
            btn = QToolButton()
            btn.setToolTip(tip)
            btn.setIcon(icons.make_icon(ikey))
            btn.setIconSize(QSize(24, 24))
            btn.setFixedSize(30, 30)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda *a, s=getattr(self, sig_name): s.emit())
            lay.addWidget(btn)

        # ---- 模块 5：视图导航 ----
        lay.addWidget(_separator())
        for key, tip, ikey, sig_name in (
            ("prev", "上一张", "prev", "prev_requested"),
            ("next", "下一张", "next", "next_requested"),
        ):
            btn = QToolButton()
            btn.setToolTip(tip)
            btn.setIcon(icons.make_icon(ikey))
            btn.setIconSize(QSize(24, 24))
            btn.setFixedSize(30, 30)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda *a, s=getattr(self, sig_name): s.emit())
            lay.addWidget(btn)

        lay.addStretch()
        self.set_mode("brush")

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