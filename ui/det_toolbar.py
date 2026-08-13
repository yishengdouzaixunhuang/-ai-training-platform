# -*- coding: utf-8 -*-
"""轻量级目标检测标注工具栏（画布上方）。

按 PRD：画矩形框开关 | 清除所有 / 删除选中 / 取消操作 | 撤销 / 重做。
极简 SVG 线条图标；画框开关互斥高亮、Hover 反馈、禁用置灰、中文 Tooltip。
"""
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QToolButton, QWidget

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
QToolButton:disabled { background: transparent; }
QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #2A3A5C; }
"""


def _separator():
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(26)
    return line


class DetToolbar(QWidget):
    """目标检测标注工具栏：画框开关 + 清除/删除/取消 + 撤销/重做。"""

    draw_toggled = pyqtSignal(bool)
    clear_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_QSS)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(4)

        # ---- 画矩形框（开关按钮，默认点亮） ----
        self.draw_btn = QToolButton()
        self.draw_btn.setCheckable(True)
        self.draw_btn.setChecked(True)
        self.draw_btn.setToolTip("画矩形框")
        self.draw_btn.setIcon(icons.make_icon("rect"))
        self.draw_btn.setIconSize(QSize(24, 24))
        self.draw_btn.setFixedSize(30, 30)
        self.draw_btn.setCursor(Qt.PointingHandCursor)
        self.draw_btn.clicked.connect(self.draw_toggled)
        lay.addWidget(self.draw_btn)

        # ---- 清除所有 / 删除选中 / 取消操作 ----
        lay.addWidget(_separator())
        for key, tip, ikey, sig_name in (
            ("clear", "清除所有", "clear_all", "clear_requested"),
            ("delete", "删除选中", "clear", "delete_requested"),
            ("cancel", "取消操作", "cancel", "cancel_requested"),
        ):
            btn = QToolButton()
            btn.setToolTip(tip)
            btn.setIcon(icons.make_icon(ikey))
            btn.setIconSize(QSize(24, 24))
            btn.setFixedSize(30, 30)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda *a, s=getattr(self, sig_name): s.emit())
            setattr(self, "btn_" + key, btn)
            lay.addWidget(btn)

        # ---- 撤销 / 重做 ----
        lay.addWidget(_separator())
        for key, tip, ikey, sig_name in (
            ("undo", "撤销 (Ctrl+Z)", "undo", "undo_requested"),
            ("redo", "重做 (Ctrl+Y)", "redo", "redo_requested"),
        ):
            btn = QToolButton()
            btn.setToolTip(tip)
            btn.setIcon(icons.make_icon(ikey))
            btn.setIconSize(QSize(24, 24))
            btn.setFixedSize(30, 30)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda *a, s=getattr(self, sig_name): s.emit())
            setattr(self, "btn_" + key, btn)
            lay.addWidget(btn)

        lay.addStretch()

    def set_draw_active(self, active):
        """外部同步画框开关（不触发 draw_toggled）。"""
        active = bool(active)
        if self.draw_btn.isChecked() != active:
            self.draw_btn.setChecked(active)

    def set_has_selection(self, has_selection):
        """删除选中按钮：仅在选中某个框时可用。"""
        self.btn_delete.setEnabled(bool(has_selection))

    def set_can_undo(self, can):
        """撤销按钮可用状态。"""
        self.btn_undo.setEnabled(bool(can))

    def set_can_redo(self, can):
        """重做按钮可用状态。"""
        self.btn_redo.setEnabled(bool(can))
