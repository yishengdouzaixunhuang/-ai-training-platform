# -*- coding: utf-8 -*-
"""轻量级目标检测标注工具栏（画布上方）。

按 PRD：画矩形框开关 | 清除所有 / 删除选中 / 取消操作 | 撤销 / 重做。
极简 SVG 线条图标；画框开关互斥高亮、Hover 反馈、禁用置灰、中文 Tooltip。
"""
from core.i18n import tr
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QToolButton, QWidget

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
    #: 把本图的**预测框**追加成标注（一次性、可撤销）
    adopt_requested = pyqtSignal()
    # 工具行展开/折叠（与分割工具栏同步，2026-10-15）
    tools_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_QSS)
        # ★ 2026-10-15 与分割工具栏同步（维护者 拍板"同样处理"）：
        #   常驻条 —— 「▣ 标注」展开开关 + 类别/图层区（由主窗口挂右侧，这里留 stretch）
        #   工具行 —— 画框开关 + 清除/删除/取消 + 撤销/重做（**默认折叠**）
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 3, 6, 3)
        outer.setSpacing(2)

        # ---- 常驻条：展开/折叠开关 ----
        chrome = QHBoxLayout(); chrome.setContentsMargins(0, 0, 0, 0); chrome.setSpacing(4)
        outer.addLayout(chrome)
        self.tools_toggle = QToolButton()
        self.tools_toggle.setObjectName("toolsToggle")
        self.tools_toggle.setCheckable(True)
        self.tools_toggle.setChecked(False)          # 默认折叠
        self.tools_toggle.setText(tr("标注"))
        self.tools_toggle.setToolTip(tr("显示 / 隐藏目标检测工具栏（画框 + 撤销重做）\n快捷键 Ctrl+B"))
        # QToolButton 默认 IconOnly ⇒ 不设这句就只有图标、没有「标注」两个字（同分割工具栏）
        self.tools_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.tools_toggle.setIcon(icons.make_icon("rect"))
        self.tools_toggle.setIconSize(QSize(16, 16))
        self.tools_toggle.setCursor(Qt.PointingHandCursor)
        self.tools_toggle.toggled.connect(self._on_tools_toggled)
        self.tools_toggle.setShortcut(QKeySequence("Ctrl+B"))
        chrome.addWidget(self.tools_toggle)
        chrome.addStretch()

        # ---- 可折叠工具行 ----
        self.tools_box = QWidget(self)
        lay = QHBoxLayout(self.tools_box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # ---- 画矩形框（开关按钮，默认点亮） ----
        self.draw_btn = QToolButton()
        self.draw_btn.setCheckable(True)
        self.draw_btn.setChecked(True)
        self.draw_btn.setToolTip(tr("画矩形框"))
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

        # ---- 预测转标注（"先粗后精"工作流的一步入口）----
        # ★ 2026-11-16：单图推理改成"预测只进预测层"之后，这里补回固化入口。
        #   只**追加**（已有同位置标注会跳过），走 BoxManager 所以 Ctrl+Z 能撤销。
        lay.addWidget(_separator())
        self.btn_adopt = QToolButton()
        self.btn_adopt.setToolTip(tr("把本图预测框追加为标注（已有框保留，可撤销）"))
        self.btn_adopt.setIcon(icons.make_icon("adopt"))
        self.btn_adopt.setIconSize(QSize(24, 24))
        self.btn_adopt.setFixedSize(30, 30)
        self.btn_adopt.setCursor(Qt.PointingHandCursor)
        self.btn_adopt.clicked.connect(lambda *a: self.adopt_requested.emit())
        lay.addWidget(self.btn_adopt)

        outer.addWidget(self.tools_box)
        self.tools_box.setVisible(False)             # 默认折叠

    # ---- 工具行展开 / 折叠 ----
    def _on_tools_toggled(self, on):
        self.tools_box.setVisible(bool(on))
        self.tools_toggled.emit(bool(on))

    def set_tools_visible(self, on):
        on = bool(on)
        if self.tools_toggle.isChecked() != on:
            self.tools_toggle.setChecked(on)
        else:
            self.tools_box.setVisible(on)

    def tools_visible(self):
        """工具行**有没有被折叠**（不是"此刻看不看得见"）。

        ⚠️ 必须用 `not isHidden()` 而不是 `isVisible()`：`isVisible()` 要求**所有祖先**
        都可见，而本工具栏在**分割工程里整体是隐藏的**（det/annot 两栏互斥）⇒
        哪怕用户把工具行展开了，`isVisible()` 也恒为 False。
        用错的后果很具体：分割工程里展开工具行后，这个 API 仍报"没展开"。
        """
        return not self.tools_box.isHidden()

    def toggle_tools(self):
        self.tools_toggle.setChecked(not self.tools_toggle.isChecked())

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
