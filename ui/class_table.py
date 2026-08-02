# -*- coding: utf-8 -*-
"""Class table widget for the annotation panel.

Matches the reference design ("类别栏" screenshot): a compact table listing
each annotation class with 序号 / 类别名 / 颜色 / 训练集 / 测试集 / 空集
columns, plus a top-right toolbar with 刷新 / 添加 / 删除 / 上移 / 下移 /
全选 buttons.

Sample counts are read from the project's class_labels.json (mapping) and
train_test_split.json (train/test/val split); 空集 counts images under the class
folder that are not (yet) labelled, which is 0 for auto-labelled projects.

Exposes the small QListWidget API the annotation panel uses:
set_classes / clear / count / currentRow / setCurrentRow / currentRowChanged.
"""
import json
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal

from core.config import CLASS_COLORS


def _hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _shade(rgb, amount):
    return tuple(max(0, min(255, int(c * (1.0 - amount)))) for c in rgb)


class ClassTableView(QWidget):
    """Table of annotation classes with per-split sample counts."""

    currentRowChanged = pyqtSignal(int)
    refresh_requested = pyqtSignal()
    add_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    move_up_requested = pyqtSignal()
    move_down_requested = pyqtSignal()
    rename_requested = pyqtSignal()

    _HEADERS = ["序号", "类别名", "颜色", "训练集", "测试集", "验证集", "空集"]

    # (top, mid, bottom, text) pastel gradients, one per toolbar pill.
    _PALETTES = [
        ((0x42, 0xC8, 0xD3), (0x8E, 0xDF, 0xE5), (0xE5, 0xFB, 0xFB), (0x1F, 0x6E, 0x78)),  # teal
        ((0xE2, 0x82, 0x82), (0xED, 0xB3, 0xB3), (0xFA, 0xE7, 0xE7), (0x8A, 0x3A, 0x3A)),  # pink
        ((0x9B, 0xE2, 0xE8), (0xC1, 0xED, 0xF1), (0xE5, 0xF9, 0xFA), (0x2E, 0x7E, 0x8A)),  # light blue
        ((0xC3, 0x9B, 0xD8), (0xDC, 0xC3, 0xE9), (0xF3, 0xEC, 0xF8), (0x5E, 0x3A, 0x78)),  # lavender
        ((0x8F, 0xCF, 0xA2), (0xB7, 0xE3, 0xC3), (0xEC, 0xF8, 0xEF), (0x2F, 0x6E, 0x46)),  # green
        ((0xF0, 0xA8, 0x68), (0xF6, 0xCB, 0x9C), (0xFD, 0xF1, 0xE0), (0x7A, 0x4A, 0x1E)),  # peach
    ]

    _TABLE_QSS = """
    QTableWidget {
        background-color: #FFFFFF;
        alternate-background-color: #F4FAFB;
        border: 1px solid #D9E9EC;
        border-radius: 8px;
        gridline-color: #EAF2F4;
        selection-background-color: #BFE9ED;
        selection-color: #14444E;
        font-size: 12px;
    }
    QTableWidget::item { padding: 2px 4px; }
    QHeaderView::section {
        background-color: #E8F6F7;
        color: #2E7E8A;
        font-weight: bold;
        border: none;
        border-right: 1px solid #D8EEF0;
        border-bottom: 1px solid #CFE7EA;
        padding: 5px 4px;
    }
    QTableCornerButton::section { background-color: #E8F6F7; border: none; }
    QScrollBar:vertical { background: transparent; width: 8px; }
    QScrollBar::handle:vertical { background: #C3DDE1; border-radius: 4px; min-height: 20px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._classes = []
        self._project_dir = None
        self._current_idx = -1
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ---- top-right toolbar ----
        tb = QHBoxLayout()
        tb.setSpacing(3)
        tb.addStretch()
        specs = [
            ("refresh", "刷新", "刷新类别统计", self.refresh_requested, 0),
            ("add", "添加", "添加类别", self.add_requested, 4),
            ("del", "删除", "删除选中类别", self.delete_requested, 1),
            ("up", "↑", "上移选中类别", self.move_up_requested, 3),
            ("down", "↓", "下移选中类别", self.move_down_requested, 2),
            ("select_all", "全选", "全选类别", None, 5),
        ]
        self._buttons = {}
        for key, text, tip, sig, palette_idx in specs:
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(self._pill_qss(palette_idx))
            if sig is not None:
                btn.clicked.connect(sig.emit)
            else:
                btn.setCheckable(True)
                btn.clicked.connect(self._toggle_select_all)
            tb.addWidget(btn)
            self._buttons[key] = btn
        layout.addLayout(tb)

        # ---- table ----
        self.table = QTableWidget(0, len(self._HEADERS))
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        self.table.setStyleSheet(self._TABLE_QSS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setHighlightSections(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        for col in (3, 4, 5, 6):
            header.setSectionResizeMode(col, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 32)
        self.table.setColumnWidth(2, 36)
        self.table.setColumnWidth(3, 42)
        self.table.setColumnWidth(4, 42)
        self.table.setColumnWidth(5, 42)
        self.table.setColumnWidth(6, 42)

        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.currentCellChanged.connect(self._on_current_cell_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self.rename_requested.emit())
        self.table.setMinimumHeight(140)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        layout.addWidget(self.table)

    # ---------- styling ----------

    @classmethod
    def _pill_qss(cls, palette_idx):
        """Toolbar pill stylesheet: pastel gradient + dark text."""
        top, mid, bottom, text = cls._PALETTES[palette_idx % len(cls._PALETTES)]
        h = (0.10, 0.08, 0.04)
        c = (0.16, 0.12, 0.06)
        return """
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 {top}, stop:0.55 {mid}, stop:1 {bottom});
            color: {text};
            font-weight: bold;
            font-size: 10px;
            border: 1px solid rgba(255, 255, 255, 180);
            border-radius: 9px;
            padding: 3px 6px;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 {ht}, stop:0.55 {hm}, stop:1 {hb});
        }}
        QPushButton:pressed, QPushButton:checked {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 {ct}, stop:0.55 {cm}, stop:1 {cb});
            border: 1px solid #FFFFFF;
        }}
        """.format(
            top=_hex(top), mid=_hex(mid), bottom=_hex(bottom), text=_hex(text),
            ht=_hex(_shade(top, h[0])), hm=_hex(_shade(mid, h[1])), hb=_hex(_shade(bottom, h[2])),
            ct=_hex(_shade(top, c[0])), cm=_hex(_shade(mid, c[1])), cb=_hex(_shade(bottom, c[2])),
        )

    # ---------- data ----------

    def set_project_dir(self, path):
        """Remember the project directory so counts can be computed."""
        self._project_dir = path
        if self._classes:
            self.set_classes(self._classes)

    def set_classes(self, classes, colors=None):
        """Rebuild the table from a list of class names."""
        prev = self._current_idx
        self.clear()
        self._classes = list(classes)
        counts = self._compute_counts()
        self.table.setRowCount(len(classes))
        for i, name in enumerate(classes):
            color = colors[i] if colors and i < len(colors) else CLASS_COLORS[i % len(CLASS_COLORS)]
            train, test, val, empty = counts.get(i, (0, 0, 0, 0))

            idx_item = QTableWidgetItem(str(i + 1))
            idx_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, idx_item)

            name_item = QTableWidgetItem(name)
            name_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            font = name_item.font()
            font.setBold(True)
            name_item.setFont(font)
            self.table.setItem(i, 1, name_item)

            swatch = QTableWidgetItem()
            swatch.setTextAlignment(Qt.AlignCenter)
            swatch.setBackground(QColor(*color))
            swatch.setToolTip("RGB({}, {}, {})".format(*color))
            self.table.setItem(i, 2, swatch)

            for col, value in ((3, train), (4, test), (5, val), (6, empty)):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, item)
        if 0 <= prev < len(classes):
            self.setCurrentRow(prev, emit=False)

    def clear(self):
        """Remove all rows and reset selection."""
        self.table.setRowCount(0)
        self._classes.clear()
        self._current_idx = -1

    def count(self):
        return len(self._classes)

    def currentRow(self):
        return self._current_idx

    def setCurrentRow(self, row, emit=True):
        """Select the class at ``row`` (like QListWidget.setCurrentRow)."""
        if row == self._current_idx:
            return
        if not (0 <= row < len(self._classes)):
            row = -1
        self._updating = True
        if row >= 0:
            self.table.setCurrentCell(row, 0)
        else:
            self.table.clearSelection()
            self.table.setCurrentCell(-1, -1)
        self._updating = False
        self._current_idx = row
        if emit:
            self.currentRowChanged.emit(row)

    # ---------- counts ----------

    def _compute_counts(self):
        """Return {class_idx: (train, test, val, empty)} from project files."""
        if not self._project_dir:
            return {}
        pd = Path(self._project_dir)
        labels = []
        mapping = {}
        label_path = pd / "annotations" / "class_labels.json"
        if label_path.exists():
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                labels = data.get("labels", [])
                mapping = data.get("mapping", {})
            except Exception:
                pass
        split_map = {}
        split_path = pd / "train_test_split.json"
        if split_path.exists():
            try:
                with open(split_path, "r", encoding="utf-8") as f:
                    split_map = json.load(f)
            except Exception:
                pass

        def norm(rel):
            rel = rel.replace("\\", "/")
            return rel[len("images/"):] if rel.startswith("images/") else rel

        def split_of(rel):
            base = os.path.splitext(rel)[0]
            for key in (base, norm(base), "images/" + norm(base), os.path.basename(base)):
                if key in split_map:
                    return split_map[key]
            return "train"

        trains = [0] * len(labels)
        tests = [0] * len(labels)
        vals = [0] * len(labels)
        empties = [0] * len(labels)
        # De-duplicate: mapping may contain both "images/x.bmp" and "x.bmp".
        base_cls = {}
        for rel, cls_id in mapping.items():
            try:
                cls_id = int(cls_id)
            except (TypeError, ValueError):
                continue
            base = os.path.splitext(norm(rel))[0]
            if base not in base_cls:
                base_cls[base] = cls_id
        for base, cls_id in base_cls.items():
            if not (0 <= cls_id < len(labels)):
                continue
            split = split_of(base)
            if split == "test":
                tests[cls_id] += 1
            elif split == "val":
                vals[cls_id] += 1
            else:
                trains[cls_id] += 1

        # 空集: images inside the class folder that are not in the mapping.
        img_dir = pd / "images"
        if img_dir.exists():
            mapped = {norm(k) for k in mapping}
            for cls_id, name in enumerate(labels):
                cls_dir = img_dir / name
                if cls_dir.is_dir():
                    for f in cls_dir.rglob("*"):
                        if f.is_file() and f.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"):
                            rel = norm(str(f.relative_to(img_dir)))
                            if rel not in mapped:
                                empties[cls_id] += 1
        return {i: (trains[i], tests[i], vals[i], empties[i]) for i in range(len(labels))}

    # ---------- slots ----------

    def _on_current_cell_changed(self, row, _col, _prev_row, _prev_col):
        if self._updating or row < 0:
            return
        if row != self._current_idx:
            self._current_idx = row
            self.currentRowChanged.emit(row)

    def _toggle_select_all(self, checked):
        if checked:
            self.table.setSelectionMode(QAbstractItemView.MultiSelection)
            self.table.selectAll()
        else:
            self.table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table.clearSelection()
