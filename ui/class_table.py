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
import re
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal

from core.config import CLASS_COLORS
from core.label_scan import scan_labels


def _hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


_IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
# 多视图命名约定：一组电芯的 N 张图 = <组名>_te_0/_te_1/...
_VIEW_SUFFIX_RE = re.compile(r"_te_\d+$")
#: 类别统计结果缓存最多留几个工程（FIFO 淘汰）
_COUNTS_CACHE_MAX = 8
#: 摘要掩码：把 hash 收进 48 位，避免整数无限增长
_HASH_MASK = (1 << 48) - 1


def _group_key(stem):
    """把 te_N 视图文件名的后缀去掉，得到"组"名（图片列表一行对应一组）。"""
    return _VIEW_SUFFIX_RE.sub("", stem)


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
    # 点击「训练集/测试集/验证集/空集」数字格子 -> (类别名, split_key)
    cell_filter_requested = pyqtSignal(str, str)

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
        background-color: #0C1220;
        alternate-background-color: #101828;
        border: 1px solid #22304A;
        border-radius: 8px;
        gridline-color: #1A2440;
        selection-background-color: #0E3A52;
        selection-color: #00E5FF;
        font-size: 12px;
    }
    QTableWidget::item { padding: 2px 4px; }
    QHeaderView::section {
        background-color: #0F1728;
        color: #00E5FF;
        font-weight: bold;
        border: none;
        border-right: 1px solid #22304A;
        border-bottom: 1px solid #2A3A5C;
        padding: 5px 4px;
    }
    QTableCornerButton::section { background-color: #0F1728; border: none; }
    QScrollBar:vertical { background: transparent; width: 8px; }
    QScrollBar::handle:vertical { background: #2A3A5C; border-radius: 4px; min-height: 20px; }
    QScrollBar::handle:vertical:hover { background: #00E5FF; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._classes = []
        self._project_dir = None
        self._current_idx = -1
        self._updating = False
        self._split_bases = {}  # (class_idx, split_key) -> set(image base names)
        # 「JSON 读不到时的兜底结果」缓存：(path) -> ((签名...), frozenset(cids))
        # 正常情况（有 labelme JSON）走的 core.label_scan 的共享缓存，不用这个。
        self._fb_cache = {}
        # 工程级「类别统计结果」缓存：{normcase(工程路径): (指纹, counts, split_bases)}
        # 见 _counts_signature() 的注释 —— 命中时跳过上千次 stat，实测 130ms → 20ms 级。
        self._counts_cache = {}

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
        # ⚠️ 数字列原来一律钉 42px，但表头是「训练集/测试集/验证集」**三个汉字**
        #    （实测需要 47px）⇒ 字被截一截。宽度该由**表头文字实际宽度**决定，不是拍整数。
        #    为什么不在这里一次量好：构造期字体**还没定型**（主题清洗在之后才跑），
        #    实测构造期量出 36px、清洗后需要 39px ⇒ 差 3px 又被截。
        #    ⇒ 抽成方法，构造期 + set_classes（工程打开后字体已定型）各调一次。
        self._fit_numeric_columns()

        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.currentCellChanged.connect(self._on_current_cell_changed)
        self.table.cellClicked.connect(self._on_cell_clicked)
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

    def set_project_dir(self, path, with_counts=True):
        """Remember the project directory so counts can be computed.

        ★ 2026-10-28：**只有路径真的变了才清缓存/重算**。
          `_refresh_class_ui()` 里是 `set_project_dir(...)` 紧跟 `set_classes(...)`
          连着调的；如果这里每次进来都清 `_seg_cache`，那次 set_classes 就会退化成
          "全量重扫" —— 而 `_refresh_class_ui` 在打开工程时会被调用多次，
          实测 `_seg_class_ids` 因此被调 1467 次（489×3）、累计 5.2 秒。
          （这是上一轮加缓存时留下的副作用：清了缓存却还指望后续命中。）

        ★ 2026-11-06：`with_counts` 参数。这里原来无条件 `set_classes(self._classes)`，
          算的是**旧类别** —— 而所有已知调用点紧接着都会用真正的类别再调一次
          `set_classes(...)`，所以这一次纯属白算（实测眉山中航 506ms、速博达 158ms）。
          调用方自己负责算统计，就传 False。
        """
        _new = str(path) if path else None
        _old = str(self._project_dir) if self._project_dir else None
        changed = (_new != _old)
        if changed:
            self._fb_cache = {}
            # 目录快照也跟着作废：否则切到新工程后，_seg_class_ids 会拿上一个工程的
            # 快照判断"文件在不在"。目前 _count_segmentation 每次都会重建、且只有一个
            # 调用点，串不了工程；但把这里清掉，将来挪到后台线程时不会踩坑。
            self._img_index = None
            self._ann_index = None
        self._project_dir = path
        if changed and self._classes:
            self.set_classes(self._classes, with_counts=with_counts)

    def _fit_numeric_columns(self):
        """四个数字列（训练集/测试集/验证集/空集）按**表头实际文字宽**定宽。

        判据是"表头放得下"，不是拍一个整数：三个汉字在 11px 字体下需要 47px，
        而原来一律 42px ⇒ 字被截（探针按 QFontMetrics 量出来钉住）。
        必须在**字体定型后**调用（构造期量到的偏小），所以构造期与 set_classes 各调一次。
        """
        from PyQt5.QtGui import QFontMetrics as _FM
        fm = _FM(self.table.horizontalHeader().font())
        need = 0
        for c in (3, 4, 5, 6):
            it = self.table.horizontalHeaderItem(c)
            if it is not None:
                need = max(need, fm.horizontalAdvance(it.text()))
        # ⚠️ 下限 48 不是随便拍的：字体在**主题清洗之后**还会变大（实测构造期量到 36px、
        #    清洗后要39px）⇒ 只按当下字体 +8 给 44px 仍然差 3px 被截。
        #    四个列一共多占不到 30px（类别名是Stretch，会自动收），换取"任何时候都不截"。
        w = max(48, need + 10)
        for c in (3, 4, 5, 6):
            self.table.setColumnWidth(c, w)

    def set_classes(self, classes, colors=None, with_counts=True):
        """Rebuild the table from a list of class names.

        ``with_counts=False``：先只把表和名字建出来，数字列留 0，
        由调用方稍后调 :meth:`refresh_counts` 补上。
        为什么需要它：``_compute_counts()`` 要遍历工程里的标注文件，
        在 1690 张图的工程上实测 559ms —— 打开工程时这段卡在主线程上，
        用户看到的就是"双击之后干等"。数字晚一拍出现，比界面假死好得多。
        """
        prev = self._current_idx
        self.clear()
        self._classes = list(classes)
        # 此时字体已定型（主题清洗在窗口构造之后才跑）⇒ 这里重量一次数字列宽
        self._fit_numeric_columns()
        counts = self._compute_counts() if with_counts else {}
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

    def refresh_counts(self):
        """只重算并更新数字列（train/test/val/empty），不重建整张表。

        配合 ``set_classes(..., with_counts=False)`` 使用：表先出来，
        这一步把统计补上。只动 item 文本，不触发整表重建。
        """
        if not self._classes or not self._project_dir:
            return
        counts = self._compute_counts()
        for i in range(len(self._classes)):
            train, test, val, empty = counts.get(i, (0, 0, 0, 0))
            for col, value in ((3, train), (4, test), (5, val), (6, empty)):
                item = self.table.item(i, col)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(i, col, item)
                item.setText(str(value))

    def clear(self):
        """Remove all rows and reset selection."""
        self.table.setRowCount(0)
        self._classes.clear()
        self._current_idx = -1

    def refresh(self):
        """Recompute per-class counts from project files and redraw the table.

        ★ 2026-11-10：先作废结果缓存再重算 —— 这是「刷新」按钮的落地路径，
          用户主动点了刷新就必须真去磁盘上重算，命中缓存会让人以为按钮坏了。
        """
        if self._classes and self._project_dir:
            self.invalidate_counts()
            self.set_classes(list(self._classes))

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
        """Return {class_idx: (train, test, val, empty)} from project files.

        Classification projects: counts come from class_labels.json (image ->
        class mapping) + train_test_split.json.
        Segmentation projects (no class_labels.json): each image's classes are
        read from its labelme JSON shapes (fallback: mask PNG pixel values).

        ★ 2026-11-10：**结果缓存（B 项）**。进来先算一次工程内容指纹，指纹没变就直接
          返回上次的结果 —— 否则每次刷新都要把上千张图的标注文件逐个 stat/读一遍
          （实测极片 1002 张 131ms、眉山中航 2001 张 500ms+，而指纹只要 20ms 级）。
          指纹的构成见 `_counts_signature()`；`_split_bases` 是这次计算的**副作用**
          （点数字格子过滤图片列表要用），所以它跟着结果一起缓存、命中时一起还原。
        """
        if not self._project_dir:
            return {}
        pd = Path(self._project_dir)
        labels = list(self._classes)
        sig, img_index, ann_index = self._counts_signature(pd, labels)
        if sig is not None:
            key = self._counts_key(pd)
            hit = self._counts_cache.get(key)
            if hit is not None and hit[0] == sig:
                # 命中就把它挪到末尾 —— dict 保持插入序，末尾 = "最近用过"，配合
                # `_trim_counts_cache()` 才是真 LRU。原来只按插入序淘汰（FIFO），
                # 而"来回切两个工程"正是这里的典型用法，先被淘汰的反而是常用的那个。
                self._counts_cache[key] = self._counts_cache.pop(key)
                # 深一层复制：调用方（bases_for）可能被外部持有，别让它改到缓存本体
                self._split_bases = {k: set(v) for k, v in hit[2].items()}
                return dict(hit[1])
        mapping = {}
        label_path = pd / "annotations" / "class_labels.json"
        if label_path.exists():
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                labels = data.get("labels", []) or labels
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
        self._split_bases = {}
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
            key = (cls_id, split)
            self._split_bases.setdefault(key, set()).add(base)
            self._split_bases[key].add(os.path.basename(base))

        # Segmentation projects: no class_labels.json mapping, so classify
        # every image under images/ from its labelme JSON / mask PNG instead.
        if not mapping:
            self._count_segmentation(pd, labels, trains, tests, vals, split_map,
                                     img_index=img_index, ann_index=ann_index)

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
                                self._split_bases.setdefault((cls_id, "empty"), set()).add(f.stem)
        result = {i: (trains[i], tests[i], vals[i], empties[i]) for i in range(len(labels))}
        if sig is not None:
            # ⚠️ 存**副本**，再 return 本地这份 result —— 与命中路径的 `dict(hit[1])` 对称。
            #    第一版这里存的是 result 本体（return 的也是它），实测：
            #    `r[1] = (999,999,999,999)` 之后，再命中缓存拿到的就是 999 ——
            #    "错且下次还错"，而且注释里明明写着"别让它改到缓存本体"。审核抓到的就是这个。
            self._counts_cache[self._counts_key(pd)] = (
                sig, dict(result), {k: set(v) for k, v in self._split_bases.items()})
            self._trim_counts_cache()
        return result

    def bases_for(self, cls_name, split_key):
        """Return the set of image base names for (class, split) — for click-to-filter.

        ★ 2026-11-10：返回**副本**。原来直接给内部 set 本体，而缓存命中时
          `_split_bases` 会被整体替换成新对象 ⇒ 外部持有的旧引用会悬空（拿到上一份数据）。
        """
        try:
            cid = self._classes.index(cls_name)
        except ValueError:
            return set()
        return set(getattr(self, "_split_bases", {}).get((cid, split_key), set()))

    # ---------- counts cache ----------

    @staticmethod
    def _counts_key(pd):
        """缓存键：工程目录（大小写归一 —— Windows 上 `D:\\X` 与 `d:\\x` 是同一目录）。"""
        try:
            return os.path.normcase(os.path.abspath(str(pd)))
        except Exception:
            return os.path.normcase(str(pd))

    def invalidate_counts(self):
        """丢掉类别统计缓存。

        「刷新」按钮要调它：用户点了刷新就是明确要求重算，命中缓存会让按钮看着没反应。
        """
        if not self._project_dir:
            self._counts_cache.clear()
            return
        self._counts_cache.pop(self._counts_key(self._project_dir), None)

    def _trim_counts_cache(self):
        """超过上限就淘汰**最早被用到**的那个（配合命中的"挪到末尾"= LRU）。

        第一版的注释写的是"只留最近用过的"，实现却只按插入序淘汰（FIFO）——
        "在两个工程间来回切"正是这里的典型用法，先插进来那个会先被淘汰，
        也就是**最常用的那个先死**。审核提的 P2-4。
        """
        while len(self._counts_cache) > _COUNTS_CACHE_MAX:
            self._counts_cache.pop(next(iter(self._counts_cache)))

    def _counts_signature(self, pd, labels):
        """工程内容指纹 + 目录快照，返回 `(sig, img_index, ann_index)`。

        ★ 2026-11-10（B 项）：`_compute_counts()` 每次都要遍历几千个标注文件判断
          "哪张图有哪些类"，而其中 99% 一个字没变。给整个工程算一份"内容没变"的证据，
          比挨个 stat 一遍便宜一个数量级（实测极片 1002 张：指纹 ~20ms / 遍历 ~110ms）。

        指纹由这些组成，任何一项变了就重算：

        | 项 | 为什么必须在里面 |
        |---|---|
        | `tuple(labels)` | 调用方临时改了类别表（还没落盘）也得重算 |
        | `images/` 扁平快照 | 加图 / 删图 / 换图 |
        | `images/<类别>/` 子目录快照 | **分类工程**的「空集」按类别目录算，不在上一层（子目录名由同一次 scandir 带回来，不多扫一遍） |
        | `annotations/` 扁平快照 | 标注 JSON、`_mask.png`、**`class_labels.json`** 全在这里 |
        | `train_test_split.json` | 划分变了，train ↔ val 的数字必须跟着变 |
        | `project.json` | 类别、视图数等工程元信息 |
        | 根目录 `class_labels.json` | 少数工程把类别真源放在根上 |

        ⚠️ 审核点过的那条：`class_labels.json` 是分类工程的**类别真源**，只挂 `images/`
           的签名会漏掉它（改了类别数字不更新，比数字算错更难被发现）。它本来就在
           `annotations/` 快照里，这里再把根目录那份单独记一笔，双保险。

        ⚠️ `sig is None` = 目录读不了、语义未知。此时**既不查缓存也不写缓存** ——
           宁可贵一次，也不能把"读失败"当成"文件都没了"。
           `images/`、`annotations/`、以及 `images/*` 子目录摘要，任何一处读不了都算。
        """
        img_dir = pd / "images"
        ann_dir = pd / "annotations"
        img_index, img_dirs = self._scan_dir(img_dir)     # 一次扫描，文件与子目录都拿到
        ann_index = self._dir_index(ann_dir)
        if img_index is None or ann_index is None:
            return None, img_index, ann_index
        sub_sig = self._subdirs_sig(img_dirs, depth=2)
        if sub_sig is None:
            return None, img_index, ann_index
        parts = [
            ("classes", tuple(labels)),
            ("images", self._flat_sig(img_index)),
            ("images/*", sub_sig),
            ("annotations", self._flat_sig(ann_index)),
        ]
        for rel in ("train_test_split.json", "project.json", "class_labels.json"):
            p = pd / rel
            try:
                st = p.stat()
                parts.append((rel, st.st_mtime_ns, st.st_size))
            except OSError:
                parts.append((rel, None))
        return tuple(parts), img_index, ann_index

    @staticmethod
    def _flat_sig(index):
        """单层目录快照的摘要 `(文件数, 累加 hash)`。

        必须**与顺序无关**：`os.scandir` 的返回顺序不做保证，按顺序拼字符串算哈希
        会把"同样的内容"判成"变了"，缓存直接失效。

        ⚠️ 用的是内置 `hash()` ⇒ **受 `PYTHONHASHSEED` 影响**（字符串哈希每个进程不同）。
           本缓存因此只能活在**进程内**：谁要把它落盘/跨进程复用，指纹一定对不上，
           而且对不上的表现是"缓存永不命中"（还算安全）或"拼出相同摘要"（就不安全了）。
           要持久化必须换成 `hashlib` 的稳定摘要。
        """
        acc = 0
        for name, item in index.items():
            acc = (acc + hash((name, item[0], item[1]))) & _HASH_MASK
        return len(index), acc

    @staticmethod
    def _subdirs_sig(dirs, depth=2):
        """给定**第一层子目录列表**，算里面所有文件的摘要 `(文件数, 累加 hash)`。

        给分类工程用：`images/OK/a.bmp`、`images/NG/b.bmp` 都不在 `_dir_index` 的
        扁平快照里（那层只收文件），而「空集」数字正是按这些目录算的。
        顶层文件由调用方的 `_flat_sig` 管，这里不碰，免得同一批文件被记两遍。

        `dirs` 由 `_scan_dir()` 在同一次 scandir 里顺带给出，所以本函数只扫子目录。

        读不了返回 None —— 调用方必须把它当成"整份指纹不可用"，不能当 0 个文件。
        第一版就是把 None 直接塞进 parts 里，结果**照样进缓存**（审核 P2-3）：
        这与"`_dir_index` 返回 None 就不缓存"是两套哲学，会静默存下不完整的指纹。
        """
        acc = 0
        total = 0
        stack = [(d, depth - 1) for d in dirs]
        while stack:
            d, lv = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        try:
                            if e.is_dir():
                                if lv > 0:
                                    stack.append((e.path, lv - 1))
                                continue
                            if not e.is_file():
                                continue
                            st = e.stat()
                            total += 1
                            acc = (acc + hash((e.name, st.st_mtime_ns, st.st_size))) & _HASH_MASK
                        except OSError:
                            continue
            except FileNotFoundError:
                continue
            except OSError:
                return None
        return total, acc

    @staticmethod
    def _scan_dir(d):
        """一次 scandir 同时拿「文件快照」和「子目录列表」：`(idx, dirs)`。

        为什么要合并：`_counts_signature()` 既要 `images/` 的文件快照、又要它的
        子目录（分类工程的 `images/<类别>/`）。分两次 scandir 就是把同一个 1002 条目的
        目录扫两遍（实测多 15ms），冷算那 +44ms 里有它一份。审核提的 P2-5 就是这个。

        语义与 `_dir_index` 一致：
          `({…}, [dirs])`  正常
          `(None, None)`   目录存在但读不了 → 语义未知，调用方不得用它下结论
        """
        idx = {}
        dirs = []
        try:
            with os.scandir(str(d)) as it:
                for e in it:
                    try:
                        if e.is_dir():
                            dirs.append(e.path)
                            continue
                        if not e.is_file():
                            continue
                        st = e.stat()
                        idx[e.name] = (st.st_mtime_ns, st.st_size)
                    except OSError:
                        continue
        except FileNotFoundError:
            return {}, []                  # 目录不存在 == 里面没有文件，空快照就对了
        except OSError:
            return None, None              # 读失败：不能当成"文件都不存在"
        return idx, dirs

    @classmethod
    def _dir_index(cls, d):
        """一次 scandir 拿 {文件名: (mtime_ns, size)}，代替逐张 os.stat。

        Windows 上 FindFirstFile/FindNextFile 一次就把目录里每个条目的属性带回来，
        所以遍历一遍目录远比 N 次 stat 便宜。

        ⚠️ 收录的是**目录下所有文件**，不只是图片：`images/` 里还放着 `_det.json`，
        签名要查它。别把这个索引当"图片索引"复用。

        返回三种情况，调用方要分清：
          {…}   正常快照（空 dict 也是有效快照 —— 目录不存在时就是它）
          None  目录存在但读不了（权限等），语义未知 → 调用方退回逐张 stat
        """
        return cls._scan_dir(d)[0]

    def _count_segmentation(self, pd, labels, trains, tests, vals, split_map,
                            img_index=None, ann_index=None):
        """Count per-class image counts by split for segmentation projects.

        多视图工程：一台电芯有 te_0/te_1/te_2 三张图，图片列表里只占**一行**，
        所以统计也必须按"组"来算。早期按文件逐个累加，结果是同一台料被数 3 次，
        Class 表里的数字点开过滤又和列表行数对不上（同一台料三行里只匹配到一行）。

        ``img_index`` / ``ann_index``：`_counts_signature()` 算指纹时已经 scandir 过一遍，
        传进来就不用再扫第二遍（传 None 时行为与旧版一致，自己扫）。
        """
        img_dir = pd / "images"
        ann_dir = pd / "annotations"
        if not img_dir.is_dir():
            return
        # 一次 scandir 拿全目录的 (mtime_ns, size)：下面 _seg_class_ids 判断缓存签名
        # 时直接查表，不再逐张 stat。2001 张图的工程能省掉 4000 多次系统调用。
        self._img_index = img_index if img_index is not None else self._dir_index(img_dir)
        if self._img_index is None:
            # images/ 目录在、但扫描失败（网络盘断连 / 权限被改 / 被别的进程独占）。
            # _dir_index 返回 None 表示"语义未知"，**不能**拿它去 for 迭代：
            # TypeError 会被 _deferred_class_counts 的 except 吞掉，数字列静默变全 0，
            # 用户以为标注丢了（比崩更糟）。这一轮直接放弃统计，等下次刷新。
            self._ann_index = None
            return
        self._ann_index = ann_index if ann_index is not None else self._dir_index(ann_dir)
        groups = {}
        for name in self._img_index:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in _IMG_EXT:
                continue
            groups.setdefault(_group_key(stem), []).append(img_dir / name)
        for key in sorted(groups):
            files = groups[key]
            # te_0 优先：列表行的 base 就是 te_0 的文件名
            files.sort(key=lambda x: (not x.stem.endswith("_te_0"), x.name))
            cids = set()
            for f in files:
                cids |= self._seg_class_ids(f, ann_dir, labels)
            if not cids:
                continue
            base = files[0].stem
            # 划分表：多视图工程按 {key}_te_0 作键，单图工程按 {key} 作键，两种都试
            split = "train"
            for cand in [f.stem for f in files] + [key]:
                s = split_map.get(cand, split_map.get("images/" + cand))
                if s:
                    split = s
                    break
            for cid in cids:
                if not (0 <= cid < len(labels)):
                    continue
                if split == "test":
                    tests[cid] += 1
                elif split == "val":
                    vals[cid] += 1
                else:
                    trains[cid] += 1
                # 记下这一组的"行 base"，点数字格子过滤图片列表时才对得上
                k = (cid, split)
                self._split_bases.setdefault(k, set()).add(base)
                self._split_bases[k].add(os.path.basename(base))

    def _seg_class_ids(self, img_path, ann_dir, labels):
        """一张分割图里出现过的类别 id 集合。

        ★★ 2026-10-28 两处改动（都是"打开工程/闭合标注卡顿"的直接原因）：
          ① **JSON 交给 `core.label_scan`**（bytes 正则 + 与 `discover_labels`
             共用的进程级缓存）。原来这里 `json.load` 整个文件，而某些工程的单个
             labelme JSON 就有 14~18MB（掩膜轮廓上万个顶点），本函数只要 label。
          ② **不再每次 stat 三个文件**。原实现在缓存外壳里对 json / mask png /
             det json 各 `stat()` 一次 —— 这个函数在打开 489 张图的工程时会被调用
             **1467 次**（`_compute_counts` 被重复触发），光 stat 就是秒级。
             现在 JSON 路径只交给 `scan_labels`（内部一次 stat），
             mask png / det json 只在**JSON 没有内容时**才去看，且单独缓存。
        """
        # ① 首选 labelme JSON（scan_labels 自带签名缓存；文件不存在返回 None）
        info = scan_labels(img_path.with_suffix(".json"))
        if info is not None:
            cids = set()
            for lab in info[0]:
                if lab in labels:
                    cids.add(labels.index(lab))
            if cids:
                return cids
        # ② 兜底：mask PNG / 检测 COCO json（少见路径，单独缓存，且只有走到这里才 stat）
        cache = getattr(self, "_fb_cache", None)
        if cache is None:
            cache = self._fb_cache = {}
        stem = img_path.stem
        mname = stem + "_mask.png"
        dname = stem + "_det.json"
        mpath = ann_dir / mname
        dpath = img_path.with_name(dname)
        idx_ann = getattr(self, "_ann_index", None)
        idx_img = getattr(self, "_img_index", None)
        sig = []
        for p, name, index in ((mpath, mname, idx_ann), (dpath, dname, idx_img)):
            if index is not None:
                # 有快照就查表：空 dict 也是有效快照（名字查不到 = 文件不存在）
                info = index.get(name)
            else:
                # 快照不可用（目录读不了）：退回逐个 stat，行为与旧版一致
                try:
                    st = p.stat()
                    info = (st.st_mtime_ns, st.st_size)
                except Exception:
                    info = None
            sig.append(info)
        sig.append(tuple(labels))
        sig = tuple(sig)
        key = str(img_path)
        hit = cache.get(key)
        if hit is not None and hit[0] == sig:
            return set(hit[1])
        cids = self._seg_class_ids_fallback(mpath, dpath, labels)
        if len(cache) > 4000:
            cache.clear()
        cache[key] = (sig, frozenset(cids))
        return set(cids)

    def _seg_class_ids_fallback(self, mask_path, det_json, labels):
        """JSON 里没读到类别时的兜底：mask PNG 像素值 → 检测 COCO json。"""
        cids = set()
        try:
            if mask_path.exists():
                import numpy as _np
                from PIL import Image as _Image
                arr = _np.array(_Image.open(str(mask_path)))
                cids = set(int(v) for v in _np.unique(arr)
                           if 0 < int(v) < len(labels) and int(v) != 255)
        except Exception:
            pass
        # Detection projects: count boxes from COCO _det.json (fallback when
        # there are no labelme shapes / mask pixels).
        if not cids:
            try:
                if det_json.exists():
                    from detection.coco_io import load_coco_json
                    boxes, _ = load_coco_json(str(det_json))
                    for b in boxes:
                        lab = b.get("category", "")
                        if lab in labels:
                            cids.add(labels.index(lab))
            except Exception:
                pass
        return cids

    # ---------- slots ----------

    def _on_current_cell_changed(self, row, _col, _prev_row, _prev_col):
        if self._updating or row < 0:
            return
        if row != self._current_idx:
            self._current_idx = row
            self.currentRowChanged.emit(row)

    def _on_cell_clicked(self, row, col):
        """Count cells (训练集/测试集/验证集/空集) filter the image list."""
        if col not in (3, 4, 5, 6):
            return
        if not (0 <= row < len(self._classes)):
            return
        split_key = {3: "train", 4: "test", 5: "val", 6: "empty"}[col]
        self.cell_filter_requested.emit(self._classes[row], split_key)

    def _toggle_select_all(self, checked):
        if checked:
            self.table.setSelectionMode(QAbstractItemView.MultiSelection)
            self.table.selectAll()
        else:
            self.table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table.clearSelection()
