# -*- coding: utf-8 -*-
"""多视图加图对话框（N 个槽位，对应工程声明的视图数）

对标友商 友商平台 的「选择图像」：绑定图片组 + 编号槽位 + 每槽位
[加载图像][加载文件夹][清空] + 计数 + 预览 + 文件列表 + 加载全部图像。

与平台既有 MixedAddImagesDialog 的区别：
  * 列数**不写死 2 列**，按工程声明的 views 动态生成（1/2/3/N 视图通用）
  * 配对按**视图后缀**分组，而不是按同名：xxx_te_0/1/2.jpg 归为同一组
  * 有预览图；未配对的文件会明确列出来，不静默丢弃

用法：
    dlg = MultiViewAddDialog(parent, views=["te_0","te_1","te_2"],
                             view_names={"te_0":"漫反射", ...})
    if dlg.exec_() == QDialog.Accepted:
        views, paths_per_view = dlg.get_view_paths()
"""
from __future__ import annotations

import os

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QListWidget, QListWidgetItem, QPushButton,
                             QToolButton, QVBoxLayout)

from ui import icons
from ui.mixed_add_dialog import STYLE

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

# 工具栏按钮样式：与平台其它工具栏（annotation_toolbar 等）保持一致
TOOLBTN_QSS = """
QToolButton { border: 1px solid #2A3A5C; border-radius: 4px; background-color: #131B2E; padding: 0px; }
QToolButton:hover { background-color: #1B2740; border-color: #00E5FF; }
QToolButton:pressed { background-color: #0E1830; }
QToolButton:disabled { background-color: #0E1420; }
"""


def stem_key(path, view):
    """把 xxx_te_1.jpg 归一成组名 xxx；没有该后缀就退回原文件名（仍可配对）。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    suf = "_" + view
    return stem[: -len(suf)] if stem.endswith(suf) else stem


class _ViewColumn:
    """一个视图槽位：列头按钮 + 预览 + 文件列表。"""

    def __init__(self, index, view, label, dlg):
        self.index = index
        self.view = view
        self.label = label
        self.files = []
        self.dlg = dlg

        self.box = QVBoxLayout()
        self.box.setSpacing(4)

        # ── 列头：序号 名称 … [图像][文件夹][清空] 计数 ──
        head = QHBoxLayout()
        head.setSpacing(4)
        # 用普通文本 + 空格，不要写 &nbsp;：
        # QLabel 默认 AutoText 只在文本含 HTML 标签时才按富文本渲染，
        # 单独一个 &nbsp; 不会被识别，会原样显示成 "&nbsp;"（已踩过）。
        title = QLabel(f"{index + 1}.  {label}")
        title.setObjectName("col_header")
        head.addWidget(title)
        head.addStretch()
        # 图标全部走平台内嵌 SVG 工厂（emoji 在本机字体下渲染不出来，会变成空白方块）
        for ikey, tip, cb, danger in (
            ("file_image", f"加载 {label} 图像文件（可多选）", self.pick_files, False),
            ("folder", f"加载 {label} 整个文件夹", self.pick_dir, False),
            ("trash", f"清空 {label} 已加载的文件", self.clear, True),
        ):
            b = QToolButton()
            b.setIcon(icons.make_icon(ikey))
            b.setIconSize(QSize(20, 20))
            b.setFixedSize(28, 26)
            b.setToolTip(tip)
            if danger:
                b.setStyleSheet("QToolButton { border:1px solid #4A2030; border-radius:4px;"
                                " background-color:#1A1016; }"
                                " QToolButton:hover { background-color:#2A1220; border-color:#FF2A6D; }")
            b.clicked.connect(cb)
            head.addWidget(b)
        self.count = QLabel("0")
        self.count.setObjectName("count_label")
        head.addWidget(self.count)
        self.box.addLayout(head)

        # ── 预览（对标友商的预览区）──
        self.preview = QLabel("（未加载）")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedHeight(140)
        self.preview.setStyleSheet(
            "QLabel { border:1px solid #2A3A5C; border-radius:4px; background:#0E1420;"
            " color:#55607A; font-size:12px; }")
        self.box.addWidget(self.preview)

        # ── 文件列表 ──
        self.list = QListWidget()
        self.list.setToolTip(f"{label} 文件列表（悬停看完整路径）")
        self.list.itemEntered.connect(self._hover)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.box.addWidget(self.list, 1)
        self._syncing = False          # 防止跨列同步时递归触发

    # ---- 交互 ----
    def pick_files(self):
        fs, _ = QFileDialog.getOpenFileNames(
            self.dlg, f"选择 {self.label} 图像", "",
            "Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff);;All (*)")
        if fs:
            self.add(fs)

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self.dlg, f"选择 {self.label} 文件夹")
        if d:
            self.add(sorted(os.path.join(d, f) for f in os.listdir(d)
                            if os.path.isfile(os.path.join(d, f))
                            and os.path.splitext(f)[1].lower() in IMG_EXTS))

    def add(self, files):
        for f in files:
            if f not in self.files:
                self.files.append(f)
        self.dlg.on_columns_changed()

    def clear(self):
        self.files = []
        self.dlg.on_columns_changed()

    def set_files(self, files):
        self.files = list(files)
        self.dlg.on_columns_changed()

    def _hover(self, item):
        p = item.data(Qt.UserRole)
        if p:
            item.setToolTip(p)

    def _on_row_changed(self, row):
        """本列选中行变化：刷新自己的预览，并让**其它列跳到同一行**。

        三视图是同一颗电芯，只有三列并排显示第 i 组，才能一眼核对配对是否正确
        （早期版本只刷新自己那一列，用户得手动点三次还容易对错号）。
        """
        self._refresh_preview()
        if not self._syncing:
            self.dlg.sync_row(self.index, row)

    def set_row_silent(self, row):
        """被其它列同步：静默设置选中行并刷新预览，不再反向触发同步。"""
        self._syncing = True
        try:
            n = self.list.count()
            if n:
                self.list.setCurrentRow(max(0, min(int(row), n - 1)))
            self._refresh_preview()
        finally:
            self._syncing = False

    def _refresh_preview(self, *_):
        row = self.list.currentRow()
        path = None
        if 0 <= row < len(self.files):
            path = self.files[row]
        elif self.files:
            path = self.files[0]
        if not path:
            self.preview.setText("（未加载）")
            self.preview.setPixmap(QPixmap())
            self.preview.setToolTip("")
            return
        group_no = row + 1 if row >= 0 else 1
        pm = QPixmap(path)
        if pm.isNull():
            self.preview.setText(f"第 {group_no} 组\n{os.path.basename(path)}（无法预览）")
            self.preview.setPixmap(QPixmap())
            self.preview.setToolTip(path)
            return
        self.preview.setPixmap(pm.scaled(QSize(self.preview.width() - 8, self.preview.height() - 8),
                                         Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview.setToolTip(f"第 {group_no} 组 · {self.label}\n{path}")

    def refresh(self):
        keep = self.list.currentRow()      # 重载后保持原来的选中位置
        self.list.blockSignals(True)
        self.list.clear()
        for i, f in enumerate(self.files):
            it = QListWidgetItem(f"{i + 1}. {os.path.basename(f)}")
            it.setData(Qt.UserRole, f)
            self.list.addItem(it)
        if self.list.count():
            first = 0 if keep < 0 else max(0, min(keep, self.list.count() - 1))
            self.list.setCurrentRow(first)
        self.list.blockSignals(False)
        self.count.setText(str(len(self.files)))
        self._refresh_preview()


class MultiViewAddDialog(QDialog):
    """多视图加图：每个视图一个槽位，可绑定成组。"""

    def __init__(self, parent=None, views=None, view_names=None,
                 title="选择图像（多视图）"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(STYLE + TOOLBTN_QSS)
        self.views = list(views or ["te_0", "te_1", "te_2"])
        self.view_names = dict(view_names or {})
        self._cols = []
        self._dropped = {}          # view -> [未配对文件名]
        self._build()
        self.setMinimumSize(960, 620)
        self.resize(1120, 700)

    def _label_of(self, v):
        cn = self.view_names.get(v)
        return f"{cn}（{v}）" if cn else v

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(16, 12, 16, 16)

        top = QHBoxLayout()
        self.bind_check = QCheckBox("绑定图片组")
        self.bind_check.setChecked(len(self.views) > 1)
        self.bind_check.setToolTip(
            "勾选后：各视图按文件名分组（xxx_te_0/1/2.jpg 归为一组），\n"
            "未凑齐的组会被剔除并列出，保证训练时每个样本的三视图齐全。")
        top.addWidget(self.bind_check)
        top.addStretch()
        tip = QLabel("命名约定：" + " / ".join(f"xxx_{v}.jpg" for v in self.views))
        tip.setStyleSheet("color:#8A93A6; font-size:12px;")
        top.addWidget(tip)
        top.addSpacing(12)
        b_all = QPushButton("加载全部图像")
        b_all.setToolTip("选一个文件夹，按 _te_N 后缀自动分到各视图槽位")
        b_all.clicked.connect(self.load_all)
        top.addWidget(b_all)
        root.addLayout(top)

        cols = QHBoxLayout()
        cols.setSpacing(12)
        for i, v in enumerate(self.views):
            c = _ViewColumn(i, v, self._label_of(v), self)
            self._cols.append(c)
            cols.addLayout(c.box, 1)
        root.addLayout(cols, 1)

        self.err = QLabel("")
        self.err.setObjectName("error_label")
        self.err.setWordWrap(True)
        self.err.setVisible(False)
        root.addWidget(self.err)

        btns = QHBoxLayout()
        btns.addStretch()
        self.ok = QPushButton("确认")
        self.ok.setObjectName("ok_btn")
        self.ok.setMinimumWidth(90)
        self.ok.clicked.connect(self._on_ok)
        cancel = QPushButton("取消")
        cancel.setObjectName("cancel_btn")
        cancel.setMinimumWidth(90)
        cancel.clicked.connect(self.reject)
        btns.addWidget(self.ok)
        btns.addWidget(cancel)
        root.addLayout(btns)

        self.bind_check.stateChanged.connect(lambda *_: self.on_columns_changed())
        self.on_columns_changed()

    # ────────── 配对 ──────────
    def _apply_binding(self, repair=False):
        """按视图后缀把各槽位文件分组；返回配对后的每组文件 + 被剔除的文件。"""
        maps = []
        for c in self._cols:
            m = {}
            for f in c.files:
                m.setdefault(stem_key(f, c.view), f)
            maps.append(m)
        if not maps:
            return [], {}
        common = set(maps[0])
        for m in maps[1:]:
            common &= set(m)
        dropped = {c.view: sorted(set(m) - common) for c, m in zip(self._cols, maps)}
        ordered = sorted(common)
        per_view = [[m[k] for k in ordered] for m in maps]
        if repair:
            for c, files in zip(self._cols, per_view):
                if c.files != files:
                    c.files = files
        return per_view, dropped

    def sync_row(self, src_index, row):
        """点某一列的某张图时，让其它列跳到同一行。

        三视图是同一颗电芯，必须三列并排显示同一组才能核对配对是否正确。
        """
        for c in self._cols:
            if c.index != src_index:
                c.set_row_silent(row)

    def on_columns_changed(self, repair=False):
        """刷新列表与校验。

        ⚠️ 默认**不改动用户的文件列表**。早期版本在每次变化时都按绑定关系修剪，
        结果是：用户往第 1 列加完图时第 2/3 列还是空的，交集为空 -> 第 1 列被整个清空
        （实测 5 个文件直接变 0）。绑定只在**确认时**真正生效，平时只做预览与提示。
        """
        for c in self._cols:
            c.refresh()
        self._validate(apply_repair=repair)

    def _validate(self, apply_repair=False):
        counts = [len(c.files) for c in self._cols]
        bind = self.bind_check.isChecked() and len(self._cols) > 1
        msg = ""
        if bind:
            per_view, dropped = self._apply_binding(repair=False)
            self._dropped = dropped
            ndrop0 = sum(len(v) for v in dropped.values())
            kept = min((len(x) for x in per_view), default=0)
            if apply_repair:
                for c, files in zip(self._cols, per_view):
                    c.files = list(files)
                for c in self._cols:
                    c.refresh()
                counts = [len(c.files) for c in self._cols]
                kept = min(counts) if counts else 0
            # 即使已经修剪过，也要把"剔除了多少"讲清楚，不能静默丢用户的图
            if ndrop0:
                msg = (f"绑定结果：可成组 {kept} 组；有 {ndrop0} 张因未凑齐将被剔除 —— "
                       + "、".join(f"{self._label_of(v)} {len(n)} 张"
                                   for v, n in dropped.items() if n))
            elif kept:
                msg = f"绑定结果：{kept} 组（各视图一一对应）"
            self.ok.setEnabled(kept > 0)
        else:
            self._dropped = {}
            if counts and counts[0] > 0:
                if len(set(counts)) > 1:
                    msg = ("各视图数量不一致（未绑定）。训练时按各视图第 i 张配对，"
                           "数量以最少的一列为准。")
                self.ok.setEnabled(min(counts) > 0)
            else:
                self.ok.setEnabled(False)
        if msg:
            self.err.setText(msg)
            self.err.setVisible(True)
        else:
            self.err.setVisible(False)

    # ────────── 批量加载 ──────────
    def load_all(self):
        d = QFileDialog.getExistingDirectory(self, "选择包含全部视图的文件夹")
        if not d:
            return
        files = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if os.path.isfile(os.path.join(d, f))
                       and os.path.splitext(f)[1].lower() in IMG_EXTS)
        buckets = {v: [] for v in self.views}
        others = []
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            hit = None
            for v in self.views:
                if stem.endswith("_" + v):
                    hit = v
                    break
            (buckets[hit] if hit else others).append(f)
        for c in self._cols:
            c.set_files(buckets.get(c.view, []))
        if others:
            self.err.setText(f"提示：{len(others)} 个文件不含视图后缀（如 _te_0），已忽略。"
                             f"示例：{os.path.basename(others[0])}")
            self.err.setVisible(True)
        self.on_columns_changed()

    def _on_ok(self):
        """确认时**才**真正按绑定关系修剪，然后检查是否还有可成组的样本。"""
        if self.bind_check.isChecked() and len(self._cols) > 1:
            self.on_columns_changed(repair=True)
        counts = [len(c.files) for c in self._cols]
        if not counts or min(counts) == 0:
            self.err.setText("每个视图至少需要加载 1 张图片。")
            self.err.setVisible(True)
            return
        if len(self._cols) > 1 and self.bind_check.isChecked() and len(set(counts)) != 1:
            return
        self.accept()

    # ────────── 结果 ──────────
    def dropped_info(self):
        """返回被剔除的文件说明，供主窗口写日志（如 ['te_0 2 张', 'te_1 1 张']）。"""
        if not self.bind_check.isChecked():
            return []
        _, dropped = self._apply_binding(repair=False)
        out = []
        for v, names in dropped.items():
            if names:
                out.append(f"{self._label_of(v)} {len(names)} 张（{'、'.join(names[:3])}"
                           + ("…" if len(names) > 3 else "") + "）")
        return out

    def get_view_paths(self):
        """返回 (views, [[第0视图的所有文件], [第1视图...], ...])，各视图已一一对应。"""
        if self.bind_check.isChecked() and len(self._cols) > 1:
            per_view, _ = self._apply_binding(repair=False)
            # 只取每组数量一致的前缀
            n = min((len(x) for x in per_view), default=0)
            return list(self.views), [x[:n] for x in per_view]
        return list(self.views), [list(c.files) for c in self._cols]
