# -*- coding: utf-8 -*-
"""数据集划分对话框（SplitDialog）。

**为什么这个文件必须存在**：`ui/main_window.py` 的 `_manage_split()`（Tools 菜单 +
图片列表右键两条入口）一直在 `from ui.split_dialog import SplitDialog`，而仓库和
git 全历史里**从来没有**这个模块 ⇒ 菜单一点就是 ModuleNotFoundError
（`crash_log.txt` 2026-10-21 14:55 实录）。这里是补上的实现。

────────────────────────────────────────────────────────────────────
数据契约（三个地方必须对齐，改任何一处都要同步这三处）
  · 文件：`<工程>/train_test_split.json`，形如 `{"<base>": "train"|"val"|"test"}`
  · 键  ：多视图工程用**首视图**文件名（`xxx_te_0`），单视图用文件名（`xxx`）
          —— 见 `training/multiview.py:128`（先查 `{key}_te_0` 再查 `{key}`）
             与 `ui/main_window.py` 状态列的 `sm.get(base)`
  · 单位：**组**，不是张。同一颗电芯的 te_0/te_1/te_2 必须同属一个集，
          否则同一场景既在训练又在验证 —— 指标虚高，而且看不出来。
────────────────────────────────────────────────────────────────────
两个容易踩的语义（本对话框的文案必须说实话，不能装作没这回事）

  1. **未划分 = 不属于任何集（空集）**，训练/验证/测试都取不到它。
     这是 2026-10-22 按用户要求改的语义 —— 之前的实现会在内存里给所有已标注样本
     补默认 `train`，于是"没划分"= "偷偷进训练集"，而界面又显示成空集。
     现在训练侧（`training/dataset.py`、`training/multiview.py`、
     `classification/dataset.py`、`classification/dual_dataset.py`）一律
     "查不到就取不到"，训练集为空会直接报错并指路到本对话框。
     ⇒ 老工程升级后如果划分表是空的，训练会停下来说"训练集为空"，
        用下面的「未划分 → 训练集」一键补齐即可。
  2. **已标注才能进训练**。取数门票是"有 json 或 mask"（`get_train_test_split`），
     没标注的组即使划了 split 也进不了训练。所以随机划分默认**只划已标注的组**，
     并且把"未标注 N 组"单独报出来。
"""
from core.i18n import tr
import json
import os
import random
import re
import shutil
import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# `xxx_te_0` -> ("xxx", "te_0")；没有视图后缀的返回 (stem, "")
_VIEW_RE = re.compile(r"^(?P<group>.+)_(?P<view>te_\d+)$")

SPLIT_LABEL = {"train": "训练", "val": "验证", "test": "测试", "": "未划分"}
SPLIT_KEYS = ("train", "val", "test")

# 「未划分」用语义色（橙）。⚠️ 不能用主题色白名单里的值（#00E5FF 等）——
# `ui/themes.py: sweep_inline_styles` 会把主题色当皮肤色剥掉，浅色主题下就看不见了。
COL_UNSET = "#E8A33D"


def _group_view(stem):
    """把一个文件名拆成 (组名, 视图后缀)。"""
    m = _VIEW_RE.match(stem)
    if m:
        return m.group("group"), m.group("view")
    return stem, ""


class SplitDialog(QDialog):
    """按**组**管理 train/val/test 划分。"""

    def __init__(self, project_dir, parent=None):
        super().__init__(parent)
        self.pd = str(project_dir)
        self.setWindowTitle("数据集划分 - %s" % os.path.basename(self.pd.rstrip("\\/")))
        self.resize(900, 640)
        self.setModal(True)

        self._rows = []          # 全部行（不受搜索框影响）
        self._view_idx = []      # 表格第 i 行 -> self._rows 的下标
        self._split_map = {}     # 盘上的原始划分表
        self._views = ()
        self._mv = False
        self._cls_mapped = set()  # 分类工程的 class_labels.json 映射（也算"已标注"）
        self._dirty = False
        self._backup_done = False
        self._img_dir = os.path.join(self.pd, "images")

        self._build_ui()
        self.reload()

    # ────────────────────────── 界面 ──────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        self.lbl_path = QLabel(self.pd)
        self.lbl_path.setToolTip(tr("划分表写到这个目录下的 train_test_split.json"))
        root.addWidget(self.lbl_path)

        note = QLabel(
            "按「组」划分：同一颗料的 te_0/te_1/te_2 三视图必须同属一个集。\n"
            "「未划分」= 不属于任何集（空集）：训练、验证、测试都不会取它。"
            "所以想让一张图进训练，必须显式设为训练集。")
        note.setWordWrap(True)
        root.addWidget(note)

        # 摘要
        self.lbl_summary = QLabel("")
        f = self.lbl_summary.font()
        f.setBold(True)
        self.lbl_summary.setFont(f)
        root.addWidget(self.lbl_summary)

        # 工具行：搜索 / 刷新 / 全选 / 反选
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText(tr("按组名过滤…"))
        self.ed_search.textChanged.connect(lambda *_: self._fill_table())
        bar.addWidget(self.ed_search, 1)
        self.btn_reload = QPushButton(tr("刷新"))
        self.btn_reload.setToolTip(tr("重新扫描 images/ 与 train_test_split.json"))
        self.btn_reload.clicked.connect(self.reload)
        bar.addWidget(self.btn_reload)
        self.btn_all = QPushButton(tr("全选"))
        self.btn_all.clicked.connect(lambda: self.table.selectAll())
        bar.addWidget(self.btn_all)
        self.btn_inv = QPushButton(tr("反选"))
        self.btn_inv.clicked.connect(self._invert_selection)
        bar.addWidget(self.btn_inv)
        root.addLayout(bar)

        # 表
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([tr("组名"), tr("视图"), tr("已标注"), tr("当前划分")])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setHighlightSections(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 64)
        self.table.setColumnWidth(2, 64)
        self.table.setColumnWidth(3, 96)
        self.table.verticalHeader().setDefaultSectionSize(24)
        root.addWidget(self.table, 1)

        # 批量设置选中项
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(QLabel(tr("选中项设为：")))
        for key in SPLIT_KEYS + ("",):
            b = QPushButton(SPLIT_LABEL[key] if key else "未划分")
            b.setToolTip("把选中的组设为「%s」（可 Ctrl/Shift 多选）" % SPLIT_LABEL[key])
            b.clicked.connect(lambda _c=False, k=key: self._set_selected(k))
            row1.addWidget(b)
        # 老工程升级用：以前"没划分"会被训练器偷偷当训练集，现在不了 ——
        # 划分表为空的工程一训练就停，这里给一条显式的迁移路径。
        self.btn_unassigned_train = QPushButton(tr("未划分 → 训练集"))
        self.btn_unassigned_train.setToolTip(
            tr("把所有「未划分」的组一次性设为训练集（只动未划分的，已划好的不碰）"))
        self.btn_unassigned_train.clicked.connect(self._assign_unassigned_train)
        row1.addWidget(self.btn_unassigned_train)
        row1.addStretch()
        root.addLayout(row1)

        # 按比例随机
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(QLabel(tr("按比例随机：")))
        self.sp_train = self._ratio_spin(70)
        self.sp_val = self._ratio_spin(15)
        self.sp_test = self._ratio_spin(15)
        for lbl, sp in (("训练", self.sp_train), ("验证", self.sp_val), ("测试", self.sp_test)):
            row2.addWidget(QLabel(lbl))
            row2.addWidget(sp)
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 999999)
        self.sp_seed.setToolTip(tr("同一个种子 + 同一批组 = 同一种划分（可复现）"))
        row2.addWidget(QLabel(tr("种子")))
        row2.addWidget(self.sp_seed)
        self.chk_ann = QCheckBox(tr("只划分已标注的组"))
        self.chk_ann.setChecked(True)
        self.chk_ann.setToolTip(tr("没有 json / mask 的组进不了训练，划了也没用"))
        row2.addWidget(self.chk_ann)
        self.btn_random = QPushButton(tr("执行随机划分"))
        self.btn_random.clicked.connect(self._random_split)
        row2.addWidget(self.btn_random)
        row2.addStretch()
        root.addLayout(row2)

        # 状态 + 底栏
        self.lbl_state = QLabel("")
        root.addWidget(self.lbl_state)

        bottom = QHBoxLayout()
        bottom.addStretch()
        self.btn_save = QPushButton(tr("保存"))
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self._on_save)
        bottom.addWidget(self.btn_save)
        self.btn_cancel = QPushButton(tr("取消"))
        self.btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(self.btn_cancel)
        root.addLayout(bottom)

    @staticmethod
    def _ratio_spin(value):
        sp = QSpinBox()
        sp.setRange(0, 100)
        sp.setSuffix("%")
        sp.setValue(value)
        sp.setFixedWidth(72)
        return sp

    # ────────────────────────── 读盘 ──────────────────────────

    def reload(self):
        """重扫目录 + 重读划分表，然后重画。未保存的改动会被丢弃（点刷新前有确认）。"""
        if self._dirty:
            r = QMessageBox.question(
                self, tr("刷新"), tr("还有未保存的改动，刷新会丢弃它们。继续？"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        self._split_map = self._read_split_map()
        self._cls_mapped = self._read_cls_mapping()
        self._views, self._mv = self._project_views()
        self._rows = self._scan_rows()
        self._dirty = False
        self._fill_table()
        self._update_summary()

    def _read_split_map(self):
        p = os.path.join(self.pd, "train_test_split.json")
        if not os.path.exists(p):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                return {str(k): str(v) for k, v in d.items()}
        except Exception:
            pass
        return {}

    def _read_cls_mapping(self):
        """分类工程的 `annotations/class_labels.json` 里出现过的图，也算已标注。"""
        p = os.path.join(self.pd, "annotations", "class_labels.json")
        if not os.path.exists(p):
            return set()
        try:
            with open(p, "r", encoding="utf-8") as f:
                mp = json.load(f).get("mapping", {}) or {}
        except Exception:
            return set()
        out = set()
        for rel in mp:
            rel = str(rel).replace("\\", "/")
            if rel.startswith("images/"):
                rel = rel[len("images/"):]
            out.add(os.path.splitext(rel)[0])
            out.add(os.path.splitext(os.path.basename(rel))[0])
        return out

    def _project_views(self):
        """(views, 是否多视图)。单视图返回 ((), False)。"""
        try:
            from training.multiview_gui import is_multiview_project, project_views
            if is_multiview_project(self.pd):
                views, _names = project_views(self.pd)
                return tuple(views), len(views) >= 2
        except Exception:
            pass
        return (), False

    def _list_images(self):
        """images/ 顶层的图。顶层为空时兜底扫一层子目录（分类工程把图放在 images/<类别>/）。"""
        img_dir = self._img_dir
        if not os.path.isdir(img_dir):
            return []
        out = []
        for f in sorted(os.listdir(img_dir)):
            fp = os.path.join(img_dir, f)
            if os.path.isfile(fp) and f.lower().endswith(IMG_EXT):
                out.append(fp)
        if out:
            return out
        for sub in sorted(os.listdir(img_dir)):
            d = os.path.join(img_dir, sub)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                fp = os.path.join(d, f)
                if os.path.isfile(fp) and f.lower().endswith(IMG_EXT):
                    out.append(fp)
        return out

    def _has_ann(self, path, rel_stem):
        base = os.path.basename(rel_stem)
        if base in self._cls_mapped or rel_stem in self._cls_mapped:
            return True
        if os.path.exists(os.path.join(self.pd, "annotations", base + "_mask.png")):
            return True
        d = os.path.dirname(path)
        if os.path.exists(os.path.join(d, base + ".json")):
            return True
        if os.path.exists(os.path.join(d, base + "_det.json")):
            return True
        return False

    def _split_of(self, base, stems, group):
        """当前划分：先查规范键，再查同组其它文件名与组名（老数据可能是别的键）。"""
        cands = [base, group] + list(stems)
        # 分类工程是 ImageFolder 布局时，键带子目录前缀（`ok/a`）；
        # 老数据也可能只写裸文件名（`a`）⇒ 两种都要认，否则明明划过了却显示未划分。
        for c in list(cands):
            b = os.path.basename(c)
            if b and b != c:
                cands.append(b)
        for cand in cands:
            if not cand:
                continue
            for k in (cand, "images/" + cand):
                v = self._split_map.get(k)
                if v in SPLIT_KEYS:
                    return v
        return ""

    def _scan_rows(self):
        """按「组」扫行。键采用 images/ 下的相对路径（扁平布局就是文件名，
        分类工程的 ImageFolder 布局是 `类别/文件名` —— 必须与
        `classification/dataset._list_images_mapping` 的 rel_key 一致，
        否则对话框写了 `a`、训练侧查 `ok/a`，明明划分过却一张都取不到）。"""
        groups = {}
        for p in self._list_images():
            # 键 = images/ 下的相对路径去扩展名：扁平布局 "a"，ImageFolder 布局 "ok/a"
            rel = os.path.splitext(os.path.relpath(p, self._img_dir))[0].replace("\\", "/")
            sub = os.path.dirname(rel)
            g, v = _group_view(os.path.basename(rel))
            gkey = ("%s/%s" % (sub, g)) if sub else g
            groups.setdefault(gkey, []).append((v, rel, p))

        rows = []
        order = {v: i for i, v in enumerate(self._views)}
        for g in sorted(groups):
            items = sorted(groups[g], key=lambda t: (order.get(t[0], 99), t[1]))
            stems = [t[1] for t in items]
            base = stems[0]                      # 首视图 = 划分表规范键
            if self._mv:
                got = len({t[0] for t in items if t[0] in order})
            else:
                got = 1
            rows.append({
                "group": g,
                "base": base,
                "stems": stems,
                "files": [t[2] for t in items],
                "n_views": got,
                "has_ann": any(self._has_ann(t[2], t[1]) for t in items),
                "split": self._split_of(base, stems, g),
            })
        return rows

    # ────────────────────────── 画面 ──────────────────────────

    def _visible_rows(self):
        kw = (self.ed_search.text() or "").strip().lower()
        if not kw:
            return [(i, r) for i, r in enumerate(self._rows)]
        return [(i, r) for i, r in enumerate(self._rows) if kw in r["group"].lower()]

    def _fill_table(self):
        sel_bases = {self._rows[i]["base"] for i in self._selected_row_idx()}
        pairs = self._visible_rows()
        self._view_idx = [i for i, _r in pairs]
        # ⚠️ 多选恢复不能用 QTableWidget.selectRow 逐个调：它会**清掉前面的选择**，
        #    刷新后只剩最后一行选中（用户会以为自己的选择丢了）。必须走 selectionModel。
        self.table.clearSelection()
        self.table.setRowCount(len(pairs))
        restore = []
        for tr, (_i, r) in enumerate(pairs):
            it_g = QTableWidgetItem(r["group"])
            it_g.setToolTip(os.path.basename(r["files"][0]) if r["files"] else "")
            self.table.setItem(tr, 0, it_g)

            txt_views = ("%d/%d" % (r["n_views"], len(self._views))) if self._mv else "1"
            it_v = QTableWidgetItem(txt_views)
            it_v.setTextAlignment(Qt.AlignCenter)
            if self._mv and r["n_views"] < len(self._views):
                it_v.setToolTip(tr("缺视图，训练会跳过这一组"))
                it_v.setForeground(Qt.red)
            self.table.setItem(tr, 1, it_v)

            it_a = QTableWidgetItem("是" if r["has_ann"] else "否")
            it_a.setTextAlignment(Qt.AlignCenter)
            if not r["has_ann"]:
                it_a.setToolTip(tr("没有 json / mask ⇒ 训练不会取到这一组"))
                try:
                    from ui.themes import fg_muted
                    it_a.setForeground(fg_muted())
                except Exception:
                    pass
            self.table.setItem(tr, 2, it_a)

            it_s = QTableWidgetItem(SPLIT_LABEL[r["split"]])
            it_s.setTextAlignment(Qt.AlignCenter)
            if not r["split"]:
                from PyQt5.QtGui import QColor
                it_s.setForeground(QColor(COL_UNSET))
                it_s.setToolTip(tr("未划分：不属于任何集 —— 训练/验证/测试都不会取它"))
            self.table.setItem(tr, 3, it_s)
            if r["base"] in sel_bases:
                restore.append(tr)
        if restore:
            from PyQt5.QtCore import QItemSelectionModel
            sm = self.table.selectionModel()
            for tr in restore:
                sm.select(self.table.model().index(tr, 0),
                          QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self._update_state()

    def _selected_row_idx(self):
        """选中的表格行 -> self._rows 下标。"""
        rows = {i.row() for i in self.table.selectionModel().selectedRows()} \
            if self.table.selectionModel() else set()
        return [self._view_idx[r] for r in sorted(rows) if 0 <= r < len(self._view_idx)]

    def _invert_selection(self):
        sel = set(self._selected_row_idx())
        self.table.clearSelection()
        for tr, idx in enumerate(self._view_idx):
            if idx not in sel:
                self.table.selectRow(tr)
        self._update_state()

    def _update_summary(self):
        n = len(self._rows)
        cnt = {k: 0 for k in SPLIT_KEYS}
        cnt[""] = 0
        for r in self._rows:
            cnt[r["split"]] = cnt.get(r["split"], 0) + 1
        n_noann = sum(1 for r in self._rows if not r["has_ann"])
        n_missing = sum(1 for r in self._rows
                        if self._mv and r["n_views"] < len(self._views))
        parts = ["共 %d 组" % n,
                 "训练 %d" % cnt["train"], "验证 %d" % cnt["val"],
                 "测试 %d" % cnt["test"], "未划分 %d（不进任何集）" % cnt[""]]
        if n_noann:
            parts.append("未标注 %d（进不了训练）" % n_noann)
        if n_missing:
            parts.append("缺视图 %d（训练会跳过）" % n_missing)
        self.lbl_summary.setText(" · ".join(parts))

    def _update_state(self):
        n_sel = len(self._selected_row_idx())
        n_dirty = sum(1 for r in self._rows
                      if r["split"] != self._split_of(r["base"], r["stems"], r["group"]))
        self._dirty = n_dirty > 0
        bits = []
        if n_sel:
            bits.append("已选 %d 组" % n_sel)
        if n_dirty:
            bits.append("已改动 %d 组（点「保存」才写入 train_test_split.json）" % n_dirty)
        self.lbl_state.setText("  ".join(bits))
        for w in (self.btn_save,):
            w.setEnabled(n_dirty > 0)
        self._update_summary()

    # ────────────────────────── 操作 ──────────────────────────

    def _set_selected(self, value):
        idx = self._selected_row_idx()
        if not idx:
            QMessageBox.information(self, tr("数据集划分"),
                                    tr("先在表里选中要设置的组（Ctrl / Shift 可多选，或点「全选」）。"))
            return
        for i in idx:
            self._rows[i]["split"] = value
        self._fill_table()

    def _assign_unassigned_train(self):
        rows = [r for r in self._rows if not r["split"]]
        if not rows:
            QMessageBox.information(self, tr("未划分 → 训练集"), tr("没有未划分的组，不用动。"))
            return
        n_noann = sum(1 for r in rows if not r["has_ann"])
        r = QMessageBox.question(
            self, tr("未划分 → 训练集"),
            "把 %d 组「未划分」一次性设为训练集？\n\n"
            "· 只动未划分的，已划到训练/验证/测试的一律不碰；\n"
            "· 其中 %d 组没有标注，设了也进不了训练（取数要求有 json 或 mask）。"
            % (len(rows), n_noann),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        for row in rows:
            row["split"] = "train"
        self._fill_table()

    def _random_split(self):
        tr, va, te = self.sp_train.value(), self.sp_val.value(), self.sp_test.value()
        if tr + va + te != 100:
            QMessageBox.warning(self, tr("比例不对"),
                                "训练 + 验证 + 测试 必须等于 100%%，当前是 %d%%。" %
                                (tr + va + te))
            return
        idx = self._selected_row_idx()
        targets = [self._rows[i] for i in idx] if idx else list(self._rows)
        if self.chk_ann.isChecked():
            targets = [r for r in targets if r["has_ann"]]
        n = len(targets)
        if n == 0:
            QMessageBox.information(self, tr("没有可划分的组"),
                                    "当前范围内没有%s组。" %
                                    ("已标注的" if self.chk_ann.isChecked() else ""))
            return
        # 覆盖提醒：val/test 是"评测的唯一依据"，被无声改写等于历史指标失真
        changed = [r for r in targets if r["split"] and r["split"] != "train"]
        if changed:
            r = QMessageBox.question(
                self, tr("会覆盖已有划分"),
                "这次随机划分会覆盖 %d 组已有的划分，其中 %d 组现在在验证/测试里。\n\n"
                "验证集/测试集是评测的唯一依据，被换掉之后新旧模型就没法公平对比了。\n"
                "继续？" % (len(targets), len(changed)),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return

        n_te = int(round(n * te / 100.0))
        n_va = int(round(n * va / 100.0))
        if n_te + n_va > n:
            n_te = min(n_te, n)
            n_va = min(n_va, n - n_te)
        n_tr = n - n_te - n_va
        # 非零比例却分到 0 组时，从训练集让一组出来（否则"设了 15% 验证集"等于没设）
        if n >= 10:
            if va > 0 and n_va == 0 and n_tr > 1:
                n_va, n_tr = 1, n_tr - 1
            if te > 0 and n_te == 0 and n_tr > 1:
                n_te, n_tr = 1, n_tr - 1

        order = list(targets)
        random.Random(self.sp_seed.value()).shuffle(order)
        for r in order[:n_te]:
            r["split"] = "test"
        for r in order[n_te:n_te + n_va]:
            r["split"] = "val"
        for r in order[n_te + n_va:]:
            r["split"] = "train"
        self._fill_table()
        QMessageBox.information(
            self, tr("随机划分完成"),
            "%d 组：训练 %d / 验证 %d / 测试 %d（种子 %d）\n\n"
            "还没写盘，确认无误后点「保存」。" %
            (n, n_tr, n_va, n_te, self.sp_seed.value()))

    # ────────────────────────── 保存 ──────────────────────────

    def _on_save(self):
        try:
            n = self._save()
        except Exception as e:
            QMessageBox.critical(self, tr("保存失败"),
                                 "写 train_test_split.json 出错：\n%s: %s" %
                                 (type(e).__name__, e))
            return
        self._saved_rows = n
        self.accept()

    def _save(self):
        """只改本工程这些组对应的键，表里没出现的旧键一律保留。"""
        from training.dataset import save_train_test_split

        raw = dict(self._read_split_map())
        other_bases = {r["base"] for r in self._rows}
        changed = 0
        for r in self._rows:
            key = r["base"]
            v = r["split"]
            # 老数据可能用组名 / 第二、三视图名做键：一并清掉，避免同一组两个键打架
            for alias in [r["group"]] + list(r["stems"]):
                if alias and alias != key and alias in raw and alias not in other_bases:
                    del raw[alias]
            if v in SPLIT_KEYS:
                if raw.get(key) != v:
                    changed += 1
                raw[key] = v
            else:
                if key in raw:
                    changed += 1
                raw.pop(key, None)

        path = os.path.join(self.pd, "train_test_split.json")
        if os.path.exists(path) and not self._backup_done:
            # 动用户数据前留一份原始划分表（划分一旦丢了，"哪些图进过测试集"就无从考证）
            try:
                shutil.copy2(path, path + ".bak_" + time.strftime("%Y%m%d-%H%M%S"))
            except Exception:
                pass
            self._backup_done = True
        save_train_test_split(self.pd, raw)
        self._split_map = raw
        self._dirty = False
        return changed
