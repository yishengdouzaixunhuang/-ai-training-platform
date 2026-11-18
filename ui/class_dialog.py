# -*- coding: utf-8 -*-
"""类别管理对话框（ClassDialog）。

**为什么这个文件必须存在**：`ui/main_window.py` 的 `_manage_classes()` 一直在
`from ui.class_dialog import ClassDialog`，而仓库和 git 全历史里从来没有这个模块
⇒ 菜单「类别管理…」一点就 ModuleNotFoundError。这里是补上的实现。

────────────────────────────────────────────────────────────────────
这个对话框**不自己造轮子**：表格直接用右栏那张 `ui/class_table.ClassTableView`
（序号 / 类别名 / 颜色 / 训练集 / 测试集 / 验证集 / 空集 七列，数字来自扫标注 JSON
与 train_test_split.json），所以两处看到的数字天生一致。

行为与右栏那张表**逐条对齐**（改这里请同步 `ui/main_window.py` 里的
`_add_class` / `_delete_class` / `_rename_class` / `_move_class`）：

  · 改动**立即写 project.json**（没有「取消」按钮 —— 说"取消了"但盘上已经变了
    比不做更糟）。关掉对话框只决定要不要让主窗刷新。
  · **编号就是掩膜值**：类别表的下标 = 掩膜 PNG 里的像素值，background 恒 0，
    255 是 ignore。所以"删除中间一项"或"上移/下移"会让后续类别**整体前移一位**，
    而已有掩膜里的数字不会跟着改 ⇒ 那些区域此后被解释成**别的类别**。
    这两种操作必须先弹窗说清楚（照抄主窗已经写好的那段文案，别再自创一句
    "Existing annotations will keep their class index" —— 那恰好把最危险的后果
    说成了"没事"）。
  · **改名要连带迁移推理参数**（`inference.params.rename_class_params`），
    否则新名字查不到参数、悄悄退回默认阈值；**删除要摘掉残项**（`drop_class_params`）。
"""
from core.i18n import tr
import json
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton,
    QVBoxLayout,
)

from ui.class_table import ClassTableView


class ClassDialog(QDialog):
    """类别增删改移。所有改动即时落盘到 <工程>/project.json。"""

    def __init__(self, project_dir, parent=None):
        super().__init__(parent)
        self.pd = str(project_dir)
        self.setWindowTitle("类别管理 - %s" % os.path.basename(self.pd.rstrip("\\/")))
        self.resize(760, 560)
        self.setModal(True)

        self._classes = []
        self._dirty = False
        self._disk_changed = False   # 盘上的 project.json 是否已被本对话框改过
        self._discovered_n = 0

        self._build_ui()
        self.reload(discover=True)

    # ────────────────────────── 界面 ──────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        lbl_path = QLabel(self.pd)
        lbl_path.setToolTip(tr("类别表写到这个目录下的 project.json"))
        root.addWidget(lbl_path)

        note = QLabel(
            "类别表的下标就是掩膜里的像素值（background 恒 0，255 = ignore）。\n"
            "删除中间一项、或上移/下移，都会让后面的类别整体前移一位，已有标注不会跟着改。")
        note.setWordWrap(True)
        # ⚠️ 高度必须按**实际换行后的行数**算，不能拍"两行"：
        #    这个 label 下面紧贴 ClassTableView 自带的工具栏，高度差几像素，
        #    第二行就会被压掉半截（渲染截图实测：末句"不会跟着改"被切）。
        note.setMinimumHeight(
            note.fontMetrics().boundingRect(
                0, 0, 720, 10000, Qt.TextWordWrap, note.text()).height() + 6)
        root.addWidget(note)

        self.table = ClassTableView(self)
        self.table.refresh_requested.connect(lambda: self.reload(discover=True))
        self.table.add_requested.connect(self._add)
        self.table.delete_requested.connect(self._delete)
        self.table.rename_requested.connect(self._rename)
        self.table.move_up_requested.connect(lambda: self._move(-1))
        self.table.move_down_requested.connect(lambda: self._move(1))
        self.table.currentRowChanged.connect(lambda *_: self._update_state())
        # 点「训练集/测试集/验证集/空集」数字格子：在主窗里会去筛图片列表，
        # 这里对话框是模态的、筛了也看不见 ⇒ 不接，免得用户以为点了没反应。
        root.addWidget(self.table, 1)

        self.lbl_state = QLabel("")
        self.lbl_state.setWordWrap(True)
        root.addWidget(self.lbl_state)

        bottom = QHBoxLayout()
        bottom.addStretch()
        self.btn_close = QPushButton(tr("关闭"))
        self.btn_close.setDefault(True)
        self.btn_close.clicked.connect(self._finish)
        bottom.addWidget(self.btn_close)
        root.addLayout(bottom)

    # ────────────────────────── 读写 ──────────────────────────

    def reload(self, discover=False):
        """重读类别表。discover=True 时按主窗同样的口径从标注 JSON 补类别。

        ⚠️ `merge_classes` 是**会写盘**的（补进来的类别直接落 project.json）。
        所以"只是打开看了一眼"也可能已经改了盘 —— 这里必须把这笔记成
        `_disk_changed`，否则关闭时返回 Rejected、主窗不刷新，
        盘上是新表、界面上是旧表，两边各说各的。
        """
        self._discovered_n = 0
        if discover:
            try:
                from core.class_discovery import discover_labels, merge_classes
                discovered = discover_labels(self.pd)
                self._classes, added = merge_classes(self.pd, discovered)
                if added:
                    self._discovered_n = len(added)
                    self._disk_changed = True
            except Exception:
                from core.class_discovery import read_project_classes
                self._classes = read_project_classes(self.pd)
        else:
            from core.class_discovery import read_project_classes
            self._classes = read_project_classes(self.pd)
        self.table.set_project_dir(self.pd, with_counts=False)   # 下面 set_classes 会算
        self.table.set_classes(list(self._classes))
        self._update_state()

    def _save(self):
        """把 self._classes 写回 project.json（保留文件里的其它键）。"""
        path = os.path.join(self.pd, "project.json")
        meta = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        meta["classes"] = list(self._classes)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        self._dirty = True
        self._disk_changed = True
        self.table.set_classes(list(self._classes))

    def _update_state(self):
        row = self.table.currentRow()
        bits = ["共 %d 个类别（编号 0-%d）" % (len(self._classes), max(0, len(self._classes) - 1))]
        if 0 <= row < len(self._classes):
            bits.append("当前：%s = 掩膜值 %d" % (self._classes[row], row))
        if self._discovered_n:
            bits.append("已从标注 JSON 补入 %d 个类别" % self._discovered_n)
        bits.append("改动即时写入 project.json" if self._dirty else "尚未改动")
        self.lbl_state.setText("  ·  ".join(bits))

    # ────────────────────────── 四个动作 ──────────────────────────

    def _add(self):
        name, ok = QInputDialog.getText(self, tr("添加类别"), tr("类别名："))
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            return
        if name in self._classes:
            QMessageBox.warning(self, tr("重复"), "类别『%s』已经在表里了。" % name)
            return
        if name == "background":
            QMessageBox.warning(self, tr("不能用"), tr("background 是背景类，恒占 0 号位。"))
            return
        self._classes = list(self._classes) + [name]
        self._save()
        self.table.setCurrentRow(len(self._classes) - 1, emit=False)
        self._update_state()

    def _delete(self):
        row = self.table.currentRow()
        if row < 0:
            # ⚠️ "没选中"和"选中了 background"是两回事，不能都说成"background 不能删"
            QMessageBox.information(self, tr("删除类别"), tr("先在表里点一行，选中要删的类别。"))
            return
        if row == 0:
            QMessageBox.information(self, tr("删除类别"), tr("background 是背景类，不能删除。"))
            return
        classes = list(self._classes)
        if row >= len(classes):
            return
        name = classes[row]
        n_tail = len(classes) - row - 1
        if n_tail == 0:
            warn = ("『%s』是最后一个类别，其余类别的编号一个都不变，"
                    "已有标注不受影响。" % name)
        else:
            warn = ("⚠️ 『%s』后面还有 %d 个类别，删掉它们会整体前移一位；"
                    "而已有标注是按编号存的、不会跟着改，"
                    "编号 ＞ %d 的区域此后会被当成别的类别（界面上看不出来）。"
                    % (name, n_tail, row))
        reply = QMessageBox.question(self, tr("删除类别"),
                                     "确定删除类别『%s』？\n\n%s\n\n"
                                     "（掩膜文件本身不会被改动；background 不能删）"
                                     % (name, warn),
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        classes.pop(row)
        self._classes = classes
        self._save()
        try:
            from inference.params import drop_class_params
            if drop_class_params(self.pd, name):
                self._dirty = True
        except Exception:
            pass
        self.table.setCurrentRow(-1, emit=False)
        self._update_state()

    def _rename(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, tr("重命名"), tr("先在表里点一行，选中要改名的类别。"))
            return
        classes = list(self._classes)
        if row >= len(classes):
            return
        old = classes[row]
        new, ok = QInputDialog.getText(self, tr("重命名类别"), tr("新名字："), text=old)
        if not ok:
            return
        new = (new or "").strip()
        if not new or new == old:
            return
        if new in classes:
            QMessageBox.warning(self, tr("重复"), "类别『%s』已经在表里了。" % new)
            return
        classes[row] = new
        self._classes = classes
        self._save()
        # 推理参数按**名字**查：不搬的话新名字查不到 → 悄悄退回默认八维阈值
        migrated = False
        try:
            from inference.params import rename_class_params
            migrated = bool(rename_class_params(self.pd, old, new))
        except Exception:
            migrated = False
        self.table.setCurrentRow(row, emit=False)
        self._update_state()
        msg = "已改名：%s → %s" % (old, new)
        if migrated:
            msg += "（推理参数已跟着迁移）"
        # 说实话：标注 JSON 里的 label 还是旧名，下次打开工程会被 discovery 当成新类别补回来
        msg += ("\n\n注意：已有标注文件（labelme json）里写的还是旧名『%s』。"
                "主窗下次「重读并重扫标注」时，旧名会被当成没登记的类别再补一行；"
                "要彻底改掉，需要同步改标注里的 label（本对话框不动标注文件）。" % old)
        QMessageBox.information(self, tr("重命名完成"), msg)

    def _move(self, delta):
        row = self.table.currentRow()
        if row < 0:
            return
        new_row = row + delta
        classes = list(self._classes)
        if not (0 <= new_row < len(classes)):
            return
        if row == 0 or new_row == 0:
            QMessageBox.information(self, tr("移动类别"), tr("background 恒占 0 号位，不能动它。"))
            return
        classes[row], classes[new_row] = classes[new_row], classes[row]
        self._classes = classes
        self._save()
        self.table.setCurrentRow(new_row, emit=False)
        self._update_state()
        self.lbl_state.setText(
            "已移动「%s」→ 编号 %d。⚠️ 编号变了，已有掩膜里的数字不会跟着改，"
            "那些区域此后按新编号解释。" % (classes[new_row], new_row))

    # ────────────────────────── 收尾 ──────────────────────────

    def _finish(self):
        # 盘上动过就返回 Accepted —— 主窗据此刷新（重扫标注约 1 秒，不白跑）
        self.done(QDialog.Accepted if self._disk_changed else QDialog.Rejected)
