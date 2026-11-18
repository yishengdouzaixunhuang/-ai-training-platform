# -*- coding: utf-8 -*-
"""列表表达式筛选对话框 —— 条目式（字段 / 条件 / 值 + AND·OR）。

每条一行，点「+ 添加条件」加一条。字段决定可选的条件与取值控件：
    str  字段 -> 文本输入框（包含/不包含/等于/不等于/正则/为空）
    num  字段 -> 文本输入框（= ≠ > ≥ < ≤ /为空）
    enum 字段 -> 下拉框（= ≠ /为空）—— 标记、数据集这类让用户选，不让用户打字

**不做自由文本表达式解析**：一是省掉语法与注入的坑，二是现场操作员
（AI 工程师或客户）不欠我们一套查询语法，选项直接摆出来就行。
"""
from __future__ import annotations
from core.i18n import tr

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QWidget, QScrollArea, QDialogButtonBox, QFrame, QRadioButton,
    QButtonGroup,
)

from core.filter_expr import (
    ENUM_CHOICES, FIELDS, FIELD_LABELS, FIELD_KINDS, OPS,
    FilterExpr, new_condition, ops_for,
)


class _CondRow(QWidget):
    """一行条件：字段 / 条件 / 值 / 删除。"""

    def __init__(self, on_remove, cond=None):
        super().__init__()
        self._on_remove = on_remove
        cond = cond or new_condition()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.cb_field = QComboBox()
        for key, label, _kind in FIELDS:
            self.cb_field.addItem(label, key)
        self.cb_field.setMinimumWidth(96)
        lay.addWidget(self.cb_field)

        self.cb_op = QComboBox()
        self.cb_op.setMinimumWidth(84)
        lay.addWidget(self.cb_op)

        self.ed_val = QLineEdit()
        self.ed_val.setPlaceholderText(tr("值"))
        self.ed_val.setMinimumWidth(110)
        lay.addWidget(self.ed_val, 1)
        self.cb_val = QComboBox()
        self.cb_val.setMinimumWidth(110)
        self.cb_val.setVisible(False)
        lay.addWidget(self.cb_val, 1)

        btn = QPushButton("×")
        btn.setFixedWidth(24)
        btn.setToolTip(tr("删除这条条件"))
        btn.clicked.connect(lambda: self._on_remove(self))
        lay.addWidget(btn)

        self.cb_field.currentIndexChanged.connect(self._on_field_changed)
        self.cb_op.currentIndexChanged.connect(self._sync_value_enabled)

        # 初值
        idx = self.cb_field.findData(cond.get("field", "fname"))
        self.cb_field.setCurrentIndex(idx if idx >= 0 else 0)
        self._rebuild_ops(cond.get("op"))
        self._set_value(cond.get("value", ""))

    # ── 内部 ──
    def _field(self):
        return self.cb_field.currentData() or "fname"

    def _kind(self):
        return FIELD_KINDS.get(self._field(), "str")

    def _rebuild_ops(self, want=None):
        ops = ops_for(self._field())
        self.cb_op.blockSignals(True)
        self.cb_op.clear()
        for k, label in ops:
            self.cb_op.addItem(label, k)
        if want:
            i = self.cb_op.findData(want)
            if i >= 0:
                self.cb_op.setCurrentIndex(i)
        self.cb_op.blockSignals(False)

    def _rebuild_values(self):
        """enum 字段换成下拉，其余用输入框。"""
        kind = self._kind()
        if kind == "enum":
            choices = ENUM_CHOICES.get(self._field(), [])
            self.cb_val.clear()
            for v, label in choices:
                self.cb_val.addItem(label, v)
            self.ed_val.setVisible(False)
            self.cb_val.setVisible(True)
        else:
            self.ed_val.setVisible(True)
            self.cb_val.setVisible(False)

    def _on_field_changed(self):
        self._rebuild_ops()
        self._rebuild_values()
        self._sync_value_enabled()

    def _sync_value_enabled(self):
        empty_op = self.cb_op.currentData() in ("empty", "!empty")
        self.ed_val.setEnabled(not empty_op)
        self.cb_val.setEnabled(not empty_op)
        if empty_op:
            self.ed_val.setText("")

    def _set_value(self, v):
        self._rebuild_values()
        if self._kind() == "enum":
            i = self.cb_val.findData("" if v is None else str(v))
            self.cb_val.setCurrentIndex(i if i >= 0 else 0)
        else:
            self.ed_val.setText("" if v is None else str(v))
        self._sync_value_enabled()

    def condition(self):
        op = self.cb_op.currentData() or "contain"
        if op in ("empty", "!empty"):
            val = ""
        elif self._kind() == "enum":
            val = self.cb_val.currentData() or ""
        else:
            val = self.ed_val.text().strip()
        return {"field": self._field(), "op": op, "value": val}


class FilterBuilderDialog(QDialog):
    def __init__(self, parent=None, expr=None):
        super().__init__(parent)
        self.setWindowTitle(tr("筛选条件"))
        self.resize(560, 360)
        root = QVBoxLayout(self)
        root.setSpacing(6)

        tip = QLabel("按「字段 / 条件 / 值」加条件；多条之间可选「同时满足(AND)」或「任一满足(OR)」。"
                     "\n留空的条目会被忽略。")
        tip.setWordWrap(True)
        root.addWidget(tip)

        logic_row = QHBoxLayout(); logic_row.setSpacing(10)
        self.rb_and = QRadioButton(tr("同时满足 (AND)"))
        self.rb_or = QRadioButton(tr("任一满足 (OR)"))
        self._grp = QButtonGroup(self)
        self._grp.addButton(self.rb_and); self._grp.addButton(self.rb_or)
        logic_row.addWidget(self.rb_and); logic_row.addWidget(self.rb_or)
        logic_row.addStretch(1)
        root.addLayout(logic_row)

        holder = QWidget(); self._rows_lay = QVBoxLayout(holder)
        self._rows_lay.setContentsMargins(0, 0, 0, 0)
        self._rows_lay.setSpacing(3)
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(QFrame.NoFrame)
        sa.setWidget(holder)
        root.addWidget(sa, 1)
        self._rows = []

        add = QPushButton(tr("+ 添加条件"))
        add.clicked.connect(lambda: self._add_row(new_condition()))
        root.addWidget(add)

        # 常用组合的快捷入口（现场最常用的那两条）
        quick = QHBoxLayout(); quick.setSpacing(4)
        self._quick = [
            ("筛过杀", [{"field": "mark", "op": "==", "value": "overkill"}], "AND"),
            ("筛漏杀", [{"field": "mark", "op": "==", "value": "escape"}], "AND"),
            ("过杀且高置信", [{"field": "mark", "op": "==", "value": "overkill"},
                              {"field": "score", "op": ">=", "value": "0.5"}], "AND"),
            ("有备注", [{"field": "mark_note", "op": "!empty", "value": ""}], "AND"),
        ]
        for label, conds, logic in self._quick:
            b = QPushButton(label)
            b.setToolTip(tr("一键填入该组合"))
            b.clicked.connect(lambda _c=False, cs=conds, lg=logic: self._apply_quick(cs, lg))
            quick.addWidget(b)
        quick.addStretch(1)
        root.addLayout(quick)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("应用"))
        bb.button(QDialogButtonBox.Cancel).setText(tr("取消"))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        src = expr or FilterExpr()
        for c in (src.conditions or [new_condition()]):
            self._add_row(c)
        if not src.conditions:
            self._add_row(new_condition())
        (self.rb_or if src.logic == "OR" else self.rb_and).setChecked(True)
        if not self._rows:
            self._add_row(new_condition())

    # ── 对外 ──
    def result_expr(self):
        conds = [r.condition() for r in self._rows]
        logic = "OR" if self.rb_or.isChecked() else "AND"
        return FilterExpr(conds, logic)

    # ── 内部 ──
    def _add_row(self, cond):
        row = _CondRow(self._remove_row, cond)
        self._rows_lay.addWidget(row)
        self._rows.append(row)

    def _remove_row(self, row):
        if row in self._rows:
            self._rows.remove(row)
            self._rows_lay.removeWidget(row)
            row.setParent(None)
            row.deleteLater()

    def _apply_quick(self, conds, logic):
        for r in list(self._rows):
            self._remove_row(r)
        for c in conds:
            self._add_row(c)
        (self.rb_or if logic == "OR" else self.rb_and).setChecked(True)
