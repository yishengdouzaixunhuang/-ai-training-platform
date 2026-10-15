# -*- coding: utf-8 -*-
"""图片列表的条目式表达式筛选。

对标 iCogtiveFlow 图 2-39 的筛选器：**字段 + 条件 + 值**，多条目之间用 AND/OR 连接。
友商是自由文本表达式，我们做**条目式**（下拉 + 输入框 + 条件下拉），
理由写在路线图 5.7：不做自由文本解析可以避开注入与语法坑，而且现场操作员
（AI 工程师/客户）不需要学一套查询语法，选项直接摆出来。

命中率最高的那条用法是把**标记**当一等字段：

    标记 = 过杀  AND  结果分数 ≥ 0.5      ->  "AI 很确信但人判是过杀的"
    标记 = 漏杀                           ->  "AI 放过的，补正样本"

字段与操作符按类型限定（字符串类给"包含/不包含"，数值类给比较，枚举类给等于/不等于/为空），
不让用户选出无意义的组合。
"""

from __future__ import annotations

import re

#: 操作符：key -> 中文名。字符串比较一律**不区分大小写**（文件名/类别名现场大小写混用）。
OPS = {
    "str": [
        ("contain", "包含"),
        ("!contain", "不包含"),
        ("==", "等于"),
        ("!=", "不等于"),
        ("=~", "正则匹配"),
        ("empty", "为空"),
        ("!empty", "非空"),
    ],
    "num": [
        ("==", "="),
        ("!=", "≠"),
        (">", ">"),
        (">=", "≥"),
        ("<", "<"),
        ("<=", "≤"),
        ("empty", "为空"),
        ("!empty", "非空"),
    ],
    "enum": [
        ("==", "="),
        ("!=", "≠"),
        ("empty", "为空"),
        ("!empty", "非空"),
    ],
}

OP_LABELS = {k: n for ops in OPS.values() for k, n in ops}

#: 字段：(key, 中文名, 类型)
FIELDS = [
    ("fname", "文件名", "str"),
    ("defect", "缺陷类型", "str"),
    ("mark", "标记", "enum"),
    ("mark_note", "标记备注", "str"),
    ("mark_by", "标记人", "str"),
    ("split", "数据集", "enum"),
    ("label", "标注类别", "str"),
    ("shape_count", "标注块数", "num"),
    ("pred_cnt", "结果数量", "num"),
    ("score", "结果分数", "num"),
    ("infer_ms", "推理耗时", "num"),
    ("img_w", "宽", "num"),
    ("img_h", "高", "num"),
    ("area_px", "面积(px)", "num"),
]

FIELD_LABELS = {k: n for k, n, _ in FIELDS}
FIELD_KINDS = {k: t for k, _, t in FIELDS}

#: 枚举字段的可选值（值 -> 显示名）
ENUM_CHOICES = {
    "mark": [
        ("normal", "正常"),
        ("overkill", "过杀"),
        ("escape", "漏杀"),
        ("doubt", "存疑"),
        ("", "无标记"),
    ],
    "split": [
        ("train", "Train"),
        ("val", "Val"),
        ("test", "Test"),
        ("", "未划分"),
    ],
}

LOGICS = ("AND", "OR")

_SPLIT_FROM_STATUS = {"[Train]": "train", "[Val]": "val", "[Test]": "test"}


def ops_for(field):
    return OPS.get(FIELD_KINDS.get(field, "str"), OPS["str"])


def ops_for_kind(kind):
    return OPS.get(kind, OPS["str"])


def new_condition(field="fname"):
    kind = FIELD_KINDS.get(field, "str")
    return {"field": field, "op": ops_for_kind(kind)[0][0], "value": ""}


# ───────────────────────── 取值 ─────────────────────────

def _to_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _split_of(record):
    """从记录里取数据集归属。优先显式 'split'，退回解析 split_status。"""
    s = record.get("split")
    if s:
        return str(s).lower()
    st = str(record.get("split_status") or "")
    for token, key in _SPLIT_FROM_STATUS.items():
        if token in st:
            return key
    return ""


def value_of(record, key):
    """把一条列表记录折算成该字段的值。取不到返回 ''（数值字段返回 None）。"""
    rec = record or {}
    kind = FIELD_KINDS.get(key, "str")
    if key == "fname":
        v = rec.get("fname") or ""
    elif key == "defect":
        v = rec.get("defect") or ""
    elif key == "mark":
        v = rec.get("mark") or ""
    elif key == "mark_note":
        v = rec.get("mark_note") or ""
    elif key == "mark_by":
        v = rec.get("mark_by") or ""
    elif key == "split":
        v = _split_of(rec)
    elif key == "label":
        labels = rec.get("shape_labels") or []
        v = ", ".join(str(x) for x in labels)
    elif key == "shape_count":
        v = _to_float(rec.get("shape_count"))
    elif key == "pred_cnt":
        v = _to_float(rec.get("pred_cnt"))
    elif key == "score":
        v = _to_float(rec.get("score_str"))
        if v is None:
            v = _to_float(rec.get("score"))
    elif key == "infer_ms":
        v = _to_float(rec.get("infer_ms"))
    elif key == "img_w":
        v = _to_float(rec.get("img_w"))
    elif key == "img_h":
        v = _to_float(rec.get("img_h"))
    elif key == "area_px":
        w = _to_float(rec.get("img_w")) or 0.0
        h = _to_float(rec.get("img_h")) or 0.0
        v = (w * h) if (w and h) else None
    else:
        v = rec.get(key)
    if kind == "num":
        return _to_float(v)
    return "" if v is None else str(v)


# ───────────────────────── 单条判定 ─────────────────────────

def match_one(record, cond):
    """一条条件是否命中。空值条件（value 为空且非 empty/!empty）视为不生效 -> True。"""
    if not isinstance(cond, dict):
        return True
    field = cond.get("field") or "fname"
    op = cond.get("op") or "contain"
    want = cond.get("value", "")
    kind = FIELD_KINDS.get(field, "str")

    if kind == "num":
        got = value_of(record, field)
        if op == "empty":
            return got is None
        if op == "!empty":
            return got is not None
        w = _to_float(want)
        if w is None or got is None:
            # 数值条件填了非数字、或该条记录没有这个值 -> 不命中
            # （"没有值"不等于"值为0"：把没跑过推理的图当成耗时0去比 "<100" 会误命中）
            return False
        if op == "==":
            return abs(got - w) < 1e-9
        if op == "!=":
            return abs(got - w) >= 1e-9
        if op == ">":
            return got > w
        if op == ">=":
            return got >= w
        if op == "<":
            return got < w
        if op == "<=":
            return got <= w
        return True

    got = value_of(record, field)
    if op == "empty":
        return got == ""
    if op == "!empty":
        return got != ""
    w = "" if want is None else str(want)
    if op in ("empty", "!empty"):
        return True
    if w == "":
        # 字符串/枚举条件没填值 -> 不生效
        return True
    g = got.lower()
    wl = w.lower()
    if op == "contain":
        return wl in g
    if op == "!contain":
        return wl not in g
    if op == "==":
        return g == wl
    if op == "!=":
        return g != wl
    if op == "=~":
        try:
            return re.search(w, got, re.IGNORECASE) is not None
        except re.error:
            # 正则写错不该让整个列表筛不出来 —— 当作不命中并保持其余条件生效
            return False
    return True


# ───────────────────────── 表达式 ─────────────────────────

class FilterExpr:
    """条目式筛选表达式：conditions + 连接词。"""

    def __init__(self, conditions=None, logic="AND"):
        self.conditions = [dict(c) for c in (conditions or []) if isinstance(c, dict)]
        self.logic = logic if logic in LOGICS else "AND"

    # -- 增删改 --
    def add(self, field="fname", op=None, value=""):
        c = {"field": field, "op": op or ops_for(field)[0][0], "value": value}
        self.conditions.append(c)
        return c

    def remove(self, idx):
        if 0 <= idx < len(self.conditions):
            self.conditions.pop(idx)

    def clear(self):
        self.conditions = []

    # -- 判定 --
    def is_empty(self):
        """没有任何**有效**条件（有效 = 值非空，或 op 是 empty/!empty）。"""
        return not self.active_conditions()

    def active_conditions(self):
        out = []
        for c in self.conditions:
            if not isinstance(c, dict):
                continue
            if c.get("op") in ("empty", "!empty") or str(c.get("value", "")) != "":
                out.append(c)
        return out

    def active_count(self):
        return len(self.active_conditions())

    def evaluate(self, record):
        act = self.active_conditions()
        if not act:
            return True
        if self.logic == "OR":
            return any(match_one(record, c) for c in act)
        return all(match_one(record, c) for c in act)

    #: 别名：语义更直白
    matches = evaluate

    def filter(self, records):
        return [r for r in (records or []) if self.evaluate(r)]

    # -- 序列化 --
    def to_dict(self):
        return {"version": 1, "logic": self.logic,
                "conditions": [dict(c) for c in self.conditions]}

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            return cls()
        conds = d.get("conditions")
        if not isinstance(conds, list):
            conds = []
        keep = []
        for c in conds:
            if not isinstance(c, dict):
                continue
            f = c.get("field")
            if f not in FIELD_KINDS:
                continue
            op = c.get("op") or "contain"
            if op not in [k for k, _ in ops_for(f)]:
                op = ops_for(f)[0][0]
            keep.append({"field": f, "op": op, "value": c.get("value", "")})
        return cls(keep, d.get("logic", "AND"))

    # -- 展示 --
    def describe(self):
        act = self.active_conditions()
        if not act:
            return ""
        joiner = " 且 " if self.logic == "AND" else " 或 "
        parts = []
        for c in act:
            f = FIELD_LABELS.get(c.get("field"), c.get("field"))
            o = OP_LABELS.get(c.get("op"), c.get("op"))
            v = c.get("value", "")
            if c.get("op") in ("empty", "!empty"):
                parts.append("%s%s" % (f, o))
            else:
                parts.append("%s %s %s" % (f, o, v))
        return joiner.join(parts)
