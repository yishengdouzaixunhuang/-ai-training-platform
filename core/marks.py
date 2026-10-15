# -*- coding: utf-8 -*-
"""图片标记 —— 对标 iCogtiveFlow 的 [正常]/[过杀]/[漏杀]/[存疑] 四态。

## 为什么要单独一条记录

iCogtiveFlow 说明书 P32 原文：

    支持人为对图片进行标记，可以进行 [正常]、[过杀]、[漏杀]、[存疑] 四种标记，
    **该标记不影响训练，只做标记使用**。

最后那句是整个设计的关键。如果把「过杀」当成一个类别塞进训练集，模型就会去学
"过杀长什么样" 这种无意义的概念 —— 过杀不是缺陷类型，是**质检口径的沉淀**。
所以标记与标签**物理分离**：

    label  -> 进训练集（annotations/<base>_mask.png + images/<base>.json）
    mark   -> 不进训练集，只做统计、筛选与难例闭环的入口（本文件的 image_marks.json）

落盘格式刻意与 train_test_split.json 平行（同目录、同 json、同一层扁平映射），
这样"划分"和"标记"两类操作在工程目录里长得一样，运维好理解：

    <project>/image_marks.json
    {
      "version": 1,
      "marks": {
        "20260910-4号机-凹点1_te_0": {
          "mark": "overkill",
          "note": "凹点，人工判为过杀（背景干净）",
          "by": "Arthur",
          "at": "2026-10-14T10:20:00"
        }
      }
    }

## 过杀 / 漏杀 的判定口径（Arthur 2026-10-14 拍板）

**以模型的识别效果为准，不以客户标准为准**：

    干净的背景，模型识别出了结果            -> 过杀 overkill
    疑似标注过的前景，模型却没有抓出        -> 漏杀 escape

也就是说判定锚点是「人工标注(前景/背景) ⇄ 模型输出」的**差集**，
而不是某个外部规格表。所以本模块提供 suggest_mark() —— 标记可以自动预填，
人工只做确认（现场每天几千张图，全靠手点不现实）。

注意这条口径与 255/ignore 的关系：过杀既然人工认定是**背景**，那就该让模型在
那里学"背景"，因此闭环动作是把该区域清成背景(0)，**不是**标成 __ignore__(255)。
（255 在 loss 里是 ignore_index，标成 255 等于告诉模型"这里别管" ——
模型拿不到信号，下次还会在同一处幻觉。理由见 annotation/convert.py。）
"""

from __future__ import annotations

import json
import os
from datetime import datetime

MARKS_FILENAME = "image_marks.json"

#: 四态。顺序即 UI 按钮顺序，也是统计表的列顺序。
MARK_KINDS = ("normal", "overkill", "escape", "doubt")

MARK_LABELS = {
    "normal": "正常",
    "overkill": "过杀",
    "escape": "漏杀",
    "doubt": "存疑",
}

#: 列表行色块 / 统计表配色。
#: 过杀用橙、漏杀用红 —— 与现场语义一致（漏检是底线问题，最刺眼）。
MARK_COLORS = {
    "normal": "#2FA84F",
    "overkill": "#E8912A",
    "escape": "#E03A3A",
    "doubt": "#8E5BD0",
    "": "#3A4256",          # 无标记（灰）
}

LABEL_TO_KIND = {v: k for k, v in MARK_LABELS.items()}

#: 提示文案（列表 tooltip / 统计面板共用）
MARK_HINTS = {
    "normal": "人工确认与模型一致（对）",
    "overkill": "干净背景被模型检出（假阳，判定锚点=标注背景）",
    "escape": "标注过的前景模型没抓出（假阴）",
    "doubt": "拿不准，先挂起来 —— 单列统计，不混进正类训练集",
}

_EMPTY_RECORD = {"mark": "", "note": "", "by": "", "at": ""}


# ───────────────────────── 落盘 ─────────────────────────

def marks_path(project_dir):
    return os.path.join(project_dir, MARKS_FILENAME)


def load_marks(project_dir):
    """读标记。缺失/损坏一律回空 dict（绝不因一个坏文件让工程打不开）。

    兼容两种形态：{"version":1,"marks":{...}} 与扁平的 {base: {...}}。
    """
    if not project_dir:
        return {}
    p = marks_path(project_dir)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    raw = d.get("marks") if isinstance(d.get("marks"), dict) else d
    out = {}
    for base, rec in (raw or {}).items():
        if base == "version" or not isinstance(rec, dict):
            continue
        kind = normalize_kind(rec.get("mark"))
        if not kind:
            continue
        out[str(base)] = {
            "mark": kind,
            "note": str(rec.get("note") or ""),
            "by": str(rec.get("by") or ""),
            "at": str(rec.get("at") or ""),
        }
    return out


def save_marks(project_dir, marks):
    if not project_dir:
        return False
    try:
        os.makedirs(project_dir, exist_ok=True)
        payload = {"version": 1, "marks": {k: dict(v) for k, v in (marks or {}).items()}}
        # 原子落盘：先写临时文件再 os.replace —— 中途崩了不会留下半个 JSON，
        # 而半个 JSON 会让下一次 load_marks 静默返回 {}（标记全丢且不报错）。
        tmp = marks_path(project_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, marks_path(project_dir))
        return True
    except Exception:
        return False


# ───────────────────────── 增删查 ─────────────────────────

def normalize_kind(kind):
    """把任意输入收敛成合法 kind；认不出来返回 ''。中文名也认。"""
    if not kind:
        return ""
    k = str(kind).strip()
    if k in MARK_KINDS:
        return k
    if k in LABEL_TO_KIND:
        return LABEL_TO_KIND[k]
    low = k.lower()
    if low in MARK_KINDS:
        return low
    return ""


def kind_label(kind):
    return MARK_LABELS.get(normalize_kind(kind), "")


def kind_color(kind):
    return MARK_COLORS.get(normalize_kind(kind), MARK_COLORS[""])


def get_mark(marks, base):
    """取某图的标记记录（没有则返回空记录，不返回 None —— 调用方少一层判空）。"""
    rec = (marks or {}).get(str(base))
    if not isinstance(rec, dict):
        return dict(_EMPTY_RECORD)
    return {
        "mark": normalize_kind(rec.get("mark")),
        "note": str(rec.get("note") or ""),
        "by": str(rec.get("by") or ""),
        "at": str(rec.get("at") or ""),
    }


def set_mark(marks, bases, kind, note=None, by=None, when=None):
    """给一批图打标记（就地改 marks）。返回实际改动的条数。

    note 传 None = 保留原有备注（只改标记时不想把备注冲掉）；
    note 传 ""   = 明确清空备注。
    """
    k = normalize_kind(kind)
    if not k:
        raise ValueError("非法标记: %r（可选 %s）" % (kind, list(MARK_KINDS)))
    if by is None:
        by = _current_user()
    if when is None:
        when = datetime.now().isoformat(timespec="seconds")
    n = 0
    for b in (bases or []):
        b = str(b)
        if not b:
            continue
        old = marks.get(b) or {}
        rec = {
            "mark": k,
            "note": (str(old.get("note") or "") if note is None else str(note)),
            "by": by,
            "at": when,
        }
        if rec != old:
            n += 1
        marks[b] = rec
    return n


def set_note(marks, bases, note, by=None, when=None):
    """只改备注，不动标记（标记为空时不写入 —— 没有标记的备注是孤儿数据）。"""
    if by is None:
        by = _current_user()
    if when is None:
        when = datetime.now().isoformat(timespec="seconds")
    n = 0
    for b in (bases or []):
        b = str(b)
        rec = marks.get(b)
        if not isinstance(rec, dict) or not normalize_kind(rec.get("mark")):
            continue
        if str(rec.get("note") or "") != str(note):
            rec["note"] = str(note)
            rec["by"] = by
            rec["at"] = when
            marks[b] = rec
            n += 1
    return n


def clear_marks(marks, bases):
    """清空一批图的标记（删记录，不留空壳）。返回删除条数。"""
    n = 0
    for b in (bases or []):
        if str(b) in (marks or {}):
            marks.pop(str(b), None)
            n += 1
    return n


def _current_user():
    try:
        import getpass
        return getpass.getuser() or ""
    except Exception:
        return ""


# ───────────────────────── 统计 ─────────────────────────

def count_marks(marks_or_records):
    """四态计数。接受 marks 映射（base->rec）或记录列表（每条含 'mark'）。"""
    out = {k: 0 for k in MARK_KINDS}
    out[""] = 0        # 无标记
    items = marks_or_records.values() if isinstance(marks_or_records, dict) else (marks_or_records or [])
    for rec in items:
        k = normalize_kind(rec.get("mark") if isinstance(rec, dict) else rec)
        out[k if k else ""] += 1
    return out


def mark_ratios(cnt):
    """把计数转成占比（以已标记总数为分母，不含无标记）。"""
    marked = sum(v for k, v in (cnt or {}).items() if k)
    out = {}
    for k in MARK_KINDS:
        n = int((cnt or {}).get(k, 0) or 0)
        out[k] = (n, (n / marked) if marked else 0.0)
    return out


def mark_summary(cnt):
    """"已标记 12（正常5/过杀4/漏杀2/存疑1）" 这样的一行。"""
    if not cnt:
        return "标记：0"
    marked = sum(v for k, v in cnt.items() if k)
    if not marked:
        return "标记：0"
    parts = "/".join("%s%d" % (MARK_LABELS[k], cnt.get(k, 0)) for k in MARK_KINDS if cnt.get(k, 0))
    return "已标记 %d（%s）" % (marked, parts)


def cross_table(records, type_key="defect"):
    """缺陷类型 × 标记 交叉表。

    records: 记录列表，每条需含 'mark' 与类型字段（默认 'defect'，即缺陷名）。
    返回 {类型: {kind: n, "": n}}，类型按总数降序、空类型记为「(无缺陷名)」。
    """
    table = {}
    for r in (records or []):
        t = str(r.get(type_key) or "").strip() or "(无缺陷名)"
        k = normalize_kind(r.get("mark"))
        row = table.setdefault(t, {kk: 0 for kk in MARK_KINDS})
        row[""] = row.get("", 0)
        if k:
            row[k] += 1
        else:
            row[""] += 1
    # 排序：按已标记数降序，其次按总数
    def _key(item):
        t, row = item
        marked = sum(v for kk, v in row.items() if kk)
        return (-marked, -sum(row.values()), t)
    return dict(sorted(table.items(), key=_key))


# ───────────────────────── 自动预判（按 Arthur 口径） ─────────────────────────

def suggest_mark(pred_cnt, shape_count, has_ann=False, overlap=None):
    """按"以模型识别效果为准"的口径给出建议标记；判不出来返回 ''。

    规则（Arthur 2026-10-14 口径）：
      · 干净背景被模型检出      -> overkill    (pred>0 且无人工前景标注)
      · 标注过的前景模型没抓出  -> escape      (有人工前景标注 且 pred==0)
      · 两者都有且重叠足够      -> normal（仅在传入 overlap 时判定，否则不猜）
      · 两者都有但完全不重叠    -> overkill（模型命中的是别处，不是这条前景）

    overlap: 可选，pred 掩膜与标注掩膜的 IoU（None = 不知道，就不猜 normal）。
      ⚠️ 不传 overlap 时**绝不**把"两者都有"判成 normal —— 那属于"AI 有检出、
      人也有标注"，但两者可能根本不在一个位置，猜 normal 会把真过杀伪装成正常。
    """
    try:
        p = int(pred_cnt or 0)
        s = int(shape_count or 0)
    except (TypeError, ValueError):
        return ""
    if p > 0 and s == 0:
        return "overkill"
    if s > 0 and p == 0:
        return "escape"
    if p > 0 and s > 0 and overlap is not None:
        try:
            ov = float(overlap)
        except (TypeError, ValueError):
            return ""
        if ov >= 0.5:
            return "normal"
        if ov <= 0.0:
            return "overkill"
    return ""


def is_consistent(mark, pred_cnt, shape_count):
    """标记与(AI 检出, 人工标注)是否自洽。用于统计面板标出可疑条目。

    返回 (bool, 说明)。空标记恒为自洽（没标记就没有矛盾）。
    """
    k = normalize_kind(mark)
    if not k:
        return True, ""
    try:
        p = int(pred_cnt or 0)
        s = int(shape_count or 0)
    except (TypeError, ValueError):
        return True, ""
    if k == "overkill" and s > 0 and p <= 0:
        return False, "标了过杀，但这张图没有 AI 检出（反而更像漏杀）"
    if k == "escape" and p > 0 and s <= 0:
        return False, "标了漏杀，但这张图没有人工前景标注"
    if k == "normal" and (p <= 0 or s <= 0):
        return False, "标了正常，但 AI 检出与人工标注没有同时存在"
    return True, ""


def records_from_marks(marks, bases=None, type_of=None):
    """把 marks 映射摊成记录列表（给 cross_table / 统计用）。

    type_of: 可调用，base -> 缺陷类型名；缺省用 base 名里的缺陷段猜。
    """
    out = []
    items = (marks or {}).items()
    if bases is not None:
        want = set(str(b) for b in bases)
        items = [(k, v) for k, v in items if k in want]
    for base, rec in items:
        out.append({
            "base": base,
            "mark": rec.get("mark", ""),
            "note": rec.get("note", ""),
            "defect": (type_of(base) if callable(type_of) else defect_of(base)),
        })
    return out


def defect_of(base):
    """从 base 名猜缺陷类型（命名规范 {date}-{机号/料号}-{面?}-{缺陷}{N}_te_{n}）。

    猜法是"尽力而为"：拿到就按它分组，拿不到归 (无缺陷名)。
    真正的缺陷名应当由调用方从标注/类别表里给出（records_from_marks 的 type_of）。
    """
    s = str(base or "")
    # 去掉 te_0/1/2 与尾部序号
    for suf in ("_te_0", "_te_1", "_te_2"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    parts = [p for p in s.split("-") if p != ""]
    if not parts:
        return ""
    tail = parts[-1]
    import re
    tail = re.sub(r"\d+$", "", tail)
    return tail or ""
