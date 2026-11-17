# -*- coding: utf-8 -*-
"""一期六项自测 —— 五态标记与难例闭环（DL Studio 对标 iCogtiveFlow）。

覆盖（对应路线图 V4「一期」1.1~1.6）：
    1.1 标记数据模型        A 组
    1.4 列表表达式筛选      B 组
    1.6 八维推理过滤        C 组
    1.5 结果转标注/难例闭环  D 组 + E 组（端到端验收链路）
    1.2/1.3 标记 UI     -> F 组（AST 级源码核验，UI 交互靠 probe 脚本）
    AST 核验             F 组

原则（沿用本项目既有铁律）：
  · 一律走**真实产品路径**，不手造中间产物：
      标注用 annotation.labelme_io 的 writer；
      预测掩膜/检测框用 flow.nodes.output 的 SegResultWriter / DetResultWriter
      （就是界面里批处理写 outputs/ 的那两个节点，不是我自己拼文件名）。
  · 源码核验一律用 **AST**，不用字符串包含 —— 注释能骗过字符串包含断言，
    而且是以 PASS 的形式骗人（比 FAIL 更危险）。
  · 环境缺依赖（如没装 torch）判 **INCONCLUSIVE**，不进失败计数，但醒目打出。
  · 断言从**临界值本身**开测（例如 max_det=2 就造 3 个连通域，而不是造 1 个）。

用法：
    python selftest_marks.py            # 全部
    python selftest_marks.py A B        # 只跑指定组
"""

import ast
import io
import json
import os
import sys
import tempfile
import time
import traceback

# 2026-11-17：本脚本从仓库根搬进 tests/manual/ 后，ROOT 指向了 manual/，
# 导致 sys.path 少一层、且 `ROOT/ui/main_window.py` 这类源码核验路径全断。
# 统一指向**仓库根**（下面的源码核验路径都按仓库根拼）。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image

# torch 提前预热：它和 cv2 各自捆绑的运行时 DLL 会打架，
# 等到 E 组（cv2 已加载很久之后）再 import torch，本机会报
#   OSError [WinError 1114] 动态链接库(DLL)初始化例程失败: ...\torch\lib\c10.dll
# 放在最前面先把它拉起来就没事。缺 torch 的机器不受影响（后面会判 INCONCLUSIVE）。
try:
    import torch as _torch_warm  # noqa: F401
except Exception:
    pass

# 工作目录放系统临时目录、**按运行唯一化**，全程不做任何删除动作：
# 本机装了 safe-delete 钩子，会 fail-closed 拦掉 shutil.rmtree / os.remove，
# 删不掉就会出现"上一轮的残留文件影响这一轮结果"（而且不报错）——
# 干脆不删，用唯一目录保证干净。
WORK = os.path.join(tempfile.gettempdir(),
                    "selftest_marks_%d_%d" % (os.getpid(), int(time.time())))

PASS, FAIL, INCON = "PASS", "FAIL", "INCONCLUSIVE"
_results = []


def ck(name, cond, detail=""):
    _results.append((PASS if cond else FAIL, name, detail))
    return bool(cond)


def incon(name, detail=""):
    _results.append((INCON, name, detail))


def section(t):
    _results.append(("SECTION", t, ""))


# ══════════════════════════ 夹具 ══════════════════════════

def fresh_project(name, classes=("background", "凹点"), size=(160, 120)):
    """造一个真工程目录（含 project.json / images / annotations / outputs）。

    不删除已有目录（安全删除钩子会拦），改用带运行序号的唯一目录名。
    """
    d = os.path.join(WORK, name)
    for sub in ("images", "annotations", "models", "outputs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    with io.open(os.path.join(d, "project.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "task_type": "语义分割", "classes": list(classes)},
                  f, ensure_ascii=False, indent=2)
    return d


def make_image(project_dir, base, size=(160, 120), bg=40):
    """造一张真图（灰度渐变的背景，便于按灰度过滤时能区分）。"""
    w, h = size
    arr = np.full((h, w), bg, dtype=np.uint8)
    for x in range(w):
        arr[:, x] = np.clip(bg + (x * 90 // max(1, w - 1)), 0, 255)
    p = os.path.join(project_dir, "images", base + ".png")
    Image.fromarray(arr).save(p)
    return p


def write_ann(project_dir, base, mask, class_names=("background", "凹点")):
    """用产品 writer 写标注（JSON + 掩膜 PNG）。"""
    from annotation.labelme_io import save_mask_to_json
    img = os.path.join(project_dir, "images", base + ".png")
    return save_mask_to_json(mask, img, project_dir, list(class_names), simplify_epsilon=1.0)


def write_pred(project_dir, base, mask):
    """用产品 writer（flow 的输出节点）写预测掩膜到 outputs/<base>_pred.png。"""
    from flow.frame import Frame
    from flow.nodes.output import SegResultWriter
    img = os.path.join(project_dir, "images", base + ".png")
    im = np.array(Image.open(img).convert("RGB"))
    fr = Frame(image=im, path=img, result={"mask": np.asarray(mask, dtype=np.uint8)})
    if not getattr(fr, "base", ""):
        fr.base = base
    node = SegResultWriter(out_dir=os.path.join(project_dir, "outputs"),
                           save_mask=True, save_overlay=False)
    node.run(fr)
    return os.path.join(project_dir, "outputs", base + "_pred.png")


def write_pred_boxes(project_dir, base, boxes, classes=("凹点",)):
    """用产品 writer（DetResultWriter）写预测检测框到 outputs/<base>_det.json。"""
    from flow.frame import Frame
    from flow.nodes.output import DetResultWriter
    img = os.path.join(project_dir, "images", base + ".png")
    im = np.array(Image.open(img).convert("RGB"))
    fr = Frame(image=im, path=img, result={"boxes": [dict(b) for b in boxes]})
    if not getattr(fr, "base", ""):
        fr.base = base
    node = DetResultWriter(out_dir=os.path.join(project_dir, "outputs"),
                           classes=",".join(classes))
    node.run(fr)
    return os.path.join(project_dir, "outputs", base + "_det.json")


def rect_mask(size, x, y, w, h, val=1):
    m = np.zeros((size[1], size[0]), dtype=np.uint8)
    m[y:y + h, x:x + w] = val
    return m


# ══════════════════════════ A. 标记数据模型 ══════════════════════════

def group_A():
    section("A. 标记数据模型（1.1）")
    from core import marks as mk

    pd = fresh_project("A_marks")

    ck("A1 未落盘的工程读标记 -> 空", mk.load_marks(pd) == {})

    marks = {}
    bases = ["cell_001_te_0", "cell_002_te_0"]
    mk.set_mark(marks, bases, "overkill", note="凹点，人工判为背景", by="me")
    mk.set_mark(marks, ["cell_003_te_0"], "escape", by="me")
    ck("A2 set_mark 写入内存", set(marks) == set(bases + ["cell_003_te_0"]))
    ck("A3 set_mark 落盘", mk.save_marks(pd, marks))
    back = mk.load_marks(pd)
    ck("A4 save/load 往返内容一致", back == marks, "back=%s" % sorted(back))
    ck("A5 备注与人/时间都在", all(back[b]["note"] == "凹点，人工判为背景" or b == "cell_003_te_0"
                                 for b in back)
       and all(back[b]["by"] == "me" and back[b]["at"] for b in back))
    ck("A6 原子落盘无 .tmp 残留", not os.path.exists(mk.marks_path(pd) + ".tmp"))

    # note=None 必须保留原备注（只改标记时不想把理由冲掉）
    mk.set_mark(marks, ["cell_001_te_0"], "doubt")
    ck("A7 只改标记（note=None）时保留原备注",
       marks["cell_001_te_0"]["note"] == "凹点，人工判为背景"
       and marks["cell_001_te_0"]["mark"] == "doubt", str(marks.get("cell_001_te_0")))
    mk.set_mark(marks, ["cell_001_te_0"], "doubt", note="")
    ck("A7b note='' 才清空备注", marks["cell_001_te_0"]["note"] == "")

    n = mk.set_note(marks, ["cell_001_te_0"], "改过的理由")
    ck("A8 set_note 改备注", n == 1 and marks["cell_001_te_0"]["note"] == "改过的理由")
    ck("A9 set_note 对无标记的图不写孤儿备注",
       mk.set_note(marks, ["not_marked"], "随便写点") == 0 and "not_marked" not in marks)

    n = mk.clear_marks(marks, ["cell_003_te_0"])
    ck("A10 clear_marks 删除记录", n == 1 and "cell_003_te_0" not in marks)

    # 坏文件容忍
    with io.open(mk.marks_path(pd), "w", encoding="utf-8") as f:
        f.write("{ 这不是 json")
    ck("A11 坏 JSON 不炸、静默回空", mk.load_marks(pd) == {})

    try:
        mk.set_mark({}, ["x"], "nonsense")
        ck("A12 非法 kind 抛 ValueError", False)
    except ValueError:
        ck("A12 非法 kind 抛 ValueError", True)

    cnt = mk.count_marks({"a": {"mark": "overkill"}, "b": {"mark": "overkill"},
                          "c": {"mark": "escape"}, "d": {}})
    ck("A13 count_marks 计数正确",
       cnt["overkill"] == 2 and cnt["escape"] == 1 and cnt[""] == 1 and cnt["normal"] == 0)
    ratios = mk.mark_ratios(cnt)
    ck("A14 占比分母=已标记（3）", abs(ratios["overkill"][1] - 2 / 3) < 1e-9)
    ck("A15 mark_summary 可读", "过杀2" in mk.mark_summary(cnt), mk.mark_summary(cnt))

    cross = mk.cross_table([
        {"defect": "凹点", "mark": "overkill"},
        {"defect": "凹点", "mark": "overkill"},
        {"defect": "划痕", "mark": "escape"},
        {"defect": "划痕", "mark": ""},
    ])
    ck("A16 交叉表分组正确",
       cross["凹点"]["overkill"] == 2 and cross["划痕"]["escape"] == 1 and cross["划痕"][""] == 1)
    ck("A17 交叉表按已标记数降序", list(cross)[0] == "凹点", list(cross))

    # suggest_mark：维护者 2026-10-14 口径
    ck("A18 干净背景被检出 -> 过杀", mk.suggest_mark(3, 0) == "overkill")
    ck("A19 标注前景没抓出 -> 漏杀", mk.suggest_mark(0, 2) == "escape")
    ck("A20 两者都有但不给 overlap -> 不猜 normal", mk.suggest_mark(3, 2) == "")
    ck("A21 两者都有且重叠 0.8 -> 正常", mk.suggest_mark(3, 2, overlap=0.8) == "normal")
    ck("A22 两者都有但完全不重叠 -> 过杀", mk.suggest_mark(3, 2, overlap=0.0) == "overkill")
    ck("A23 都没有 -> 不猜", mk.suggest_mark(0, 0) == "")

    ck("A24 标过杀却有漏杀特征 -> 判不自洽", mk.is_consistent("overkill", 0, 2)[0] is False)
    ck("A25 标漏杀却有检出无标注 -> 判不自洽", mk.is_consistent("escape", 2, 0)[0] is False)
    ck("A26 空标记恒自洽", mk.is_consistent("", 0, 0)[0] is True)
    ck("A27 正常且两者都有 -> 自洽", mk.is_consistent("normal", 1, 1)[0] is True)

    ck("A28 从命名规范取缺陷段", mk.defect_of("20260910-4号机-凹点3_te_0") == "凹点",
       mk.defect_of("20260910-4号机-凹点3_te_0"))
    ck("A29 带面的命名也能取到", mk.defect_of("20260910-5号机-大面-划痕12_te_1") == "划痕",
       mk.defect_of("20260910-5号机-大面-划痕12_te_1"))


# ══════════════════════════ B. 表达式筛选 ══════════════════════════

def _rec(**kw):
    """按主窗口 _filter_images 产出记录的同一套字段构造一条记录。"""
    r = {"fname": "", "base": "", "defect": "", "mark": "", "mark_note": "", "mark_by": "",
         "split": "", "split_status": "", "shape_labels": [], "shape_count": 0,
         "pred_cnt": 0, "score_str": "", "infer_ms": "", "img_w": 0, "img_h": 0}
    r.update(kw)
    return r


def group_B():
    section("B. 表达式筛选（1.4）")
    from core.filter_expr import FilterExpr, match_one, value_of, FIELD_KINDS, new_condition

    r = _rec(fname="20260910-4号机-凹点3_te_0.png", base="20260910-4号机-凹点3_te_0",
             defect="凹点", mark="overkill", mark_note="背景干净", mark_by="me",
             split="train", split_status=" [Train]", shape_labels=["凹点", "划痕"],
             shape_count=2, pred_cnt=5, score_str="0.87", infer_ms="42",
             img_w=160, img_h=120)

    ck("B1 defect 字段", value_of(r, "defect") == "凹点")
    ck("B2 mark 字段", value_of(r, "mark") == "overkill")
    ck("B3 label 字段合并多类别", value_of(r, "label") == "凹点, 划痕")
    ck("B4 score 从 score_str 解析", abs(value_of(r, "score") - 0.87) < 1e-9)
    ck("B5 infer_ms 解析为数值", value_of(r, "infer_ms") == 42.0)
    ck("B6 area_px 由宽高算", value_of(r, "area_px") == 160 * 120)
    ck("B7 split 直接取", value_of(r, "split") == "train")
    ck("B7b split 缺失时回退解析 split_status",
       value_of(_rec(split_status=" [Val]"), "split") == "val")

    ck("B8 contain 命中", match_one(r, {"field": "fname", "op": "contain", "value": "凹点"}))
    ck("B9 contain 大小写不敏感",
       match_one(_rec(fname="Cell_A.TIF"), {"field": "fname", "op": "contain", "value": "cell_a"}))
    ck("B10 !contain", match_one(r, {"field": "fname", "op": "!contain", "value": "划痕"}))
    ck("B11 == 精确", match_one(r, {"field": "mark", "op": "==", "value": "overkill"}))
    ck("B12 != 精确", match_one(r, {"field": "mark", "op": "!=", "value": "escape"}))
    ck("B13 正则匹配", match_one(r, {"field": "fname", "op": "=~", "value": r"凹点\d"}))
    ck("B14 非法正则不抛、判不命中",
       match_one(r, {"field": "fname", "op": "=~", "value": "("}) is False)

    ck("B15 num > 生效", match_one(r, {"field": "pred_cnt", "op": ">", "value": 4}))
    ck("B16 num >= 临界值命中（从临界值开测）",
       match_one(r, {"field": "pred_cnt", "op": ">=", "value": 5}))
    ck("B17 num <= 临界值命中",
       match_one(r, {"field": "score", "op": "<=", "value": 0.87}))
    ck("B18 num == 浮点容差",
       match_one(r, {"field": "pred_cnt", "op": "==", "value": 5}))
    ck("B19 num != 浮点容差",
       match_one(r, {"field": "pred_cnt", "op": "!=", "value": 6}))

    # 空值语义：没有耗时 ≠ 耗时 0
    r0 = _rec(fname="a.png", infer_ms="")
    ck("B20 无耗时不被当成 0（<100 不命中）",
       match_one(r0, {"field": "infer_ms", "op": "<", "value": 100}) is False)
    ck("B21 empty 判空", match_one(r0, {"field": "infer_ms", "op": "empty"}))
    ck("B22 !empty 判非空", match_one(r, {"field": "infer_ms", "op": "!empty"}))
    ck("B23 枚举 empty（无标记）",
       match_one(_rec(mark=""), {"field": "mark", "op": "empty"}))
    ck("B24 字符串 empty（无备注）",
       match_one(_rec(mark_note=""), {"field": "mark_note", "op": "empty"}))
    ck("B25 数值条件填非数字 -> 不命中",
       match_one(r, {"field": "pred_cnt", "op": ">", "value": "abc"}) is False)
    ck("B26 字符串条件留空 -> 不生效（恒真）",
       match_one(r, {"field": "fname", "op": "contain", "value": "  "}) or
       match_one(r, {"field": "fname", "op": "contain", "value": ""}))

    # AND / OR
    f_and = FilterExpr([{"field": "mark", "op": "==", "value": "overkill"},
                        {"field": "score", "op": ">=", "value": 0.5}], "AND")
    ck("B27 AND 全真才真", f_and.evaluate(r))
    f_and2 = FilterExpr([{"field": "mark", "op": "==", "value": "overkill"},
                         {"field": "score", "op": ">=", "value": 0.95}], "AND")
    ck("B28 AND 一假即假", f_and2.evaluate(r) is False)
    f_or = FilterExpr([{"field": "mark", "op": "==", "value": "escape"},
                       {"field": "score", "op": ">=", "value": 0.95}], "OR")
    ck("B29 OR 一真即真", f_or.evaluate(r) is False)   # 两条都假
    f_or2 = FilterExpr([{"field": "mark", "op": "==", "value": "overkill"},
                        {"field": "score", "op": ">=", "value": 0.95}], "OR")
    ck("B30 OR 命中一条即真", f_or2.evaluate(r))

    # 空表达式
    empty = FilterExpr()
    ck("B31 空表达式 is_empty", empty.is_empty())
    ck("B32 空表达式恒真（不筛掉任何图）", empty.evaluate(_rec()) is True)
    only_blank = FilterExpr([{"field": "fname", "op": "contain", "value": ""}])
    ck("B33 只有空条件的表达式视为空", only_blank.is_empty())
    only_empty_op = FilterExpr([{"field": "mark_note", "op": "empty", "value": ""}])
    ck("B34 empty 操作符算有效条件", only_empty_op.is_empty() is False)

    # 序列化
    rt = FilterExpr.from_dict(f_and.to_dict())
    ck("B35 to_dict/from_dict 往返", rt.logic == "AND" and len(rt.conditions) == 2
       and rt.conditions[1]["value"] == 0.5, rt.to_dict())
    bad = FilterExpr.from_dict({"logic": "XOR", "conditions": [
        {"field": "不存在的字段", "op": "==", "value": 1},
        {"field": "mark", "op": "不存在的操作符", "value": "overkill"},
    ]})
    ck("B36 非法字段被剔除", all(c["field"] != "不存在的字段" for c in bad.conditions))
    ck("B37 非法操作符被收敛为合法值",
       bad.conditions and bad.conditions[0]["op"] in [k for k, _ in
                                                     __import__("core.filter_expr",
                                                                fromlist=["ops_for"]).ops_for(
                                                         bad.conditions[0]["field"])])
    ck("B38 非法 logic 收敛为 AND", bad.logic == "AND")
    ck("B39 describe 可读", "标记" in f_and.describe() and "分数" in f_and.describe(),
       f_and.describe())

    # 批量 filter
    rows = [r, _rec(mark="escape", defect="划痕"), _rec(mark="overkill", defect="划痕")]
    ck("B40 filter 批量返回 2 条", len(FilterExpr(
        [{"field": "mark", "op": "==", "value": "overkill"}]).filter(rows)) == 2)

    ck("B41 字段表含「标记」（一等字段）", "mark" in FIELD_KINDS and "mark_note" in FIELD_KINDS)
    ck("B42 new_condition 默认合法",
       new_condition("mark")["op"] in [k for k, _ in __import__(
           "core.filter_expr", fromlist=["ops_for"]).ops_for("mark")])


# ══════════════════════════ C. 八维推理过滤 ══════════════════════════

def group_C():
    section("C. 八维推理过滤（1.6）")
    from inference import params as P

    keys = [k for k, _n, _d in P.RANGE_KEYS]
    ck("C1 按类别七维齐全（含半径）",
       keys == ["score", "area", "width", "height", "angle", "gray", "radius"], keys)
    ck("C2 全局含中心半径（第 8 维）", "center_radius" in P.GLOBAL_DEFAULTS)
    d = P.default_class_params()
    ck("C3 默认全开（各维 on=True）", all(d[P.ENABLE_KEY][k] is True for k in keys), d[P.ENABLE_KEY])

    # 向后兼容：造一份**旧版**参数（没有 on、没有 radius）
    pd = fresh_project("C_params", classes=("background", "凹点"))
    old = {"version": 1,
           "global": {"hole_fill": 0, "center_radius": 0, "sort": "面积降序", "max_det": 99999},
           "classes": {"凹点": {"score": [0.0, -1.0], "area": [5.0, -1e9],
                                "width": [1.0, -1e9], "height": [1.0, -1e9],
                                "angle": [-180.0, -180.0], "gray": [0.0, -255.0]}}}
    with io.open(P.params_path(pd), "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False)
    lp = P.load_params(pd, ["background", "凹点"])
    ck("C4 旧参数加载不炸", isinstance(lp["classes"]["凹点"], dict))
    ck("C5 旧参数补齐半径维", "radius" in lp["classes"]["凹点"])
    ck("C6 旧参数缺 on -> 视为全开",
       P._enabled_of(lp["classes"]["凹点"], "radius") is True
       and P._enabled_of(lp["classes"]["凹点"], "score") is True)
    ck("C7 旧参数的区间值被保留", lp["classes"]["凹点"]["area"] == [5.0, -1e9])

    # 关掉某一维后落盘再读回来，必须还是关的
    lp["classes"]["凹点"][P.ENABLE_KEY]["radius"] = False
    P.save_params(pd, lp)
    lp2 = P.load_params(pd, ["background", "凹点"])
    ck("C8 on 开关能落盘并读回", P._enabled_of(lp2["classes"]["凹点"], "radius") is False)
    P.save_params(pd, P.default_params(["background", "凹点"]))

    # 类别被删掉之后：JSON 里那条记录不能继续跟着走
    #   （面板只列当前类别 ⇒ 残项永远看不见，却会被 `save_params` 原样写回、
    #     一次一次传下去；2026-10-14 用户问「Atd 能删不」时挖出来的）
    pd3 = fresh_project("C_drop", classes=("background", "凹点", "划痕"))
    P.save_params(pd3, P.default_params(["background", "凹点", "划痕"]))
    ck("C29 三个类别的参数都落了盘",
       set(P.load_params(pd3, ["background", "凹点", "划痕"])["classes"]) == {"凹点", "划痕"},
       str(list(P.load_params(pd3, ["background", "凹点", "划痕"])["classes"])))
    ck("C30 类别被删后 load_params 不再把残项带出来",
       "划痕" not in P.load_params(pd3, ["background", "凹点"])["classes"],
       str(list(P.load_params(pd3, ["background", "凹点"])["classes"])))
    P.drop_class_params(pd3, "划痕")
    with io.open(P.params_path(pd3), encoding="utf-8") as f:
        _d3 = json.load(f)
    ck("C31 外科式摘键：只掉那一个键，其余类别与 global 原样还在",
       "划痕" not in _d3["classes"] and "凹点" in _d3["classes"] and bool(_d3["global"]),
       str(sorted(_d3["classes"])))
    ck("C32 摘一个本来就没有的键 -> 返回 False（不假装成功）",
       P.drop_class_params(pd3, "不存在的类别") is False)
    # 注意：C11 已经把它摘掉了，这里先把文件写回去再验"不裁剪"
    P.save_params(pd3, P.default_params(["background", "凹点", "划痕"]))
    ck("C33 不给 classes 时不做裁剪（不猜用户的类别表）",
       "划痕" in P.load_params(pd3)["classes"],
       str(sorted(P.load_params(pd3)["classes"])))
    # C14 类别改名：那条参数必须**跟着搬家**（面板按名字查参数，不搬就查不到
    #     ⇒ 悄悄给一组默认值，用户调好的八维阈值没了）
    _p3 = P.load_params(pd3, ["background", "凹点", "划痕"])
    _p3["classes"]["划痕"]["area"] = [7.0, -1e9]
    P.save_params(pd3, _p3)
    P.rename_class_params(pd3, "划痕", "划痕2")
    with io.open(P.params_path(pd3), encoding="utf-8") as f:
        _d4 = json.load(f)
    ck("C34 改名后参数原样搬到新名字底下（阈值不丢、旧名字不留）",
       "划痕2" in _d4["classes"] and "划痕" not in _d4["classes"]
       and _d4["classes"]["划痕2"]["area"] == [7.0, -1e9],
       str(sorted(_d4["classes"])))

    # ── 逐维生效：造真实掩膜 ──
    H, W = 120, 160
    CLS = ["background", "凹点"]
    gray = np.full((H, W), 60, dtype=np.uint8)

    def _mk(pred, cls_cfg=None, gcfg=None):
        p = P.default_params(CLS)
        if cls_cfg:
            p["classes"]["凹点"].update(cls_cfg)
        if gcfg:
            p["global"].update(gcfg)
        return P.filter_prediction(pred, classes=CLS, params=p, gray=gray)

    def _count(mask):
        return len([v for v in np.unique(mask).tolist() if v not in (0, 255)])

    # 面积：3 个块 10x10(100px) / 20x20(400px) / 30x30(900px)
    pred = np.zeros((H, W), dtype=np.uint8)
    pred[5:15, 5:15] = 1
    pred[5:25, 30:50] = 1
    pred[5:35, 60:90] = 1
    out, rep = _mk(pred, cls_cfg={"area": [0.0, 500.0]})
    ck("C9 面积上限生效（3->2，扣掉 900px）", rep["kept"]["凹点"] == 2 and rep["dropped"]["凹点"] == 1,
       str(rep["kept"]))
    out2, rep2 = _mk(pred, cls_cfg={"area": [0.0, 500.0], P.ENABLE_KEY: {"area": False}})
    ck("C10 关掉面积维 -> 不再过滤（对照组）", rep2["kept"]["凹点"] == 3 and rep2["dropped"]["凹点"] == 0,
       str(rep2["kept"]))

    # 宽度：块宽 10 / 20 / 30
    out, rep = _mk(pred, cls_cfg={"width": [0.0, 15.0]})
    ck("C11 宽度上限生效（3->1）", rep["kept"]["凹点"] == 1, str(rep["kept"]))
    out, rep = _mk(pred, cls_cfg={"width": [0.0, 15.0], P.ENABLE_KEY: {"width": False}})
    ck("C12 关掉宽度维 -> 不过滤", rep["kept"]["凹点"] == 3)

    # 高度：块高 10 / 20 / 30
    out, rep = _mk(pred, cls_cfg={"height": [0.0, 25.0]})
    ck("C13 高度上限生效（3->2）", rep["kept"]["凹点"] == 2, str(rep["kept"]))

    # 半径（新维）：等半径 10/20/30 -> 半径 sqrt(A/pi) = 5.64 / 11.28 / 16.92
    out, rep = _mk(pred, cls_cfg={"radius": [10.0, -1.0]})
    ck("C14 半径下限生效（3->2，扣掉等效半径 5.64 的小块）",
       rep["kept"]["凹点"] == 2 and rep["dropped"]["凹点"] == 1, str(rep["kept"]))
    out, rep = _mk(pred, cls_cfg={"radius": [10.0, -1.0], P.ENABLE_KEY: {"radius": False}})
    ck("C15 关掉半径维 -> 不过滤（新维的对照组）", rep["kept"]["凹点"] == 3, str(rep["kept"]))

    # 灰度：原图左暗右亮（write_image 造的渐变），这里 gray 全 60 改成分区实测
    g2 = np.full((H, W), 60, dtype=np.uint8)
    g2[5:35, 60:90] = 200          # 只有最大那个块所在列是亮区
    p = P.default_params(CLS)
    p["classes"]["凹点"]["gray"] = [0.0, 100.0]
    out, rep = P.filter_prediction(pred, classes=CLS, params=p, gray=g2)
    ck("C16 灰度上限生效（扣掉亮块）", rep["kept"]["凹点"] == 2, str(rep["kept"]))
    p["classes"]["凹点"][P.ENABLE_KEY]["gray"] = False
    out, rep = P.filter_prediction(pred, classes=CLS, params=p, gray=g2)
    ck("C17 关掉灰度维 -> 不过滤", rep["kept"]["凹点"] == 3)

    # 角度：用一条 45° 斜带
    pask = np.zeros((H, W), dtype=np.uint8)
    for i in range(40):
        pask[20 + i, 20 + i] = 1
        pask[20 + i, 21 + i] = 1
    p = P.default_params(CLS)
    p["classes"]["凹点"]["angle"] = [-5.0, 5.0]     # 只留横平竖直
    out, rep = P.filter_prediction(pask, classes=CLS, params=p, gray=gray)
    ck("C18 角度区间生效（45° 斜带被丢）", rep["kept"]["凹点"] == 0, str(rep["kept"]))
    p["classes"]["凹点"][P.ENABLE_KEY]["angle"] = False
    out, rep = P.filter_prediction(pask, classes=CLS, params=p, gray=gray)
    ck("C19 关掉角度维 -> 不过滤", rep["kept"]["凹点"] == 1)

    # 中心半径（全局第 8 维）
    out, rep = _mk(pred, gcfg={"center_radius": 10})
    ck("C20 中心半径全局下限生效（3->2）", rep["kept"]["凹点"] == 2, str(rep["kept"]))
    out, rep = _mk(pred, gcfg={"center_radius": 10, "center_radius_on": False})
    ck("C21 关掉中心半径维 -> 不过滤", rep["kept"]["凹点"] == 3)

    # 最大检测数：造 3 个块，上限 2
    out, rep = _mk(pred, gcfg={"max_det": 2})
    ck("C22 最大检测数生效（3->2）", rep["kept"]["凹点"] == 2, str(rep["kept"]))
    ck("C23 超限计入丢弃原因",
       rep["reasons"].get("超过最大检测个数", 0) == 1, str(dict(rep["reasons"])))

    # 分数：有/无概率图
    prob = np.zeros((H, W), dtype=np.float32)
    prob[5:15, 5:15] = 0.3
    prob[5:25, 30:50] = 0.6
    prob[5:35, 60:90] = 0.9
    p = P.default_params(CLS)
    p["classes"]["凹点"]["score"] = [0.5, -1.0]
    out, rep = P.filter_prediction(pred, classes=CLS, params=p, gray=gray, prob=prob)
    ck("C24 有概率图时分数维生效（扣掉 0.3 那块）", rep["kept"]["凹点"] == 2, str(rep["kept"]))
    ck("C25 score_checked 标记为真", rep.get("score_checked") is True)
    p = P.default_params(CLS)
    p["classes"]["凹点"]["score"] = [0.5, -1.0]
    out, rep = P.filter_prediction(pred, classes=CLS, params=p, gray=gray, prob=None)
    ck("C26 无概率图时明确记 score_skipped（不假装过滤过）", rep.get("score_skipped") is True)

    # summarize 必须把"未启用"写出来
    p = P.default_params(CLS)
    p["classes"]["凹点"][P.ENABLE_KEY]["radius"] = False
    out, rep = P.filter_prediction(pred, classes=CLS, params=p, gray=gray)
    ck("C27 summarize 报出未启用的维", "未启用" in P.summarize(rep) and "半径" in P.summarize(rep),
       P.summarize(rep))

    # 默认参数（全开 + 宽松）不应改变结果
    out, rep = _mk(pred)
    ck("C28 默认参数下三块全保留（全开但宽松）", rep["kept"]["凹点"] == 3, str(rep["kept"]))


# ══════════════════════════ D. 结果转标注（单动作） ══════════════════════════

def group_D():
    section("D. 结果转标注（1.5，单动作）")
    from annotation import convert as CV
    from annotation import labelme_io

    pd = fresh_project("D_convert", classes=("background", "凹点"))
    CLS = ["background", "凹点"]
    size = (160, 120)
    base = "d1_te_0"
    make_image(pd, base, size)

    # 人工标注：一个 20x20 块（x30-50, y30-50）
    ann = rect_mask(size, 30, 30, 20, 20, 1)
    write_ann(pd, base, ann, CLS)

    # 模型预测：另一个 20x20 块（x90-110, y30-50）—— 与标注完全不重叠
    pred = rect_mask(size, 90, 30, 20, 20, 1)
    write_pred(pd, base, pred)

    ck("D1 预测掩膜按产品约定落在 outputs/<base>_pred.png",
       os.path.exists(CV.pred_mask_path(pd, base)), CV.pred_mask_path(pd, base))
    pm = CV.load_pred_mask(pd, base)
    ck("D2 load_pred_mask 读到的前景像素数正确",
       pm is not None and int((pm > 0).sum()) == 400, None if pm is None else int((pm > 0).sum()))

    am = CV.load_ann_mask(pd, base, CLS)
    ck("D3 load_ann_mask 由 JSON 现算", am is not None and int((am > 0).sum()) == 400,
       None if am is None else int((am > 0).sum()))

    ck("D4 IoU 完全不重叠 = 0", abs(CV.iou_masks(pm, am)) < 1e-9)
    ck("D5 IoU 完全相同 = 1", abs(CV.iou_masks(am, am) - 1.0) < 1e-9)

    mm = CV.miss_mask(pd, base, CLS)
    ck("D6 漏检区域 = 标注 ∩ ¬预测（400 px 全漏）",
       mm is not None and int((mm > 0).sum()) == 400, None if mm is None else int((mm > 0).sum()))

    ious = CV.mask_iou_matrix(pm, am)
    ck("D7 逐类 IoU 表给到 1 号类", 1 in ious and ious[1] == 0.0, str(ious))

    # Convert Label：把预测转成标注
    r = CV.convert_result_to_annotation(pd, base, source="segment", class_names=CLS)
    ck("D8 convert_result_to_annotation 成功", r["ok"] and r["added"] >= 1, str(r))
    shapes, _ = labelme_io.load_labelme_json(CV.ann_json_path(pd, base))
    ck("D9 标注里现在有两块（人工 1 + AI 1）", len(shapes) == 2, "shapes=%d" % len(shapes))
    ck("D10 AI 那块用了掩膜里的类别名", any(s["label"] == "凹点" for s in shapes),
       [s.get("label") for s in shapes])

    r2 = CV.convert_result_to_annotation(pd, base, source="segment", class_names=CLS)
    ck("D11 幂等：再跑一次不重复加", r2["ok"] and r2["added"] == 0, str(r2))

    # erase：过杀 -> 扣掉与预测重叠的标注（AI 那块），保留人工那块
    r = CV.erase_result_from_annotation(pd, base, source="segment", class_names=CLS)
    ck("D12 erase 扣掉与预测重叠的 AI 标注", r["ok"] and r["removed"] == 1 and r["kept"] == 1, str(r))
    shapes2, _ = labelme_io.load_labelme_json(CV.ann_json_path(pd, base))
    ck("D13 人工那块还在（只扣 AI 的）", len(shapes2) == 1, "shapes=%d" % len(shapes2))

    # rebuild：漏杀 -> 由 JSON 重建掩膜
    r = CV.rebuild_mask_from_annotation(pd, base, class_names=CLS)
    ck("D14 rebuild 成功", r["ok"], str(r))
    am2 = np.array(Image.open(CV.ann_mask_path(pd, base)))
    ck("D15 重建掩膜前景像素与人标注一致", int((am2 > 0).sum()) == 400,
       int((am2 > 0).sum()))
    ck("D16 漏检区域图已落盘", os.path.exists(CV.miss_mask_path(pd, base)))

    # 全扣完 -> 全背景负样本
    pd2 = fresh_project("D_neg", classes=("background", "凹点"))
    b2 = "d2_te_0"
    make_image(pd2, b2, size)
    pred2 = rect_mask(size, 40, 40, 20, 20, 1)
    write_pred(pd2, b2, pred2)
    # 人工标注**正好**就是 AI 那块（模拟"上一轮 Convert Label 把 AI 结果固化了"）
    write_ann(pd2, b2, pred2, CLS)
    r = CV.erase_result_from_annotation(pd2, b2, source="segment", class_names=CLS)
    shapes3, _ = labelme_io.load_labelme_json(CV.ann_json_path(pd2, b2))
    m3 = np.array(Image.open(CV.ann_mask_path(pd2, b2)))
    ck("D17 过杀全扣 -> 标注清空", r["ok"] and len(shapes3) == 0, str(r))
    ck("D18 过杀全扣 -> 掩膜全背景（真负样本，不是 255 ignore）",
       int(m3.sum()) == 0 and 255 not in np.unique(m3).tolist(),
       "sum=%d uniq=%s" % (int(m3.sum()), np.unique(m3).tolist()))

    # 已经是空标注 -> 幂等
    r = CV.erase_result_from_annotation(pd2, b2, source="segment", class_names=CLS)
    ck("D19 已经是全背景时 erase 幂等且不报错", r["ok"], str(r))

    # ── 检测通路 ──
    pd3 = fresh_project("D_det", classes=("background", "凹点"))
    b3 = "d3_te_0"
    make_image(pd3, b3, size)
    # 人工标注框：两个框（一个与预测重合、一个不重合）
    from detection.coco_io import save_coco_json, load_coco_json
    ann_boxes = [{"category": "凹点", "bbox": [30, 30, 20, 20], "score": None, "area": 400},
                 {"category": "凹点", "bbox": [10, 90, 15, 15], "score": None, "area": 225}]
    save_coco_json(CV.ann_det_path(pd3, b3), CV.find_image_path(pd3, b3), size[0], size[1],
                   ann_boxes, ["凹点"], bbox_type="hbb")
    # 预测框：与第一个标注完全重合
    write_pred_boxes(pd3, b3, [{"category": "凹点", "bbox": [30, 30, 20, 20], "score": 0.9,
                                "area": 400}])
    ck("D20 检测预测按产品约定落在 outputs/<base>_det.json",
       os.path.exists(CV.pred_det_path(pd3, b3)))
    ck("D21 检测标注在 images/<base>_det.json（目录不同、名字相同的坑）",
       os.path.exists(CV.ann_det_path(pd3, b3)) and
       CV.ann_det_path(pd3, b3) != CV.pred_det_path(pd3, b3))
    ck("D22 框 IoU 完全重合 = 1",
       abs(CV.box_iou([30, 30, 20, 20], [30, 30, 20, 20]) - 1.0) < 1e-9)
    ck("D23 框 IoU 不重叠 = 0",
       abs(CV.box_iou([30, 30, 20, 20], [100, 100, 10, 10])) < 1e-9)

    r = CV.convert_result_to_annotation(pd3, b3, source="detect")
    ab, _ = load_coco_json(CV.ann_det_path(pd3, b3))
    ck("D24 检测 Convert 幂等（预测框已有对应标注，不加）",
       r["ok"] and r["added"] == 0 and len(ab) == 2, str(r))
    r = CV.erase_result_from_annotation(pd3, b3, source="detect")
    ab, _ = load_coco_json(CV.ann_det_path(pd3, b3))
    ck("D25 检测 erase 扣掉与预测重合的框、保留另一个",
       r["ok"] and r["removed"] == 1 and len(ab) == 1, str(r))
    ck("D26 保留的是不重合那个框", ab and abs(ab[0]["bbox"][0] - 10) < 1e-6, str(ab))

    # 调色板(P)模式掩膜不能串色
    pd4 = fresh_project("D_palette", classes=("background", "凹点"))
    b4 = "d4_te_0"
    make_image(pd4, b4, size)
    m = rect_mask(size, 20, 20, 10, 10, 1)
    imp = Image.fromarray(m, mode="L").convert("P")
    p4 = CV.pred_mask_path(pd4, b4)
    os.makedirs(os.path.dirname(p4), exist_ok=True)
    imp.save(p4)
    got = CV.load_pred_mask(pd4, b4)
    ck("D27 P 模式掩膜读到的是类别号（不是调色板颜色）",
       got is not None and int((got > 0).sum()) == 100 and set(np.unique(got).tolist()) == {0, 1},
       None if got is None else str(np.unique(got).tolist()))


# ══════════════════════════ E. 端到端验收链路 ══════════════════════════

def group_E():
    section("E. 验收链路：筛过杀 → 转负样本 → 进下一轮训练集")
    from core import marks as mk
    from core.filter_expr import FilterExpr
    from annotation import convert as CV
    from annotation import labelme_io

    pd = fresh_project("E_chain", classes=("background", "凹点"))
    CLS = ["background", "凹点"]
    size = (160, 120)

    # 三张图，分别对应 过杀 / 漏杀 / 正常
    #  过杀：AI 有检出(400px)，无人工前景标注  -> 干净背景被检出
    #  漏杀：AI 无检出，人工有标注(400px)
    #  正常：AI 检出 与 人工标注 完全重合
    base_ok = "e1-凹点1_te_0"
    base_es = "e2-凹点2_te_0"
    base_nm = "e3-凹点3_te_0"
    for b in (base_ok, base_es, base_nm):
        make_image(pd, b, size)

    write_pred(pd, base_ok, rect_mask(size, 20, 20, 20, 20, 1))          # AI 说有
    write_ann(pd, base_es, rect_mask(size, 20, 20, 20, 20, 1), CLS)      # 人工说有（AI 没说）
    m = rect_mask(size, 60, 60, 20, 20, 1)
    write_pred(pd, base_nm, m)
    write_ann(pd, base_nm, m, CLS)

    # 1) 打标记（走真实落盘）
    marks = {}
    mk.set_mark(marks, [base_ok], "overkill", note="背景干净，AI 幻觉", by="selftest")
    mk.set_mark(marks, [base_es], "escape", note="人工标了，AI 没抓出", by="selftest")
    mk.set_mark(marks, [base_nm], "normal", note="AI 与人工一致", by="selftest")
    mk.save_marks(pd, marks)

    # 2) 模拟"重开工程"：从盘上重新读
    marks2 = mk.load_marks(pd)
    ck("E1 三张图的标记都落盘且能读回",
       marks2.get(base_ok, {}).get("mark") == "overkill"
       and marks2.get(base_es, {}).get("mark") == "escape"
       and marks2.get(base_nm, {}).get("mark") == "normal", str(sorted(marks2)))

    # 3) 「筛出所有 AI 检出但我判为过杀的图」—— 用真实表达式
    expr = FilterExpr([{"field": "mark", "op": "==", "value": "overkill"}], "AND")
    recs = [{"base": b, "mark": marks2.get(b, {}).get("mark", "")}
            for b in (base_ok, base_es, base_nm)]
    hit = [r["base"] for r in expr.filter(recs)]
    ck("E2 表达式能筛出过杀那批", hit == [base_ok], str(hit))
    expr2 = FilterExpr([{"field": "mark", "op": "==", "value": "overkill"},
                        {"field": "pred_cnt", "op": ">=", "value": 1}], "AND")
    ck("E2b 组合条件（过杀 + 有检出）也命中",
       len(expr2.filter([{"base": base_ok, "mark": "overkill", "pred_cnt": 1}])) == 1)

    # 4) 一键闭环
    rep = CV.convert_by_mark(pd, [base_ok, base_es, base_nm], marks2, source="segment")
    ck("E3 convert_by_mark 三张都处理成功",
       rep["summary"]["done"] == 3 and rep["summary"]["failed"] == 0, str(rep["summary"]))
    ck("E4 归入训练集计数>0", rep["train_added"] >= 1, str(rep.get("train_added")))

    # 5) 逐图核验
    sh_ok, _ = labelme_io.load_labelme_json(CV.ann_json_path(pd, base_ok))
    m_ok = np.array(Image.open(CV.ann_mask_path(pd, base_ok)))
    ck("E5 过杀图 -> 标注为空、掩膜全背景（负样本）",
       len(sh_ok or []) == 0 and int(m_ok.sum()) == 0, "shapes=%d sum=%d" % (len(sh_ok or []), int(m_ok.sum())))
    ck("E6 过杀图掩膜里没有 255（不是 ignore，模型拿得到「背景」的梯度）",
       255 not in np.unique(m_ok).tolist(), str(np.unique(m_ok).tolist()))

    m_es = np.array(Image.open(CV.ann_mask_path(pd, base_es)))
    ck("E7 漏杀图 -> 掩膜重建，正样本完整（400px）", int((m_es > 0).sum()) == 400,
       int((m_es > 0).sum()))

    sh_nm, _ = labelme_io.load_labelme_json(CV.ann_json_path(pd, base_nm))
    ck("E8 正常图 -> AI 检出已固化成标注（1 条）", len(sh_nm or []) == 1, "shapes=%d" % len(sh_nm or []))

    sfile = os.path.join(pd, "train_test_split.json")
    smap = json.load(io.open(sfile, encoding="utf-8")) if os.path.exists(sfile) else {}
    ck("E9 三张图都进了 train（下一轮训练集）",
       all(smap.get(b) == "train" for b in (base_ok, base_es, base_nm)), str(smap))

    # 6) 训练侧真实取数（不需要手抄规则）
    try:
        from training.dataset import SegmentationDataset
        ds = SegmentationDataset(pd, split="train", image_size=(64, 64))
        got = sorted({os.path.splitext(os.path.basename(p))[0] for p, _j, _m in ds._image_samples})
        want = sorted({base_ok, base_es, base_nm})
        ck("E10 训练集真实取数含这三张", all(b in got for b in want),
           "取到=%s" % got)
        ck("E11 过杀那张取到的掩膜是全背景",
           int(ds._get_mask([os.path.splitext(os.path.basename(p))[0]
                             for p, _j, _m in ds._image_samples].index(base_ok)).sum()) == 0)
    except ImportError as e:
        incon("E10/E11 训练集真实取数（缺 torch）", str(e))
    except Exception as e:
        ck("E10 训练集真实取数", False, "%s: %s" % (type(e).__name__, e))

    # 7) 已在 Val 的图不被偷偷挪进训练集
    from training.dataset import get_train_test_split, save_train_test_split
    smap2, _ = get_train_test_split(pd)
    smap2[base_es] = "val"
    save_train_test_split(pd, smap2)
    CV.add_to_train(pd, [base_es, base_ok])
    smap3, _ = get_train_test_split(pd)
    ck("E12 已在 Val 的图不被挪进 train（保住模型对比的可信度）",
       smap3.get(base_es) == "val" and smap3.get(base_ok) == "train", str(smap3))

    # 8) 失败必须可见
    rep2 = CV.convert_by_mark(pd, ["not_exist_te_0"], {"not_exist_te_0": {"mark": "escape"}},
                              source="segment")
    ck("E13 不存在的图 -> 计入 failures（不静默成功）",
       rep2["summary"]["failed"] == 1 and rep2["failures"][0]["base"] == "not_exist_te_0",
       str(rep2["summary"]))
    ck("E14 summarize_report 输出可读", "共 1 张" in CV.summarize_report(rep2),
       CV.summarize_report(rep2))

    # 9) doubt 不进训练集
    rep3 = CV.convert_by_mark(pd, [base_nm], {base_nm: {"mark": "doubt"}}, source="segment")
    ck("E15 存疑态按设计被跳过", rep3["summary"]["skipped"] == 1 and rep3["summary"]["done"] == 0,
       str(rep3["summary"]))


# ══════════════════════════ F. AST 源码核验 ══════════════════════════

def _parse(path):
    with io.open(path, encoding="utf-8-sig") as f:
        return ast.parse(f.read(), path)


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _attr_names(node):
    """函数体内用到的**属性名**集合。

    刻意不用 `ast.dump` + 字符串包含：dump 里属性节点的形态是 `attr='mark_filter'`，
    写成 `"'attr', 'mark_filter'"` 永远匹配不上 —— 断言看着有道理、实际永远 FAIL，
    和"被注释骗过的假绿"是同一类坑的两面（一个永远红、一个永远绿，都不是断言）。
    """
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute):
            names.add(n.attr)
    return names


def _global_names(node):
    """函数体里引用到的**全局/模块级符号名**（如 ENABLE_KEY、MARK_KINDS）。"""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
    return out


def _str_consts(node):
    """函数体内出现的字符串字面量集合（注释不在 AST 里，天然骗不过）。"""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
    return out


def _const_values(node):
    """函数体内出现的常量值集合（含 True/False/数字）。"""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and not isinstance(n.value, str):
            out.add(n.value)
    return out


def _callee_attrs(node):
    """被调用到的属性名（如 self._save_shapes(...) 里的 _save_shapes）。"""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            out.add(n.func.attr)
    return out


def _callee_names(node):
    """被调用到的**所有**函数名（含直接调用的模块级函数，如 overlap_of_shape(...)）。"""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


def _arg_names(node):
    """函数的形参名集合（含 keyword-only 与 *args/**kwargs）。"""
    a = node.args
    names = set()
    for grp in (a.posonlyargs, a.args, a.kwonlyargs):
        for x in grp:
            names.add(x.arg)
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names


def group_F():
    section("F. AST 源码核验（防「注释骗过断言」）")

    # F1：主窗口的筛选热路径真的接了标记
    mw = os.path.join(ROOT, "ui", "main_window.py")
    tree = _parse(mw)
    fif = _func(tree, "_filter_images")
    ck("F1 _filter_images 存在", fif is not None)
    if fif is not None:
        attrs = _attr_names(fif)
        ck("F2 _filter_images 里真的用了 self._mark_filter 做筛选",
           "_mark_filter" in attrs, str(sorted(attrs))[:300])
        ck("F3 _filter_images 里真的读了 self._marks", "_marks" in attrs)
        ck("F4 _filter_images 里真的调了表达式筛选 evaluate",
           "evaluate" in _callee_attrs(fif))
        keys5 = _str_consts(fif)
        ck("F5 _filter_images 里真的把标记写进了记录（'mark'/'mark_note' 键）",
           "mark" in keys5 and "mark_note" in keys5, str(sorted(keys5))[:300])

    for fn, desc in (("_on_mark_requested", "标记入口"),
                     ("_on_mark_note", "备注入口"),
                     ("_run_convert_by_mark", "闭环一键入口"),
                     ("_on_mark_cell_filter", "交叉表点格子筛选"),
                     ("_open_filter_builder", "表达式筛选入口")):
        ck("F6 主窗口有 %s（%s）" % (fn, desc), _func(tree, fn) is not None)

    # F7：标记列真的被渲染（配色 + 文案）
    rp = _func(tree, "_render_page")
    ck("F7 _render_page 渲染了标记列（文案+配色都取自 core/marks）",
       rp is not None and "kind_label" in _callee_attrs(rp) and "kind_color" in _callee_attrs(rp))

    # F8：推理参数面板真的把「启用开关」写进参数
    ip = os.path.join(ROOT, "ui", "infer_params.py")
    t2 = _parse(ip)
    ovc = _func(t2, "_on_value_changed")
    ck("F8 infer_params._on_value_changed 真的写 ENABLE_KEY",
       ovc is not None and "ENABLE_KEY" in _global_names(ovc))
    init2 = _func(t2, "__init__")
    ck("F9 infer_params 面板有各维启用勾选框",
       init2 is not None and "_enables" in _attr_names(init2),
       str(sorted(_attr_names(init2)))[:200] if init2 else "")

    # F10：params.py 真的把 radius 加进了 RANGE_KEYS，并把 on 默认全 True
    tp = _parse(os.path.join(ROOT, "inference", "params.py"))
    rk = None
    for node in ast.walk(tp):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "RANGE_KEYS" for t in node.targets):
            rk = node
    keys = []
    if rk is not None:
        for el in rk.value.elts:
            if isinstance(el, ast.Tuple) and el.elts and isinstance(el.elts[0], ast.Constant):
                keys.append(el.elts[0].value)
    ck("F10 RANGE_KEYS 字面量就是七维（含 radius）",
       keys == ["score", "area", "width", "height", "angle", "gray", "radius"], str(keys))
    dcp = _func(tp, "default_class_params")
    ck("F11 default_class_params 里 on 默认全 True",
       dcp is not None and "ENABLE_KEY" in _global_names(dcp)
       and True in _const_values(dcp))

    # F12：过杀走"背景"而不是 255/ignore —— 用 AST 看真实的字符串常量，
    #      注释里出现 "ignore" 不影响判定（这正是字符串包含断言会假绿的原因）
    tc = _parse(os.path.join(ROOT, "annotation", "convert.py"))
    lits = set()
    for node in ast.walk(tc):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lits.add(node.value.strip())
    bad_lits = [s for s in lits if s.lower() in ("ignore", "__ignore__", "gnore")]
    # 排除"判断 ignore 的读路径"（labelme_io 里有对 ignore 的判定，本文件不该有写入 ignore 的常量）
    ck("F12 convert.py 没有把 ignore/__ignore__ 当标签写的字符串常量",
       not bad_lits, "发现=%s" % bad_lits)

    # F13：core/marks.py 的四态字面量正确
    tm = _parse(os.path.join(ROOT, "core", "marks.py"))
    kinds = None
    for node in ast.walk(tm):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "MARK_KINDS" for t in node.targets):
            kinds = [e.value for e in node.value.elts]
    ck("F13 MARK_KINDS 字面量 = 四态",
       kinds == ["normal", "overkill", "escape", "doubt"], str(kinds))

    # F14：convert.py 的 erase 用的判定阈值与 erase 语义对得上（真的按重叠扣）
    er = _func(tc, "erase_result_from_annotation")
    ck("F14 erase 里真的做了「重叠 >= tau 才扣」",
       er is not None and "tau" in _arg_names(er)
       and "overlap_of_shape" in _callee_names(er),
       ("args=%s callees=%s" % (sorted(_arg_names(er)), sorted(_callee_names(er)))[:240]
        if er is not None else ""))
    ck("F15 erase 真的会写回（调用 _save_shapes / _save_boxes）",
       er is not None and bool({"_save_shapes", "_save_boxes"} & _callee_names(er)),
       str(sorted(_callee_names(er)))[:240] if er is not None else "")

    # F16：add_to_train 真的读盘上的原始 split 文件 —— 只看内存默认值会让
    #      验收链路最后一环（"进入下一轮训练集"）静默失效，而且不报错。
    att = _func(tc, "add_to_train")
    ck("F16 add_to_train 读盘上 train_test_split.json（不是只看内存默认值）",
       att is not None and "train_test_split.json" in _str_consts(att),
       str(sorted(_str_consts(att)))[:220])

    # F17/F18：标记列不只是"多了一列表头"，第 9 个格子必须真的被**建出来**。
    #   真窗口探针抓到的 bug：_render_page 只 setColumnCount(9)，items_data 里却
    #   只有 8 项 -> item(row, 8) 恒为 None -> 标记列永远空白（配色代码成了死代码）。
    #   这类"表头有、格子没有"的洞，AST 只看函数名是看不见的，必须数 items_data。
    n_items, last_is_mark = None, False
    if rp is not None:
        for node in ast.walk(rp):
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", "") == "items_data" for t in node.targets)
                    and isinstance(node.value, ast.List)):
                n_items = len(node.value.elts)
                if node.value.elts:
                    last_is_mark = "kind_label" in {
                        c.attr for c in ast.walk(node.value.elts[-1])
                        if isinstance(c, ast.Attribute)}
    ck("F17 _render_page 的 items_data 是 9 项（标记列的格子真的被建出来了）",
       n_items == 9, "len=%s" % n_items)
    ck("F18 items_data 最后一项就是标记列的 kind_label(...)", last_is_mark)

    # F19/F20：**跨模块属性核验** —— 所有 `模块别名.属性` 引用的属性必须在被引模块里
    #   真实存在。这一条能静态抓住"函数改名了、调用点没改"这类错误：
    #   真窗口探针抓到的第 2 个 bug 就是 `marks_mod._guess_defect`（真名 defect_of），
    #   它被 try/except 兜住后只写一行日志 —— 统计面板永远是空的，而且**不报错**。
    def _attrs_on(tree_, alias):
        out = set()
        for node in ast.walk(tree_):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == alias):
                out.add(node.attr)
        return out

    def _existence_check(ck_name, mod, alias, tree_, desc):
        try:
            have = set(dir(mod))
        except Exception as e:
            ck(ck_name, False, "模块导入失败: %r" % (e,))
            return
        used = _attrs_on(tree_, alias)
        miss = sorted(a for a in used if a not in have)
        ck(ck_name, bool(used) and not miss,
           ("%s：引用了不存在的 %s.%s" % (desc, alias, miss) if miss
            else "只用到 %s" % sorted(used)))

    from core import marks as _mkmod
    _existence_check("F19 ui/main_window.py 里 marks_mod.<x> 全部真实存在",
                     _mkmod, "marks_mod", tree, "main_window")

    tp2 = _parse(os.path.join(ROOT, "ui", "mark_panel.py"))
    _existence_check("F20 ui/mark_panel.py 里 mk.<x> 全部真实存在",
                     _mkmod, "mk", tp2, "mark_panel")

    # F21：正面断言 —— 缺陷归类必须走 core/marks.defect_of
    dob = _func(tree, "_defect_of_base")
    ck("F21 _defect_of_base 走的是 marks_mod.defect_of",
       dob is not None and "defect_of" in _callee_attrs(dob),
       str(sorted(_callee_attrs(dob)))[:200] if dob is not None else "")


# ══════════════════════════ 主流程 ══════════════════════════

GROUPS = {"A": group_A, "B": group_B, "C": group_C, "D": group_D,
          "E": group_E, "F": group_F}


def main(argv):
    want = [a.upper() for a in argv if a.upper() in GROUPS] or list(GROUPS)
    os.makedirs(WORK, exist_ok=True)
    print("=" * 74)
    print("一期六项自测（标记 / 筛选 / 八维 / 转标注 / 闭环 / AST 核验）")
    print("工作目录: %s" % WORK)
    print("=" * 74)
    for g in want:
        try:
            GROUPS[g]()
        except Exception:
            # limit 给足：截断的堆栈会让人以为"根因就是最后一行"，实际根因常在被截掉的那几帧
            _results.append((FAIL, "组 %s 执行异常" % g, traceback.format_exc(limit=40)))
    n_pass = sum(1 for s, _n, _d in _results if s == PASS)
    n_fail = sum(1 for s, _n, _d in _results if s == FAIL)
    n_incon = sum(1 for s, _n, _d in _results if s == INCON)
    for s, n, d in _results:
        if s == "SECTION":
            print("\n── %s ──" % n)
        elif s == PASS:
            print("  %-4s %s" % (PASS, n))
        elif s == INCON:
            print("  %-4s %s   [%s]" % (INCON, n, d))
        else:
            print("  %-4s %s" % (FAIL, n))
            if d:
                print("       %s" % str(d)[:2200])
    print("\n" + "=" * 74)
    print("结果：PASS %d ｜ FAIL %d ｜ INCONCLUSIVE %d ｜ 合计 %d"
          % (n_pass, n_fail, n_incon, n_pass + n_fail + n_incon))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
