# -*- coding: utf-8 -*-
"""标记 UI 探针 —— 在**真窗口**里走一遍标记的完整产品路径。

为什么要真窗口（而不是只做 AST 核验）：
    AST 只能证明"代码里写了 self._mark_filter"，证明不了"点下去真的有反应"。
    本项目已有先例：掩膜筛选那条链在 AST 上完全正确，但 UI 里 `_send()` 少传一个
    参数，人在窗口里怎么操作都解不开（死锁）。UI 层的盲区只有真窗口能覆盖。

⚠️ 两个本机坑（都实测踩过）：
  1. **torch 必须在 PyQt5 之前导入**。本机 PyQt5 先加载后，torch 的
     c10.dll 会报 `OSError [WinError 1114] 动态链接库(DLL)初始化例程失败`。
     产品的 `ui/main_window.py` 恰好是 torch 在前（第 3 行 vs 第 8 行），
     所以产品没事；探针也必须照这个顺序，否则会在 import 阶段就崩。
  2. `QT_QPA_PLATFORM=offscreen` 在 Windows 上**没有字体引擎** —— 气泡/表格画得出来
     但文字全是空白。要看文字必须用 windows 平台。

用法：
    python probe_mark_ui.py            # 跑断言 + 出截图
"""

import os
import sys

# ① torch 先于 PyQt5（见文件头坑 1）
try:
    import torch  # noqa: F401
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ② 真窗口才有字体（见文件头坑 2）。必须在 QApplication 创建前设好。
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

import io
import json
import ast
import re
import tempfile
import time

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QCheckBox, QHeaderView, QLabel, QSizePolicy,
                            QToolButton)
from PyQt5.QtCore import Qt, QTimer

_results = []
_T0 = time.time()


def stage(msg):
    """打点：探针跑得慢的时候，光看最后一行汇总完全不知道卡在哪一段。"""
    print("  [%6.1fs] %s" % (time.time() - _T0, msg), flush=True)


def ck(name, cond, detail=""):
    _results.append((bool(cond), name, detail))
    return bool(cond)


def _fields(txt):
    """把一行计数里的字段名抠出来（`Train: 28 组 | Ann: 33 组` -> {'Train','Ann'}）。

    两行信息是否冗余，本质上就是"字段集有没有交集"，用眼睛看不如让机器算。
    """
    return {m.group(1) for m in re.finditer(r"([A-Za-z]+):", txt or "")}


def _infer_times_is_ms():
    """AST 核验：`_infer_times[...]` 的赋值右侧必须出现 `* 1000`（毫秒）。

    为什么不写成字符串包含：这段代码里"秒"和"毫秒"的区别**只是一个因子**，
    任何字符串断言都抓不住；而且注释里一提"1000"就会造成假绿（本项目踩过：
    探针断言的字面量命中的是**修复说明注释**）。所以按语法树取"赋值语句的右子树"。
    """
    try:
        src = open(os.path.join(ROOT, "ui", "main_window.py"), encoding="utf-8-sig").read()
        tree = ast.parse(src)
    except Exception as e:
        return False, "读不了 ui/main_window.py: %s: %s" % (type(e).__name__, e)
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute)
                    and t.value.attr == "_infer_times"):
                writes.append(node.value)
    if not writes:
        return False, "没找到 _infer_times 的写入点（名字改了？）"

    def _has_1000(n):
        for x in ast.walk(n):
            if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Mult):
                for side in (x.left, x.right):
                    if isinstance(side, ast.Constant) and side.value == 1000:
                        return True
        return False
    bad = [w for w in writes if not _has_1000(w)]
    return (not bad), "写入点 %d 个，未乘 1000 的 %d 个" % (len(writes), len(bad))


def _emit_state_sig_has_nviews():
    """AST 核验：`MultiViewPreview._emit_state` 的去重签名里必须含 `n_views`。

    为什么单独钉这一条（2026-10-14 实测踩到）：视图数从 1 变成 3 时，
    签名里那五项（三个开关 + 类别 + 缩放）**一个都没变** ⇒ 少了 n_views 就
    永远不发那一拍 ⇒ 工具栏右侧那三个图层开关与类别格**永远不显示**，
    而界面上完全看不出"少了东西"（它们本来就该在那儿，只是没有对照）。
    这类"元组里少一个元素"的错：字符串断言抓不住（少个元素既不报错也不影响语法），
    肉眼看代码也极易滑过去 —— 只有取语法树数元素才钉得住。
    """
    try:
        src = open(os.path.join(ROOT, "ui", "multiview_preview.py"), encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception as e:
        return False, "读不了 ui/multiview_preview.py: %s: %s" % (type(e).__name__, e)
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_emit_state"]:
        for node in ast.walk(fn):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "sig"):
                if not isinstance(node.value, ast.Tuple):
                    return False, "sig 不是元组字面量（%s）" % type(node.value).__name__
                txt = ast.dump(node.value)
                return ("n_views" in txt), "sig 有 %d 个元素，含 n_views=%s" % (
                    len(node.value.elts), "n_views" in txt)
    return False, "没找到 _emit_state 里的 sig 赋值（函数改名了？）"


def _status_info_text_params():
    """AST 核验：`status_info_text` 不该再带 `full` 参数。

    那条 `full=False` 分支是给多视图顶栏用的；顶栏整行拆掉后它**没有调用者**了。
    留一条没人走的路，下一个人想接就会直接接上 —— 于是又多出一处"同一个状态显示两遍"，
    而那正是这次重构要消掉的毛病。所以这里钉"参数表"，不是钉"分支里写了什么"。
    """
    try:
        src = open(os.path.join(ROOT, "annotation", "mask_editor.py"), encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception as e:
        return None, "读不了 annotation/mask_editor.py: %s: %s" % (type(e).__name__, e)
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "status_info_text"]:
        names = [a.arg for a in fn.args.args]
        return names, "参数表 = %s" % names


def _theme_has_slider_style():
    """AST 核验：主题表里必须有 QSlider 的槽/手柄规则。

    为什么钉这条（2026-10-14 实测踩到，代价是"一个像素都不画"）：
    底栏缩放滑块把配色写在了**控件自己的内联样式**里，而那三个色
    （#2A3A5C / #0E3A52 / #00E5FF）**全在 THEME_COLORS 白名单里** ⇒
    `sweep_inline_styles` 判定"这段只用了主题皮肤色"，把颜色声明整段剥掉
    （度量留下）。于是槽和手柄**没有背景色**，而主题表当时没有 QSlider
    规则可兜底 ⇒ 滑块占着 84px 宽、画出来是一段空槽。
    功能断言（value / geometry / valueChanged / zoom_level）当时**全绿**。
    规则：**内联样式只写度量，颜色写进主题表**（主题表的规则 sweep 不会碰）。

    取的是**字符串常量节点的值**（AST），不是源码文本 —— 注释里写一遍
    "QSlider::groove" 不能算数（本仓库有过"注释骗过断言"的假绿）。
    """
    try:
        src = open(os.path.join(ROOT, "ui", "themes.py"), encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception as e:
        return False, "读不了 ui/themes.py: %s: %s" % (type(e).__name__, e)
    vals = [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    joined = "\n".join(vals)
    ok = ("QSlider::groove" in joined) and ("QSlider::handle" in joined)
    return ok, "主题表里 QSlider::groove=%s / QSlider::handle=%s（共 %d 个字符串常量）" % (
        "QSlider::groove" in joined, "QSlider::handle" in joined, len(vals))


def _mainwindow_has_no_slider_qss():
    """AST 核验：`ui/main_window.py` 的**模块级常量里不许再有滑块的 QSS**。

    滑块配色只许有一处真相源（主题表）。窗口里再放一份 `*_SLIDER_QSS` 常量，
    下一个人就会顺手 `setStyleSheet()` 上去 —— 然后颜色又被 sweep 剥掉、
    又是一段空槽，而且这次没人会想到去 themes.py 找。
    """
    try:
        src = open(os.path.join(ROOT, "ui", "main_window.py"), encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception as e:
        return False, "读不了 ui/main_window.py: %s: %s" % (type(e).__name__, e)
    bad = []
    for n in tree.body:                                   # 只看模块级赋值
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and "SLIDER_QSS" in t.id:
                    bad.append(t.id)
    return (not bad), ("模块级滑块 QSS 常量 = %s" % (bad or "无"))


def _mainwindow_no_proj_toggle():
    """AST 核验：左栏那个「▼ 折叠工程列表」按钮**不该再存在**。

    2026-10-14 晚左栏重构后，"显示/隐藏工程列表"由竖排页签接管（点一下收起）。
    再留一个 ▼ 干同一件事，就是两个入口、两种状态判断 —— 迟早打架
    （典型症状：▼ 显示成 ▼ 但列表是收起的）。所以这里钉死它不复存在。

    用 AST 而不是 `"_proj_toggle_btn" in src`：字符串包含会被**注释**骗过，
    而这轮修的注释里恰好写着这个名字（老坑："注释能骗过的断言 = 没有断言"）。
    """
    try:
        src = open(os.path.join(ROOT, "ui", "main_window.py"), encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception as e:
        return False, "读不了 ui/main_window.py: %s: %s" % (type(e).__name__, e)
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "_proj_toggle_btn":
            hits.append("self._proj_toggle_btn @line %d" % n.lineno)
        elif isinstance(n, ast.Name) and n.id == "proj_toggle_btn":
            hits.append("%s @line %d" % (n.id, n.lineno))
    return (not hits), ("残留引用 = %s" % (hits or "无"))


def _mw_tree():
    """读 `ui/main_window.py` 的语法树（本文件里所有 AST 核验共用）。"""
    src = open(os.path.join(ROOT, "ui", "main_window.py"), encoding="utf-8-sig").read()
    return ast.parse(src)


def _mw_no_dead_image_props():
    """AST：右栏「图像管理」那页的工厂函数与控件属性必须**彻底消失**。

    2026-10-14 晚该页整页删除。删页而留控件，就是"看着像能用、其实挂不上去"的死代码：
    下一轮谁想复用 `self.img_info_name`，会先看到"没报错"、再看到"界面上根本没有"。
    按语法树取 Attribute / FunctionDef 名，不数字符串 —— 本轮注释里恰好把这些名字
    写了一遍（写明"已删除"），字符串包含必然假绿（老坑："注释能骗过的断言 = 没有断言"）。
    """
    try:
        tree = _mw_tree()
    except Exception as e:
        return False, "读不了 ui/main_window.py: %s: %s" % (type(e).__name__, e)
    props = {"img_info_name", "img_info_size", "img_info_ann",
             "split_count_label", "out_count_label",
             # 2026-10-14 晚追加：它们在左栏底部短暂存在过一行，Arthur 判该行冗余后删除
             "split_manage_btn", "open_out_btn"}
    funcs = {"_make_image_info_section", "_make_split_section", "_make_project_ops_section"}
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in props:
            hits.append("self.%s @line %d" % (n.attr, n.lineno))
        elif isinstance(n, ast.FunctionDef) and n.name in funcs:
            hits.append("def %s @line %d" % (n.name, n.lineno))
    return (not hits), ("残留 = %s" % (hits or "无"))


def _mw_menu_has_output_dir():
    """AST：两个右键菜单的**函数体里**真的调了 `self._open_output_dir`。

    「输出目录」原先是右栏「图像管理」页上的一个按钮，删页时必须搬走（否则丢功能）。
    这里取的是 `ast.Call` 节点（谁调了它），不是"源码里出现过这个名字" ——
    后者会被本轮的说明注释骗过：我在 `_on_image_context_menu` 上面就写了一句
    "「输出目录」原先在右栏「图像管理」页里"，那句话里没有下划线名字，
    但同类注释迟早会写上，所以不给自己留这个坑。
    """
    try:
        tree = _mw_tree()
    except Exception as e:
        return False, "读不了 ui/main_window.py: %s: %s" % (type(e).__name__, e)
    want = {"_on_project_context_menu", "_on_image_context_menu"}
    found = {k: 0 for k in want}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in want:
            for c in ast.walk(n):
                if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                        and c.func.attr == "_open_output_dir"):
                    found[n.name] += 1
    ok = all(v > 0 for v in found.values())
    return ok, "各菜单里的调用次数 = %s" % found


def _mw_manage_split_entrypoints():
    """AST：`_manage_split`（数据集划分对话框）必须仍有**多个**入口活着。

    为什么单独钉它：它先前只挂在左栏那个小按钮上，Arthur 判定"那一行冗余"要删 ——
    一删，这个功能在界面上就**一个入口都不剩**（`SplitDialog` 再也打不开），
    而且不会有任何报错，只是"某个功能静默消失"。所以删按钮的同时必须补右键/菜单入口，
    这条断言就是钉"补上了"。

    数的是"哪个函数体里出现了 `self._manage_split`"——AST 里没有注释节点，
    所以本文件/产品文件里那些"原先是按钮"的说明文字一概不算数。
    """
    try:
        tree = _mw_tree()
    except Exception as e:
        return [], "读不了 ui/main_window.py: %s: %s" % (type(e).__name__, e)
    funcs = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for c in ast.walk(n):
                if isinstance(c, ast.Attribute) and c.attr == "_manage_split":
                    funcs.add(n.name)
                    break
    return sorted(funcs), "出现它的函数 = %s" % sorted(funcs)


def _render_panel_tabs_match_index():
    """AST：`scripts/render_panel.py::ALL_TABS` 必须与 `_right_tab_index` 的键**完全一致**。

    这个脚本是"改布局前后先看一眼"的离屏截图工具，页签名写错时 `show_right_tab()`
    会**静默 return**，于是你会对着「标注管理」的截图分析「图像管理」的布局
    —— 2026-10-13 真踩过。删页之后正是最容易脱钩的时刻，所以机器比对一次。
    """
    try:
        mw = _mw_tree()
        rp = ast.parse(open(os.path.join(ROOT, "scripts", "render_panel.py"),
                            encoding="utf-8-sig").read())
    except Exception as e:
        return False, "读源码失败: %s: %s" % (type(e).__name__, e)
    idx_keys = None
    for n in ast.walk(mw):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Attribute)
                and n.targets[0].attr == "_right_tab_index"
                and isinstance(n.value, ast.Dict)):
            idx_keys = [k.value for k in n.value.keys if isinstance(k, ast.Constant)]
    tabs = None
    for n in ast.walk(rp):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ALL_TABS" for t in n.targets)
                and isinstance(n.value, ast.List)):
            tabs = [e.value for e in n.value.elts if isinstance(e, ast.Constant)]
    if idx_keys is None or tabs is None:
        return False, "_right_tab_index=%s ALL_TABS=%s" % (idx_keys, tabs)
    return (sorted(idx_keys) == sorted(tabs)), "_right_tab_index=%s ALL_TABS=%s" % (idx_keys, tabs)


def _painted_cnt(w):
    """数"非底色像素"：w 抓图后，减去出现最多的那个颜色（=底色）。

    判"控件画出来了没有"必须数像素 —— 同底栏滑块那次：占着 84px 宽、一像素不画，
    而 value/geometry/信号断言全绿。
    """
    from collections import Counter as _C2
    _img = w.grab().toImage()
    _cc = _C2()
    for _y in range(_img.height()):
        for _x in range(_img.width()):
            _c = _img.pixelColor(_x, _y)
            _cc[(_c.red(), _c.green(), _c.blue())] += 1
    if not _cc:
        return 0
    return _img.width() * _img.height() - _cc.most_common(1)[0][1]


def _check_panel_handles(win, pump, ck, app):
    """左右栏折叠（2026-10-15 第三版）：折叠=**面板整个藏起来**（画布吃满整宽），
    恢复按钮变成**贴在画布边缘的浮层小按钮**（不进布局 ⇒ 不占宽度）。

    写成模块级函数而不是内联块：内联块在这份 2000+ 行探针里已经出过三次缩进事故
    （Edit 插入的 8 空格 vs 上下文 4 空格 ⇒ IndentationError，探针直接跑不起来）。
    """
    for side in ("left", "right"):
        panel = win._left_panel if side == "left" else win._right_panel
        inner = getattr(win, "_%s_toggle_btn" % side)
        edge = getattr(win, "_edge_left_btn" if side == "left" else "_edge_right_btn")
        center = win._center_panel
        w_before, c_before = panel.width(), center.width()
        # ── 折叠 ──
        inner.click(); pump(app, 0.45)
        ck("%s 栏折叠后**面板整个隐藏**（宽度归零，画布能占满整宽）" % side,
           panel.isHidden() and panel.width() == 0,
           "hidden=%s width=%d" % (panel.isHidden(), panel.width()))
        ck("%s 栏折叠后画布真的变宽了（+%dpx，不是留一条空白）" % (
               side, center.width() - c_before),
           center.width() > c_before + 20,
           "画布 %d -> %d" % (c_before, center.width()))
        ck("%s 栏折叠后**边缘恢复按钮出现**（不是消失）" % side,
           (not edge.isHidden()),
           "edgeHidden=%s text=%r" % (edge.isHidden(), edge.text()))
        ck("%s 栏折叠后面板内那枚箭头已隐藏（避免两个入口重复）" % side,
           inner.isHidden(), "innerHidden=%s" % inner.isHidden())
        ck("%s 栏折叠后边缘按钮**贴在画布%s边缘**（左=0，右=贴右）" % (
               side, "左" if side == "left" else "右"),
           edge.x() == 0 if side == "left"
           else edge.x() + edge.width() >= center.width() - 2,
           "x=%d w=%d 画布宽=%d" % (edge.x(), edge.width(), center.width()))
        # ── 用边缘按钮展开 ──
        edge.click(); pump(app, 0.45)
        ck("%s 栏点边缘按钮真能展开（面板回来、画布回到折叠前的宽度）" % side,
           (not panel.isHidden()) and abs(center.width() - c_before) <= 2,
           "hidden=%s 画布 %d（折叠前 %d）" % (panel.isHidden(), center.width(), c_before))
        ck("%s 栏展开后边缘按钮隐藏、面板内箭头回来" % side,
           edge.isHidden() and (not inner.isHidden()),
           "edgeHidden=%s innerHidden=%s" % (edge.isHidden(), inner.isHidden()))
        ck("%s 栏展开后宽度复原为用户自己拖过的 %d（不是写死 240）" % (side, w_before),
           abs(panel.width() - w_before) <= 2,
           "now=%d before=%d" % (panel.width(), w_before))


def _dominant_color(w, rect=None):
    """取控件（或某块区域）里**占比最大**的颜色，返回 (r, g, b)。

    为什么图层开关要用它、而不是数"非底色像素"：
    `QCheckBox.grab()` 的透明边距算进来后，**占比最大的是透明像素**，
    于是勾选/未勾选的"非底色像素数"完全相同（实测 338/338，两态一模一样）⇒
    断言会**假绿**。真正要看的是 indicator 那块**主色变了没有**
    （勾上=青色实心，没勾=深色空框）。
    """
    from collections import Counter as _C3
    img = w.grab().toImage()
    r = rect or img.rect()
    cc = _C3()
    for y in range(r.top(), r.bottom() + 1):
        for x in range(r.left(), r.right() + 1):
            if 0 <= x < img.width() and 0 <= y < img.height():
                c = img.pixelColor(x, y)
                if c.alpha() > 0:                    # 透明像素不算"颜色"
                    cc[(c.red(), c.green(), c.blue())] += 1
    if not cc:
        return None
    return cc.most_common(1)[0][0]


def _register_in_memory(pm, proj_dir):
    """把夹具工程只登记到**内存**，绝不碰用户的 `.custom_projects.json`。

    原来直接调 `pm.register_external_project()` —— 它会 `_save_custom_dirs()` **落盘**，
    于是每跑一次探针，用户的工程列表里就多两条指向 `%TEMP%` 的垃圾工程。
    2026-10-14 用户截图里那两条 `probe_mv` / `probe_ui` 就是这么来的；而且 Temp 迟早会被
    清掉，留下两条点不开的死条目。这里只复刻真正必需的两件事：修 parent_dir + 写内存注册表。
    """
    d = os.path.abspath(proj_dir)
    meta_path = os.path.join(d, "project.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["parent_dir"] = os.path.dirname(d)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    name = meta.get("name") or os.path.basename(d)
    pm._custom_dirs[name] = d
    return name


def _load_product_base_qss():
    """从 `main.py` 里 AST 抠出产品的基础 QSS（那段几百行的深色表）。

    为什么必须抠：`apply_theme` 是「**基础表** + 主题覆盖」两段拼起来的，基础表由
    `main.py` 用 `set_base_qss()` 提前登记（就写在它的 `main()` 里）。探针只调
    `apply_theme` 的话 `base_qss()` 是空的 ⇒ 只有薄薄一层 `DARK_INDICATOR` 生效
    ⇒ 界面看起来仍是系统默认的浅色，**拿这种截图去谈"好不好看"完全无效**
    （2026-10-14 实测踩到：以为 dark 主题下两个 tab 按钮是空的，其实是探针没初始化）。

    用 AST 而不是字符串截取：这段 QSS 里含 `{}`、引号、注释，正则一碰就碎；
    而且要取的是 `app.setStyleSheet(<最长字符串常量>)`，按语法树取才稳。
    """
    import ast
    try:
        # ⚠️ 必须 utf-8-sig：main.py 开头带 UTF-8 BOM（U+FEFF），用普通 utf-8 读进来
        #    ast.parse 会直接抛 `invalid non-printable character U+FEFF`。
        #    第一版就是这么栽的，而且被下面那个 except 静默吞成"空 QSS"⇒ 截图配色
        #    白得像没应用主题，我还以为是 apply_theme 的问题（典型的沉默错误）。
        src = open(os.path.join(ROOT, "main.py"), encoding="utf-8-sig").read()
        tree = ast.parse(src)
    except Exception as e:
        # 不静默：抠不到就等于"截图配色不代表产品"，必须让人看见
        print("  ⚠ 基础 QSS 抠取失败: %s: %s（截图配色将与产品不一致）"
              % (type(e).__name__, e), flush=True)
        return ""
    best = ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "setStyleSheet"):
            continue
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and len(a.value) > len(best):
                best = a.value
    return best


def _apply_product_theme(app, force=None):
    """按产品路径应用主题：登记基础表 → apply_theme → （主窗口构造完再）sweep。

    探针漏掉这步的代价（2026-10-14 差点误报成产品 bug）：
    `ui/main_window.py` 里左栏两个 tab 按钮的内联样式写死了 `color: #D6E4FF`
    （深色主题的浅蓝），没有 sweep 把颜色剥掉时，压在浅底上就是"两个空按钮"，
    看着像文字没渲染出来。产品里不会这样 —— sweep 会把"只含主题皮肤色"的内联样式
    剥色、交给主题表接管（`#d6e4ff` 正在 THEME_COLORS 白名单里）。

    顺带：不应用主题时，探针截图的**配色根本不是用户看到的样子**，
    拿它去讨论"好不好看"是无效的。所以截图前必须走这一步。
    """
    try:
        from ui.themes import apply_theme, set_base_qss, base_qss
    except Exception as e:
        print("  主题模块加载失败: %s: %s" % (type(e).__name__, e), flush=True)
        return None
    _base = _load_product_base_qss()
    if _base and not base_qss():
        set_base_qss(_base)
    name = force
    if not name:
        try:
            with open(os.path.join(ROOT, ".ui_settings.json"), encoding="utf-8") as f:
                name = json.load(f).get("theme", "dark")
        except Exception:
            name = "dark"
    # ⚠️ apply_theme 会顺手把主题**写回** .ui_settings.json。探针不许改用户设置
    #    （跑一次 --theme dark 就把用户的浅色改掉，属于"探测行为留下副作用"），
    #    所以先留原文、调用后还原。用覆盖写而不是删除 —— 本机的 safe-delete 钩子
    #    会拦 os.remove/unlink。
    _ui = os.path.join(ROOT, ".ui_settings.json")
    _bak = None
    try:
        if os.path.exists(_ui):
            with open(_ui, "rb") as f:
                _bak = f.read()
    except Exception:
        _bak = None
    apply_theme(app, name)
    # main.py 是「apply_theme 之后用 QTimer.singleShot(0) 再 sweep 一次」（那时控件树才建好）。
    # 探针在 MainWindow 构造完才调这里，直接同步 sweep 一次，效果等同。
    try:
        from ui.themes import sweep_inline_styles
        sweep_inline_styles(app, restore=False)
    except Exception:
        pass
    if _bak is not None:
        try:
            with open(_ui, "wb") as f:
                f.write(_bak)
        except Exception:
            pass
    return name


def pump(app, seconds=0.4):
    """跑一会儿事件循环（异步扫图/信号回灌都要它）。"""
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait_until(app, pred, timeout=25.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.02)
    return False


def build_project(name):
    d = os.path.join(tempfile.gettempdir(), "probe_mark_ui_%d" % int(time.time()), name)
    for sub in ("images", "annotations", "models", "outputs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    with io.open(os.path.join(d, "project.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "task_type": "语义分割",
                   "classes": ["background", "凹点"]}, f, ensure_ascii=False, indent=2)
    # 4 张图：2 张凹点、2 张划痕（缺陷类型从文件名推断）
    specs = [("p1-凹点1_te_0", "凹点"), ("p2-凹点2_te_0", "凹点"),
             ("p3-划痕1_te_0", "划痕"), ("p4-划痕2_te_0", "划痕")]
    for base, _defect in specs:
        arr = np.full((90, 140), 70, dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(d, "images", base + ".png"))
    return d, [b for b, _ in specs]


def dump_results(title):
    n_pass = sum(1 for ok_, _n, _d in _results if ok_)
    n_fail = len(_results) - n_pass
    lines = ["=" * 74, title, "=" * 74]
    for ok_, n, d in _results:
        lines.append("  %-4s %s" % ("PASS" if ok_ else "FAIL", n))
        if not ok_ and d:
            lines.append("       %s" % str(d)[:400])
    lines.append("-" * 74)
    lines.append("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (n_pass, n_fail, len(_results)))
    lines.append("=" * 74)
    _text = "\n".join(lines)
    print(_text, flush=True)

    # ★ 自动落盘（2026-10-14 加）：这个探针出现过**偶发** FAIL —— 同一份代码，
    #   一次 188/188、下一次 175/11，而当时是用 `| grep` 管道跑的、**没留日志**，
    #   11 条明细全丢了、事后归因不了（只能记成"疑似外部窗口抢焦点"，无法证实）。
    #   以后无论怎么调用（管道 / 重定向 / 双击 bat），明细都在 `_ui_probe_shots/` 里：
    #     · 有失败 -> `probe_last_FAIL.log`（**不被后续成功覆盖**，证据留得住）
    #     · 全通过 -> `probe_last_ok.log`
    try:
        _d = os.path.join(ROOT, "_ui_probe_shots")
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, "probe_last_%s.log" % ("FAIL" if n_fail else "ok")),
                  "w", encoding="utf-8") as f:
            f.write("# %s  （python probe_mark_ui.py %s）\n"
                    % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(sys.argv[1:])))
            f.write(_text + "\n")
    except Exception:
        pass
    return n_fail


# 多视图夹具里三个视图的**底色值**：刻意各不相同。
# 要是三格都写成同一个值，"底栏读的是光标底下那一格"这条断言就恒为真 ——
# 读错了格子（比如一律读主视图）也照样绿，等于没测（典型的"测试区分不出来"）。
# 现场那三张图也是各不相同的（漫反射是灰度分布，形状图1/2 是高度/梯度分布）。
_VIEW_FILL = {"te_0": 60, "te_1": 100, "te_2": 140}


def build_mv_project(name, groups=2, views=("te_0", "te_1", "te_2")):
    """造一个**真·多视图工程**。

    判据就一条：`project.json` 里声明了 `views`（`is_multiview_project()` 只认这个）。
    这里刻意按孚能现场那张截图的结构造：N 组 × 3 视图，文件名 `{组名}_te_{n}.png`。
    """
    d = os.path.join(tempfile.gettempdir(), "probe_mark_ui_%d" % int(time.time()), name)
    for sub in ("images", "annotations", "models", "outputs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    with io.open(os.path.join(d, "project.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "task_type": "语义分割",
                   "classes": ["background", "凹点"],
                   "views": list(views), "view_count": len(views),
                   "in_channels": 3 * len(views),
                   "view_names": {"te_0": "漫反射", "te_1": "形状图1", "te_2": "形状图2"}},
                  f, ensure_ascii=False, indent=2)
    bases = []
    for g in range(groups):
        b = "20260910-6号机-大面-凹点%d" % (g + 1)
        bases.append(b)
        for v in views:
            arr = np.full((90, 140), _VIEW_FILL.get(v, 60) + g, dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(d, "images", "%s_%s.png" % (b, v)))
    return d, bases, len(views)


def build_cls_project(name):
    """造一个专验「类别增删」的工程：5 个类别 + 用指定类别号画好的掩膜 + 推理参数。

    为什么不复用 `build_project` / `build_mv_project`：它们都只有 2 个类别
    （`["background", "凹点"]`），而第 0 行是 background、删不得 ⇒ **「删中间那个类别」
    这条路径根本走不到**。可那正是唯一会把已有标注**重新解释**的动作
    （掩膜里存的是类别号 = 类别表里的位置），也正是最该被守住的那条。

    掩膜刻意用了编号 1 / 3 / 4（跳过 2）：删掉编号 2 之后，
    "原来编号 3 的那片区域"是否落到了合法范围之外，才是个有内容的判据。
    """
    d = os.path.join(tempfile.gettempdir(), "probe_mark_ui_%d" % int(time.time()), name)
    for sub in ("images", "annotations", "models", "outputs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    classes = ["background", "A", "B", "C", "D"]
    with io.open(os.path.join(d, "project.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "task_type": "语义分割", "classes": classes},
                  f, ensure_ascii=False, indent=2)
    for i, cid in enumerate((1, 3, 4)):          # 3 组图，各含一个类别
        base = "cls%d_te_0" % (i + 1)
        Image.fromarray(np.full((40, 60), 80, dtype=np.uint8)).save(
            os.path.join(d, "images", base + ".png"))
        m = np.zeros((40, 60), dtype=np.uint8)
        m[5:20, 5:25] = cid
        Image.fromarray(m).save(os.path.join(d, "annotations", base + "_mask.png"))
    # 推理参数：每个类别一条 —— 删类别时那条同名记录必须跟着消失
    from inference.params import default_params, save_params
    save_params(d, default_params(classes))
    return d, classes


def _read_json(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    from ui.main_window import MainWindow
    from core import marks as mk

    win = MainWindow()
    win.resize(1500, 900)
    win.show()
    pump(app, 0.3)

    # 主题必须在 MainWindow 构造之后应用（sweep_inline_styles 要遍历已建好的控件树）。
    # 默认跟 .ui_settings.json；用 `--theme dark|light` 可覆盖（对比两张截图时用）。
    _theme_arg = ""
    if "--theme" in sys.argv:
        _i = sys.argv.index("--theme")
        if _i + 1 < len(sys.argv):
            _theme_arg = sys.argv[_i + 1]
    _theme_used = _apply_product_theme(app, _theme_arg)
    # 截图按主题分名：同一份探针跑两种主题时，后跑的会把前一张**覆盖**掉
    # （2026-10-14 实测：浅色那轮把深色的 _shot_with_info_bar.png 盖了，对比图就没了）。
    _tsuf = "_%s" % _theme_used if _theme_used in ("dark", "light") else ""
    pump(app, 0.3)
    stage("主题已应用：%s" % _theme_used)

    # 探针全程不许改动用户的工程注册表 —— 记下原文，结尾逐字节比对
    _reg_path = win.pm._custom_registry_path()
    _reg_before = _reg_path.read_text(encoding="utf-8") if _reg_path.exists() else ""

    pd, bases = build_project("probe_ui")
    stage("窗口已建好")
    ck("构造窗口成功（真 QMainWindow）", win is not None)
    ck("图片列表有 9 列且第 9 列是「标记」",
       win.image_list_widget.columnCount() == 9
       and win.image_list_widget.horizontalHeaderItem(8).text() == "标记",
       "cols=%d hdr8=%s" % (win.image_list_widget.columnCount(),
                            win.image_list_widget.horizontalHeaderItem(
                                win.image_list_widget.columnCount() - 1).text()))
    ck("标记工具条存在且是四个态按钮 + 清空 + 备注",
       hasattr(win, "mark_toolbar")
       and len(win.mark_toolbar._btns) == 4
       and win.mark_toolbar.btn_clear is not None
       and win.mark_toolbar.btn_note is not None,
       str(sorted(win.mark_toolbar._btns)) if hasattr(win, "mark_toolbar") else "无工具条")
    # ── 2026-10-15「7 个按钮挤一行」：判据是**文字有没有被裁**，不是"控件在不在布局里" ──
    _tb = getattr(win, "mark_toolbar", None)
    if _tb is not None:
        _all = list(_tb._btns.values()) + [_tb.btn_clear, _tb.btn_note, _tb.btn_stats]
        _clipped = [(b.text(), b.width(), b.sizeHint().width())
                    for b in _all if b.width() < b.sizeHint().width()]
        ck("★ 标记工具条 7 个按钮文字都没被裁（实宽 >= sizeHint 宽）",
           not _clipped, str(_clipped))
        # 量纲对齐：附加按钮原本走全局默认样式（sizeHint 80px），与四态（40px）差一个量级，
        # 硬 minWidth 加起来 294px 已超过左栏可用 ~300px ⇒ Qt 必然牺牲某几个按钮的宽度。
        _need = sum(b.sizeHint().width() for b in _all) + 2 * (len(_all) - 1) + 9
        _avail = _tb.width() if _tb.width() > 0 else 300
        ck("★ 标记工具条一行放得下（最小需求 <= 可用宽度，有余量不靠挤压）",
           _need <= _avail, "需求 %d / 可用 %d（差 %+d）" % (_need, _avail, _avail - _need))
        ck("★ 四态按钮是 Expanding（面板拉宽时均分富余，不右半截留白）",
           all(b.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
               for b in _tb._btns.values()),
           str([b.sizePolicy().horizontalPolicy() for b in _tb._btns.values()]))
        ck("★ 三个附加按钮有紧凑度量（内联 padding，非全局默认 5px 12px）",
           all("padding" in (b.styleSheet() or "") for b in (_tb.btn_clear, _tb.btn_note, _tb.btn_stats)),
           str([(b.text(), b.styleSheet()) for b in (_tb.btn_clear, _tb.btn_note, _tb.btn_stats)]))
    ck("标记统计面板存在", hasattr(win, "mark_stats"))
    ck("左侧手风琴有两页（图片列表 + 标记统计）",
       win._accordion_stack.count() == 2, str(win._accordion_stack.count()))
    # ── 2026-10-15 默认页改「工程管理」：必须在这里核（构造完、还没打开工程）──
    #    后面所有检查都是在"工程已打开"之后跑的，那时页签已被自动切走，量不到默认值。
    _d0 = win.left_tab_state()
    ck("★ 启动默认落在「工程管理」（页索引 0、工程树可见）",
       _d0["page"] == 0 and _d0["content_visible"] and _d0["checked"] == ["proj"],
           str(_d0))
    # ★ 页签**互斥**：勾选状态必须由页栈下标唯一决定。
    #   2026-10-15 踩到：show_left_tab() 原来只勾目标、从不去取消别的 ⇒
    #   "默认 proj + 打开工程自动切 img"之后两个按钮**同时勾着**，
    #   页签亮两颗 = 用户不知道现在在哪一页。（右栏用 QButtonGroup 独占，没这毛病；
    #   左栏为了"再点一次收起"特意用非独占，才暴露出来。）
    ck("★ 页签互斥：任意时刻**最多一颗**勾着",
           len(win.left_tab_state()["checked"]) <= 1,
           str(win.left_tab_state()["checked"]))
    _check_panel_handles(win, pump, ck, app)
    win.show_left_tab("img"); pump(app, 0.3)
    ck("★ 从「工程管理」切到「图像管理」后只剩 img 勾着（proj 自动弹起）",
       win.left_tab_state()["checked"] == ["img"], str(win.left_tab_state()))
    win.show_left_tab("proj"); pump(app, 0.3)
    # 配套：默认页改了以后，**打开工程必须自动切到「图像管理」**，否则点开工程后
    # 用户还盯着一页空图片列表。源码级守卫（AST 取调用节点，注释骗不过）。
    _src_mw_early = open(os.path.join(ROOT, "ui", "main_window.py"),
                         encoding="utf-8").read()
    _open_fn = [n for n in ast.walk(ast.parse(_src_mw_early))
                if isinstance(n, ast.FunctionDef) and n.name == "_open_project_by_name"]
    _has_switch = False
    if _open_fn:
        for n in ast.walk(_open_fn[0]):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "show_left_tab"):
                _has_switch = any(isinstance(a, ast.Constant) and a.value == "img"
                                  for a in n.args)
    ck("★ AST：`_open_project_by_name` 里有 `show_left_tab(\"img\")`（打开工程自动切图片列表）",
       _has_switch, "没找到该调用")
    # ── 2026-10-15 第二轮减负：页头整行（图片列表标题 + 标记统计按钮）已删 ──
    ck("★ 页头假标题已删干净（win 上不该再有 _img_list_title_label）",
       not hasattr(win, "_img_list_title_label"),
       str(getattr(win, "_img_list_title_label", None)))
    ck("★ 页头「标记统计」入口按钮已删干净（win 上不该再有 _accordion_btns）",
       not hasattr(win, "_accordion_btns"),
       str(getattr(win, "_accordion_btns", None)))
    # 入口只留一处（标记工具条「统计」按钮）；返回只留一处（统计页「← 返回列表」）——
    # 用真按钮点一遍，确认这两条路真通（删入口 ≠ 删功能，接错信号 = 点了没反应）。
    win._switch_accordion(0); pump(app, 0.2)
    _n_recv = win.mark_toolbar.btn_stats.receivers(win.mark_toolbar.btn_stats.clicked)
    ck("★ 标记工具条「统计」按钮的 clicked 真连着槽", _n_recv > 0, "receivers=%d" % _n_recv)
    win.mark_toolbar.btn_stats.click(); pump(app, 0.3)
    ck("点「统计」→ 切到标记统计页（accordion=1）",
       win._accordion_stack.currentIndex() == 1, str(win._accordion_stack.currentIndex()))
    _n_back = win.mark_stats.btn_back.receivers(win.mark_stats.btn_back.clicked)
    ck("★ 统计页「← 返回列表」按钮存在且 clicked 真连着槽",
       hasattr(win.mark_stats, "btn_back") and _n_back > 0,
       "存在=%s receivers=%d" % (hasattr(win.mark_stats, "btn_back"), _n_back))
    win.mark_stats.btn_back.click(); pump(app, 0.3)
    ck("点「← 返回列表」→ 切回图片列表页（accordion=0）",
       win._accordion_stack.currentIndex() == 0, str(win._accordion_stack.currentIndex()))

    # ── 打开工程（只登记到内存，见 _register_in_memory）──
    stage("开始打开工程")
    pname = _register_in_memory(win.pm, pd)
    win._open_project_by_name(pname)
    ok = wait_until(app, lambda: len(getattr(win, "_all_images", []) or []) >= len(bases), 40)
    stage("工程已扫描")
    ck("工程打开后图片列表被异步扫出来", ok,
       "拿到 %d 张" % len(getattr(win, "_all_images", []) or []))
    ck("列表渲染出 4 行", win.image_list_widget.rowCount() == len(bases),
       str(win.image_list_widget.rowCount()))
    ck("未标记时标记列是空的",
       all((win.image_list_widget.item(r, 8).text() or "") == "" for r in range(win.image_list_widget.rowCount())))

    # ── 单选打标记：过杀 ──
    win.image_list_widget.item(0, 0).setSelected(True)
    pump(app, 0.2)
    win._on_mark_requested("overkill")
    pump(app, 0.3)
    it = win.image_list_widget.item(0, 8)
    ck("点「过杀」后标记列文字=过杀", it is not None and it.text() == "过杀",
       it.text() if it else "无单元格")
    ck("标记格子带上了过杀色底",
       it is not None and it.background().color().name().lower() == mk.MARK_COLORS["overkill"].lower(),
       it.background().color().name() if it else "")
    # 序号列（永远在可视区）同步上色 —— 标记列在最右侧，窄左栏不横滚就看不见
    i0 = win.image_list_widget.item(0, 0)
    ck("序号格同步上同色底（不横向滚动也能一眼看到标记）",
       i0 is not None and i0.background().color().name().lower() == mk.MARK_COLORS["overkill"].lower(),
       i0.background().color().name() if i0 else "无单元格")
    mp = mk.marks_path(pd)
    ck("标记已落盘 image_marks.json", os.path.exists(mp))
    saved = mk.load_marks(pd) if os.path.exists(mp) else {}
    ck("落盘内容正确（mark=过杀 + 标记人 + 时间）",
       saved.get(bases[0], {}).get("mark") == "overkill"
       and saved.get(bases[0], {}).get("by") and saved.get(bases[0], {}).get("at"),
       str(saved.get(bases[0])))

    # ── 多选批量：把剩下 3 张分别标成 漏杀/正常/存疑 ──
    for row, kind, label in ((1, "escape", "漏杀"), (2, "normal", "正常"), (3, "doubt", "存疑")):
        win.image_list_widget.clearSelection()
        win.image_list_widget.item(row, 0).setSelected(True)
        pump(app, 0.1)
        win._on_mark_requested(kind)
        pump(app, 0.25)
        got = win.image_list_widget.item(row, 8).text()
        ck("第 %d 行标成「%s」" % (row + 1, label), got == label, got)

    saved = mk.load_marks(pd)
    ck("四态都落盘了", sorted(v["mark"] for v in saved.values())
       == ["doubt", "escape", "normal", "overkill"], str(sorted(v["mark"] for v in saved.values())))

    stage("四态标记已打完")
    # ── 统计面板 ──
    win._switch_accordion(1)
    pump(app, 0.3)
    ck("统计页被切到前台", win._accordion_stack.currentIndex() == 1)
    chips = {k: win.mark_stats._chips[k].text() for k in mk.MARK_KINDS}
    ck("四态计数正确（各 1）",
       all(chips[k].endswith(" 1") for k in mk.MARK_KINDS), str(chips))
    ck("统计面板交叉表按缺陷类型分行（凹点/划痕）",
       win.mark_stats.table.rowCount() == 2, str(win.mark_stats.table.rowCount()))
    names = [win.mark_stats.table.item(r, 0).text() for r in range(win.mark_stats.table.rowCount())]
    ck("交叉表行名是缺陷类型", set(names) == {"凹点", "划痕"}, str(names))
    ck("占比文字已生成", "占比" in win.mark_stats.lbl_ratio.text(), win.mark_stats.lbl_ratio.text())

    # ── 点交叉表格子 = 筛选 ──
    ck("交叉表里能查到「凹点」这一行", "凹点" in names, str(names))
    # 逐格核数字：截图肉眼看容易把列对齐看错，代码断言才是判据
    def _cell(rname, kind):
        if rname not in names:
            return "?"
        it = win.mark_stats.table.item(names.index(rname),
                                       1 + list(mk.MARK_KINDS).index(kind))
        return it.text() if it is not None else ""
    grid = {(rn, k): _cell(rn, k) for rn in ("凹点", "划痕") for k in mk.MARK_KINDS}
    ck("交叉表数字对得上（凹点=过杀1+漏杀1；划痕=正常1+存疑1）",
       grid[("凹点", "overkill")] == "1" and grid[("凹点", "escape")] == "1"
       and grid[("划痕", "normal")] == "1" and grid[("划痕", "doubt")] == "1"
       and grid[("划痕", "overkill")] == "" and grid[("凹点", "normal")] == "",
       str(grid))
    if "凹点" in names:
        r_ao = names.index("凹点")
        win.mark_stats._on_cell(r_ao, 2)          # 第 2 列 = 过杀
        pump(app, 0.3)
        f = [r["base"] for r in win._filtered_data]
        ck("点交叉表格子 -> 只剩「凹点 + 过杀」那张", f == [bases[0]], str(f))
        ck("列表上方回显了筛选条件", win.mark_filter_label.isVisible()
           and "过杀" in win.mark_filter_label.text(), win.mark_filter_label.text())
        ck("筛选后列表只渲染 1 行", win.image_list_widget.rowCount() == 1,
           str(win.image_list_widget.rowCount()))

    # ── 表达式筛选叠加 ──
    from core.filter_expr import FilterExpr
    win._filter_expr = FilterExpr([{"field": "mark", "op": "==", "value": "escape"}], "AND")
    win._mark_filter = ""
    win._defect_filter = ""
    win._refresh_after_mark_change()
    pump(app, 0.3)
    f = [r["base"] for r in win._filtered_data]
    ck("表达式筛选（标记=漏杀）生效", f == [bases[1]], str(f))
    win._clear_mark_filters()
    pump(app, 0.3)
    ck("清除筛选后 4 张全回来", len(win._filtered_data) == len(bases),
       str(len(win._filtered_data)))

    # ── 筛选构建对话框（不 exec，只验能构造与出表达式）──
    try:
        from ui.filter_builder import FilterBuilderDialog
        dlg = FilterBuilderDialog(win, expr=win._filter_expr)
        ex = dlg.result_expr()
        ck("筛选对话框能构造并产出表达式", ex is not None and ex.logic in ("AND", "OR"))
        ck("对话框有四个常用快捷入口预设",
           len(dlg._quick) >= 4, str([q[0] for q in dlg._quick]))
        dlg._apply_quick([{"field": "mark", "op": "==", "value": "overkill"}], "AND")
        ex2 = dlg.result_expr()
        ck("快捷入口「筛过杀」填出的条件正确",
           ex2.active_count() == 1 and ex2.conditions[0]["field"] == "mark"
           and ex2.conditions[0]["value"] == "overkill", str(ex2.to_dict()))
        dlg.close()
    except Exception as e:
        ck("筛选对话框能构造并产出表达式", False, "%s: %s" % (type(e).__name__, e))

    stage("筛选相关已验完")
    # ── 一键闭环（走**真按钮**，不看代码）：过杀 → 负样本 → 进训练集 ──
    #    这里必须把 QMessageBox 的确认框换成不阻塞的桩，否则探针会卡在模态框上
    #    （第一次跑 240s 超时就是这个原因之外的另一个地雷）。
    from PyQt5.QtWidgets import QMessageBox
    _orig = {k: getattr(QMessageBox, k) for k in ("question", "information", "critical", "warning")}
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    try:
        # 造一份"AI 分割结果"：p1（凹点，标了过杀）有一块前景
        from flow.nodes.output import SegResultWriter
        seg = np.zeros((90, 140), dtype=np.uint8)
        seg[30:60, 40:100] = 1
        Image.fromarray(seg).save(os.path.join(pd, "outputs", bases[0] + "_pred.png"))

        win.image_list_widget.clearSelection()
        win.image_list_widget.item(0, 0).setSelected(True)
        pump(app, 0.2)
        # 原走「结果转标注…」面板按钮；2026-10-15 该按钮收进图片列表右键菜单，
        # 面板上删了 ⇒ 这里直接调按钮背后的槽（右键菜单是模态 exec_，探针不方便触发）。
        win._run_convert_by_mark()
        pump(app, 0.8)
        ann = os.path.join(pd, "images", bases[0] + ".json")
        ck("一键闭环后，过杀图落下了标注文件（负样本才有东西可训练）", os.path.exists(ann))
        shapes = []
        if os.path.exists(ann):
            with io.open(ann, encoding="utf-8") as f:
                shapes = (json.load(f) or {}).get("shapes") or []
        ck("过杀图转成的是**负样本**（AI 那块前景被扣成背景，不是 255）", shapes == [], str(shapes))
        sp = os.path.join(pd, "train_test_split.json")
        sm = {}
        if os.path.exists(sp):
            with io.open(sp, encoding="utf-8") as f:
                sm = json.load(f) or {}
            sm = sm.get("split", sm)
        ck("该图已真写进 train_test_split.json 的 train", sm.get(bases[0]) == "train", str(sm))
        ck("日志区出现了「结果转标注」的汇总（说明信号回灌到了真窗口）",
           "结果转标注" in win.log_widget.toPlainText(),
           win.log_widget.toPlainText()[-300:])
    finally:
        for _k, _v in _orig.items():
            setattr(QMessageBox, _k, _v)

    # ── 八维面板：勾选框真的在，且关掉一维会写进参数 ──
    panel = getattr(win, "infer_params", None)
    if panel is not None:
        ck("推理参数面板有 7 个区间维的启用勾选框",
           len(getattr(panel, "_enables", {})) == 7, str(len(getattr(panel, "_enables", {}))))
        ck("中心半径维也有启用勾选框", hasattr(panel, "center_radius_chk"))
        ck("默认全开（勾选框全部 checked）",
           all(c.isChecked() for c in panel._enables.values()))
        ck("面板识别到半径这一行", "radius" in panel._spins, str(sorted(panel._spins)))
    else:
        ck("推理参数面板存在", False, "win.infer_params 不存在")

    stage("闭环已验完")
    # ── 多视图工程：列表"按组成行"，计数必须分「组 / 张」两个量纲 ──
    #    背景（2026-10-14 用户实测）：孚能三视图工程 99 张 = 33 组，列表 33 行，
    #    状态栏写 "Showing 33/99" ⇒ 用户以为少了 66 张。根因是拿**行数**去除**文件数**。
    stage("开始验多视图列表")
    pd2, gbases, nviews = build_mv_project("probe_mv")
    n_files2 = len(gbases) * nviews
    win._open_project_by_name(_register_in_memory(win.pm, pd2))
    ok = wait_until(app, lambda: len(getattr(win, "_all_images", []) or []) >= n_files2, 40)
    ck("多视图工程扫出全部 %d 张（%d 组 × %d 视图）" % (n_files2, len(gbases), nviews), ok,
       "拿到 %d 张" % len(getattr(win, "_all_images", []) or []))
    ck("工程被识别为多视图", bool(getattr(win, "_mv_mode_list", False)),
       str(getattr(win, "_mv_view_meta", None)))
    ck("列表按组归并：%d 张 → %d 行" % (n_files2, len(gbases)),
       win.image_list_widget.rowCount() == len(gbases)
       and len(win._filtered_data) == len(gbases),
       "rows=%d filtered=%d" % (win.image_list_widget.rowCount(), len(win._filtered_data)))
    ck("「图片名」格里堆了 %d 行文件名（点进去能看到每张图）" % nviews,
       all((win.image_list_widget.item(r, 1).text() or "").count("\n") == nviews - 1
           for r in range(win.image_list_widget.rowCount())))
    _s1 = win.stats_summary.text()
    _s2 = win.list_status.text()
    ck("未筛选时第一行不再出现 Showing x/y（老写法在这里误报 33/99）",
       "Showing" not in _s1, _s1)
    ck("第一行（分页条）同时给出「组」和「张」两个量纲",
       "组" in _s1 and ("%d 张" % n_files2) in _s1, _s1)
    ck("第二行（状态栏）给数据集划分，且「组」单位带上了",
       re.search(r"Ann: \d+ 组", _s2) is not None, _s2)
    # ★ 去冗余（2026-10-14 用户："这个地方信息冗余了"）
    #   两行曾各自报一遍 Total / Train / Ann，数字一字不差。现在钉死分工：
    #   第一行 = 列表范围（多少组/多少张），第二行 = 数据集统计（Train/Val/Ann/Blocks）。
    ck("第一行不报 Train / Ann（那是第二行的职责）",
       "Train" not in _s1 and "Ann" not in _s1, _s1)
    ck("第二行不再重复总量（Total 段已移走）", "Total" not in _s2, _s2)
    ck("两行的字段零重叠（信息不重复）",
       not (_fields(_s1) & _fields(_s2)),
       "%s ∩ %s = %s" % (sorted(_fields(_s1)), sorted(_fields(_s2)),
                         sorted(_fields(_s1) & _fields(_s2))))
    # 分页标签曾写"全部 33"：与左边"共 33 组"是同一个数字的第二遍，
    # 且窄面板下会被截断成"全部 3"（用户截图里就是这样）。
    ck("分页标签不再重复报行数（也不会被截断）",
       re.search(r"\d", win.page_info_label.text()) is None,
       win.page_info_label.text())

    # 真筛一次：筛完必须能分清"筛掉的是组还是张"
    win._mark_filter = "overkill"
    win._refresh_after_mark_change()
    pump(app, 0.3)
    _s1f = win.stats_summary.text()
    ck("筛选后第一行说明筛掉多少（组/张 双量纲）",
       "筛选" in _s1f and "组" in _s1f and "张" in _s1f, _s1f)
    ck("筛选后第二行仍只讲数据集统计（不掺列表范围）",
       "Total" not in win.list_status.text() and "筛选" not in win.list_status.text(),
       win.list_status.text())
    win._clear_mark_filters()
    pump(app, 0.3)

    # 多视图行的标记落在该组**主视图**（te_0）base 上 —— 与 _row_bases 的口径一致
    win.image_list_widget.clearSelection()
    win.image_list_widget.item(0, 0).setSelected(True)
    pump(app, 0.2)
    win._on_mark_requested("overkill")
    pump(app, 0.3)
    saved2 = mk.load_marks(pd2)
    ck("多视图行的标记落在该组主视图 base（_te_0）上，不是整组名",
       list(saved2.keys()) == [gbases[0] + "_te_0"], str(list(saved2.keys())))
    ck("多视图行的序号格也上了标记色",
       win.image_list_widget.item(0, 0).background().color().name().lower()
       == mk.MARK_COLORS["overkill"].lower())

    # ── 截图（肉眼看文字是否有渲染）──
    win._switch_accordion(1)
    pump(app, 0.4)
    shot_dir = os.path.join(ROOT, "_ui_probe_shots")
    os.makedirs(shot_dir, exist_ok=True)
    p1 = os.path.join(shot_dir, "_shot_mark_stats" + _tsuf + ".png")
    win.grab().save(p1)
    win._switch_accordion(0)
    pump(app, 0.4)
    p2 = os.path.join(shot_dir, "_shot_mark_list" + _tsuf + ".png")
    win.grab().save(p2)
    # 左栏特写：探针窗口 1500 宽（左栏被压到 ~200px），跟用户实际（~900px 窗口、
    # 左栏占一半左右）差太远，看不出"控件挤不挤、有没有被截断"。按用户的比例再抓一张。
    p3 = os.path.join(shot_dir, "_shot_left_panel" + _tsuf + ".png")
    try:
        win._main_splitter.setSizes([470, 1030])
        pump(app, 0.4)
        win._left_panel.grab().save(p3)
    except Exception as e:
        print("  左栏特写失败: %s: %s" % (type(e).__name__, e), flush=True)
    ck("截图已落盘（供肉眼看文字是否渲染）",
       os.path.exists(p1) and os.path.getsize(p1) > 20000
       and os.path.exists(p2) and os.path.getsize(p2) > 20000
       and os.path.exists(p3) and os.path.getsize(p3) > 10000,
       "%s / %s / %s" % (p1, p2, p3))

    stage("截图已存")

    # ── ★ 本轮（2026-10-14 下午）：底部「图片信息栏」+ 顶栏去重 + 耗时单位 ──
    #    用户给的两张图：一张是成熟视觉软件的图片信息栏（文件/宽高/坐标/RGB/已标注/
    #    已检出/耗时/平均耗时），一张是我们自己的多视图顶栏。这里把它钉死成断言，
    #    免得下次改布局时又被"顺手"去掉。
    stage("核底部图片信息栏")
    _bar_names = ("_info_file", "_info_dims", "_info_xy", "_info_rgb",
                  "_info_ann", "_info_det", "_info_ms", "_info_avg")
    ck("底栏八个字段控件都在", all(hasattr(win, n) for n in _bar_names),
       ",".join(n for n in _bar_names if not hasattr(win, n)) or "8/8")
    _bar = getattr(win, "_img_info_bar", None)
    ck("底栏挂在图像区**下方**（不是上面、也不是被塞进状态栏）",
       _bar is not None and win._viewer_stack.parentWidget().layout().indexOf(_bar)
       > win._viewer_stack.parentWidget().layout().indexOf(win._viewer_stack),
       "bar=%s" % (type(_bar).__name__ if _bar is not None else None))
    # ⚠️ 必须连**高度**一起判：只查 isVisible() 会被"有父对象但没进布局"的空壳骗过
    #    （实测整条栏只剩 3px 高、标签一个都看不见，isVisible 依然为 True）。
    ck("底栏在真窗口里可见，且真的有高度（不是 3px 空壳）",
       _bar is not None and _bar.isVisible() and _bar.height() >= 16,
       "visible=%s h=%d" % (None if _bar is None else _bar.isVisible(),
                            -1 if _bar is None else _bar.height()))

    # 底栏的数据必须与**图片列表那一行**同源（列表说几块、底栏就得是几块）
    win.image_list_widget.setCurrentCell(0, 1)      # 真路径：换 currentCell -> 载图
    pump(app, 0.4)
    ck("选行后画布真的载入了图（底栏才有东西可报）",
       bool(getattr(win.canvas, "current_image_path", ""))
       and os.path.exists(win.canvas.current_image_path),
       str(getattr(win.canvas, "current_image_path", "")))
    _base0 = win._filtered_data[0]["base"] if win._filtered_data else ""
    _rec0 = win._img_info_record(_base0)
    ck("底栏「已标注/已检出」就是列表那条记录里的数字（不另算一份）",
       _rec0 is not None and win._info_ann.text() == "已标注 %s" % _rec0["shape_count"]
       and win._info_det.text() == "已检出 %s" % _rec0["pred_cnt"],
       "%s / %s（记录 %s / %s）" % (win._info_ann.text(), win._info_det.text(),
                                   _rec0["shape_count"] if _rec0 else None,
                                   _rec0["pred_cnt"] if _rec0 else None))
    ck("底栏「宽高」用的是记录里的尺寸（与列表「大小」列同一个数）",
       (not _rec0) or win._info_dims.text() == "宽高 %d×%d" % (_rec0["img_w"], _rec0["img_h"]),
       win._info_dims.text())
    ck("底栏文件名字段带全路径 tooltip（中间省略后仍可查完整名）",
       _base0.split("_")[0] in win._info_file.fullText() and bool(win._info_file.toolTip()),
       "%s | tip=%s" % (win._info_file.text(), win._info_file.toolTip()))

    # 光标坐标 + RGB：走**真事件**进 mouseMoveEvent（不是直接调 _on_canvas_cursor_moved）
    from PyQt5.QtCore import QPoint as _QP, QPointF as _QPF, QEvent as _QE
    from PyQt5.QtGui import QMouseEvent as _QME

    def _move_widget_to(w, pos, verify=None, tries=8):
        """把一次真 MouseMove 送进**任意控件**（优先 QTest，退化 sendEvent，未生效会重试）。

        ⚠️ 必须走真事件，不能直接调 `_on_canvas_cursor_moved`：参考窗曾经因为
           内层 QLabel 默认不开鼠标跟踪而**一个事件都收不到**（用户 2026-10-14 报的
           "光标停在形状1/形状2 上时值是空的"就是它）。直接调函数会把这个坑整个绕开 ——
           全绿而现场照样空白。

        ⚠️ 为什么要重试：**冷启动那一跑**面板可能还没完成首次渲染
           （`_view_state()` 拿不到映射就直接不上报）。实测同一条命令第一次跑 9 条 FAIL、
           紧接着第二次跑只剩 1 条 —— 这种 flaky 比 FAIL 更害人：
           真 FAIL 会被当成噪声"再跑一遍就绿了"糊过去。

        ⚠️ `verify` **不要**写成"底栏不是空"：底栏这时多半停着**上一格**的值，
           一判就为真 ⇒ 事件其实没送到也报"成功"（这条 flaky 就是这么来的）。
           要验就验"**这一格自己的上报发生了**"（调用方传 `len(_src_log) > k`）。
        """
        how = "none"
        for i in range(tries):
            if i % 2 == 0:
                try:
                    from PyQt5.QtTest import QTest
                    QTest.mouseMove(w, _QP(pos[0], pos[1]), 10)
                    how = "QTest.mouseMove"
                except Exception as e:
                    print("  QTest.mouseMove(%s) 不可用：%s"
                          % (type(w).__name__, e), flush=True)
            else:
                QApplication.instance().sendEvent(
                    w, _QME(_QE.MouseMove, _QPF(pos[0], pos[1]),
                            Qt.NoButton, Qt.NoButton, Qt.NoModifier))
                how = "sendEvent(QMouseEvent)"
            pump(app, 0.4 if i == 0 else 0.25)
            if verify is None or verify():
                return how
        return how + "(重试 %d 次仍未生效)" % tries

    # 主画布那格也挂个"上报计数器"：`_move_to` 的 verify 同样不能只看底栏非空
    # （同一个坑：底栏可能停着别的值，一判就真）。
    _cv_log = []
    try:
        win.canvas.cursor_moved.connect(lambda *a: _cv_log.append(a))
    except Exception:
        pass

    def _move_to(pos):
        _k = len(_cv_log)
        return _move_widget_to(win.canvas, pos, lambda: len(_cv_log) > _k)

    def _check_ref_hover(tag, require_distinct=True):
        """三格（漫反射 / 形状图1 / 形状图2）轮流停一遍光标，核底栏那两项。

        为什么抽成函数：夹具工程是我自己造的图，"三格值不同"是刻意安排的；
        而用户报的是**真实工程**（1175×3739 的 L 灰度 JPEG × 99 张）。两边都得跑，
        否则等于只测了"我造的那个世界"。

        判据分两层，缺一层都不算数：
          ① 报出来的值 == **那一格自己的文件**在 (x,y) 的像素（独立来源对拍，
             不复用产品里的换算公式 —— 复用公式的话公式错了也测不出来）
          ② （夹具）三格在同一坐标上的值互不相同 —— 只断言"非空"的话，
             一律错读成主视图也照样绿。真实工程不强求，只把三档打印出来肉眼核。
        """
        _panels = list(getattr(win.mv_preview, "_panels", []) or [])
        _n = len(getattr(win.mv_preview, "_items", []) or [])
        from PIL import Image as _PILImage      # 只读文件头/取像素，用来做独立来源对拍
        ck("%s：多视图建出了 %d 个参考窗" % (tag, _n - 1),
           _n >= 2 and len(_panels) == _n - 1, "views=%d panels=%d" % (_n, len(_panels)))
        ck("%s：参考窗两层都开了鼠标跟踪（不开就一个 MouseMove 都收不到，坐标必空）"
           % tag,
           bool(_panels) and all(p.hasMouseTracking() and p.view.hasMouseTracking()
                                 for p in _panels),
           str([(p.hasMouseTracking(), p.view.hasMouseTracking()) for p in _panels]))
        ck("%s：容器有光标汇总信号（底栏只挂这一个钩子）" % tag,
           hasattr(win.mv_preview, "cursor_moved"),
           "没有它 ⇒ 底栏对参考窗必然是空的")
        if not _panels:
            return
        # ── 契约断言：像素值的**形态**由视图角色定，不由文件 mode 定 ──
        # 用户 2026-10-14 的原话："形状1和形状2都要显示RGB"。用**真调用**锁死行为：
        # 夹具与真实工程的形状图源文件都是 L 灰度，旧实现会返回 int（底栏写「灰度 N」），
        # 现在参考窗必须返回三元组。⚠️ 这里不能写成"读源码里有没有 RGB 字样"——
        # 注释里写一遍就把断言骗绿了（本项目已有先例）。
        _pp = _panels[0].pixel_at(5, 5)
        ck("%s：参考窗 pixel_at 给的是 (r,g,b) 三元组（形状图固定 RGB 形态）" % tag,
           isinstance(_pp, tuple) and len(_pp) == 3, repr(_pp))
        # 主画布那格：`AnnotationCanvas` 加载图时一律 `.convert("RGB")`（mask_editor.py），
        # 所以它**本来就是 RGB 形态**。用户看到的"漫反射 RGB、形状图两格灰度"正是三格不齐，
        # 本次改的就是把形状图那两格拉齐到 RGB。这条断言钉住"三格形态一致"。
        # ⚠️ 别以为主画布会给 int：`canvas.image` 不是"文件原样"，是转换后的 RGB。
        _cv = type(win)._pixel_at(win.canvas, 5, 5)
        ck("%s：主画布（canvas.image 已 convert RGB）也给三元组 ⇒ 三格形态一致" % tag,
           isinstance(_cv, tuple) and len(_cv) == 3,
           "%r (%s)" % (_cv, type(_cv).__name__))
        # 探针槽：记下**每次上报的来源是谁**。比"信号连没连上"强 ——
        # 连上了但三格都自称主画布，一样是坏的，而那种坏只有看来源才看得见。
        _src_log = []
        win.mv_preview.cursor_moved.connect(
            lambda x, y, s: _src_log.append(
                (x, y, "canvas" if s is win.canvas else "ref")))
        # 先把上一轮的停留状态清干净：面板会去重（同一个整数像素不重复发），
        # 带着旧状态进来可能整段"什么都不发"，被误判成没接上。
        for _p in _panels:
            QApplication.instance().sendEvent(_p.view, _QE(_QE.Leave))
        _move_widget_to(win.canvas, (12, 12))   # 画布同理，先挪开
        pump(app, 0.15)

        _seen = {}
        _targets = [("漫反射（主画布）", win.canvas, 0)] + \
                   [("形状图%d" % (i + 1), p.view, i + 1) for i, p in enumerate(_panels)]
        for _lbl, _w, _idx in _targets:
            _vp = (max(8, _w.width() // 2), max(8, _w.height() // 2))
            # verify 用"这一格自己上报了没"（`_src_log` 增长），不用"底栏不是空" ——
            # 底栏此刻停着上一格的值，用后者等于不验。
            _k0 = len(_src_log)
            _hw = _move_widget_to(_w, _vp, lambda: len(_src_log) > _k0)
            _src = _src_log[-1][2] if _src_log else None
            ck("%s / %s：上报来源标的是这一格（%s）" % (tag, _lbl, _src),
               _src == ("canvas" if _idx == 0 else "ref"), str(_src_log[-3:]))
            _sxy, _sv = win._info_xy.text(), win._info_rgb.text()
            ck("%s / %s：光标停上去就报出坐标（%s）" % (tag, _lbl, _hw),
               _sxy.startswith("坐标 ") and _sxy != "坐标 --", _sxy)
            ck("%s / %s：像素值也一并出来了（不再是空的）" % (tag, _lbl),
               _sv not in ("RGB --", "灰度 --"), "%s | %s" % (_sxy, _sv))
            _m = re.match(r"坐标 (\d+),(\d+)$", _sxy)
            if not _m:
                ck("%s / %s：坐标能解析（后面要靠它去对拍像素）" % (tag, _lbl), False, _sxy)
                continue
            _mx, _my = int(_m.group(1)), int(_m.group(2))
            # 独立来源：主画布走画布里的 PIL 图；参考窗走它**自己那个文件**
            if _idx == 0:
                _pv = win.canvas.image.getpixel((_mx, _my))
                _from = "canvas.image"
            else:
                _path = getattr(_panels[_idx - 1], "_path", "")
                if not _path or not os.path.exists(_path):
                    ck("%s / %s：面板记得住自己的图片路径" % (tag, _lbl), False, str(_path))
                    continue
                with _PILImage.open(_path) as _im:
                    _pv = _im.getpixel((_mx, _my))
                _from = os.path.basename(_path)
            # 值**形态**（产品口径，Arthur 2026-10-14："形状1和形状2都要显示RGB"）：
            #   三格**统一 RGB**。主画布那格本来就是 —— `AnnotationCanvas.image` 加载时
            #   就 `.convert("RGB")` 了；参考窗那两格早先按文件 mode 走（L 灰度写「灰度 N」），
            #   本次改成一律三元组。
            # 期望串按各格**自己的独立来源**算（主画布走 canvas.image、参考窗走它自己的文件），
            # 不复用产品里的换算公式 —— 复用公式的话公式错了也测不出来。
            if _idx == 0:
                _exp = ("灰度 %d" % int(_pv)) if isinstance(_pv, int) else \
                       ("RGB %d,%d,%d" % (int(_pv[0]), int(_pv[1]), int(_pv[2])))
                _form = "灰度" if isinstance(_pv, int) else "RGB"
            else:
                # 形状图那两格：源文件可能是 L 灰度（三值相等），也可能本身是彩色的。
                # 两种情况都按 RGB 形态写 —— 形态由**格子**定，不由文件 mode 定。
                _rgb = (_pv, _pv, _pv) if isinstance(_pv, int) else (_pv[0], _pv[1], _pv[2])
                _exp = "RGB %d,%d,%d" % (int(_rgb[0]), int(_rgb[1]), int(_rgb[2]))
                _form = "RGB"
            ck("%s / %s：报的就是**它自己那张图**在 (%d,%d) 的像素（%s）"
               % (tag, _lbl, _mx, _my, _from), _sv == _exp,
               "期望 %s，实际 %s" % (_exp, _sv))
            ck("%s / %s：值形态是「%s *」（三格统一 RGB，形状图不再写「灰度」）"
               % (tag, _lbl, _form), _sv.startswith(_form + " "), _sv)
            _seen[_lbl] = (_mx, _my, _sv)
        print("  [%s] 三格读数：%s" % (tag, _seen), flush=True)
        if require_distinct:
            _vals = [v[2] for v in _seen.values()]
            ck("%s：三格报出来的像素值互不相同（说明真的各读各的图，不是都在读主视图）"
               % tag, len(_vals) == _n and len(set(_vals)) == _n, str(_seen))
        # 三窗同步的意义就在这里：同一个坐标在三格里指的是同一处。
        # 差个一两像素是缩放取整带来的（render_view 的目标矩形是 round 过的），可接受；
        # 哪天换算公式改了、三格对不上，这条会立刻红。
        _coords = [(v[0], v[1]) for v in _seen.values()]
        if _coords:
            _dx = max(c[0] for c in _coords) - min(c[0] for c in _coords)
            _dy = max(c[1] for c in _coords) - min(c[1] for c in _coords)
            ck("%s：三格在同一窗口位置给出的坐标是同一个（三窗同步要保证的就是这个）"
               % tag, _dx <= 2 and _dy <= 2, "偏差 dx=%d dy=%d（%s）" % (_dx, _dy, _seen))
        # 离开 -> 清空；再回来 -> 重新上报（节流缓存得清干净）
        _last = _panels[-1]
        QApplication.instance().sendEvent(_last.view, _QE(_QE.Leave))
        pump(app, 0.15)
        ck("%s：光标离开形状图后坐标清空（不留最后一次的假读数）" % tag,
           win._info_xy.text() == "坐标 --" and win._info_rgb.text() == "RGB --",
           "%s / %s" % (win._info_xy.text(), win._info_rgb.text()))
        _k1 = len(_src_log)
        _move_widget_to(_last.view,
                        (max(8, _last.view.width() // 2),
                         max(8, _last.view.height() // 2)),
                        lambda: len(_src_log) > _k1)
        ck("%s：离开后再回到形状图同一点仍会重新上报（节流缓存清干净了）" % tag,
           win._info_xy.text() != "坐标 --", win._info_xy.text())

        # ── 顺手守住"在参考窗上按住拖动也能平移" ──
        # 本次把光标上报插进了 eventFilter 里原本 `obj is self.view and
        # self._on_wheel is not None` 那个判断，改了组织结构：别把平移转发弄丢。
        # 平移是拖动才触发的（不按键就没有 MouseMove 的语义），所以这里必须
        # 真发 press + move + release 三件套，而不是只发一个 MouseMove。
        _p = _panels[0].view
        _c = win.canvas
        _sx, _sy = (float(getattr(_c, "pan_offset_x", 0.0)),
                    float(getattr(_c, "pan_offset_y", 0.0)))
        _o = _QP(max(10, _p.width() // 2), max(10, _p.height() // 2))
        for _t, _pt, _bt in ((_QE.MouseButtonPress, _o, Qt.LeftButton),
                             (_QE.MouseMove, _o + _QP(30, 12), Qt.NoButton),
                             (_QE.MouseButtonRelease, _o + _QP(30, 12), Qt.LeftButton)):
            _btn = Qt.LeftButton if _t != _QE.MouseMove else Qt.NoButton
            _btns = Qt.LeftButton if _t != _QE.MouseButtonRelease else Qt.NoButton
            QApplication.instance().sendEvent(
                _p, _QME(_t, _QPF(_pt), _btn, _btns, Qt.NoModifier))
        pump(app, 0.3)
        _nx = float(getattr(_c, "pan_offset_x", 0.0))
        ck("%s：在参考窗上按住拖动仍能平移画布（本次改了 eventFilter，守住它）" % tag,
           abs(_nx - _sx) > 1.0, "pan_offset_x %.1f -> %.1f" % (_sx, _nx))
        # 复位，别把平移后的画面留给后面的截图
        _c.pan_offset_x, _c.pan_offset_y = _sx, _sy
        try:
            _c.update()
        except Exception:
            pass
        win.mv_preview._sync_from_canvas(force=True)
        pump(app, 0.25)
        # ★ 视图动过之后再停一次：读数必须是按**当前**映射算出来的，不是历史缓存。
        #   判据：和拖动前同一个窗口位置（view 中心）给出的坐标应当完全一致 ——
        #   平移复位了，映射就该回到原处。若哪天换算被改坏，或读数变成"最后一次的值"
        #   不跟着映射走，这条会立刻红。
        _tgt = _panels[0]
        _vp0 = (max(8, _tgt.view.width() // 2), max(8, _tgt.view.height() // 2))
        _k2 = len(_src_log)
        _move_widget_to(_tgt.view, _vp0, lambda: len(_src_log) > _k2)
        _before = _seen.get("形状图1")
        ck("%s：视图变过（拖动+复位）后再停上去，坐标立刻按新映射给出（不是历史值）" % tag,
           _before is None or win._info_xy.text() == "坐标 %d,%d" % (_before[0], _before[1]),
           "拖动前 %s，复位后 %s" % (_before, win._info_xy.text()))

    def _check_bars_merge(tag, multi):
        """★ 顶栏并进底栏：工具栏右侧状态区 + 底栏右端缩放控件。

        用户 2026-10-14 的原话："我感觉咱们顶栏和底栏可以合二为一，你可以参考一下友商的"。
        重点不是"控件在不在"，而是这几条**只有真窗口才验得出来**的：
          · 三个图层开关在多视图模式下必须真的露面 —— 它们原先全屏只有顶栏那一处显示，
            只能靠 Ctrl+Space / Ctrl+Alt 盲切，删掉顶栏就等于删掉了唯一的可见入口；
          · 点按钮要真的动到画布（走 `toggle_*`），而不是只改了自己的 checked；
          · 反向也要成立：画布（快捷键那条路）改了状态，按钮要跟着变；
          · 滑块拖完**不能弹回去** —— `AnnotationCanvas.resizeEvent` 是无条件
            `_fit_to_window()`，只要底栏显隐在抖就会把刚设的缩放重置（冒烟实测踩到过）。
        """
        mv = win.mv_preview
        tb = win.annotation_toolbar
        # ── 2026-10-15 工具行折叠（友商"点标注显示隐藏工具栏"）──
        # 判据不是"按钮还在不在"，是三件量化的事：
        #   ① 默认折叠；② 点一次真展开、再点真收起；③ **折叠后画布真的变高**
        #      （折叠一个布局兄弟控件，画布理应把这 30px 拿走 —— 不变高 = 折叠是假的）
        ck("★ 工具行默认折叠（Arthur 拍板的默认态）", not tb.tools_visible(),
           "tools_visible=%s" % tb.tools_visible())
        _h0 = win.canvas.height()
        tb.tools_toggle.click(); pump(app, 0.4)
        ck("★ 点「标注」-> 工具行展开", tb.tools_visible(), "tools_visible=%s" % tb.tools_visible())
        _h1 = win.canvas.height()
        tb.tools_toggle.click(); pump(app, 0.4)
        ck("★ 再点「标注」-> 工具行收起（往返可用）", not tb.tools_visible(),
           "tools_visible=%s" % tb.tools_visible())
        _h2 = win.canvas.height()
        ck("★ 折叠后画布真的变高了（纵向空间还给图像，不是假折叠）",
           _h2 > _h1 + 5, "收起 %d / 展开 %d（差 %+d px）" % (_h2, _h1, _h2 - _h1))
        # 上一张/下一张**全应用没有快捷键** —— 藏进折叠区 = 图片导航失联（且不报错）
        _nav = [tb.findChild(QToolButton, "nav" + n) for n in ("Prev", "Next", "Save")]
        ck("★ 上一张/下一张/保存都在（findChild 真找到，不是 None）",
           all(b is not None for b in _nav), str([b is not None for b in _nav]))
        ck("★ 上一张/下一张/保存**不在**折叠区里（折叠后仍可点，否则导航失联）",
           all(b is not None and b.parentWidget() is not tb.tools_box and b.isVisible()
               for b in _nav),
           str([(b.objectName(), b.parentWidget().__class__.__name__) if b else None for b in _nav]))
        ck("★ 「显示」标签与三个图层开关在折叠态仍然可见（常驻条右侧没被一起藏掉）",
           tb._display_label.isVisible()
           and all(b.isVisible() for b in tb._layer_btns.values()),
           "label=%s %s" % (tb._display_label.isVisible(),
                            str([(k, b.isVisible()) for k, b in tb._layer_btns.items()])))
        # 快捷键：折叠后用键盘也能展开（鼠标不在画布上时的唯一出路）
        ck("★ 「标注」按钮挂了 Ctrl+B 快捷键", bool(tb.tools_toggle.shortcut().toString()),
           tb.tools_toggle.shortcut().toString())
        # QToolButton 默认 ToolButtonIconOnly ⇒ 设了 text 也**一个字都不显示**（实测踩过：
        # 折叠态抓图只剩一个斜杠图标，看不出哪个是折叠开关）。必须显式 TextBesideIcon。
        ck("★ 折叠开关**有文字**（QToolButton 默认 IconOnly 会把文字吞掉）",
           tb.tools_toggle.toolButtonStyle() == Qt.ToolButtonTextBesideIcon
           and bool(tb.tools_toggle.text().strip()),
           "style=%s text=%r" % (tb.tools_toggle.toolButtonStyle(), tb.tools_toggle.text()))
        ck("★ 折叠开关真画出来了（未勾选态非底色像素 > 30，不是一块空白）",
           _painted_cnt(tb.tools_toggle) > 30,
           "非底色像素 %d" % _painted_cnt(tb.tools_toggle))
        # 展开态排版：QLabel 默认 Preferred 会**吞掉整行余量**（实测「宽度」被撑到 605px，
        # 文字左对齐 ⇒ 看着像"标签和输入框裂开一大段"）。
        # ⚠️ 断言必须量"**标签自己有没有被撑开**"，不能量"标签右边缘到输入框的间距" ——
        #    后者在标签被撑开时依然是 4px，会**假绿 PASS**（真踩过一次）。
        tb.set_tools_visible(True); pump(app, 0.35)
        _lb = [w for w in tb.findChildren(QLabel) if w.text().strip() == "宽度"]
        _lbw = _lbsh = -1
        if _lb:
            _lbw, _lbsh = _lb[0].width(), _lb[0].sizeHint().width()
        ck("★ 展开态「宽度」标签没被撑开（宽度 ≈ sizeHint，不是 Preferred 吞余量）",
           _lb and 0 < _lbw <= _lbsh + 2, "实宽 %s / sizeHint %s" % (_lbw, _lbsh))
        ck("★ 展开态数字框保持紧凑（Fixed 策略，宽 <= sizeHint+2）",
           tb.width_spin.width() <= tb.width_spin.sizeHint().width() + 2,
           "实宽 %d / sizeHint %d" % (tb.width_spin.width(), tb.width_spin.sizeHint().width()))
        tb.set_tools_visible(False); pump(app, 0.25)
        # 检测工具栏同款处理
        _dt = win.det_toolbar
        ck("★ 检测工具栏同样默认折叠", not _dt.tools_visible(), "tools_visible=%s" % _dt.tools_visible())
        _dt.tools_toggle.click(); pump(app, 0.35)
        ck("★ 检测工具栏点一下真展开（画框开关在 tools_box 里）",
           _dt.tools_visible() and not _dt.draw_btn.isHidden(),
           "tools=%s draw显式隐藏=%s" % (_dt.tools_visible(), _dt.draw_btn.isHidden()))
        _dt.tools_toggle.click(); pump(app, 0.3)
        # 两态出图（2026-10-15 折叠功能的观感核验：只看断言不够，得看画面）
        try:
            _tb = win.annotation_toolbar
            _tb.set_tools_visible(False); pump(app, 0.35)
            win.annotation_toolbar.grab().save(
                os.path.join(shot_dir, "_shot_toolbar_collapsed" + _tsuf + ".png"))
            _tb.set_tools_visible(True); pump(app, 0.35)
            _tb.grab().save(
                os.path.join(shot_dir, "_shot_toolbar_expanded" + _tsuf + ".png"))
            _tb.set_tools_visible(False); pump(app, 0.3)
            win.grab().save(os.path.join(shot_dir, "_shot_win_toolbar_collapsed" + _tsuf + ".png"))
            ck("工具栏两态截图已出（collapsed/expanded）", True, "")
        except Exception as e:
            ck("工具栏两态截图已出（collapsed/expanded）", False,
               "%s: %s" % (type(e).__name__, e))
        # 主题表兜底：这三个 objectName 的配色必须在 themes.py 里，
        # 否则 sweep_inline_styles 剥掉内联颜色后退回通用档（滑块"空槽"同型坑）
        _th = io.open(os.path.join(ROOT, "ui", "themes.py"), encoding="utf-8-sig").read()
        for _on in ("toolsToggle", "layerSw", "classPick"):
            ck("★ 主题表里有 #%s 的配色规则（剥色后有兜底）" % _on,
               _th.count("#" + _on) >= 2, "themes.py 命中 %d 次" % _th.count("#" + _on))
        ck("%s：工具栏右侧状态区的显隐 == 是否多视图（单图模式画布上已叠着同一份状态）"
           % tag, tb.layers_visible() == bool(multi),
           "layers_visible=%s multi=%s items=%d"
           % (tb.layers_visible(), multi, len(mv._items)))
        ck("%s：三个图层开关都在，且 tooltip 写明快捷键（原来是只能盲切的）" % tag,
           len(tb._layer_btns) == 3
           and all(("Ctrl" in b.toolTip() or "Space" in b.toolTip())
                   for b in tb._layer_btns.values()),
           str({k: b.toolTip() for k, b in tb._layer_btns.items()}))
        # ── 2026-10-15 照友商改成「显示 + 真复选框」──
        ck("★ 图层开关已是 QCheckBox（不再是带边框的方块按钮）",
           all(isinstance(b, QCheckBox) for b in tb._layer_btns.values()),
           str({k: type(b).__name__ for k, b in tb._layer_btns.items()}))
        ck("★ 前面有「显示」分组标签（不然看不出这三个是可见性开关）",
           getattr(tb, "_display_label", None) is not None
           and tb._display_label.text().strip() == "显示",
           str(getattr(tb, "_display_label", None)))
        # ★ 勾选/未勾选**必须画得出差别** —— 这是"占位不画"同型坑：
        #   复选框的 indicator 颜色写在 themes.py，若被剥掉就只剩一个 12px 空方块，
        #   而 isChecked() 照样 True、toggled 照样发，功能断言全绿。
        if all(b.isVisible() for b in tb._layer_btns.values()):
            _b0 = tb._layer_btns["annot"]
            from PyQt5.QtCore import QRect as _QR2
            _ind = _QR2(1, 1, 14, max(2, _b0.height() - 2))     # 左侧 indicator 那块
            _on = _dominant_color(_b0, _ind)
            _b0.setChecked(False); pump(app, 0.25)
            _off = _dominant_color(_b0, _ind)
            _b0.setChecked(True); pump(app, 0.2)
            ck("★ 复选框勾选/未勾选**画得出差别**（indicator 主色不同，非空方块）",
               _on is not None and _off is not None and _on != _off,
               "勾选主色 %s / 未勾主色 %s" % (_on, _off))
            ck("★ 勾选态 indicator 是**高饱和**色（青块，对齐友商实心方块）",
               _on is not None and (max(_on) - min(_on)) >= 60,
               "勾选主色 %s（通道极差 %s）"
               % (_on, (max(_on) - min(_on)) if _on else None))
        # 宽度不溢出：友商方案实测 191px vs 原 203px，中央列可用 ~757px
        _rb = tb.right_box.sizeHint().width()
        ck("★ 图层区宽度没把常驻条撑爆（<= 260px）", _rb <= 260, "right_box %dpx" % _rb)
        # ── 2026-10-15「平均灰度」三视图并列（用合成区域测，不依赖有没有掩膜）──
        _vps = win._view_paths_of_current_image()
        _exp = 3 if multi else 1
        ck("★ 当前图所在组的视图路径取到了（%s 工程应为 %d 条）" % (tag, _exp),
           len(_vps) == _exp, "取到 %d 条：%s" % (len(_vps), [os.path.basename(p) for p in _vps]))
        _img = getattr(win.canvas, "image", None)
        if _img is not None and _vps:
            import numpy as _np2
            _h, _w = _img.size[1], _img.size[0]
            _reg = _np2.zeros((_h, _w), dtype=bool)
            _reg[_h // 3:_h // 3 + 20, _w // 3:_w // 3 + 20] = True
            _txt = win._gray_text_for_region(_reg)
            _parts = [p for p in (_txt or "").split("，") if p.strip()]
            ck("★ 平均灰度按视图并列（%s 工程应为 %d 个值，用「，」隔开）" % (tag, _exp),
               len(_parts) == _exp, "实得 %r" % _txt)
            try:
                [_float_ok] = [all(0.0 <= float(p) <= 255.0 for p in _parts)] if _parts else [False]
            except Exception:
                _float_ok = False
            ck("★ 并列的灰度每一段都是合法数值（不是占位/空串）",
               _float_ok, "实得 %r" % _txt)
            # ★ 口径守卫：并列的第一个值必须**等于从前那套算法**（canvas._cached_gray，
            #   RGB 三通道等权平均）算出来的数 —— 口径一漂，用户拿这一列卡灰度差阈值就废了。
            _cg = getattr(win.canvas, "_cached_gray", None)
            if _cg is not None and _np2 is not None and _cg.shape[:2] == _reg.shape[:2]:
                _old = float(_cg[_reg].mean())
                _first = float(_parts[0])
                ck("★ 并列的第一个值与老口径一致（|差| <= 0.05，等权平均而非亮度加权）",
                   abs(_first - _old) <= 0.05,
                   "新=%s 老=%.3f 差 %.3f" % (_parts[0], _old, abs(_first - _old)))
        # 两张统计表的「平均灰度」列也要跟着并列（它们才是用户盯着看的那两列）。
        # ⚠️ 列位置**不能写死**：Result 表「平均灰度」在倒数第 2 列（后面还有"交并比"），
        #    Annotation 表它在**最后一列**；而且检测/分类模式下整张表会换列。
        #    ⇒ 一律**按列头文字找**，找不到就把真实列头打进失败信息。
        for _tname, _tbl in (("Annotation", win.ann_table), ("Result", win.stats_table)):
            _heads = [_tbl.horizontalHeaderItem(i).text()
                      for i in range(_tbl.columnCount())]
            if "平均灰度" in _heads:
                _ci = _heads.index("平均灰度")
                _cell = (_tbl.item(0, _ci).text() if _tbl.rowCount() > 0 else "")
                _n = len([p for p in (_cell or "").split("，") if p.strip()])
                ck("★ %s 表「平均灰度」列也是 %d 个并列值" % (_tname, _exp),
                   _tbl.rowCount() == 0 or _n == _exp,
                   "行数=%d 首行=%r 列头=%s" % (_tbl.rowCount(), _cell, _heads))
            else:
                ck("★ %s 表存在「平均灰度」列" % _tname, False,
                   "真实列头=%s" % _heads)

        # ① 点按钮 -> 画布真的翻（不是只改了个 checked）
        for _k, _f in (("annot", "_show_annotation"), ("ignore", "_show_ignore"),
                       ("pred", "_show_prediction")):
            _b = tb._layer_btns[_k]
            _st0 = bool(getattr(win.canvas, _f))
            _b.setChecked(not _b.isChecked())          # 等价于用户点一下
            pump(app, 0.45)
            _st1 = bool(getattr(win.canvas, _f))
            ck("%s：点「%s」-> canvas.%s 真的翻转（%s -> %s）"
               % (tag, _b.text(), _f, _st0, _st1), _st0 != _st1,
               "按钮 checked=%s" % _b.isChecked())
            ck("%s：「%s」按钮的亮灭与画布实际状态一致" % (tag, _b.text()),
               _b.isChecked() == _st1, "btn=%s canvas=%s" % (_b.isChecked(), _st1))
            _b.setChecked(_st0)                        # 复原
            pump(app, 0.3)

        # ② 反向：画布自己改（快捷键那条路）-> 按钮要跟着变
        _b2 = tb._layer_btns["ignore"]
        _q0 = _b2.isChecked()
        win.canvas.toggle_ignore_visibility()
        pump(app, 0.55)
        ck("%s：画布自己改状态（快捷键那条路）-> 按钮跟着变" % tag,
           _b2.isChecked() != _q0, "btn %s -> %s" % (_q0, _b2.isChecked()))
        win.canvas.toggle_ignore_visibility()
        pump(app, 0.35)

        # ③ 当前类别与选类别（2026-10-15 工具栏那一格删除后，改核**替代路径**）
        #    删除前这里核的是"工具栏类别格文字 = 画布当前类别"+"菜单项数 = 类别数"
        #    +"点菜单第 N 项真切换"。现在那格没了 ⇒ 必须把同样的保证转移到还剩下的两处：
        #      · 显示：右栏类别表的**选中行**（与 canvas 同源）
        #      · 选择：数字键 1-9（`_on_canvas_class_select`），它同时同步选中行
        _cls = None
        try:
            _cls = win.canvas.label_manager.classes[win.canvas.current_class_id]
        except Exception:
            pass
        ck("%s：★ 工具栏那格「类别」已删干净（class_btn / _class_menu 都不该再有）" % tag,
           not hasattr(tb, "class_btn") and not hasattr(tb, "_class_menu"),
           str([hasattr(tb, n) for n in ("class_btn", "_class_menu")]))
        _cur = win.class_list.currentRow()
        ck("%s：当前类别改由**右栏类别表选中行**承担（与 canvas 同一份）" % tag,
           _cur == int(getattr(win.canvas, "current_class_id", -1)),
           "class_list.currentRow=%s canvas.current_class_id=%s"
           % (_cur, getattr(win.canvas, "current_class_id", None)))
        _nc = len(getattr(win.canvas.label_manager, "classes", []) or [])
        _old = int(getattr(win.canvas, "current_class_id", 0) or 0)
        _pick = 1 if _old != 1 else 0
        if _nc >= 2:
            # 走数字键这条替代路径：按下→ canvas 切 + 右栏选中行跟着动
            win._on_canvas_class_select(_pick)
            pump(app, 0.4)
            _row = win.class_list.currentRow()      # ⚠️ 别写 `int(x or -1)`：
            # 行号 0 是 falsy，会被当成"没有选中"→ 自己造出一个假 FAIL（踩过）。
            ck("%s：数字键选类别 -> 画布切过去且右栏选中行同步" % tag,
               int(win.canvas.current_class_id) == _pick and _row == _pick,
               "canvas=%s class_list=%s" % (win.canvas.current_class_id, _row))
            win._on_canvas_class_select(_old)
            pump(app, 0.3)

        # ④ 底栏右端：−/滑块/+ 与读数（照友商摆在同一栏最右端）
        ck("%s：底栏右端有 −/滑块/+ 与百分比读数" % tag,
           all(hasattr(win, n) for n in ("_zoom_out_btn", "_zoom_slider",
                                         "_zoom_in_btn", "_zoom_pct")), "缺件")
        from ui.main_window import (zoom_pct_to_pos as _z2p,
                                    zoom_pos_to_pct as _p2z, ZOOM_STEP_POS as _STEP)

        # ★ 刻度不变量：100%（= 适应窗口）必须在**轨道中段**。
        #   线性 5..800 时 100% 落在最左 12% —— 手柄贴着左端、右边空一大截，
        #   看着像坏了；而且"缩小"的 20 倍行程挤在 12% 里，往左一碰就掉到 5%。
        #   对数刻度把它放到中点（实测 pos≈525/1000）。
        _p100 = _z2p(100)
        ck("%s：100%%（适应窗口）落在滑块**轨道中段**（不是贴左端）" % tag,
           0.40 <= _p100 / 1000.0 <= 0.65,
           "pos=%d/1000 = %.0f%%" % (_p100, _p100 / 10.0))
        ck("%s：位置<->百分比单调可逆（回写同一位置不会漂）" % tag,
           all(_p2z(_z2p(p)) == p for p in (10, 25, 50, 100, 200, 400, 800))
           and _p2z(0) < _p2z(500) < _p2z(1000),
           "100%%->%d 200%%->%d 800%%->%d" % (_p100, _z2p(200), _z2p(800)))

        _fit = float(getattr(win.canvas, "_fit_zoom", 0) or 0)
        win._zoom_slider.setValue(_z2p(200))
        pump(app, 0.5)
        ck("%s：滑块拖到 200%% -> zoom_level 精确变成 fit×2" % tag,
           _fit > 0 and abs(win.canvas.zoom_level - _fit * 2.0) < _fit * 0.02,
           "zoom=%.6f 期望=%.6f" % (win.canvas.zoom_level, _fit * 2.0))
        ck("%s：百分比读数与画布同一个口径（不是另算一份）" % tag,
           win._zoom_pct.text() == "%d%%" % mv.zoom_percent(),
           "%s vs %d%%" % (win._zoom_pct.text(), mv.zoom_percent()))
        # ★ 守住"拖完不弹回去"：底栏显隐若与视图模式判断打架，画布会反复 resize，
        #   而 resizeEvent 是无条件 _fit_to_window() ⇒ 刚设的缩放被重置。
        _zz = float(win.canvas.zoom_level)
        pump(app, 1.0)
        ck("%s：稳定 1s 后缩放不回退（底栏显隐没在抖 -> 画布没被 resize）" % tag,
           abs(win.canvas.zoom_level - _zz) < 1e-9,
           "%.6f -> %.6f" % (_zz, win.canvas.zoom_level))

        # ★★ 像素级：控件"在布局里、信号也对"**不等于画出来了**。
        #    底栏滑块曾经因为颜色写在内联样式里、且全在 THEME_COLORS 白名单里，
        #    被 sweep_inline_styles 剥成"没有背景色"，而主题表当时没有 QSlider
        #    规则兜底 ⇒ 它占着 84px 宽、画出来是一段空槽。
        #    上面每一条断言（value / zoom_level / 读数 / 不回退）当时**全绿**。
        #    所以这里把底栏 grab 出来数像素 —— 槽(#2A3A5C 深 / #C9D3E0 浅)、
        #    手柄(#00E5FF 深 / #0077B6 浅)必须真有。
        _im = win._info_bar.grab().toImage()
        _n_groove = _n_handle = 0
        for _y in range(_im.height()):
            for _x in range(_im.width()):
                _c = _im.pixelColor(_x, _y)
                _k = (_c.red(), _c.green(), _c.blue())
                if _k in ((42, 58, 92), (201, 211, 224)):
                    _n_groove += 1
                elif _k in ((0, 229, 255), (0, 119, 182)):
                    _n_handle += 1
        _sl_vis = bool(win._zoom_slider.isVisible())
        ck("%s：滑块**真的画出来了**（可见 且 槽/手柄像素数 > 0，不是一段空槽）" % tag,
           _sl_vis and _n_groove > 50 and _n_handle > 20,
           "visible=%s 槽 %dpx / 手柄 %dpx" % (_sl_vis, _n_groove, _n_handle))

        # ★ ± 一格 = 滑块位置走 _STEP：说明它**没有另开一条缩放路径**，
        #   而且对数刻度下"一格"是固定倍率 —— 在低倍和高倍处点一下幅度必须一致
        #   （线性档位在 20% 处点一下是 +50%、在 700% 处几乎不动，手感差一个数量级）。
        def _one_step(pct, tag=tag):
            win._zoom_slider.setValue(_z2p(pct))
            pump(app, 0.35)
            _p0, _b = int(win._zoom_slider.value()), mv.zoom_percent()
            win._zoom_in_btn.click()
            pump(app, 0.35)
            return _b, mv.zoom_percent(), int(win._zoom_slider.value()) - _p0

        _b1, _a1, _d1 = _one_step(40)
        _b2, _a2, _d2 = _one_step(500)
        # 这里要求**精确**等于 _STEP（不是"差不多"）：曾经差 1~2 —— 画布上报的是
        # 取整后的百分比，`_apply_zoom_ui` 无条件拿它反算位置，把滑块从 356 挪到 358。
        # 所以这条同时守着"回写不许无故拖动滑块"（见 _apply_zoom_ui 的说明）。
        ck("%s：+ 一格就是滑块位置固定走 %d（没有另开一条缩放路径，回写也没挪它）"
           % (tag, _STEP), _d1 == _STEP and _d2 == _STEP, "走了 %d / %d" % (_d1, _d2))
        if _b1 > 0 and _b2 > 0:
            _r1, _r2 = _a1 / float(_b1), _a2 / float(_b2)
            ck("%s：± 一格是**固定倍率**（%d%%→%d%% 与 %d%%→%d%% 倍率一致）"
               % (tag, _b1, _a1, _b2, _a2), abs(_r1 - _r2) < 0.02,
               "×%.3f vs ×%.3f" % (_r1, _r2))
        win._zoom_slider.setValue(_z2p(100))
        pump(app, 0.45)

        # ⑤ 「视图布局」按钮：只在多视图模式出现；菜单齐全且真能切布局
        _vl = getattr(win, "_view_layout_btn", None)
        ck("%s：底栏「视图布局」按钮只在多视图模式出现" % tag,
           _vl is not None and bool(_vl.isVisible()) == bool(multi),
           "visible=%s multi=%s" % (_vl.isVisible() if _vl is not None else None, multi))
        if multi and _vl is not None:
            ck("%s：「视图布局」菜单列出了全部模式（不是空菜单）" % tag,
               len(win._layout_menu.actions()) == len(mv.layout_modes()),
               "%d vs %d" % (len(win._layout_menu.actions()), len(mv.layout_modes())))
            win._on_layout_pick("3*1")
            pump(app, 0.9)
            ck("%s：选「3*1」后 layout_mode 真的变了" % tag, mv.layout_mode() == "3*1",
               mv.layout_mode())
            win._on_layout_pick("自动（N*1）")
            pump(app, 0.9)

        # ⑥ 底栏的读数项一个都不能被右端控件挤掉
        _vis = [lb.text() for lb in (win._info_dims, win._info_xy, win._info_rgb,
                                     win._info_ann, win._info_det, win._info_ms,
                                     win._info_avg) if lb.isVisible()]
        ck("%s：底栏 7 个读数项全部可见（没被右端新控件挤掉）" % tag,
           len(_vis) == 7, "可见 %d 个：%s" % (len(_vis), _vis))

    # ── ★ 左栏竖排页签（2026-10-14 晚：工程管理 / 图像管理 两页，点一下收起）──
    def _check_left_panel():
        """左栏重构后的行为核验。

        为什么要钉这些：这一栏原来是"工程树钉死 150px + 图片列表吃剩下的那点高度"，
        改成两页之后，**最典型的三种错法在界面上都不报错**：
          · 页签点了没反应 / 切错页（页栈没跟着走）；
          · 收起之后**点不回来**（把竖条本身也一起藏了）；
          · 页签"占着位却不画字"—— 与底栏滑块那次同源：内联样式里的颜色被
            `sweep_inline_styles` 剥掉、主题表又没有规则兜底。
        最后一条只有**数像素**才抓得到，所以下面专门留了一条像素级断言。
        """
        btns = getattr(win, "_left_tab_btns", {}) or {}
        ck("左栏页签是两个（工程管理 / 图像管理）",
           set(btns) == {"proj", "img"}, str(sorted(btns)))
        ck("左栏页签文字是**竖排**（一个字一行，与右栏同款）",
           bool(btns) and all("\n" in b.text() for b in btns.values()),
           str({k: b.text().replace("\n", "|") for k, b in btns.items()}))
        ck("左栏页栈是两页（工程列表 / 图片列表）",
           win._left_stack.count() == 2, str(win._left_stack.count()))

        # ① 切页本身可用（默认值在更早处断言 —— 那里是**构造完、还没打开工程**的真·默认态）
        win.show_left_tab("img")
        pump(app, 0.4)
        _st = win.left_tab_state()
        ck("切到「图像管理」生效（页索引 1、内容可见、按钮勾着）",
           _st["page"] == 1 and _st["content_visible"] and _st["checked"] == ["img"],
           str(_st))

        # ② 点「工程管理」-> 切到工程列表，且工程树拿到**整栏高度**（原来被钉死 150px）
        btns["proj"].click()                       # 走真实点击路径，不直接设状态
        pump(app, 0.5)
        _st = win.left_tab_state()
        _th = win.project_tree.height()
        ck("点「工程管理」-> 切到页 0、内容可见、按钮变勾选",
           _st["page"] == 0 and _st["content_visible"] and _st["checked"] == ["proj"],
           str(_st))
        ck("工程列表拿到**整栏高度**（不再被 setMaximumHeight(150) 钉死）",
           win.project_tree.maximumHeight() > 150 and _th > 200,
           "maxH=%d 实高=%d" % (win.project_tree.maximumHeight(), _th))

        # ③ 再点一次「工程管理」-> 内容收起（照友商"点一下显示/再点一下隐藏"）
        btns["proj"].click()
        pump(app, 0.4)
        _st = win.left_tab_state()
        ck("再点一次「工程管理」-> 内容收起（两个按钮都弹起）",
           (not _st["content_visible"]) and _st["checked"] == [], str(_st))
        ck("★ 收起后**竖排页签仍然可见**（否则就再也点不回来了）",
           all(b.isVisible() for b in btns.values())
           and all(b.height() > 20 for b in btns.values()),
           str({k: (b.isVisible(), b.height()) for k, b in btns.items()}))

        # ④ 收起状态下点「图像管理」-> 内容回来、切到页 1
        btns["img"].click()
        pump(app, 0.4)
        _st = win.left_tab_state()
        ck("收起状态下点「图像管理」-> 内容回来并切到页 1",
           _st["content_visible"] and _st["page"] == 1 and _st["checked"] == ["img"],
           str(_st))

        # ⑤ ★ 像素级：页签**真的把字画出来了**，不是一条空按钮
        #    （同底栏滑块那次：占位不画，功能断言全绿也发现不了）
        from collections import Counter as _Cnt

        def _painted(w):
            _i = w.grab().toImage()
            _cc = _Cnt()
            for _y in range(_i.height()):
                for _x in range(_i.width()):
                    _c = _i.pixelColor(_x, _y)
                    _cc[(_c.red(), _c.green(), _c.blue())] += 1
            if not _cc:
                return 0, 0
            return (_i.width() * _i.height() - _cc.most_common(1)[0][1],
                    _i.width() * _i.height())

        _pn, _pt = _painted(btns["proj"])      # 未勾选：底色之外的像素只能是**文字**
        ck("★ 竖排页签真的画出来了（未勾选那颗里非底色像素 > 50）",
           _pn > 50, "%d / %d 像素" % (_pn, _pt))
        _pn2, _pt2 = _painted(btns["img"])     # 勾选：文字 + 选中边框
        ck("勾选中的页签有可见的选中态（非底色像素多于未勾选那颗）",
           _pn2 > _pn, "勾选 %d / 未勾选 %d" % (_pn2, _pn))

        # ⑥ 页签条必须在页栈**左边**（左右栏别搞反）
        _bar = btns["img"].parentWidget()
        ck("左栏页签条在页栈左侧（tab_bar.x < stack.x）",
           _bar is not None and _bar.x() < win._left_stack.x(),
           "bar.x=%s stack.x=%s" % (_bar.x() if _bar else "?", win._left_stack.x()))

        # ⑦ 收尾：还原成「图像管理」，别把状态留给后面的检查
        win.show_left_tab("img")
        pump(app, 0.3)

    # ── ★ 右栏「图像管理」整页删除（2026-10-14 晚：左栏已有同名页 ⇒ 右侧冗余）──
    def _check_right_panel_no_image():
        """右栏 4 页 → 3 页的核验。

        为什么钉：删页之后有 4 种错法**界面上都不报错** ——
          ① 页栈少了一页、`_right_tab_index` 没同步 ⇒ 点页签跳到别的页或空白页；
          ② 老名字 `show_right_tab("image")` 变成**静默 return**（这是预期行为，
             所以断言的是"它不再改变当前页"，而不是"它不报错"——静默本身就是约定）；
          ③ 那两个**唯一入口**（管理划分… / 输出目录）没搬走 ⇒ 功能丢了还没人发现，
             这是删页真正的风险点（其余内容在底栏/计数行/右键菜单里各有一份）。
             随后 Arthur 又判"面板上那一行按钮本身冗余"（2026-10-14 20:39）⇒ 按钮删掉、
             动作改挂右键/菜单 ⇒ 所以下面既断言"按钮属性不存在"，也断言"两个槽函数还活着"；
          ④ 渲染脚本的 ALL_TABS 还留着 "image" ⇒ 出图时对着别的页分析（见 AST 那条）。
        """
        btns = getattr(win, "_right_tab_btns", {}) or {}
        ck("★ 右栏页签只剩 3 个（参数 / 标注 / 模型），「图像管理」已不在",
           set(btns) == {"params", "annot", "model"}, str(sorted(btns)))
        ck("右栏页栈也是 3 页", win._right_stack.count() == 3, str(win._right_stack.count()))
        ck("★ 右栏下标表与页栈页数一致（不一致 = 点了页签跳错页，且不报错）",
           len(getattr(win, "_right_tab_index", {})) == win._right_stack.count(),
           "index=%s 页数=%d" % (sorted(getattr(win, "_right_tab_index", {})),
                                win._right_stack.count()))

        # 老 key 已失效，且**不能把页跳乱**
        win.show_right_tab("model")
        pump(app, 0.3)
        _before = win._right_stack.currentIndex()
        win.show_right_tab("image")
        pump(app, 0.2)
        ck("旧页签名 show_right_tab(\"image\") 已失效，且不会把当前页跳乱",
           win._right_stack.currentIndex() == _before,
           "before=%s after=%s" % (_before, win._right_stack.currentIndex()))
        win.show_right_tab("annot")
        pump(app, 0.3)

        # ① 那行按钮**已按 Arthur 要求删掉**（2026-10-14 晚实机反馈："这两个按钮冗余的"）
        for attr in ("split_manage_btn", "open_out_btn"):
            ck("★ 左栏那行小按钮已删干净（win 上不该再有 %s）" % attr,
               not hasattr(win, attr), "hasattr=%s" % hasattr(win, attr))
        ck("★ 删按钮没把状态行一起带走（list_status 仍在图片列表页里）",
           getattr(win, "list_status", None) is not None and win.list_status.parent() is not None,
           str(getattr(win, "list_status", None)))
        # ★ 删的是**按钮**，不是功能：两个槽函数必须还活着（入口位置由下面 AST 那几条钉）
        ck("★ 两个动作的槽函数仍在（删按钮 ≠ 删功能）",
           callable(getattr(win, "_manage_split", None))
           and callable(getattr(win, "_open_output_dir", None)),
           "_manage_split=%s _open_output_dir=%s"
           % (callable(getattr(win, "_manage_split", None)),
              callable(getattr(win, "_open_output_dir", None))))
        # ⚠️ 这里**不能真调** `_open_output_dir()`：工程里只要有 `outputs/`，它就会
        #    `os.startfile` **弹出资源管理器窗口** —— 探针的第一铁律是"不打扰小千"，
        #    而且新窗口会**抢焦点**，把后面的检查带崩（2026-10-14 实测：连跑两次时
        #    出现过 11 条偶发 FAIL，正是这一类外部窗口干扰；第一版就是这么写的）。
        #    改法：把 `_open_dir` 换成记录器再调 —— 既不弹窗，又能证明"真的走到了它"。
        _opened = []
        _real_open_dir = win._open_dir
        win._open_dir = lambda p: _opened.append(p)
        try:
            win._open_output_dir()
            pump(app, 0.2)
            _bad = [p for p in _opened if not str(p).replace("\\", "/").endswith("outputs")]
            ck("★ 调 _open_output_dir() 走到 _open_dir 且**不弹窗**（已换记录器），目标以 outputs 结尾",
               not _bad, "记录 = %s（空 = 当前没打开工程，也算通过）" % _opened)
        except Exception as e:
            ck("★ 调 _open_output_dir() 走到 _open_dir 且**不弹窗**（已换记录器），目标以 outputs 结尾",
               False, "%s: %s" % (type(e).__name__, e))
        finally:
            win._open_dir = _real_open_dir

    # ── ★ 类别删除：删完画布必须跟着换表（2026-10-14 用户问「Atd 能删不」时挖出来的坑）──
    def _check_class_delete():
        """类别号 = **类别表里的位置**（掩膜里存的就是这个号），所以删一次要动三处。

        旧实现 `del classes[row]` 之后只刷新了 Class 表与菜单，**没让画布重读类别表**：
        画布的 `label_manager.classes` 是 `load()` 时抓的一份普通 list，于是它还拿着旧表、
        `current_class_id` 停在已失效的编号上 ⇒ 继续画就把**不存在的类别号**写进掩膜。
        实测（真工程 新松_233…，删 Atd）：project.json 只剩 0..4，
        而画布仍是 6 张表、当前编号 5、工具栏按钮还写着「类别:Atd」、菜单 5 项一个都没勾
        —— **界面上看不出任何异常**，这是典型的静默错。

        ⚠️ 这条路径必须用"有 3 个以上类别"的夹具才走得到：2 个类别的工程
        （第 0 行 background 删不得）根本没有"中间那个"可选。
        """
        pd, classes = build_cls_project("probe_cls")
        win._open_project_by_name(_register_in_memory(win.pm, pd))
        _ok = wait_until(app, lambda: len(getattr(win, "_all_images", []) or []) >= 3, 40)
        ck("类别删除夹具：3 组图都扫到了", _ok,
           "拿到 %d 张" % len(getattr(win, "_all_images", []) or []))

        from PyQt5.QtWidgets import QMessageBox as _QMB
        _orig_q = _QMB.question
        _msgs = []

        def _yes(_parent, _title, _text, *a, **k):
            _msgs.append(_text)
            return _QMB.Yes

        _QMB.question = staticmethod(_yes)
        tb = win.annotation_toolbar

        def _sel(row):
            """选中类别表第 row 行。

            ⚠️ 不能只调一次就往下走：开工程会触发一次类别统计重扫，重扫
            （`ClassTableView.set_classes` → `clear()`）会把 `currentRow` 重置成 -1；
            而 `_delete_class` 读的正是 `currentRow()` ⇒ 点删除等于"没选中"，
            旧代码还会打一句误导的「background 不能删」。这里重试到选中为止，
            并把"到底选中了没有"单独断言一条，别让后面 8 条断言替它背锅。
            """
            for _ in range(8):
                win.class_list.setCurrentRow(row)
                pump(app, 0.25)
                if win.class_list.currentRow() == row:
                    return True
            return False

        try:
            ck("类别删除夹具：类别表就是 %s" % classes,
               list(win._cached_classes) == classes, str(win._cached_classes))

            # ① 删**最后一个**类别（D）：编号谁都不用动，属于安全的那种
            ck("① 在 Class 表里选中 D 那一行（currentRow 真落在 4 上）", _sel(4),
               "currentRow=%s" % win.class_list.currentRow())
            ck("① 选中 D -> 画布当前类别切到 4",
               int(win.canvas.current_class_id) == 4, str(win.canvas.current_class_id))
            win._delete_class()
            pump(app, 0.45)
            _e1 = ["background", "A", "B", "C"]
            ck("① 删最后一个类别 -> project.json 落盘 == %s" % _e1,
               _read_json(os.path.join(pd, "project.json")).get("classes") == _e1,
               str(_read_json(os.path.join(pd, "project.json")).get("classes")))
            ck("① **画布类别表跟着换了**（旧实现会一直停在删之前那份）",
               list(getattr(win.canvas.label_manager, "classes", []) or []) == _e1,
               str(getattr(win.canvas.label_manager, "classes", None)))
            ck("① 当前类别号被夹回合法范围（不能停在已失效的 4）",
               0 <= int(win.canvas.current_class_id) < len(_e1),
               "cid=%s，合法范围 [0,%d)" % (win.canvas.current_class_id, len(_e1)))
            ck("① 右栏类别表也只剩 %d 行（当前类别的显示责任在这张表上）" % len(_e1),
               int(win.class_list.count() or 0) == len(_e1),
               "class_list.count=%s" % win.class_list.count())
            ck("① 提示语说的是「最后一个类别」（编号不用动）",
               bool(_msgs) and "最后一个类别" in _msgs[-1] and "⚠️" not in _msgs[-1],
               (_msgs[-1][:70] if _msgs else "根本没弹窗"))
            _ip = _read_json(os.path.join(pd, "inference_params.json"))
            ck("① 推理参数文件里那条已删类别的记录被摘掉了（且没伤到别处）",
               sorted((_ip.get("classes") or {}).keys()) == ["A", "B", "C"]
               and bool(_ip.get("global")),
               str(sorted((_ip.get("classes") or {}).keys())))
            ck("① 掩膜文件一个都没被改动（3 个 mask 还在）",
               len([x for x in os.listdir(os.path.join(pd, "annotations"))
                    if x.endswith("_mask.png")]) == 3)

            # ② 删**中间**那个（B）：它后面的 C 会前移一位，选中项必须跟着挪
            #    现在表是 [background, A, B, C]，选中 2（=B）再删
            ck("② 选中中间的 B（编号 2）", _sel(2) and int(win.canvas.current_class_id) == 2,
               "currentRow=%s cid=%s" % (win.class_list.currentRow(),
                                         win.canvas.current_class_id))
            _n0 = len(_msgs)
            win._delete_class()
            pump(app, 0.45)
            _e2 = ["background", "A", "C"]
            ck("② 删中间类别 -> project.json 落盘 == %s" % _e2,
               _read_json(os.path.join(pd, "project.json")).get("classes") == _e2,
               str(_read_json(os.path.join(pd, "project.json")).get("classes")))
            ck("② 画布类别表跟着换了（编号 2 现在指的是 C）",
               list(getattr(win.canvas.label_manager, "classes", []) or []) == _e2,
               str(getattr(win.canvas.label_manager, "classes", None)))
            ck("② 当前类别号合法且指的正是「顶上来的」那个 C",
               0 <= int(win.canvas.current_class_id) < len(_e2)
               and _e2[int(win.canvas.current_class_id)] == "C",
               "cid=%s -> %s" % (win.canvas.current_class_id,
                                 _e2[int(win.canvas.current_class_id)]
                                 if 0 <= int(win.canvas.current_class_id) < len(_e2) else "越界"))
            ck("② 提示语明确警告「后面的类别会前移」（旧文案把最危险的情况说成了「没事」）",
               len(_msgs) > _n0 and "前移" in _msgs[-1] and "⚠️" in _msgs[-1],
               (_msgs[-1][:70] if _msgs else "根本没弹窗"))

            # ③ background 不能删：不弹窗、不改表
            ck("③ 选中 background（第 0 行）", _sel(0), "currentRow=%s" % win.class_list.currentRow())
            _n1 = len(_msgs)
            win._delete_class()
            pump(app, 0.25)
            ck("③ 选中 background 时删除被拒（不弹窗、类别表一字未动）",
               len(_msgs) == _n1 and list(win._cached_classes) == _e2,
               "弹窗=%d 表=%s" % (len(_msgs) - _n1, win._cached_classes))

            # ④ 改名：走真实触发路径（类别名那一格双击 -> rename_requested）
            from PyQt5.QtWidgets import QInputDialog as _QID
            _orig_i = _QID.getText
            _QID.getText = staticmethod(lambda *a, **k: ("C2", True))
            try:
                _sel(2)                                  # 现在是 [background, A, C]
                win.class_list.table.cellDoubleClicked.emit(2, 1)
                pump(app, 0.45)
            finally:
                _QID.getText = staticmethod(_orig_i)
            _e3 = ["background", "A", "C2"]
            ck("④ 双击类别名那格改名 -> project.json 落盘 == %s" % _e3,
               _read_json(os.path.join(pd, "project.json")).get("classes") == _e3,
               str(_read_json(os.path.join(pd, "project.json")).get("classes")))
            ck("④ 改名 -> 画布类别表也跟着换（工具栏与画布不会各说各的）",
               list(getattr(win.canvas.label_manager, "classes", []) or []) == _e3,
               str(getattr(win.canvas.label_manager, "classes", None)))
            _ip2 = _read_json(os.path.join(pd, "inference_params.json"))
            ck("④ 改名 -> 推理参数那条记录跟着搬家（不搬就悄悄退回默认阈值）",
               "C2" in (_ip2.get("classes") or {}) and "C" not in (_ip2.get("classes") or {}),
               "keys=%s" % sorted((_ip2.get("classes") or {}).keys()))

            # ⑤ 上移：换的是位置，而位置就是掩膜的类别号 —— 对应关系变了，画布必须重读
            _sel(2)
            win.class_list._buttons["up"].click()        # 真按钮
            pump(app, 0.45)
            _e4 = ["background", "C2", "A"]
            ck("⑤ 点「↑」上移 -> project.json 落盘 == %s" % _e4,
               _read_json(os.path.join(pd, "project.json")).get("classes") == _e4,
               str(_read_json(os.path.join(pd, "project.json")).get("classes")))
            ck("⑤ 上移 -> 画布类别表也跟着换（编号 1 现在指 C2）",
               list(getattr(win.canvas.label_manager, "classes", []) or []) == _e4,
               str(getattr(win.canvas.label_manager, "classes", None)))
        finally:
            _QMB.question = staticmethod(_orig_q)

    _pos = (max(8, win.canvas.width() // 2), max(8, win.canvas.height() // 2))
    _how = _move_to(_pos)
    _pt = win.canvas._window_to_image(_QP(_pos[0], _pos[1]))
    ck("光标进了画布就把像素坐标报到表上（%s）" % _how,
       _pt is not None and win._info_xy.text() == "坐标 %d,%d" % (_pt[0], _pt[1]),
       "期望 %s，实际 %s" % (_pt, win._info_xy.text()))
    if _pt is not None:
        _v = win.canvas.image.getpixel((_pt[0], _pt[1]))
        if isinstance(_v, (tuple, list)):
            _exp_rgb = "RGB %d,%d,%d" % (int(_v[0]), int(_v[1]), int(_v[2]))
        else:
            _exp_rgb = "灰度 %d" % int(_v)
        ck("底栏 RGB 就是画布上那一点的像素值（显示什么读什么，不是另读一张图）",
           win._info_rgb.text() == _exp_rgb, "期望 %s，实际 %s" % (_exp_rgb, win._info_rgb.text()))
    # 光标离开控件：坐标要清掉（留着"最后一次"的值会像是当前位置）
    QApplication.instance().sendEvent(win.canvas, _QE(_QE.Leave))
    pump(app, 0.15)
    ck("光标离开画布后坐标清空（不留假读数）",
       win._info_xy.text() == "坐标 --" and win._info_rgb.text() == "RGB --",
       "%s / %s" % (win._info_xy.text(), win._info_rgb.text()))
    # 咬过的坑：缓存只在"值真的变了"时才发信号 ⇒ 离开后若不把缓存清空，
    # 再回到同一点就不会上报（显示永远停在 "--"）。
    _how2 = _move_to(_pos)
    ck("离开后再回到**同一点**仍会重新上报（信号节流缓存清干净了，%s）" % _how2,
       _pt is not None and win._info_xy.text() == "坐标 %d,%d" % (_pt[0], _pt[1]),
       win._info_xy.text())

    # ── ★ 参考窗（形状图1 / 形状图2）同样要报坐标与像素 ──
    #    用户 2026-10-14 的原话："目前光标停留在形状1和形状2上时，这些值都是空的"。
    #    根因**不是**信号没接：参考窗那两层（QFrame + 内层 QLabel）连鼠标跟踪都没开，
    #    QLabel 默认只在按住键时才产生 MouseMove ⇒ 一个事件都没有，谁也无从上报。
    _check_ref_hover("夹具")
    # 顶栏并进底栏：这一刻窗口里是 **probe_mv（多视图）** ——
    # `_check_ref_hover("夹具")` 之所以能真的测到参考窗，就是因为它是紧接多视图夹具
    # 之后跑的（单图那会儿 `_panels` 是空的，那条函数第一行就 return 了）。
    _check_bars_merge("夹具(多视图)", multi=True)
    # 再切回单图：工具栏右侧状态区**要自动隐藏** —— 单图模式下画布自己叠着同一份
    # 状态文本，两处同时显示就是这次要消掉的毛病。这条比"静态看某个模式"更有价值：
    # 它测的是"跟着视图模式走"这件事本身（冒烟的 bug 正是两处判断打架 ⇒ 反复显隐
    # ⇒ 画布反复 resize ⇒ 刚拖的缩放被 resizeEvent::_fit_to_window 重置）。
    _mv_items = list(getattr(win.mv_preview, "_items", []) or [])
    win.mv_preview.clear_views()
    pump(app, 0.8)
    ck("切回单图后工具栏右侧状态区自动隐藏（不同屏显示两份状态）",
       not win.annotation_toolbar.layers_visible(),
       "layers_visible=%s" % win.annotation_toolbar.layers_visible())
    if _mv_items:
        win.mv_preview.set_views(_mv_items)
        pump(app, 1.0)
        ck("再切回多视图后状态区又自动出现（来回都跟得上）",
           win.annotation_toolbar.layers_visible(),
           "layers_visible=%s" % win.annotation_toolbar.layers_visible())


    # 耗时：单位必须是毫秒（曾按秒存，导致"耗时 < 1000"筛选全命中）
    # ⚠️ 喂值之后必须走**真实路径**重建记录（`_filter_images`）——推理完产品就是这么刷的。
    #    第一版只改 `_pred_count` 就直接断言，结果底栏还是旧值：底栏读的是**列表那条记录**，
    #    记录不重建它就不该变（这正是"同源"该有的样子），错的是断言不是产品。
    _save_pc = dict(getattr(win, "_pred_count", None) or {})
    _save_it = dict(getattr(win, "_infer_times", None) or {})
    try:
        win._pred_count = {_base0: 7}
        win._infer_times = {_base0: 250.0}
        win._filter_images(win.search_input.text())
        pump(app, 0.5)
        ck("底栏「已检出」跟着记录走（喂 7 就显示 7）", win._info_det.text() == "已检出 7",
           win._info_det.text())
        ck("底栏「耗时」按**毫秒**显示（值 250.0 -> 250ms）", win._info_ms.text() == "耗时 250ms",
           win._info_ms.text())
        ck("底栏「平均耗时」同样按毫秒显示", win._info_avg.text() == "平均 250ms",
           win._info_avg.text())
        ck("列表「耗时」列与底栏是同一个数（同源，不各算一份）",
           win.image_list_widget.item(0, 7).text() == "250.0",
           win.image_list_widget.item(0, 7).text())
    finally:
        win._pred_count = _save_pc
        win._infer_times = _save_it
        win._filter_images(win.search_input.text())
        pump(app, 0.4)

    _ok_ms, _ms_detail = _infer_times_is_ms()
    ck("AST：_infer_times 的写入点真的乘了 1000（秒→毫秒）", _ok_ms, _ms_detail)
    ck("列表「耗时」列头带上了单位", 
       win.image_list_widget.horizontalHeaderItem(7).text() == "耗时ms",
       win.image_list_widget.horizontalHeaderItem(7).text())

    # ── ★ 顶栏整行已拆（2026-10-14 傍晚）：状态改由「工具栏右侧 + 底栏」承载 ──
    #    用户原话："我感觉咱们顶栏和底栏可以合二为一，你可以参考一下友商的"。
    #    实测那条栏装不下两行的信息（三段文本 726px + 下拉 150px ≈ 976px，
    #    而中央列只有 757~950px），所以**不是搬位置**，而是按"这项有没有别处显示"
    #    重新分派：三个图层开关 + 类别 -> 工具栏右侧；缩放 -> 底栏右端（照友商）；
    #    "当前工具 Pan" 与 "Brush: 15" -> 删（工具栏上本来就高亮 / 就是那个 15 px）；
    #    "3 视图：漫反射/形状图1/形状图2" -> 删（每格标题上已写着）。
    _full = win.canvas.status_info_text()
    ck("画布叠加层仍带尺寸与缩放（它没有别的栏兜底，去掉就没地方看了）",
       re.match(r"^\d+x\d+ \| ", _full) is not None and "Zoom:" in _full, _full)
    ck("笔宽叫法统一成 Brush（Width 字样已消失）",
       "Brush:" in _full and "Width" not in _full, _full)
    ck("多视图顶栏整行已拆：bar/combo/zoom_lbl/hint/info_lbl 五个成员都不再存在",
       not any(hasattr(win.mv_preview, n)
               for n in ("bar", "combo", "zoom_lbl", "hint", "info_lbl")),
       "仍在：" + str([n for n in ("bar", "combo", "zoom_lbl", "hint", "info_lbl")
                       if hasattr(win.mv_preview, n)]))
    _sp, _sp_d = _status_info_text_params()
    ck("status_info_text 不再留 full 参数（那条分支已无调用者，留着就是下一次冗余的入口）",
       _sp == ["self"], _sp_d)
    _sn, _sn_d = _emit_state_sig_has_nviews()
    ck("AST：_emit_state 的去重签名含 n_views（少它则视图数变化时不发信号，"
       "工具栏那三个开关永远不显示）", _sn, _sn_d)

    _th, _th_d = _theme_has_slider_style()
    ck("AST：滑块配色写在主题表里（写在内联样式里会被 sweep 剥掉 => 一段空槽）", _th, _th_d)
    _mq, _mq_d = _mainwindow_has_no_slider_qss()
    ck("AST：main_window 里不再有模块级滑块 QSS 常量（配色只留一处真相源）", _mq, _mq_d)
    _pt, _pt_d = _mainwindow_no_proj_toggle()
    ck("AST：左栏那个「▼ 折叠工程列表」按钮已彻底移除"
       "（折叠已交给竖排页签，留两份就是两个入口、两种状态）", _pt, _pt_d)
    _src_mw = open(os.path.join(ROOT, "ui", "main_window.py"), encoding="utf-8").read()
    # 用 AST 数字符串**常量**，不数源码里的字：注释里写一遍不该算一处
    # （老坑："注释能骗过的断言 = 没有断言"）。
    _lit = [n.value for n in ast.walk(ast.parse(_src_mw))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and "QPushButton:checked { background-color: #0E3A52" in n.value]
    ck("竖排页签样式只有一份真相源（AST 数字符串常量，恰 1 处；右栏已改用它）",
       len(_lit) == 1, "命中 %d 处" % len(_lit))

    # ── ★ 右栏删页的 AST 侧守卫（2026-10-14 晚）──
    _dead, _dead_d = _mw_no_dead_image_props()
    ck("AST：那页的工厂函数与控件属性已彻底消失（不留'看着能用、其实挂不上'的死代码）",
       _dead, _dead_d)
    _menu, _menu_d = _mw_menu_has_output_dir()
    ck("AST：两个右键菜单的函数体里都**真的调用了** _open_output_dir（取 Call 节点，注释骗不过）",
       _menu, _menu_d)
    _rp_ok, _rp_d = _render_panel_tabs_match_index()
    ck("AST：渲染脚本 ALL_TABS 与 _right_tab_index 完全一致（不一致会对着别的页分析布局）",
       _rp_ok, _rp_d)
    _ms_funcs, _ms_d = _mw_manage_split_entrypoints()
    ck("★ AST：`_manage_split` 仍有 ≥2 处入口（图片列表右键 + Tools 菜单）"
       "—— 面板按钮删掉后，少一处就等于这功能静默消失",
       ("_on_image_context_menu" in _ms_funcs) and len(_ms_funcs) >= 2, _ms_d)

    # 左栏表格高度自适应：行少时收成内容高度
    win._fit_list_height()
    _rows = win.image_list_widget.rowCount()
    _body = sum(win.image_list_widget.rowHeight(r) for r in range(_rows))
    ck("行少时表格收成内容高度（下方不再留大片空白）",
       win.image_list_widget.maximumHeight() != 16777215
       and abs(win.image_list_widget.maximumHeight()
               - (_body + win.image_list_widget.horizontalHeader().height()
                  + 2 * win.image_list_widget.frameWidth() + 6)) <= 2,
       "rows=%d maxH=%d body=%d" % (_rows, win.image_list_widget.maximumHeight(), _body))

    _bar_shot = os.path.join(shot_dir, "_shot_img_info_bar" + _tsuf + ".png")
    pump(app, 0.4)
    _bsize = (_bar.width(), _bar.height()) if _bar is not None else (0, 0)
    try:
        _bar.grab().save(_bar_shot)
        ck("底栏特写截图已落盘", os.path.exists(_bar_shot) and os.path.getsize(_bar_shot) > 1000,
           "%d bytes，控件 %dx%d" % ((os.path.getsize(_bar_shot) if os.path.exists(_bar_shot) else 0),
                                    _bsize[0], _bsize[1]))
        _win_shot = os.path.join(shot_dir, "_shot_with_info_bar" + _tsuf + ".png")
        win.grab().save(_win_shot)
        ck("整窗截图（含底栏）已落盘", os.path.exists(_win_shot)
           and os.path.getsize(_win_shot) > 20000,
           "%d bytes" % (os.path.getsize(_win_shot) if os.path.exists(_win_shot) else 0))
    except Exception as e:
        ck("底栏特写截图已落盘", False, "%s: %s" % (type(e).__name__, e))
    stage("底栏/顶栏/耗时单位核完")

    # ── （可选）在**真实工程**上核一遍数字：`--real-project <工程名>` ──
    #    为什么要有这个开关：夹具工程是自造的，"33 组 / 99 张"这种量纲问题只有真实
    #    三视图工程才复现得出来。这里**只读**：不打标记、不转标注、不调 close()
    #    （closeEvent 会回写分类面板参数 ⇒ 会改到用户工程，所以走硬杀路径）。
    _real = ""
    if "--real-project" in sys.argv:
        _i = sys.argv.index("--real-project")
        if _i + 1 < len(sys.argv):
            _real = sys.argv[_i + 1]
    killed_early = False
    if _real:
        stage("打开真实工程（只读）：%s" % _real)
        win._open_project_by_name(_real)
        # ⚠️ 不能只等"列表非空" —— 上一个夹具工程的图还在列表里，会**拿旧数据冒充**真实工程
        #    （第一版就是这么写的，结果报出"真实工程 6 张 2 行"= 夹具的数字）。
        #    判据必须是"列表里的路径全部落在真实工程目录下"（异步扫图落了盘才算数）。
        _rdir = str(win.pm.get_project_dir(_real)).replace("\\", "/").rstrip("/").lower()

        def _landed():
            imgs = getattr(win, "_all_images", None) or []
            return bool(imgs) and all(
                str(p).replace("\\", "/").lower().startswith(_rdir) for p in imgs)

        ok = wait_until(app, _landed, 120)
        ck("能打开真实工程 %s（列表已换成它的图）" % _real, ok,
           "拿到 %d 张：%s" % (len(getattr(win, "_all_images", []) or []),
                              (getattr(win, "_all_images", []) or [""])[0]))
        if ok:
            nf = len(win._all_images)
            ng = len(win._filtered_data)
            nv = len(getattr(win, "_mv_view_meta", []) or [])
            ck("真实工程：%d 张被归成 %d 行，且 %d = %d × %d（一张没少）"
               % (nf, ng, nf, ng, nv),
               (not getattr(win, "_mv_mode_list", False)) or nf == ng * nv,
               "images=%d rows=%d views=%d" % (nf, ng, nv))
            ck("真实工程的计数栏不再出现误导性的 Showing x/y",
               "Showing" not in win.stats_summary.text(), win.stats_summary.text())
            # 表格高度自适应必须**双向**成立：行少要收、行多要撑满。
            # 只测"收"会漏掉"把长列表也收成 4 行"这种更糟的回归。
            win.image_list_widget.setCurrentCell(0, 1)
            pump(app, 0.5)
            win._fit_list_height()
            pump(app, 0.2)
            ck("真实工程（%d 行）表格照旧撑满，没被自适应误收起来" % ng,
               win.image_list_widget.maximumHeight() == 16777215,
               "maxH=%d rows=%d" % (win.image_list_widget.maximumHeight(),
                                    win.image_list_widget.rowCount()))
            _rbase = win._filtered_data[0]["base"] if win._filtered_data else ""
            ck("真实工程的底栏与列表第 1 行同源（文件/尺寸都对得上）",
               bool(_rbase) and _rbase in win._info_file.fullText()
               and win._info_dims.text() != "宽高 --",
               "%s | %s" % (win._info_file.fullText(), win._info_dims.text()))
            print("  底栏     ：%s" % " | ".join(
                lb.text() for lb in (win._info_dims, win._info_xy, win._info_rgb, win._info_ann,
                                     win._info_det, win._info_ms, win._info_avg)), flush=True)
            # ★ 用户报的现象就在这个工程上（99 张真图 × 3 格）：
            #   "光标停留在形状1和形状2上时，这些值都是空的"。
            #   夹具那两张 90×140 的 PNG 说明不了真实尺寸下的换算，必须在这里也跑一遍。
            #   require_distinct=False：真图三格在同一个点上的值**可能**天然相同
            #   （比如大片平坦区），拿它当判据会出假 FAIL；真正的判据是上面那条
            #   "值 == 它自己那张文件的像素"。
            _check_ref_hover("真实工程", require_distinct=False)
            _check_bars_merge("真实工程(多视图)", multi=True)
            print("  底栏(参考窗后)：%s" % " | ".join(
                lb.text() for lb in (win._info_xy, win._info_rgb)), flush=True)
            print("  计数栏   ：%s" % win.stats_summary.text(), flush=True)
            print("  状态栏   ：%s" % win.list_status.text(), flush=True)
            print("  分组日志 ：%s" % (win._mv_view_meta,), flush=True)
            rp = os.path.join(shot_dir, "_shot_real_project_counts" + _tsuf + ".png")
            win.grab().save(rp)
            ck("真实工程截图已落盘", os.path.exists(rp) and os.path.getsize(rp) > 20000, rp)
            # 把「平均灰度」三视图并列的**真实数字**打成文本 + 单独出表截图：
            # 断言只能证明"有 3 段、能解析成 float"，证明不了"这三段确实不同、
            # 且量级合理"（真实三视图亮度差极大：te_2 形状图2 几乎全黑）。
            try:
                _heads = [win.ann_table.horizontalHeaderItem(i).text()
                          for i in range(win.ann_table.columnCount())]
                if "平均灰度" in _heads and win.ann_table.rowCount() > 0:
                    _ci = _heads.index("平均灰度")
                    _lines = []
                    for _r in range(min(6, win.ann_table.rowCount())):
                        _cat = win.ann_table.item(_r, 1)
                        _g = win.ann_table.item(_r, _ci)
                        _lines.append("%-6s %s" % (_cat.text() if _cat else "?",
                                                   _g.text() if _g else "?"))
                    print("  真实工程 平均灰度（漫反射，形状图1，形状图2）:")
                    for _l in _lines:
                        print("     " + _l, flush=True)
                    _d = 0
                    for _r in range(win.ann_table.rowCount()):
                        _g = win.ann_table.item(_r, _ci)
                        if not _g:
                            continue
                        _ps = [p for p in (_g.text() or "").split("，") if p.strip()]
                        if len(_ps) == 3:
                            try:
                                if max(float(p) for p in _ps) - min(float(p) for p in _ps) > 1.0:
                                    _d += 1
                            except Exception:
                                pass
                    ck("★ 真实工程：三段灰度**确实各不相同**（不是同一数字重复三遍）",
                       _d > 0, "三段有明显差异的行数 = %d / %d"
                       % (_d, win.ann_table.rowCount()))
                    _ap = os.path.join(shot_dir, "_shot_real_ann_table" + _tsuf + ".png")
                    # 这张表所有列都是 Stretch 模式 ⇒ **没有横向滚动可言**，
                    # scrollToColumn 不起作用，grab 出来的只有被挤在左边那几列。
                    # 抓图时临时把表撑到"各列 sizeHint 之和"（抓完立刻还原），
                    # 目的只有一个：让「平均灰度」那一列**出现在截图里**——截图里没有
                    # 的东西等于没核（这跟"看不见的列不能算数"是同一条纪律）。
                    _old_w = win.ann_table.width()
                    try:
                        _need = sum(max(60, win.ann_table.columnWidth(i))
                                    for i in range(win.ann_table.columnCount())) + 8
                        win.ann_table.setFixedWidth(int(_need))
                        win.ann_table.resizeSection(0, 46)
                        win.ann_table.resizeSection(1, 70)
                        # 灰度列要放得下 "86.2，188.2，68.3" 这种三段值。
                        # ⚠️ 该表所有列都是 Stretch 模式 ⇒ **resizeSection 对它无效**
                        #    （Stretch 强制均分），必须先把这列临时切成 Interactive。
                        try:
                            _hdr = win.ann_table.horizontalHeader()
                            _hdr.setSectionResizeMode(_ci, QHeaderView.Interactive)
                            _hdr.resizeSection(_ci, 142)
                        except Exception:
                            pass
                        app.processEvents()
                    except Exception:
                        pass
                    win.ann_table.grab().save(_ap)
                    try:
                        win.ann_table.setFixedWidth(_old_w)
                        app.processEvents()
                    except Exception:
                        pass
                    ck("真实标注表截图已落盘（**仅证明非空**；灰度列是否完整显示由上面的数值打印与"
                       "「三段各不相同」断言负责，Qt 的 Stretch 列模式在 grab 里压不宽）",
                       os.path.exists(_ap) and os.path.getsize(_ap) > 2000, _ap)
            except Exception as e:
                ck("真实工程平均灰度并列核验", False, "%s: %s" % (type(e).__name__, e))
        stage("真实工程核完")
        killed_early = True          # 不调 close()：closeEvent 会回写用户工程

    # ── ★ 类别删除（**必须放在最后**）──
    #    它自己要开一个新工程，会把"当前工程"换掉 ⇒ 插在中间时，后面那些
    #    "夹具(多视图)"断言全都会跑在一个单视图工程上、集体假 FAIL
    #    （2026-10-14 实测：24 条 FAIL 全是这一个位置错误引起的）。
    _check_class_delete()
    stage("类别删除已验完")

    # ── ★ 左栏竖排页签（放在最后：它不换工程，收尾会还原成「图像管理」）──
    _check_left_panel()
    stage("左栏页签已验完")

    # ── ★ 右栏「图像管理」删页（放最后：它会切右栏页签，也要在真实工程之后跑）──
    _check_right_panel_no_image()
    stage("右栏删页已验完")

    if not killed_early:
        win.close()
        pump(app, 0.2)
        stage("窗口已关")

    # ★ 探针不许往用户的工程注册表里写东西（用户 2026-10-14 截图里那两条
    #   probe_mv / probe_ui 就是旧版探针写进去的，删的时候才发现）
    _reg_now = _reg_path.read_text(encoding="utf-8") if _reg_path.exists() else ""
    ck("探针没有改动用户的工程注册表（.custom_projects.json 逐字节未变）",
       _reg_now == _reg_before, "%d → %d 字节" % (len(_reg_before), len(_reg_now)))

    return dump_results("标记 UI 探针（真窗口）")


if __name__ == "__main__":
    # ⚠️ 探针崩了也要把「已经跑完的断言」打出来 —— 否则前面全绿也看不见，
    #    只能看到一行 traceback，等于白跑（第一次就是这么栽的）。
    _code = 1
    try:
        _code = main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        print(flush=True)
        dump_results("标记 UI 探针（中途崩溃，以下为已跑完的部分）")
        _code = 1
    # ⚠️ 必须 os._exit：窗口关掉后 Qt 线程/扫描线程还挂着，正常 sys.exit 会
    #    在解释器收尾阶段**卡死**（实测被 timeout 240 杀掉，退出码 124）。
    # ⚠️ 必须用 TerminateProcess 硬杀，连解释器收尾和 DLL 卸载都不能走。
    #    实测（本机 2026-10-12，最小复现：只 `MainWindow()` 不 show 也一样）：
    #      · `sys.exit(code)`  → 退不出去
    #      · `os._exit(code)`  → **到达了这行**、进程依然活着
    #    卡点在 ExitProcess 的 DLL_PROCESS_DETACH（MainWindow 的依赖链里有 DLL 的
    #    卸载钩子会挂住 —— 与"本机 PyQt5 先于 torch 导入就报 c10.dll 初始化失败"
    #    是同一类 DLL 问题）。表现极具欺骗性：断言全打完了，退出码却是 124
    #    （被 timeout 杀掉），看上去像"探针卡死"。
    #    另：ctypes 取伪句柄必须把 restype 设成 c_void_p，否则句柄被截断，
    #    TerminateProcess 返回 0 / GetLastError=6（ERROR_INVALID_HANDLE）——
    #    这个坑我踩过一次，写下免得下次又以为是"硬杀也没用"。
    sys.stdout.flush()
    try:
        import ctypes
        _k = ctypes.windll.kernel32
        _k.GetCurrentProcess.restype = ctypes.c_void_p
        _k.TerminateProcess(ctypes.c_void_p(_k.GetCurrentProcess()), int(_code))
    except Exception:
        pass
    os._exit(_code)
