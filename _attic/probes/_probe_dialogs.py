# -*- coding: utf-8 -*-
"""数据集划分 / 类别管理两个对话框的自测探针。

背景：`ui/split_dialog.py` 与 `ui/class_dialog.py` 是 2026-10-22 补写的
（此前仓库里只有调用点、没有实现，点菜单必崩）。这个文件钉住它们的**数据契约**：

  · 划分按「组」不按「张」，键 = 首视图文件名（多视图 `xxx_te_0`）
  · 没标注 / 缺视图的组必须如实标出来，不能混进随机划分当正常样本
  · 改类别 = 改掩膜编号语义，删除中间项必须先说"后面整体前移"
  · 改名要把推理参数搬走，删除要摘掉残项

用法：
    "C:/aitp_venv/Scripts/python.exe" _probe_dialogs.py
输出直接打屏（纯 Qt 控件操作，不需要真窗口），末行给 PASS / FAIL 合计。
"""
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OK_N = [0]
BAD = []


def ck(name, cond, detail=""):
    if cond:
        OK_N[0] += 1
        print("  PASS %s" % name)
    else:
        BAD.append(name)
        print("  FAIL %s     <<< %s" % (name, detail))


# ── 把三个会阻塞的模态弹窗换成记录器（探针里没人点按钮）──────────────
POPUPS = []


def _fake(kind):
    def _f(parent, title, text, *a, **k):
        POPUPS.append((kind, str(title), str(text)))
        return QMessageBox.Yes if kind == "question" else QMessageBox.Ok
    return staticmethod(_f)


QMessageBox.information = _fake("information")
QMessageBox.warning = _fake("warning")
QMessageBox.critical = _fake("critical")
QMessageBox.question = _fake("question")


def _popup_text(kind=None, since=0):
    # 标题也算进来：有些信息落在标题上（如「随机划分完成」），只查正文会误判成没弹
    return "\n".join("%s\n%s" % (ti, t) for k, ti, t in POPUPS[since:]
                     if kind is None or k == kind)


def _write_img(path, color=0):
    Image.fromarray(np.full((8, 8, 3), color, np.uint8)).save(path)


def _write_mask(path, value):
    Image.fromarray(np.full((8, 8), value, np.uint8)).save(path)


def _write_json(path, labels):
    json.dump({"version": "5.0.1", "shapes": [{"label": l, "points": [[1, 1]], "shape_type": "point"}
                                              for l in labels],
               "imagePath": os.path.basename(path).replace(".json", ".png"),
               "imageHeight": 8, "imageWidth": 8},
              open(path, "w", encoding="utf-8"))


# ══════════════════════════ 造工程 ══════════════════════════

def make_multiview_project(root):
    """4 组三视图：g1 有 json / g2 有 mask / g3 无标注 / g4 缺 te_2。"""
    pd = os.path.join(root, "proj_mv")
    os.makedirs(os.path.join(pd, "images"))
    os.makedirs(os.path.join(pd, "annotations"))
    json.dump({"classes": ["background", "A"], "views": ["te_0", "te_1", "te_2"]},
              open(os.path.join(pd, "project.json"), "w", encoding="utf-8"), indent=2)
    views_by_group = {"g1": ["te_0", "te_1", "te_2"], "g2": ["te_0", "te_1", "te_2"],
                      "g3": ["te_0", "te_1", "te_2"], "g4": ["te_0", "te_1"]}
    for g, views in views_by_group.items():
        for v in views:
            _write_img(os.path.join(pd, "images", "%s_%s.png" % (g, v)))
    _write_json(os.path.join(pd, "images", "g1_te_0.json"), ["A"])
    _write_mask(os.path.join(pd, "annotations", "g2_te_0_mask.png"), 1)
    return pd


def make_flat_project(root, n=20):
    """n 张单图 + 每张都有 json（用来验比例与可复现）。"""
    pd = os.path.join(root, "proj_flat")
    os.makedirs(os.path.join(pd, "images"))
    os.makedirs(os.path.join(pd, "annotations"))
    json.dump({"classes": ["background", "A"]},
              open(os.path.join(pd, "project.json"), "w", encoding="utf-8"), indent=2)
    for i in range(n):
        stem = "s%02d" % i
        _write_img(os.path.join(pd, "images", stem + ".png"))
        _write_json(os.path.join(pd, "images", stem + ".json"), ["A"])
    return pd


def _select(dlg, groups):
    """按组名多选（走 selectionModel，和用户 Ctrl 点一样）。"""
    from PyQt5.QtCore import QItemSelectionModel
    sm = dlg.table.selectionModel()
    sm.clearSelection()
    want = set(groups)
    for tr, idx in enumerate(dlg._view_idx):
        if dlg._rows[idx]["group"] in want:
            sm.select(dlg.table.model().index(tr, 0),
                      QItemSelectionModel.Select | QItemSelectionModel.Rows)


def _read_split(pd):
    p = os.path.join(pd, "train_test_split.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p, "r", encoding="utf-8"))


# ══════════════════════════ 划分对话框 ══════════════════════════

def test_split_multiview(pd):
    from ui.split_dialog import SplitDialog
    print("\n[1] 多视图工程：按组列出、状态如实")
    dlg = SplitDialog(pd)
    ck("扫出 4 个「组」（12 张图不能算成 12 行）", len(dlg._rows) == 4,
       "rows=%d" % len(dlg._rows))
    ck("表格行数与组数一致（且没有搜索过滤）", dlg.table.rowCount() == 4,
       "rowCount=%d" % dlg.table.rowCount())
    by = {r["group"]: r for r in dlg._rows}
    ck("完整三视图的组标 3/3", by["g1"]["n_views"] == 3 and by["g2"]["n_views"] == 3,
       "g1=%d g2=%d" % (by["g1"]["n_views"], by["g2"]["n_views"]))
    ck("缺 te_2 的组标 2/3（不是被偷偷丢掉）", by["g4"]["n_views"] == 2,
       "g4=%d" % by["g4"]["n_views"])
    ck("有 json 的算已标注", by["g1"]["has_ann"])
    ck("有 mask 的算已标注", by["g2"]["has_ann"])
    ck("没标注的如实标否", not by["g3"]["has_ann"] and not by["g4"]["has_ann"])
    s = dlg.lbl_summary.text()
    ck("摘要报「共 4 组 / 未划分 4」", "共 4 组" in s and "未划分 4" in s, s)
    ck("摘要单独报「未标注 2（进不了训练）」", "未标注 2" in s, s)
    ck("摘要单独报「缺视图 1」", "缺视图 1" in s, s)
    ck("初始全是「未划分」（不会擅自替用户决定）",
       all(r["split"] == "" for r in dlg._rows))
    ck("规范键 = 首视图文件名（te_0）", by["g1"]["base"] == "g1_te_0",
       by["g1"]["base"])

    print("\n[2] 全选设为训练 → 保存落盘")
    dlg.table.selectAll()
    n_sel = len(dlg._selected_row_idx())
    ck("全选拿到 4 行", n_sel == 4, "n=%d" % n_sel)
    dlg._set_selected("train")
    ck("刷新后多选**没有丢**（selectRow 会清前面的选择，已改走 selectionModel）",
       len(dlg._selected_row_idx()) == 4, "n=%d" % len(dlg._selected_row_idx()))
    ck("改动被识别为 dirty（保存按钮才亮）", dlg._dirty and dlg.btn_save.isEnabled())
    dlg._save()
    sp = _read_split(pd)
    ck("落盘 4 个键，全是首视图名",
       set(sp) == {"g1_te_0", "g2_te_0", "g3_te_0", "g4_te_0"}, str(sorted(sp)))
    ck("值全是 train", all(v == "train" for v in sp.values()), str(sp))

    print("\n[3] 单组改成验证")
    dlg.reload()
    _select(dlg, ["g2"])
    ck("只选中 1 行", len(dlg._selected_row_idx()) == 1,
       str(dlg._selected_row_idx()))
    dlg._set_selected("val")
    dlg._save()
    sp = _read_split(pd)
    ck("g2 变 val、其余仍 train",
       sp.get("g2_te_0") == "val" and sp.get("g1_te_0") == "train", str(sp))

    print("\n[4] 老数据用「组名」做键时：显示要认、保存要归一")
    with open(os.path.join(pd, "train_test_split.json"), "w", encoding="utf-8") as f:
        json.dump({"g1": "val", "g9_te_0": "train"}, f)
    dlg.reload()
    by = {r["group"]: r for r in dlg._rows}
    ck("按组名做键的老数据仍然显示为「验证」", by["g1"]["split"] == "val",
       by["g1"]["split"])
    dlg._save()
    sp = _read_split(pd)
    ck("保存后归一成 te_0 键", sp.get("g1_te_0") == "val", str(sp))
    ck("别名键 g1 被清掉（同一组两个键会打架）", "g1" not in sp, str(sp))
    ck("与本工程无关的旧键 g9_te_0 原样保留", sp.get("g9_te_0") == "train", str(sp))

    print("\n[5] 「未划分」= 从划分表里删掉这一项")
    dlg.reload()
    _select(dlg, ["g3", "g4"])
    dlg._set_selected("")
    dlg._save()
    sp = _read_split(pd)
    ck("设为未划分的组不再出现在表里",
       "g3_te_0" not in sp and "g4_te_0" not in sp, str(sorted(sp)))
    ck("其它组不受影响", sp.get("g1_te_0") == "val", str(sp))
    dlg.close()


def test_split_ratio(pd):
    from ui.split_dialog import SplitDialog
    print("\n[6] 单图工程：比例、可复现、只划已标注")
    dlg = SplitDialog(pd)
    ck("20 张单图 = 20 行", len(dlg._rows) == 20, "rows=%d" % len(dlg._rows))
    ck("单图工程「视图」列显示 1", all(r["n_views"] == 1 for r in dlg._rows))
    ck("单图工程的键就是文件名", dlg._rows[0]["base"] == "s00", dlg._rows[0]["base"])

    dlg.sp_train.setValue(70)
    dlg.sp_val.setValue(15)
    dlg.sp_test.setValue(15)
    dlg.sp_seed.setValue(1234)
    since = len(POPUPS)
    dlg._random_split()
    cnt = {"train": 0, "val": 0, "test": 0, "": 0}
    for r in dlg._rows:
        cnt[r["split"]] = cnt.get(r["split"], 0) + 1
    ck("20 组按 70/15/15 分完，没有漏网的未划分组", cnt[""] == 0, str(cnt))
    ck("训练 ~14 / 验证 3 / 测试 3（四舍五入到组）",
       cnt["train"] == 14 and cnt["val"] == 3 and cnt["test"] == 3, str(cnt))
    first = {r["base"]: r["split"] for r in dlg._rows}
    ck("划分完成有弹窗告知（不是悄悄改了）",
       "随机划分完成" in _popup_text(since=since), _popup_text(since=since))

    dlg2 = SplitDialog(pd)
    dlg2.sp_train.setValue(70)
    dlg2.sp_val.setValue(15)
    dlg2.sp_test.setValue(15)
    dlg2.sp_seed.setValue(1234)
    dlg2._random_split()
    same = {r["base"]: r["split"] for r in dlg2._rows}
    ck("同种子 = 同划分（可复现）", same == first, "两次不一致")
    dlg2.close()

    print("\n[7] 比例不合 100% 必须拦下来")
    dlg.sp_val.setValue(25)
    before = {r["base"]: r["split"] for r in dlg._rows}
    since = len(POPUPS)
    dlg._random_split()
    after = {r["base"]: r["split"] for r in dlg._rows}
    ck("比例 70+25+15=110 被拦（弹窗说清原因）",
       "必须等于 100" in _popup_text(since=since), _popup_text(since=since))
    ck("被拦时一行都没改", before == after)
    dlg.close()


def make_nested_cls_project(root):
    """分类工程 ImageFolder 布局：图在 images/<类别>/ 下，划分键必须带前缀（ok/a）。"""
    pd = os.path.join(root, "proj_nested")
    for sub in ("ok", "ng"):
        os.makedirs(os.path.join(pd, "images", sub))
    os.makedirs(os.path.join(pd, "annotations"))
    json.dump({"classes": ["background", "ok", "ng"]},
              open(os.path.join(pd, "project.json"), "w", encoding="utf-8"), indent=2)
    for stem, sub in (("a", "ok"), ("b", "ok"), ("c", "ng")):
        _write_img(os.path.join(pd, "images", sub, stem + ".png"))
    json.dump({"labels": ["ok", "ng"],
               "mapping": {"images/ok/a": 0, "images/ok/b": 0, "images/ng/c": 1}},
              open(os.path.join(pd, "annotations", "class_labels.json"), "w", encoding="utf-8"),
              indent=2)
    return pd


def test_split_nested(root):
    """ImageFolder 布局：对话框写的键必须与训练侧查询的键**逐字相同**。"""
    from ui.split_dialog import SplitDialog
    from classification.dataset import _default_split_map, _list_images_mapping
    print("\n[9] 分类工程（images/<类别>/ 布局）：键要带目录前缀")
    pd = make_nested_cls_project(root)
    dlg = SplitDialog(pd)
    ck("扫出 3 行（每张图一行，不是按目录算）", len(dlg._rows) == 3,
       "rows=%d" % len(dlg._rows))
    ck("组名带类别前缀", {r["group"] for r in dlg._rows} == {"ok/a", "ok/b", "ng/c"},
       str(sorted(r["group"] for r in dlg._rows)))
    ck("分类工程的图算「已标注」（class_labels.json 有映射）",
       all(r["has_ann"] for r in dlg._rows))
    dlg.table.selectAll()
    dlg._set_selected("train")
    dlg._save()
    sp = _read_split(pd)
    ck("落盘键 = ok/a、ok/b、ng/c（不是裸文件名）",
       set(sp) == {"ok/a", "ok/b", "ng/c"}, str(sorted(sp)))
    tr, _n = _list_images_mapping(pd, "train", _default_split_map(pd))
    ck("★ 训练侧按这些键真能取到 3 张（写 a 而查 ok/a 就会一张都取不到）",
       len(tr) == 3, "训练取到 %d 张" % len(tr))
    dlg.close()


def test_split_empty(root):
    from ui.split_dialog import SplitDialog
    print("\n[8] 空工程不能崩")
    pd = os.path.join(root, "proj_empty")
    os.makedirs(os.path.join(pd, "images"))
    json.dump({"classes": ["background"]},
              open(os.path.join(pd, "project.json"), "w", encoding="utf-8"))
    dlg = SplitDialog(pd)
    ck("没有图片时 0 行、摘要写 0 组", len(dlg._rows) == 0 and "共 0 组" in dlg.lbl_summary.text(),
       dlg.lbl_summary.text())
    dlg._set_selected("train")   # 没选中就设 -> 只该弹提示，不该崩
    ck("没选任何行时设划分只提示不崩", True)
    dlg.close()


# ══════════════════════════ 类别对话框 ══════════════════════════

def test_class_dialog(pd, pd_flat):
    from ui.class_dialog import ClassDialog
    from core.class_discovery import read_project_classes

    # 造一份推理参数，验"改名要搬家、删除要摘残项"
    json.dump({"version": 1, "global": {}, "classes": {"A": {"min_area": [10, 900]}}},
              open(os.path.join(pd, "inference_params.json"), "w", encoding="utf-8"))

    print("\n[9] 类别表初始化")
    dlg = ClassDialog(pd)
    ck("读到 project.json 的类别表", dlg._classes == ["background", "A"], str(dlg._classes))
    ck("表格行数与类别数一致", dlg.table.table.rowCount() == len(dlg._classes),
       str(dlg.table.table.rowCount()))
    ck("类名那一列写的是「A」", dlg.table.table.item(1, 1).text() == "A",
       dlg.table.table.item(1, 1).text())

    print("\n[10] 添加 / 改名 / 删除 / 移动")
    _orig_get = QInputDialog.getText
    QInputDialog.getText = staticmethod(lambda *a, **k: ("B", True))
    dlg._add()
    ck("添加 B 后立即落盘", read_project_classes(pd) == ["background", "A", "B"],
       str(read_project_classes(pd)))

    QInputDialog.getText = staticmethod(lambda *a, **k: ("A2", True))
    dlg.table.setCurrentRow(1, emit=False)
    dlg._rename()
    ck("改名 A→A2 落盘", read_project_classes(pd) == ["background", "A2", "B"],
       str(read_project_classes(pd)))
    params = json.load(open(os.path.join(pd, "inference_params.json"), encoding="utf-8"))
    ck("推理参数跟着搬到 A2（不搬就悄悄退回默认阈值）",
       "A2" in (params.get("classes") or {}) and "A" not in (params.get("classes") or {}),
       str(params.get("classes")))
    ck("改名有弹窗说明「标注里的旧名还在」",
       "旧名" in _popup_text(kind="information"), _popup_text(kind="information"))

    QInputDialog.getText = staticmethod(lambda *a, **k: ("C", True))
    dlg._add()
    ck("再加 C", read_project_classes(pd) == ["background", "A2", "B", "C"],
       str(read_project_classes(pd)))

    since = len(POPUPS)
    dlg.table.setCurrentRow(2, emit=False)     # 删中间的 B
    dlg._delete()
    txt = _popup_text(since=since)
    ck("删中间项前警告「整体前移一位」", "整体前移一位" in txt, txt)
    ck("删完落盘", read_project_classes(pd) == ["background", "A2", "C"],
       str(read_project_classes(pd)))

    since = len(POPUPS)
    dlg.table.setCurrentRow(0, emit=False)     # background
    dlg._delete()
    ck("background 不能删（且说清是背景类）",
       "不能删除" in _popup_text(since=since), _popup_text(since=since))
    ck("background 没被删掉", read_project_classes(pd)[0] == "background")

    dlg.table.setCurrentRow(-1, emit=False)
    since = len(POPUPS)
    dlg._delete()
    ck("没选中时说「先点一行」，而不是说成 background 不能删",
       "先" in _popup_text(since=since) and "background" not in _popup_text(since=since),
       _popup_text(since=since))

    dlg.table.setCurrentRow(2, emit=False)     # C 上移
    dlg._move(-1)
    ck("上移一位落盘", read_project_classes(pd) == ["background", "C", "A2"],
       str(read_project_classes(pd)))

    since = len(POPUPS)
    dlg._move(-1)                              # C 已经在 1 号位，再上移会撞 background
    ck("撞到 background 时拒绝并说明", "恒占 0 号位" in _popup_text(since=since),
       _popup_text(since=since))

    QInputDialog.getText = _orig_get

    print("\n[11] 关闭语义：盘上动过才让主窗刷新")
    ck("改过 => 返回 Accepted（主窗据此刷新）",
       dlg._dirty and dlg._disk_changed)
    dlg.close()
    # 干净的工程：标注 JSON 里的 label 与类别表完全对得上 ⇒ 不该无意义地刷主窗
    clean = ClassDialog(pd_flat)
    ck("干净工程没改盘 => 关闭走 Rejected（不白跑 1 秒重扫）",
       not clean._disk_changed and not clean._dirty)
    clean.close()
    # 上面把 A 改成了 A2，而 g1_te_0.json 里仍写着 A ⇒ 重开会把 A「补回来」。
    # 这不是 bug 而是"改名不动标注文件"的必然后果，这里钉住它，避免有人以为改名已彻底生效。
    again = ClassDialog(pd)
    ck("改名不动标注 => 重开时旧名被 discovery 补回（提示文案已说实话）",
       again._disk_changed and "background" in again._classes and "A" in again._classes,
       str(again._classes))
    ck("补入动作写进状态行（不是悄悄发生）",
       "补入" in again.lbl_state.text(), again.lbl_state.text())
    again.close()


# ══════════════════════════ main ══════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    root = tempfile.mkdtemp(prefix="aitp_dlg_")
    try:
        pd_mv = make_multiview_project(root)
        pd_flat = make_flat_project(root, 20)
        test_split_multiview(pd_mv)
        test_split_ratio(pd_flat)
        test_split_nested(root)
        test_split_empty(root)
        test_class_dialog(pd_mv, pd_flat)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\n" + "=" * 70)
    print("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (OK_N[0], len(BAD), OK_N[0] + len(BAD)))
    if BAD:
        for b in BAD:
            print("  FAILED: %s" % b)
    print("=" * 70)
    return 0 if not BAD else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("CRASH:\n" + traceback.format_exc())
        sys.exit(2)
