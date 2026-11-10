# -*- coding: utf-8 -*-
"""`ClassTableView._compute_counts()` 结果缓存的回归测试（B 项）。

缓存最容易出的错不是"没命中"，而是**该失效的时候没失效** —— 用户改了标注、
加了图、换了划分，数字却纹丝不动，而界面上看不出任何异常。所以这里的重点是
"改了什么就必须跟着变"，不是"命中多快"。

底线：
    1. 命中缓存时**绝不再扫一遍**标注（否则缓存没意义）
    2. 标注 / 加图 / 划分 / 类别真源（class_labels.json）任何一处变了，必须重算
    3. 目录读不了（语义未知）时**不进缓存**，宁可重算
    4. `_split_bases`（点数字格子过滤图片列表用）在命中时也要还原成同一份

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_counts_cache.py
"""
import json
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ⚠️ 与 main.py 一致：torch 必须先于 PyQt5，否则 WinError 1114 c10.dll
try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print(f"[warn] torch 加载失败：{_e}")

from PyQt5.QtWidgets import QApplication  # noqa: E402

import ui.class_table as ct  # noqa: E402

failures = []
_APP = None


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------- helpers

def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, ensure_ascii=False)


def _wbytes(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _labelme(*labels):
    return {
        "version": "5.0.1",
        "shapes": [{"label": lb, "points": [[0, 0], [1, 1]], "shape_type": "polygon"}
                   for lb in labels],
        "imagePath": "x.png",
        "imageData": None,
    }


def make_seg_project(tmp, classes=("background", "Lb", "Ps")):
    """最小分割工程：JSON 与图并排放在 images/（labelme 约定，真实工程也是这样）。"""
    pd = os.path.join(tmp, "segproj")
    _w(os.path.join(pd, "project.json"),
       {"name": "segproj", "task_type": "semantic_segmentation", "classes": list(classes)})
    _wbytes(os.path.join(pd, "images", "a.png"))
    _w(os.path.join(pd, "images", "a.json"), _labelme("Lb"))
    _wbytes(os.path.join(pd, "images", "b.png"))
    _w(os.path.join(pd, "images", "b.json"), _labelme("Ps"))
    _wbytes(os.path.join(pd, "annotations", "a_mask.png"))
    _wbytes(os.path.join(pd, "annotations", "b_mask.png"))
    _w(os.path.join(pd, "train_test_split.json"), {"a": "train", "b": "val"})
    return pd


def make_cls_project(tmp, classes=("OK", "NG")):
    """最小分类工程：类别真源在 annotations/class_labels.json（mapping）。"""
    pd = os.path.join(tmp, "clsproj")
    _w(os.path.join(pd, "project.json"),
       {"name": "clsproj", "task_type": "classification", "classes": []})
    # ⚠️ 类别目录名与类别名必须一致（真实 ImageFolder 工程就是这样）；
    #    目录名与类别名只差大小写时，「空集」会因为比较区分大小写而多算 —— 已知旧问题，本轮没动。
    for rel in ("OK/a.bmp", "OK/b.bmp", "NG/c.bmp"):
        _wbytes(os.path.join(pd, "images", rel))
    _w(os.path.join(pd, "annotations", "class_labels.json"),
       {"labels": list(classes),
        "mapping": {"images/OK/a.bmp": 0, "images/NG/c.bmp": 1}})
    _w(os.path.join(pd, "train_test_split.json"), {})
    return pd


def new_table(pd, classes):
    tv = ct.ClassTableView()
    tv.set_project_dir(pd, with_counts=False)
    tv.set_classes(list(classes), with_counts=False)
    return tv


class _CountScans:
    """把 ui.class_table 里的 scan_labels 换成计数器，用来证明"没有重扫"。"""

    def __init__(self):
        self.n = 0
        self._orig = ct.scan_labels

    def __enter__(self):
        def wrapped(p):
            self.n += 1
            return self._orig(p)
        ct.scan_labels = wrapped
        return self

    def __exit__(self, *a):
        ct.scan_labels = self._orig
        return False


# ---------------------------------------------------------------- tests

def test_counts_and_split():
    """数字本身要对：train / val / test 分别归到哪个桶。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    c = tv._compute_counts()
    assert c[1] == (1, 0, 0, 0), f"Lb 应 train=1：{c[1]}"
    assert c[2] == (0, 0, 1, 0), f"Ps 应 val=1：{c[2]}"
    assert c[0] == (0, 0, 0, 0), f"background 不该有数：{c[0]}"


def test_cache_hit_no_rescan():
    """第二次必须走缓存：不重扫标注、结果一致、耗时明显更低。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    first = tv._compute_counts()
    with _CountScans() as cs:
        second = tv._compute_counts()
    assert cs.n == 0, f"命中缓存后不该再扫标注，实际扫了 {cs.n} 次"
    assert first == second, f"缓存结果与原结果不一致：{first} vs {second}"


def test_cache_hit_restores_split_bases():
    """命中时 `_split_bases` 必须一起还原 —— 否则点数字格子过滤不出图。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    tv._compute_counts()
    fresh = {k: set(v) for k, v in tv._split_bases.items()}
    tv._compute_counts()  # 命中
    got = {k: set(v) for k, v in tv._split_bases.items()}
    assert got == fresh, f"_split_bases 未还原：{got} != {fresh}"
    assert tv.bases_for("Lb", "train"), "bases_for 应有 Lb/train 的集合"


def test_annotation_edit_invalidates():
    """改了标注 JSON（多画一个 Ps）→ 数字必须变。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    before = tv._compute_counts()
    assert before[2] == (0, 0, 1, 0), before[2]
    time.sleep(0.01)
    _w(os.path.join(pd, "images", "a.json"), _labelme("Lb", "Ps", "Ps"))
    after = tv._compute_counts()
    assert after[1] == (1, 0, 0, 0), f"Lb 仍应 train=1：{after[1]}"
    assert after[2] == (1, 0, 1, 0), f"a 加了 Ps ⇒ train=1 且 val=1：{after[2]}"


def test_new_image_invalidates():
    """往 images/ 加一张带标注的图 → 数字必须变。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    before = tv._compute_counts()
    assert before[1] == (1, 0, 0, 0), before[1]
    time.sleep(0.01)
    _wbytes(os.path.join(pd, "images", "c.png"))
    _w(os.path.join(pd, "images", "c.json"), _labelme("Lb"))
    after = tv._compute_counts()
    assert after[1] == (2, 0, 0, 0), f"Lb 应 train=2：{after[1]}"


def test_split_change_invalidates():
    """改 train_test_split.json（a 从 train 挪到 test）→ train/test 必须跟着变。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    assert tv._compute_counts()[1] == (1, 0, 0, 0)
    time.sleep(0.01)
    _w(os.path.join(pd, "train_test_split.json"), {"a": "test", "b": "val"})
    after = tv._compute_counts()
    assert after[1] == (0, 1, 0, 0), f"a 改判 test 后 Lb 应 test=1：{after[1]}"


def test_labels_change_invalidates():
    """类别表顺序变了（Lb ↔ Ps 换位）→ 缓存不能按旧索引返回。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    assert tv._compute_counts()[1] == (1, 0, 0, 0)
    tv.set_classes(["background", "Ps", "Lb"], with_counts=False)
    after = tv._compute_counts()
    assert after[1] == (0, 0, 1, 0), f"换位后 index1 是 Ps，应 val=1：{after[1]}"
    assert after[2] == (1, 0, 0, 0), f"换位后 index2 是 Lb，应 train=1：{after[2]}"


def test_class_labels_json_invalidates():
    """★ 审核点名的那条：改 class_labels.json 的 mapping → 数字必须变。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_cls_project(tmp)
    tv = new_table(pd, ["OK", "NG"])
    c0 = tv._compute_counts()
    assert c0[0] == (1, 0, 0, 1), f"OK: mapping 里 1 张、目录里还有 1 张没标（空集）：{c0[0]}"
    assert c0[1] == (1, 0, 0, 0), f"NG train=1：{c0[1]}"
    time.sleep(0.01)
    _w(os.path.join(pd, "annotations", "class_labels.json"),
       {"labels": ["OK", "NG"],
        "mapping": {"images/OK/a.bmp": 0, "images/OK/b.bmp": 0, "images/NG/c.bmp": 1}})
    c1 = tv._compute_counts()
    assert c1[0] == (2, 0, 0, 0), f"b 补进 mapping 后 OK train=2、空集 0：{c1[0]}"


def test_class_image_added_invalidates():
    """分类工程往类别目录里塞图（class_labels.json 没动）→ 空集必须变。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_cls_project(tmp)
    tv = new_table(pd, ["OK", "NG"])
    assert tv._compute_counts()[0] == (1, 0, 0, 1)
    time.sleep(0.01)
    _wbytes(os.path.join(pd, "images", "OK", "d.bmp"))
    after = tv._compute_counts()
    assert after[0] == (1, 0, 0, 2), f"OK/ 下多一张未标注 ⇒ 空集 2：{after[0]}"


def test_invalidate_forces_rescan():
    """「刷新」按钮走 invalidate_counts()，之后必须真的重扫。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    tv._compute_counts()
    tv.invalidate_counts()
    assert not tv._counts_cache, "invalidate_counts 后缓存应为空"
    with _CountScans() as cs:
        tv._compute_counts()
    assert cs.n >= 2, f"作废缓存后应重新扫描标注，实际 {cs.n} 次"


def test_unreadable_dir_no_cache():
    """目录读不了 → 指纹 None → 不进缓存（不能把"读失败"当"全没了"）。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    orig = ct.ClassTableView.__dict__["_dir_index"]   # 存 staticmethod 本体，别存解开后的函数
    ct.ClassTableView._dir_index = staticmethod(lambda d: None)
    try:
        sig, img_i, ann_i = tv._counts_signature(ct.Path(pd), ["background", "Lb", "Ps"])
        assert sig is None and img_i is None, "读不了时指纹必须是 None"
        tv._compute_counts()
        assert not tv._counts_cache, "指纹不可用时不该写缓存"
    finally:
        ct.ClassTableView._dir_index = orig


def test_switch_back_is_cheap():
    """切走再切回同一工程要命中缓存（这是本轮优化的主要收益）。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd1 = make_seg_project(tmp)
    pd2 = make_cls_project(tmp)
    tv = new_table(pd1, ["background", "Lb", "Ps"])
    tv._compute_counts()
    tv.set_project_dir(pd2, with_counts=False)
    tv.set_classes(["OK", "NG"], with_counts=False)
    tv._compute_counts()
    tv.set_project_dir(pd1, with_counts=False)
    tv.set_classes(["background", "Lb", "Ps"], with_counts=False)
    with _CountScans() as cs:
        tv._compute_counts()
    assert cs.n == 0, f"切回原工程应命中缓存，实际重扫 {cs.n} 次"


def test_cache_is_capped():
    """缓存不能无限长：超过上限淘汰最早的。"""
    tmp = tempfile.mkdtemp(prefix="cc_")
    tv = new_table(os.path.join(tmp, "none"), [])
    tv._counts_cache.clear()
    for i in range(ct._COUNTS_CACHE_MAX + 3):
        tv._counts_cache["k{}".format(i)] = ("sig", {}, {})
        while len(tv._counts_cache) > ct._COUNTS_CACHE_MAX:
            tv._counts_cache.pop(next(iter(tv._counts_cache)))
    assert len(tv._counts_cache) == ct._COUNTS_CACHE_MAX


def main():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    print("=== 类别统计缓存 回归测试 ===")
    checks = [
        ("数字与划分正确", test_counts_and_split),
        ("命中缓存不重扫", test_cache_hit_no_rescan),
        ("命中时还原 _split_bases", test_cache_hit_restores_split_bases),
        ("改标注 JSON 会失效", test_annotation_edit_invalidates),
        ("加图会失效", test_new_image_invalidates),
        ("改划分会失效", test_split_change_invalidates),
        ("改类别顺序会失效", test_labels_change_invalidates),
        ("改 class_labels.json 会失效", test_class_labels_json_invalidates),
        ("类别目录加图会失效", test_class_image_added_invalidates),
        ("刷新按钮绕过缓存", test_invalidate_forces_rescan),
        ("目录读不了不缓存", test_unreadable_dir_no_cache),
        ("切回工程命中缓存", test_switch_back_is_cheap),
        ("缓存有上限", test_cache_is_capped),
    ]
    for name, fn in checks:
        check(name, fn)

    print()
    if failures:
        print(f"❌ {len(failures)}/{len(checks)} 项失败")
        return 1
    print(f"✅ {len(checks)}/{len(checks)} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
