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


def make_seg_project(tmp, classes=("background", "Lb", "Ps"), name="segproj"):
    """最小分割工程：JSON 与图并排放在 images/（labelme 约定，真实工程也是这样）。"""
    pd = os.path.join(tmp, name)
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
    """目录读不了 → 指纹 None → 不进缓存（不能把"读失败"当"全没了"）。

    两种情况都要覆盖：`images/` 读不了（`_scan_dir` 返回 None）、
    `annotations/` 读不了（`_dir_index` 返回 None）。
    """
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    cls = ["background", "Lb", "Ps"]

    for name, attr, value in (("images 读不了", "_scan_dir", staticmethod(lambda d: (None, None))),
                              ("annotations 读不了", "_dir_index", staticmethod(lambda d: None))):
        tv = new_table(pd, cls)
        tv._counts_cache.clear()
        orig = ct.ClassTableView.__dict__[attr]   # 存 staticmethod 本体，别存解开后的函数
        setattr(ct.ClassTableView, attr, value)   # __dict__ 是 mappingproxy，不能直接赋值
        try:
            sig = tv._counts_signature(ct.Path(pd), cls)[0]
            assert sig is None, "{}：指纹必须是 None".format(name)
            tv._compute_counts()
            assert not tv._counts_cache, "{}：指纹不可用时不该写缓存".format(name)
        finally:
            setattr(ct.ClassTableView, attr, orig)


def test_subdirs_sig_failure_no_cache():
    """子目录摘要读不了（返回 None）也必须算"指纹不可用"。

    ★ 审核 P2-3：第一版把 None 直接塞进指纹列表 ⇒ 指纹照样非 None ⇒ **照样进缓存**，
      与"`_dir_index` 返回 None 就不缓存"是两套哲学，会静默存下不完整的指纹。
    """
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    cls = ["background", "Lb", "Ps"]
    tv = new_table(pd, cls)
    tv._counts_cache.clear()
    orig = ct.ClassTableView.__dict__["_subdirs_sig"]
    setattr(ct.ClassTableView, "_subdirs_sig", staticmethod(lambda dirs, depth=2: None))
    try:
        sig = tv._counts_signature(ct.Path(pd), cls)[0]
        assert sig is None, "子目录摘要读不了时指纹必须是 None"
        tv._compute_counts()
        assert not tv._counts_cache, "指纹不可用时不该写缓存"
    finally:
        setattr(ct.ClassTableView, "_subdirs_sig", orig)


def test_return_value_is_a_copy():
    """返回的 dict / `_split_bases` 都必须是副本，改返回值不能污染缓存。

    ★ 审核 P1-2：写入路径第一版存的是 result 本体，`r[1] = (999,...)` 之后
      再命中缓存拿到的就是 999 —— 错且下次还错。
    """
    tmp = tempfile.mkdtemp(prefix="cc_")
    pd = make_seg_project(tmp)
    tv = new_table(pd, ["background", "Lb", "Ps"])
    r = tv._compute_counts()
    key = tv._counts_key(ct.Path(pd))
    assert r is not tv._counts_cache[key][1], "返回值不该是缓存本体"
    r[1] = (999, 999, 999, 999)
    assert tv._counts_cache[key][1][1] == (1, 0, 0, 0), "改返回值污染了缓存本体"

    # 两条路径都要防：写缓存那条（上面）和**命中**那条（下面）
    r2 = tv._compute_counts()                      # 这一拍是命中
    assert r2 is not tv._counts_cache[key][1], "命中返回值也不该是缓存本体"
    assert r2[1] == (1, 0, 0, 0), "命中缓存后拿到了被污染的数字"
    r2[1] = (888, 888, 888, 888)
    assert tv._compute_counts()[1] == (1, 0, 0, 0), "改命中返回值污染了缓存本体"

    # _split_bases 同理：外部拿到的集合不能改到内部
    bases = tv.bases_for("Lb", "train")
    bases.add("ZZZ_不该进来的")
    assert "ZZZ_不该进来的" not in tv.bases_for("Lb", "train"), "bases_for 返回了内部 set 本体"


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


def _make_projects(tmp, n):
    """造 n 个互不相同的小工程，返回 [(pd, classes), ...]。"""
    out = []
    for i in range(n):
        pd = make_seg_project(tmp, name="p{}".format(i))
        out.append((pd, ["background", "Lb", "Ps"]))
    return out


def _touch(tv, pd, classes):
    """切到某个工程并算一次（= 真实路径 set_project_dir + set_classes + compute）。"""
    tv.set_project_dir(pd, with_counts=False)
    tv.set_classes(list(classes), with_counts=False)
    tv._compute_counts()


def test_cache_is_capped():
    """缓存有上限：真的走 `_compute_counts()`，淘汰真的由生产代码执行。

    ⚠️ 第一版这条是**自证测试**：测试体里自己塞 dict、自己跑淘汰循环、再断言自己淘汰对了，
      从头到尾没调 `_compute_counts()` —— 把生产代码里那几行淘汰整段删掉它也照样 PASS。
      审核指出后改成真跑（P1-1）。
    ⚠️ 还有一处更隐蔽的错：`new_table()` 是 `with_counts=False`，**不会算统计**。
      所以第一版里"第一个工程"压根没进缓存，`keys[0] not in cache` 是句废话断言。
      现在每个工程都显式 `_touch()` 一次。
    """
    tmp = tempfile.mkdtemp(prefix="cc_")
    mx = ct._COUNTS_CACHE_MAX
    projs = _make_projects(tmp, mx + 3)
    tv = new_table(*projs[0])
    keys = []
    for pd, classes in projs:            # 11 个工程全部真算一遍
        _touch(tv, pd, classes)
        keys.append(tv._counts_key(ct.Path(pd)))

    assert len(tv._counts_cache) == mx, "缓存条数应被压到上限：{}".format(len(tv._counts_cache))
    assert keys[-1] in tv._counts_cache, "刚算完的工程必须还在"
    assert keys[0] not in tv._counts_cache, "最早的那个必须被淘汰"
    assert keys[2] not in tv._counts_cache, "超出的那些都该淘汰"
    assert keys[3] in tv._counts_cache, "界内最老的那个应保留"
    assert len(tv._counts_cache) == len(set(keys[-mx:])), "留下的应该是最后 mx 个"


def test_lru_keeps_hot_project():
    """淘汰的是「最久没用」的，不是「最早插入」的 —— 来回切的两个工程不能被自己挤掉。

    ★ 审核 P2-4。这条测试第一版是假绿的：`new_table()` 不算统计，所以"再访问 p0"
      其实是 miss 而不是命中，有没有 LRU 挪位结果都一样。现在改成：
      先让缓存真的填满 8 个，再断言**命中后 p0 被挪到末尾**（机制本身），最后看淘汰对象。
    """
    tmp = tempfile.mkdtemp(prefix="cc_")
    mx = ct._COUNTS_CACHE_MAX
    projs = _make_projects(tmp, mx + 1)
    tv = new_table(*projs[0])
    keys = []
    for pd, classes in projs[:mx]:       # 正好填满 mx 个
        _touch(tv, pd, classes)
        keys.append(tv._counts_key(ct.Path(pd)))
    assert len(tv._counts_cache) == mx
    assert list(tv._counts_cache)[0] == keys[0], "刚填满时 p0 应在最前（最久未用）"

    _touch(tv, *projs[0])                # 真命中：p0 应被挪到末尾
    assert list(tv._counts_cache)[-1] == keys[0], "命中后 p0 必须挪到末尾（LRU 的判据）"

    _touch(tv, *projs[mx])               # 再来一个新工程，触发淘汰
    assert keys[0] in tv._counts_cache, "刚用过的 p0 不该被淘汰"
    assert keys[1] not in tv._counts_cache, "最久未用的 p1 才该被淘汰"


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
        ("子目录摘要读不了不缓存", test_subdirs_sig_failure_no_cache),
        ("返回值是副本", test_return_value_is_a_copy),
        ("切回工程命中缓存", test_switch_back_is_cheap),
        ("缓存有上限", test_cache_is_capped),
        ("淘汰最久未用的（LRU）", test_lru_keeps_hot_project),
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
