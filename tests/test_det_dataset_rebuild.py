# -*- coding: utf-8 -*-
"""`build_yolo_dataset()` 重建数据集：**不能留下上一轮的副本**。

## 这条 bug 长什么样

图片按 base 名落在 `images/train|val/`，拷贝是 `if not dst_img.exists()` 跳过。
而划分是 `random.seed(42)` + `shuffle(按文件名排序的列表)`：

| 你做了什么 | 列表长度 | 划分结果 |
|---|---|---|
| 什么都没变，重训 | 28 → 28 | 完全相同 |
| 补了 3 张图 | 28 → 31 | **置换整体改变，大面积换边** |
| 删了 1 张图 | 28 → 27 | 同上 |

换边之后，上一轮留在另一侧的那份副本**永远不会被删** ⇒ 同一张图同时出现在
train 和 val（训练见过、验证又考它）⇒ mAP 虚高，界面上看不出来。
删图方向同理：工程里已删的图还在训练集里继续被训练。

## ⚠️ 上一版测试为什么白写（写在这里免得再犯）

上一版想用"改动某张图的标注内容后重建，断言不重叠"。**改内容不改变文件名集合**，
`sorted(...)` 之后的列表一模一样，shuffle 结果完全一样，划分压根没变 ——
所以在没修的代码上也是绿的。以后写这类用例，**必须先证明前提成立**：
本文件里 `_assert_split_changed()` 就是这个作用，前提消失就大声失败。

## 守住的底线

1. 重建后磁盘上的文件数 == 报告里的 train/val 数（方案 B，确定性最强）；
2. 补图/删图后重建：train ∩ val = ∅，且已删图的两侧都不再有；
3. 清理失败（文件被上次训练占用）必须抛错，不许"静默继续"；
4. 输出目录传错（= 工程根 / 上级目录 / 不像数据集的非空目录）必须拒绝且不删源图。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_det_dataset_rebuild.py
"""
import json
import os
import shutil
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import detection.dataset as ds_mod  # noqa: E402
from detection.coco_io import save_coco_json  # noqa: E402

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def _png(path, w=64, h=32):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), (20, 20, 20)).save(path)


def add_image(pd, base, boxes=(("1", [4, 4, 20, 20]),), w=64, h=32):
    """往工程里加一张图 + 它的 _det.json（模拟"用户又标了一张"）。"""
    img = os.path.join(pd, "images", base + ".png")
    _png(img, w, h)
    save_coco_json(os.path.join(pd, "images", base + "_det.json"), img, w, h,
                   [{"category": c, "bbox": list(b)} for c, b in boxes],
                   ["1", "A"])


def make_project(tmp, n=10, name="proj"):
    pd = os.path.join(tmp, name)
    os.makedirs(os.path.join(pd, "images"), exist_ok=True)
    with open(os.path.join(pd, "project.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "task_type": "目标检测"}, f)
    for i in range(n):
        add_image(pd, "img%02d" % i)
    return pd


def _bases(pd, subset):
    d = os.path.join(pd, "yolo_dataset", "images", subset)
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(".png")}


def _count(pd, subset):
    d = os.path.join(pd, "yolo_dataset", "images", subset)
    return len(os.listdir(d)) if os.path.isdir(d) else 0


def _assert_counts_match(pd, rep):
    """磁盘上的文件数必须 == 报告里的数量（残留副本一旦存在就必然不等）。"""
    for subset, key in (("train", "train_images"), ("val", "val_images")):
        got, want = _count(pd, subset), rep[key]
        assert got == want, \
            "★ %s 目录 %d 个文件，报告只有 %d 个 —— 残留了上一轮的副本" % (subset, got, want)
        n_lbl = len(os.listdir(os.path.join(pd, "yolo_dataset", "labels", subset)))
        assert n_lbl == want, "★ labels/%s 有 %d 个，报告 %d 个" % (subset, n_lbl, want)


def _assert_no_overlap(pd):
    tr, va = _bases(pd, "train"), _bases(pd, "val")
    both = sorted(tr & va)
    assert not both, "★ 同一张图同时在 train 和 val 里：%r" % both[:5]


def _assert_split_changed(before, after, what):
    """前提自检：划分**真的**变了，否则"不重叠"这类断言是空转。"""
    if before == after:
        raise AssertionError(
            "★ 前提失效：%s 之后 train 集合没变化，这条用例测不到东西。"
            "（实现改了划分算法？那要重写本用例，别让它静默变绿）" % what)


# ------------------------------------------------------------------ 方案 B

def test_counts_match_report_first_build():
    """首次构建：文件数 == 报告数，train/val 不重叠。"""
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 10)
        rep = ds_mod.build_yolo_dataset(pd, val_split=0.2)
        assert rep["train_images"] == 8 and rep["val_images"] == 2, rep
        _assert_counts_match(pd, rep)
        _assert_no_overlap(pd)
        assert rep["removed_files"] == 0, "首次构建不该清理任何东西：%r" % rep
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rebuild_same_data_no_duplicates():
    """同一批数据连建两次：数量不变、不重复。"""
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 10)
        ds_mod.build_yolo_dataset(pd, val_split=0.2)
        rep2 = ds_mod.build_yolo_dataset(pd, val_split=0.2)
        _assert_counts_match(pd, rep2)
        _assert_no_overlap(pd)
        assert rep2["removed_files"] > 0, "第二次构建应当清掉上一轮产物"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 方案 A

def test_add_image_then_rebuild():
    """★ 补一张图（列表 12→13，划分整体重排）后重建：不许重叠、不许残留。

    ⚠️ 张数是**算过**的，不是随便挑的：`shuffle` 的置换并不保证"长度一变就重排"。
    实测 n=5 / 10 / 20 补一张图后 train 集合**完全不变**，那种规模下这条用例
    就是空转（我第一次用 10 张，被下面的前提自检当场抓出来）。
    """
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 12)
        ds_mod.build_yolo_dataset(pd, val_split=0.2)
        before_train = set(_bases(pd, "train"))

        add_image(pd, "imgNEW")
        rep = ds_mod.build_yolo_dataset(pd, val_split=0.2)
        assert rep["train_images"] + rep["val_images"] == 13, rep

        _assert_counts_match(pd, rep)          # 方案 B：残留必然导致数字不等
        _assert_no_overlap(pd)                 # 方案 A：任意一张都不许两边都有
        # 前提自检：长度变了，划分应该真的重排过；万一没变，说明这条用例空转
        _assert_split_changed(before_train, set(_bases(pd, "train")), "补一张图")
        assert "imgNEW" in (_bases(pd, "train") | _bases(pd, "val"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_delete_image_then_rebuild_drops_stale():
    """★ 删一张图后重建：它的旧副本不许留在训练集里继续被训练。"""
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 10)
        ds_mod.build_yolo_dataset(pd, val_split=0.2)
        tr, va = _bases(pd, "train"), _bases(pd, "val")
        gone = sorted(tr)[0]
        assert gone in (tr | va), "前提：删的这张本来就在数据集里"

        for p in (os.path.join(pd, "images", gone + ".png"),
                  os.path.join(pd, "images", gone + "_det.json")):
            os.remove(p)

        rep = ds_mod.build_yolo_dataset(pd, val_split=0.2)
        assert rep["train_images"] + rep["val_images"] == 9, rep
        _assert_counts_match(pd, rep)
        _assert_no_overlap(pd)
        assert gone not in (_bases(pd, "train") | _bases(pd, "val")), \
            "★ 已从工程删掉的 %s 还留在数据集里" % gone
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 失败必须报

def test_wipe_failure_raises_not_silent():
    """清理失败（文件被占用）必须抛，不许 ignore_errors 之后假装重建成功。

    ⚠️ 不能靠 patch `shutil.rmtree` 来造失败：那样连 `ignore_errors=True` 也被一起
    替身掉了，变异"把 ignore_errors 加上"照样绿（第一版就这么漏的）。
    这里改用**真的把文件占住** —— Windows 上 open 着的文件不允许删除，
    这正是现场"上次训练还没退出"的形态。
    """
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 6)
        ds_mod.build_yolo_dataset(pd, val_split=0.34)      # 先造一份旧的
        victim = None
        for root, _dirs, files in os.walk(os.path.join(pd, "yolo_dataset")):
            if files:
                victim = os.path.join(root, sorted(files)[0])
                break
        assert victim and os.path.exists(victim), "前提：旧数据集里得有文件"

        raised = None
        fh = open(victim, "rb")                            # 占住，不关
        try:
            try:
                ds_mod.build_yolo_dataset(pd, val_split=0.34)
            except RuntimeError as e:
                raised = e
            except Exception as e:                          # noqa: BLE001
                raise AssertionError("★ 抛的不是 RuntimeError：%r" % (e,))
        finally:
            fh.close()

        assert raised is not None, "★ 清理失败被吞了（ignore_errors？）：调用方以为重建成功了"
        assert "清理" in str(raised), "错误信息没说清是清理失败：%s" % raised
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _expect_value_error(fn, must_contain):
    """断言抛 ValueError 且信息里点明了是哪条护栏 —— 别让别的护栏替它挡枪。"""
    try:
        fn()
    except ValueError as e:
        assert must_contain in str(e), \
            "抛的 ValueError 不是这条护栏：期望含 %r，实际 %s" % (must_contain, e)
        return
    except Exception as e:                                  # noqa: BLE001
        raise AssertionError("★ 抛的不是 ValueError：%r" % (e,))
    raise AssertionError("★ 没拦住")


OUTSIDE = "必须在工程目录内"      # 不变式：数据集目录必须在工程里面
NOT_DATASET = "不像构建出来的数据集"


def test_refuse_project_root():
    """output_dir 传成工程根：必须拒绝，且源图一张不少。"""
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 4)
        n_before = len(os.listdir(os.path.join(pd, "images")))
        _expect_value_error(
            lambda: ds_mod.build_yolo_dataset(pd, output_dir=pd, val_split=0.5),
            OUTSIDE)
        n_after = len(os.listdir(os.path.join(pd, "images")))
        assert n_before == n_after == 8, "源图被动了：%d → %d" % (n_before, n_after)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refuse_parent_dir():
    """output_dir 传成工程的**上级**目录：必须拒绝（能一口气删掉整个工程）。"""
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 4)
        n_before = len(os.listdir(os.path.join(pd, "images")))
        _expect_value_error(
            lambda: ds_mod.build_yolo_dataset(pd, output_dir=tmp, val_split=0.5),
            OUTSIDE)
        assert os.path.exists(os.path.join(pd, "project.json")), "★ 工程被删了"
        assert len(os.listdir(os.path.join(pd, "images"))) == n_before, "源图被删了"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refuse_other_project():
    """★ 用户复核指出的缺口：output_dir 误传成**另一个工程根**。

    另一个工程也有 `images/`，按"三个标记任一存在"的旧判据会被当成数据集整个删掉。
    不变式（必须在本工程内）一条就挡住了，而且**不许动人家一张图**。
    """
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 4, name="proj")            # 本工程
        other = make_project(tmp, 3, name="neighbour")    # 隔壁工程
        other_images = os.path.join(other, "images")
        n_before = len(os.listdir(other_images))
        assert n_before == 6, "前提：隔壁工程要有图片 + _det.json"

        _expect_value_error(
            lambda: ds_mod.build_yolo_dataset(pd, output_dir=other, val_split=0.5),
            OUTSIDE)
        assert os.path.isdir(other_images), "★ 隔壁工程的 images/ 被删了"
        assert len(os.listdir(other_images)) == n_before, "★ 隔壁工程的图被删了"
        assert os.path.exists(os.path.join(other, "project.json")), "★ 隔壁工程被删了"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refuse_own_images_dir():
    """工程**内**的误传：output_dir = proj/images。有图但不像数据集 → 不许删。"""
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 4)
        imgs = os.path.join(pd, "images")
        n_before = len(os.listdir(imgs))
        _expect_value_error(
            lambda: ds_mod.build_yolo_dataset(pd, output_dir=imgs, val_split=0.5),
            NOT_DATASET)
        assert len(os.listdir(imgs)) == n_before == 8, "★ 工程的源图被删了"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refuse_dir_with_only_images_subdir():
    """★ 钉住判据的严格度：有 `images/` 子目录、但没有 `labels/` 和 `data.yaml` 的不算数据集。

    这条防的是把判据退回 `any(...)` —— 那样**任何带 images/ 的目录**都会被当成
    数据集删掉（另一个工程根正是这个形状）。
    """
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 4)
        fake = os.path.join(pd, "exports")               # 工程内、不在 proj.parents 之外
        os.makedirs(os.path.join(fake, "images"))
        keep = os.path.join(fake, "images", "a.png")
        _png(keep)
        _expect_value_error(
            lambda: ds_mod.build_yolo_dataset(pd, output_dir=fake, val_split=0.5),
            NOT_DATASET)
        assert os.path.exists(keep), "★ 只有 images/ 的目录被当成数据集删了"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refuse_non_dataset_dir():
    """工程内一个随手建的目录（非空、无标记）：必须拒绝，且不删人家东西。"""
    tmp = tempfile.mkdtemp()
    try:
        pd = make_project(tmp, 4)
        other = os.path.join(pd, "my_precious")
        os.makedirs(other)
        keep = os.path.join(other, "报告.txt")
        with open(keep, "w", encoding="utf-8") as f:
            f.write("别删我")
        _expect_value_error(
            lambda: ds_mod.build_yolo_dataset(pd, output_dir=other, val_split=0.5),
            NOT_DATASET)
        assert os.path.exists(keep), "★ 用户目录里的文件被删了"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    checks = [
        ("★ 首次构建：文件数 == 报告数", test_counts_match_report_first_build),
        ("同数据重建两次：不重复", test_rebuild_same_data_no_duplicates),
        ("★ 补图后重建：不重叠、不残留（含前提自检）", test_add_image_then_rebuild),
        ("★ 删图后重建：旧副本被清掉", test_delete_image_then_rebuild_drops_stale),
        ("★ 清理失败必须抛错，不许静默", test_wipe_failure_raises_not_silent),
        ("护栏：输出目录 = 工程根 → 拒绝", test_refuse_project_root),
        ("护栏：输出目录 = 工程上级目录 → 拒绝", test_refuse_parent_dir),
        ("★ 护栏：输出目录 = 另一个工程 → 拒绝（不动人家源图）", test_refuse_other_project),
        ("★ 护栏：输出目录 = 工程自己的 images/ → 拒绝", test_refuse_own_images_dir),
        ("★ 护栏：只有 images/ 子目录的目录 → 不算数据集", test_refuse_dir_with_only_images_subdir),
        ("护栏：工程内无关的非空目录 → 拒绝", test_refuse_non_dataset_dir),
    ]
    for name, fn in checks:
        check(name, fn)

    print()
    if failures:
        print("❌ 有失败：%d / %d" % (len(failures), len(checks)))
        for name, exc in failures:
            print("   - %s: %s" % (name, exc))
        return 1
    print("✅ 全部通过  %d/%d 项通过" % (len(checks), len(checks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
