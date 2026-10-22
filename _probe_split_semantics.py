# -*- coding: utf-8 -*-
"""「未划分 = 不属于任何集」语义自测。

背景（2026-10-22，用户要求）：以前 `train_test_split.json` 里没登记的图
会被**偷偷补成 train**：
  · `training/dataset.get_train_test_split()` 在内存里给所有已标注样本补 "train"
  · `training/multiview.py` 用 `split_map.get(key, "train")`
  · 分类侧三处 `split_map.get(..., "train")`
结果是"没划分"= "进训练集"，而界面显示成空集 —— 同一张图两个说法。

现在统一改成：**查不到就取不到**（训练/验证/测试都拿不到），训练集为空直接报错。

这个探针钉住四件容易改回去的事：
  ① 默认值不许回潮（三个取数入口都要按"未划分不算 train"过滤）
  ② 训练集为空要明确报错，不能静默训 0 张
  ③ `annotation/convert.add_to_train()` 必须**显式写盘**（它以前靠 ① 的默认值才写得进去）
  ④ 训练集只包含**显式划到 train** 的组

用法：
    "C:/aitp_venv/Scripts/python.exe" _probe_split_semantics.py
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


def _img(path, shape=(16, 16)):
    Image.fromarray(np.zeros(shape, np.uint8)).save(path)


def _mask(path, value=1, shape=(16, 16)):
    Image.fromarray(np.full(shape, value, np.uint8)).save(path)


def _json_labels(path, labels=("A",)):
    json.dump({"version": "5.0.1", "imagePath": "", "imageHeight": 16, "imageWidth": 16,
               "shapes": [{"label": l, "points": [[2, 2], [8, 8]], "shape_type": "rectangle"}
                          for l in labels]},
              open(path, "w", encoding="utf-8"))


# ───────────────────────── 造工程 ─────────────────────────

def make_multiview(root):
    """4 组三视图：t1/t2 划到 train，v1 划到 val，u1/u2 **未划分**。"""
    pd = os.path.join(root, "mv")
    os.makedirs(os.path.join(pd, "images"))
    os.makedirs(os.path.join(pd, "annotations"))
    json.dump({"classes": ["background", "A"], "views": ["te_0", "te_1", "te_2"]},
              open(os.path.join(pd, "project.json"), "w", encoding="utf-8"), indent=2)
    groups = ["t1", "t2", "v1", "u1", "u2"]
    for g in groups:
        for v in ("te_0", "te_1", "te_2"):
            _img(os.path.join(pd, "images", "%s_%s.png" % (g, v)))
        _mask(os.path.join(pd, "annotations", "%s_te_0_mask.png" % g), 1)
    json.dump({"t1_te_0": "train", "t2_te_0": "train", "v1_te_0": "val"},
              open(os.path.join(pd, "train_test_split.json"), "w", encoding="utf-8"), indent=2)
    return pd, groups


def make_single(root):
    """3 张单图：s1/s2 → train，s3 **未划分**。"""
    pd = os.path.join(root, "sv")
    os.makedirs(os.path.join(pd, "images"))
    os.makedirs(os.path.join(pd, "annotations"))
    json.dump({"classes": ["background", "A"]},
              open(os.path.join(pd, "project.json"), "w", encoding="utf-8"), indent=2)
    for s in ("s1", "s2", "s3"):
        _img(os.path.join(pd, "images", s + ".png"))
        _mask(os.path.join(pd, "annotations", s + "_mask.png"), 1)
    json.dump({"s1": "train", "s2": "train"},
              open(os.path.join(pd, "train_test_split.json"), "w", encoding="utf-8"), indent=2)
    return pd


def make_empty_split(root):
    """有图有标注，但划分表一条都没有（升级后最容易撞上的状态）。"""
    pd = os.path.join(root, "empty")
    os.makedirs(os.path.join(pd, "images"))
    os.makedirs(os.path.join(pd, "annotations"))
    json.dump({"classes": ["background", "A"]},
              open(os.path.join(pd, "project.json"), "w", encoding="utf-8"), indent=2)
    for s in ("e1", "e2"):
        _img(os.path.join(pd, "images", s + ".png"))
        _mask(os.path.join(pd, "annotations", s + "_mask.png"), 1)
    return pd


# ───────────────────────── 用例 ─────────────────────────

def test_get_split(pd_sv):
    from training.dataset import get_train_test_split
    print("\n[1] get_train_test_split 不再补默认 train")
    sm, samples = get_train_test_split(pd_sv)
    ck("返回值里只有盘上登记的键（s1/s2）", set(sm) == {"s1", "s2"}, str(sm))
    ck("未划分的 s3 不在返回表里", "s3" not in sm, str(sm))
    ck("samples 仍然列出全部已标注样本（3 张，供上层显示用）",
       sorted(samples) == ["s1", "s2", "s3"], str(samples))


def test_segmentation_dataset(pd_sv, pd_mv):
    from training.dataset import SegmentationDataset
    print("\n[2] 单视图取数：未划分的图进不了任何集")
    tr = SegmentationDataset(pd_sv, split="train", image_size=(16, 16))
    bases = [os.path.splitext(os.path.basename(s[0]))[0] for s in tr._image_samples]
    ck("train 只有 s1/s2（s3 未划分被排除）", sorted(bases) == ["s1", "s2"], str(bases))
    va = SegmentationDataset(pd_sv, split="val", image_size=(16, 16))
    ck("val 为空（没人划到 val）", len(va) == 0 or len(va._image_samples) == 0,
       str(len(va._image_samples)))
    al = SegmentationDataset(pd_sv, split="all", image_size=(16, 16))
    ck("split=all 仍能看到全部 3 张（查看全部这条路径不受影响）",
       len(al._image_samples) == 3, str(len(al._image_samples)))

    print("\n[3] 多视图取数：按组过滤，未划分的组整组排除")
    mv = SegmentationDataset(pd_mv, split="train", image_size=(16, 16),
                             views=("te_0", "te_1", "te_2"))
    keys = sorted(os.path.splitext(os.path.basename(s[0]))[0] for s in mv._image_samples)
    ck("train 只拿到 t1/t2_te_0", keys == ["t1_te_0", "t2_te_0"], str(keys))
    mvv = SegmentationDataset(pd_mv, split="val", image_size=(16, 16),
                              views=("te_0", "te_1", "te_2"))
    ck("val 只拿到 v1_te_0（不是所有未划分的都算 val）",
       len(mvv._image_samples) == 1, str(len(mvv._image_samples)))


def test_multiview_trainer(pd_mv):
    print("\n[4] 三视图训练器（training/multiview.py）同样不默认 train")
    from training.multiview import MultiViewSegDataset
    ds = MultiViewSegDataset(pd_mv, split="train", image_size=64, views=("te_0", "te_1", "te_2"))
    keys = sorted(s[0] for s in ds.samples)
    ck("train 只含 t1/t2（u1/u2 未划分 → 不取）", keys == ["t1", "t2"], str(keys))
    ds_val = MultiViewSegDataset(pd_mv, split="val", image_size=64, views=("te_0", "te_1", "te_2"))
    ck("val 只含 v1", sorted(s[0] for s in ds_val.samples) == ["v1"],
       str([s[0] for s in ds_val.samples]))


def test_empty_train(pd_empty):
    print("\n[5] 训练集为空必须当场报错（不许静默训 0 张）")
    from training.dataset import SegmentationDataset
    try:
        SegmentationDataset(pd_empty, split="train", image_size=(16, 16))
        ck("划分表为空时构造 train 数据集报错", False, "没报错")
    except RuntimeError as e:
        msg = str(e)
        ck("划分表为空时构造 train 数据集报错", True)
        ck("报错里指路「数据集划分管理」", "数据集划分管理" in msg, msg[:80])
        ck("报错里说清「标注过 ≠ 进训练集」", "标注过" in msg or "未划分" in msg, msg[:120])
    except Exception as e:
        ck("划分表为空时构造 train 数据集报错", False, "%s: %s" % (type(e).__name__, e))


def test_add_to_train(pd_sv):
    print("\n[6] add_to_train 必须显式写盘（以前靠默认值才写得进去）")
    from annotation.convert import add_to_train
    sp = os.path.join(pd_sv, "train_test_split.json")
    ck("前置：s3 现在没在划分表里", "s3" not in json.load(open(sp, encoding="utf-8")))
    n = add_to_train(pd_sv, ["s3"])
    ck("返回值 n=1（真的算作改动）", n == 1, str(n))
    disk = json.load(open(sp, encoding="utf-8"))
    ck("★ 盘上真的多了 s3=train（去掉默认值后最容易退化的地方）",
       disk.get("s3") == "train", str(disk))
    ck("已有的 val 没被顺手改成 train", disk.get("s1") == "train", str(disk))

    # 已经在 val 的不许偷挪
    json.dump({"s1": "train", "s2": "val", "s3": "train"},
              open(sp, "w", encoding="utf-8"), indent=2)
    add_to_train(pd_sv, ["s2"])
    ck("已划到 val 的图不会被 add_to_train 挪进 train",
       json.load(open(sp, encoding="utf-8")).get("s2") == "val")


def test_classification(root):
    print("\n[7] 分类侧同一口径")
    pd = os.path.join(root, "cls")
    os.makedirs(os.path.join(pd, "images", "ok"))
    os.makedirs(os.path.join(pd, "images", "ng"))
    os.makedirs(os.path.join(pd, "annotations"))
    for stem, sub in (("a", "ok"), ("b", "ok"), ("c", "ng")):
        _img(os.path.join(pd, "images", sub, stem + ".png"))
    json.dump({"labels": ["ok", "ng"],
               "mapping": {"images/ok/a": 0, "images/ok/b": 0, "images/ng/c": 1}},
              open(os.path.join(pd, "annotations", "class_labels.json"), "w", encoding="utf-8"),
              indent=2)
    # 键是 images/ 下的相对路径（ImageFolder 布局 = "类别/文件名"），不是裸文件名
    json.dump({"ok/a": "train", "ok/b": "train"},
              open(os.path.join(pd, "train_test_split.json"), "w", encoding="utf-8"), indent=2)
    from classification.dataset import _default_split_map, _list_images_mapping
    sm = _default_split_map(pd)
    tr, _names = _list_images_mapping(pd, "train", sm)
    ck("分类 train 只有 a/b（c 未划分被排除）",
       sorted(os.path.splitext(os.path.basename(p[1]))[0] for p in tr) == ["a", "b"],
       str([p[0] for p in tr]))
    al, _n2 = _list_images_mapping(pd, "all", sm)
    ck("分类 split=all 仍能看到 3 张", len(al) == 3, str(len(al)))

    from classification.dataset import ClassificationDataset
    try:
        # 把 a/b 的划分也拿掉 -> 训练集空
        json.dump({}, open(os.path.join(pd, "train_test_split.json"), "w", encoding="utf-8"))
        ClassificationDataset(pd, split="train")
        ck("分类训练集为空时报错", False, "没报错")
    except RuntimeError as e:
        ck("分类训练集为空时报错", True)
        ck("分类报错同样指路数据集划分", "数据集划分管理" in str(e), str(e)[:80])


def test_right_click(pd_sv):
    """右键「Set as Val」走的是 get → 改一条 → save。去掉默认值后它不该顺手写一堆 train。"""
    from training.dataset import get_train_test_split, save_train_test_split
    print("\n[8] 右键设划分不会顺手把没划分的图写成 train")
    sp = os.path.join(pd_sv, "train_test_split.json")
    json.dump({"s1": "train", "s2": "train"},
              open(sp, "w", encoding="utf-8"), indent=2)
    sm, _ = get_train_test_split(pd_sv)
    sm["s3"] = "val"                      # 等价于右键把 s3 设为验证
    save_train_test_split(pd_sv, sm)
    disk = json.load(open(sp, encoding="utf-8"))
    ck("盘上只有 s1/s2/s3 三条（没有凭空多出来的 train）",
       set(disk) == {"s1", "s2", "s3"}, str(sorted(disk)))
    ck("s3 = val", disk.get("s3") == "val", str(disk))


def main():
    root = tempfile.mkdtemp(prefix="aitp_semantics_")
    try:
        pd_mv, _g = make_multiview(root)
        pd_sv = make_single(root)
        pd_empty = make_empty_split(root)
        test_get_split(pd_sv)
        test_segmentation_dataset(pd_sv, pd_mv)
        test_multiview_trainer(pd_mv)
        test_empty_train(pd_empty)
        test_add_to_train(pd_sv)
        test_right_click(pd_sv)
        test_classification(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\n" + "=" * 70)
    print("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (OK_N[0], len(BAD), OK_N[0] + len(BAD)))
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
