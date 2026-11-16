# -*- coding: utf-8 -*-
"""「导入图片 + 导入标注 → 直接训练」这条路的回归测试（目标检测）。

背景（真实工程踩到的）：
    平台的目标检测训练只认 `images/<base>_det.json`，而 iSenseFlow / labelme
    导出的标注是 `images/<base>.json`（shapes 里是 rectangle）。界面「导入标注」
    原来只生成**分割掩膜** `_mask.png`，一个 `_det.json` 都不产出 —— 于是用户
    "导入图片 + 导入标注"做完，训练仍然报 `No detection annotations found`。

这里守住的底线：
    1. labelme 矩形 → bbox 的像素值**必须等于**源 JSON 的点（不是大概对）；
    2. 一批文件的 `category_id` 由同一份排序类别表决定（与训练侧 sorted 一致）；
    3. 脏数据（坏 JSON / 没矩形 / 没原图）不抛异常，但要能报出原因；
    4. 端到端：转换后 `get_annotated_count()` 必须从 0 变正数，
       `build_yolo_dataset()` 能建出数据集，且 class_names 与类别表一致。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_det_import.py
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

from detection.labelme_rect import (  # noqa: E402
    boxes_from_labelme, categories_of, collect, convert_items, convert_one,
    find_image, rect_to_bbox,
)
from detection.coco_io import load_coco_json  # noqa: E402
from detection.dataset import build_yolo_dataset, get_annotated_count  # noqa: E402
from core.project_manager import read_task_type  # noqa: E402

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, ensure_ascii=False)


def _png(path, w=400, h=80):
    """真写一张 PNG —— 缺 imageWidth/Height 时靠 PIL 兜底，不能拿假文件糊弄。"""
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), (30, 30, 30)).save(path)


def _labelme(pairs, img_name="a.png", w=400, h=80, with_size=True):
    """pairs: [(label, [x0,y0,x1,y1]), ...] → labelme dict（两点矩形）"""
    d = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [{"label": lb, "shape_type": "rectangle",
                    "points": [[p[0], p[1]], [p[2], p[3]]],
                    "description": "", "group_id": None}
                   for lb, p in pairs],
        "imagePath": img_name,
        "imageData": None,
    }
    if with_size:
        d["imageWidth"] = w
        d["imageHeight"] = h
    return d


def _mkproject(tmp, name="detproj", task="目标检测", n=2, labels=(("1", [10, 10, 30, 50]),
                                                                  ("A", [60, 8, 92, 46]))):
    """造一个"已导入图片+标注"的检测工程：images/<base>.{png,json}"""
    pd = os.path.join(tmp, name)
    os.makedirs(os.path.join(pd, "images"), exist_ok=True)
    _w(os.path.join(pd, "project.json"),
       {"name": name, "task_type": task, "classes": ["background"], "parent_dir": tmp})
    bases = []
    for i in range(n):
        base = "img%d" % (i + 1)
        _png(os.path.join(pd, "images", base + ".png"), 400, 80)
        _w(os.path.join(pd, "images", base + ".json"),
           _labelme(labels, img_name=base + ".png"))
        bases.append(base)
    return pd, bases


# ------------------------------------------------------------------ 基础转换

def test_rect_to_bbox():
    # 两个角点顺序不固定，必须都给出 [x,y,w,h]
    assert rect_to_bbox([[10, 20], [40, 60]]) == [10, 20, 30, 40]
    assert rect_to_bbox([[40, 60], [10, 20]]) == [10, 20, 30, 40]
    assert rect_to_bbox([[5.5, 5.5], [5.5, 5.5]]) == [5.5, 5.5, 0.0, 0.0]


def test_boxes_from_labelme_filters():
    d = {"shapes": [
        {"label": "1", "shape_type": "rectangle", "points": [[1, 2], [11, 22]]},
        {"label": "2", "shape_type": "polygon", "points": [[0, 0], [5, 0], [5, 5]]},
        {"label": "  ", "shape_type": "rectangle", "points": [[0, 0], [9, 9]]},
        {"label": "A", "shape_type": "rectangle", "points": [[3, 3], [7, 7]]},
    ]}
    boxes, other = boxes_from_labelme(d)
    assert [c for c, _ in boxes] == ["1", "A"], boxes
    assert other == {"polygon": 1}, other


def test_convert_one_writes_exact_bbox():
    tmp = tempfile.mkdtemp()
    try:
        _png(os.path.join(tmp, "a.png"), 400, 80)
        jp = os.path.join(tmp, "a.json")
        _w(jp, _labelme([("1", [18.5, 13.5, 34.5, 56.5]),
                         ("A", [224.5, 34.5, 300.0, 70.0])]))
        rep = convert_one(jp, os.path.join(tmp, "a.png"))
        assert rep["written"] is True, rep
        assert rep["boxes"] == 2, rep
        boxes, info = load_coco_json(os.path.join(tmp, "a_det.json"))
        got = [(b["category"], b["bbox"]) for b in boxes]
        assert got == [("1", [18.5, 13.5, 16.0, 43.0]),
                       ("A", [224.5, 34.5, 75.5, 35.5])], got
        assert info["imageWidth"] == 400 and info["imageHeight"] == 80, info
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_convert_one_category_id_follows_batch_table():
    """★ 类别顺序**故意与排序不同**（H 在前）。

    如果实现偷懒"自己按名字再排一遍"，结果刚好也是 ["1","H"] —— 那样这条用例
    永远绿、什么也没守住（变异测试实拍：把 `if categories is None` 改成
    `if True` 时它照样 PASS）。所以这里必须让调用方顺序 ≠ 字典序。
    """
    tmp = tempfile.mkdtemp()
    try:
        _png(os.path.join(tmp, "a.png"), 400, 80)
        jp = os.path.join(tmp, "a.json")
        _w(jp, _labelme([("H", [1, 1, 5, 5]), ("1", [6, 6, 9, 9])]))
        cats = ["H", "1"]          # 调用方给的顺序就是 category_id，不能自己重排
        convert_one(jp, os.path.join(tmp, "a.png"), categories=cats)
        boxes, _info = load_coco_json(os.path.join(tmp, "a_det.json"))
        by_name = {b["category"]: b["category_id"] for b in boxes}
        assert by_name == {"H": 0, "1": 1}, by_name
        with open(os.path.join(tmp, "a_det.json"), "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert [c["name"] for c in saved["categories"]] == cats, saved["categories"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_convert_one_size_fallback_to_pil():
    tmp = tempfile.mkdtemp()
    try:
        _png(os.path.join(tmp, "a.png"), 137, 39)          # 非整十的尺寸
        jp = os.path.join(tmp, "a.json")
        _w(jp, _labelme([("1", [1, 1, 9, 9])], with_size=False))
        rep = convert_one(jp, os.path.join(tmp, "a.png"))
        assert rep["written"] is True, rep
        assert rep["size"] == (137, 39), rep["size"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_convert_one_empty_and_bad_json():
    tmp = tempfile.mkdtemp()
    try:
        _png(os.path.join(tmp, "a.png"), 100, 50)
        # ① 只有多边形 → 不写文件，但要说清原因
        poly = {"shapes": [{"label": "1", "shape_type": "polygon",
                            "points": [[0, 0], [5, 0], [5, 5]]}],
                "imagePath": "a.png", "imageWidth": 100, "imageHeight": 50}
        _w(os.path.join(tmp, "a.json"), poly)
        rep = convert_one(os.path.join(tmp, "a.json"), os.path.join(tmp, "a.png"))
        assert rep["written"] is False and "矩形" in rep["skipped"], rep
        assert not os.path.exists(os.path.join(tmp, "a_det.json"))
        # ② 坏 JSON → 不抛异常，报类型
        _w(os.path.join(tmp, "b.json"), "{not json")
        _png(os.path.join(tmp, "b.png"), 100, 50)
        rep2 = convert_one(os.path.join(tmp, "b.json"), os.path.join(tmp, "b.png"))
        assert rep2["written"] is False and rep2["skipped"], rep2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_categories_of_sorted_and_merged():
    tmp = tempfile.mkdtemp()
    try:
        _w(os.path.join(tmp, "a.json"), _labelme([("H", [0, 0, 1, 1]), ("1", [0, 0, 1, 1])]))
        _w(os.path.join(tmp, "b.json"), _labelme([("A", [0, 0, 1, 1]), ("1", [0, 0, 1, 1])]))
        cats = categories_of([os.path.join(tmp, "a.json"), os.path.join(tmp, "b.json")])
        assert cats == ["1", "A", "H"], cats       # 去重 + 按名字排序
        # 坏文件不能带崩整批
        _w(os.path.join(tmp, "c.json"), "x")
        again = categories_of([os.path.join(tmp, "c.json")] + [os.path.join(tmp, "a.json")])
        assert again == ["1", "H"], again
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collect_reports_missing_image_and_empty():
    tmp = tempfile.mkdtemp()
    try:
        labels = os.path.join(tmp, "labels")
        images = os.path.join(tmp, "images")
        os.makedirs(images, exist_ok=True)
        _png(os.path.join(images, "ok.png"), 100, 50)
        _w(os.path.join(labels, "ok.json"), _labelme([("1", [1, 1, 9, 9])], img_name="ok.png"))
        _w(os.path.join(labels, "noimg.json"), _labelme([("1", [1, 1, 9, 9])],
                                                        img_name="ghost.png"))
        _w(os.path.join(labels, "empty.json"),
           {"shapes": [], "imagePath": "ok.png", "imageWidth": 100, "imageHeight": 50})
        items, rep = collect(labels, images)
        assert rep["items"] == 1, rep
        assert len(rep["skipped"]["no_image"]) == 1, rep["skipped"]
        assert len(rep["skipped"]["empty"]) == 1, rep["skipped"]
        n = convert_items(items, categories_of([i[0] for i in items]))
        assert n == 1, n
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_find_image_any_ext():
    tmp = tempfile.mkdtemp()
    try:
        _w(os.path.join(tmp, "a.bmp"), "not really a bmp")
        assert find_image(tmp, "a").endswith("a.bmp")
        assert find_image(tmp, "nope") is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 端到端

def test_end_to_end_import_then_train_ready():
    """导入后：annotated_count 0→2，build_yolo_dataset 能建，类别表与训练侧一致。"""
    tmp = tempfile.mkdtemp()
    try:
        pd, bases = _mkproject(tmp, n=3)
        assert get_annotated_count(pd) == 0, "造工程时不该有 _det.json"
        cats = categories_of([os.path.join(pd, "images", b + ".json") for b in bases])
        for b in bases:
            rep = convert_one(os.path.join(pd, "images", b + ".json"),
                              os.path.join(pd, "images", b + ".png"),
                              categories=cats)
            assert rep["written"], rep
        assert get_annotated_count(pd) == 3, get_annotated_count(pd)

        ds = build_yolo_dataset(pd, val_split=0.34, seed=1)
        assert ds["num_classes"] == len(cats), ds
        assert ds["class_names"] == cats, (ds["class_names"], cats)
        assert ds["train_images"] + ds["val_images"] == 3, ds

        # 标签文件里的框必须能反推回源坐标（误差只来自 6 位小数归一化）
        label_txt = os.path.join(pd, "yolo_dataset", "labels", "train", "img1.txt")
        if not os.path.exists(label_txt):     # 划分可能把 img1 放 val
            label_txt = os.path.join(pd, "yolo_dataset", "labels", "val", "img1.txt")
        lines = [l.split() for l in open(label_txt, encoding="utf-8").read().split("\n") if l]
        assert len(lines) == 2, lines
        for parts, (name, (x0, y0, x1, y1)) in zip(lines, [("1", (10, 10, 30, 50)),
                                                           ("A", (60, 8, 92, 46))]):
            cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            assert cats[cid] == name, (cats, cid, name)
            back = [(cx - w / 2) * 400, (cy - h / 2) * 80, w * 400, h * 80]
            want = [x0, y0, x1 - x0, y1 - y0]
            for a, b in zip(back, want):
                assert abs(a - b) < 0.05, (back, want)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_read_task_type_both_spellings():
    tmp = tempfile.mkdtemp()
    try:
        pd = os.path.join(tmp, "p")
        os.makedirs(pd, exist_ok=True)
        _w(os.path.join(pd, "project.json"), {"task_type": "目标检测"})
        assert read_task_type(pd) == "detection"
        _w(os.path.join(pd, "project.json"), {"task_type": "object_detection"})
        assert read_task_type(pd) == "detection"
        _w(os.path.join(pd, "project.json"), {"task_type": "语义分割"})
        assert read_task_type(pd) == "segmentation"
        # 缺文件 / 坏文件 / 未知类型 → 兜底，绝不返回 None
        assert read_task_type(os.path.join(tmp, "nope")) == "segmentation"
        _w(os.path.join(pd, "project.json"), "{bad")
        assert read_task_type(pd) == "segmentation"
        _w(os.path.join(pd, "project.json"), {"task_type": "火星任务"})
        assert read_task_type(pd, default="segmentation") == "segmentation"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_detection_package_init_does_not_need_qt():
    """`import detection.coco_io` 不该把 PyQt5 拽进来（工具/测试/无显示环境）。"""
    import subprocess
    code = ("import sys; import detection.coco_io; "
            "print('PyQt5' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.strip() == "False", r.stdout


def main():
    checks = [
        ("矩形两点 → bbox（含逆序）", test_rect_to_bbox),
        ("只收 rectangle，多边形/空标签跳过", test_boxes_from_labelme_filters),
        ("转换写出像素级一致的 bbox", test_convert_one_writes_exact_bbox),
        ("category_id 跟批次类别表走", test_convert_one_category_id_follows_batch_table),
        ("缺 imageWidth 时用 PIL 兜底", test_convert_one_size_fallback_to_pil),
        ("没矩形 / 坏 JSON：不写、不抛、有原因", test_convert_one_empty_and_bad_json),
        ("批次类别表：去重 + 排序 + 抗坏文件", test_categories_of_sorted_and_merged),
        ("目录扫描：缺图 / 空标注进报告", test_collect_reports_missing_image_and_empty),
        ("按 base 找图（扩展名不限）", test_find_image_any_ext),
        ("★ 端到端：导完标注就能建数据集训练", test_end_to_end_import_then_train_ready),
        ("★ 工程类型判定：中英两种写法 + 兜底", test_read_task_type_both_spellings),
        ("★ detection 包初始化不依赖 Qt", test_detection_package_init_does_not_need_qt),
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
