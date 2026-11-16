# -*- coding: utf-8 -*-
"""图片列表「标注」列 = **标注块数**，同一批标注只能算一次。

## 用户报的现象（2026-11-16）

"这张图我就 10 个字符，为啥列表显示标了 20 个？"

## 根因

检测工程的标注有**两份文件**描述同一批框：

```
images/<base>.json       ← labelme 原始标注（导入时落在 images/ 里）
images/<base>_det.json   ← 平台检测标注（由上面那份转出来）
```

`_filter_images` 的统计原来两份都累加：

```python
if _det_exists:  shape_count += len(det_boxes)        # 10
if _json_exists: shape_count += len(json_labels)      # 再 +10
```

于是 10 → 20；底栏 `Blocks` 也是 2 倍（实测 300 个框报 600）；
「标注块数」筛选表达式（core/filter_expr.py）跟着一起错。

现在按 **`_det.json` > labelme JSON > 掩膜** 只取一个来源。

运行：C:\\aitp_venv\\Scripts\\python.exe tests\\test_ann_count_source.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:                                  # 与 main.py 一致：torch 先于 PyQt5
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

from detection.coco_io import save_coco_json  # noqa: E402

failures = []
_APP = None
_WIN = None
_TMP = None
_MADE = []

ANN_COL = 4          # 「标注」列的下标（序号 0 / 图片名 1 / 大小 2 / 数据集 3 / 标注 4）


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def window():
    global _APP, _WIN, _TMP
    if _WIN is not None:
        return _WIN
    from PyQt5.QtWidgets import QApplication
    import ui.task_queue as tq

    fd, p = tempfile.mkstemp(prefix="aitp_anncnt_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    _APP = QApplication.instance() or QApplication([])
    init_ui_invoker()
    _TMP = tempfile.mkdtemp(prefix="aitp_anncnt_proj_")
    _WIN = MainWindow()
    return _WIN


def wait_until(pred, timeout=40.0, desc="条件"):
    end = time.time() + timeout
    while time.time() < end:
        _APP.processEvents()
        time.sleep(0.02)
        if pred():
            return
    raise AssertionError("等待超时：%s" % desc)


def _png(path, w=64, h=32):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), (20, 20, 20)).save(path)


def _labelme(n, img_name, w=64, h=32):
    return {"version": "5.0.1", "flags": {},
            "shapes": [{"label": "0", "shape_type": "rectangle",
                        "points": [[2 + i * 4, 4], [6 + i * 4, 20]]} for i in range(n)],
            "imagePath": img_name, "imageData": None,
            "imageWidth": w, "imageHeight": h}


def make_project(task="目标检测", n_images=2, labelme_n=3, det_n=3, det_empty=False):
    """建工程：每张图可分别带 labelme JSON 和 _det.json（数量可不同，用来测优先级）。

    det_empty=True：写一个**合法但 0 个框**的 `_det.json`（良品图的形态）。
    """
    w = window()
    name = "_anncnt_%d" % (len(_MADE) + 1)
    w.pm.create_project(name, task_type=task, parent_dir=_TMP)
    _MADE.append(name)
    pd = str(w.pm.get_project_dir(name))
    for i in range(n_images):
        base = "img%d" % (i + 1)
        img = os.path.join(pd, "images", base + ".png")
        _png(img)
        if labelme_n:
            with open(os.path.join(pd, "images", base + ".json"), "w", encoding="utf-8") as f:
                json.dump(_labelme(labelme_n, base + ".png"), f)
        if det_empty:
            save_coco_json(os.path.join(pd, "images", base + "_det.json"), img, 64, 32, [], ["0"])
        elif det_n:
            save_coco_json(
                os.path.join(pd, "images", base + "_det.json"), img, 64, 32,
                [{"category": "0", "bbox": [2 + j * 4, 4, 4, 16]} for j in range(det_n)],
                ["0"])
    return name, pd


def open_project(name, wait_warm=True):
    w = window()
    w._open_project_by_name(name)
    wait_until(lambda: w.list_status.text().find("Blocks:") >= 0
               and len(getattr(w, "_filtered_data", []) or []) > 0,
               desc="列表与统计就绪")
    if wait_warm:
        # ★★ 必须等预热完成后的那次"补刷"。
        #   首轮建表时 `_allow_decode` 还是 False（后台掩膜统计没跑完），labelme JSON
        #   那一支被跳过 —— 此时统计只有 _det.json 的数量，"两份都累加"的 bug 在这个
        #   时刻**看不见**。变异实拍：不等补刷，"恢复成两份都累加"照样全绿。
        wait_until(lambda: getattr(w, "_mask_stats_ready_gen", -1) == getattr(w, "_project_gen", -2),
                   timeout=40, desc="掩膜预热完成（补刷后统计才含 labelme JSON）")
        _APP.processEvents()
    return w


def counts():
    """{base: shape_count}"""
    return {r.get("base"): r.get("shape_count") for r in (window()._filtered_data or [])}


def blocks_in_status():
    t = window().list_status.text()
    for part in t.split("|"):
        part = part.strip()
        if part.startswith("Blocks:"):
            return int(part.split(":")[1].strip())
    raise AssertionError("状态栏没有 Blocks 字段：%r" % t)


def test_det_and_labelme_not_doubled():
    """★ 两份文件都在（各 3 个）→ 只算 3，不是 6。"""
    name, _pd = make_project("目标检测", n_images=2, labelme_n=3, det_n=3)
    open_project(name)
    got = counts()
    assert got, "列表没数据"
    bad = {k: v for k, v in got.items() if v != 3}
    assert not bad, "★ 标注数翻倍了（应为 3）：%r" % bad
    assert blocks_in_status() == 6, "★ 底栏 Blocks 应为 6（2 张 × 3），实际 %d" % blocks_in_status()


def test_list_column_matches():
    """列表「标注」列与内部计数同源。"""
    w = window()
    cells = []
    for row in range(w.image_list_widget.rowCount()):
        it = w.image_list_widget.item(row, ANN_COL)
        cells.append(it.text() if it else None)
    assert cells and all(c == "3" for c in cells), "★ 列表「标注」列不对：%r" % cells


def test_det_only():
    """只有 _det.json → 3。"""
    name, _pd = make_project("目标检测", n_images=1, labelme_n=0, det_n=3)
    open_project(name)
    assert counts() == {"img1": 3}, counts()


def test_labelme_only():
    """只有 labelme JSON（分割工程的老路子）→ 3，没被这次改动带偏。"""
    name, _pd = make_project("语义分割", n_images=1, labelme_n=3, det_n=0)
    open_project(name)
    wait_until(lambda: all((v or 0) > 0 for v in counts().values()),
               timeout=30, desc="labelme JSON 解码（预热）完成")
    assert counts() == {"img1": 3}, counts()


def test_det_wins_over_labelme():
    """★ 优先级：_det.json 3 个 + labelme 5 个 → 报 3（检测侧才是权威）。"""
    name, _pd = make_project("目标检测", n_images=1, labelme_n=5, det_n=3)
    open_project(name)
    wait_until(lambda: counts().get("img1") is not None, desc="首行就绪")
    got = counts()["img1"]
    assert got == 3, "★ 应优先 _det.json（3），实际 %r（5=只读 json，8=两份相加）" % got
    assert blocks_in_status() == 3, "底栏 Blocks 应为 3，实际 %d" % blocks_in_status()


def test_empty_det_shows_zero_and_does_not_fall_back():
    """★ 空 `_det.json`（0 个框、合法）：算**已标注**，列表显式写 "0"，不退回 labelme。

    这是产品定过的语义（2026-11-16）：
    · 空 = 人已经过目、确认这张没有缺陷（良品图），所以是"已标注"；
    · 但以前列表留**空白**，看着和"根本没标"一样，而 Annotated Only 又算它已标注
      ⇒ 两处矛盾。现在显式写 0（带 tooltip）。
    · labelme 里就算还有 3 个形状，也**不许**退回去 —— 那会把用户删掉的框复现。
    """
    name, _pd = make_project("目标检测", n_images=1, labelme_n=3, det_n=0, det_empty=True)
    w = open_project(name)
    rec = (w._filtered_data or [{}])[0]
    assert rec.get("shape_count") == 0, "★ 空 _det.json 不该退回 labelme：%r" % rec.get("shape_count")
    assert rec.get("has_ann") is True, "★ 空标注仍算「已标注」（产品口径）"
    cell = w.image_list_widget.item(0, ANN_COL)
    assert cell is not None and cell.text() == "0", \
        "★ 空标注应显式写 0，实际 %r（空白会让人以为没标）" % (cell.text() if cell else None)
    assert "空标注" in (cell.toolTip() or ""), "★ 0 没有提示说明：%r" % cell.toolTip()


def test_no_annotation_at_all_is_blank():
    """真·没标注（既无 _det.json 也无 labelme）→ 「标注」列留空，别显示 0。"""
    name, pd = make_project("目标检测", n_images=1, labelme_n=0, det_n=0)
    w = open_project(name)
    rec = (w._filtered_data or [{}])[0]
    assert rec.get("has_ann") is False, "前提：这张图不该算已标注"
    cell = w.image_list_widget.item(0, ANN_COL)
    assert cell is not None and cell.text() == "", \
        "★ 没标注的图不该显示 0：%r" % (cell.text() if cell else None)


def cleanup():
    w = _WIN
    if w is not None:
        for name in _MADE:
            try:
                w.pm.delete_project(name)
            except Exception:  # noqa: BLE001
                pass
    if _TMP:
        shutil.rmtree(_TMP, ignore_errors=True)


def main():
    checks = [
        ("★ 两份标注文件都在时不翻倍", test_det_and_labelme_not_doubled),
        ("★ 列表「标注」列与计数同源", test_list_column_matches),
        ("只有 _det.json", test_det_only),
        ("只有 labelme JSON（分割）", test_labelme_only),
        ("★ _det.json 优先于 labelme", test_det_wins_over_labelme),
        ("★ 空 _det.json：显示 0、算已标注、不退回 labelme", test_empty_det_shows_zero_and_does_not_fall_back),
        ("真没标注的图留空（不显示 0）", test_no_annotation_at_all_is_blank),
    ]
    try:
        for name, fn in checks:
            check(name, fn)
    finally:
        cleanup()

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
