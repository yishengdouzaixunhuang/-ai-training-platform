# -*- coding: utf-8 -*-
"""批量检测推理后，「分数」「耗时ms」两列必须回填，且**重启平台后还在**。

## 用户报的现象（2026-11-16）

"推理完后，分数和耗时都没有回填" —— 结果列有数（10/14/15），分数与耗时整列空白。

## 根因（两条，各自独立）

1. **分数**：列表读的是 `self._img_conf[base]`（平均置信度）。分割那支从
   `<base>_prob.png` 算得有；**检测那支从来没算过** `img_conf`，只填了 `pred_count`。
2. **耗时**：只存在 `self._infer_times`（内存）。检测批量推理连这个都不写，
   输出 JSON 里也没有任何时间字段 ⇒ 重启平台必然空白。

## 修法

- `DetInfer` 给 `frame.meta["infer_ms"]` 记纯推理耗时；
- `DetResultWriter` 调 `save_coco_json(meta={"infer_ms": ...})` 落盘；
- `_load_output_stats_async` 的检测分支：平均 `annotations[].score` → `img_conf`，
  读 `meta.infer_ms` → 补进 `_infer_times`（**只补内存里没有的**，不覆盖本会话刚跑的值）。

运行：C:\\aitp_venv\\Scripts\\python.exe tests\\test_det_infer_stats_backfill.py
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

try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

import detection.trainer as dt_mod  # noqa: E402
from flow.batch import BatchRunner, build_det_flow  # noqa: E402
from flow.frame import Frame  # noqa: E402

failures = []
_APP = None
_WIN = None
_TMP = None
_MADE = []

SCORES = (0.98, 0.99)            # 平均 0.985 → 列表显示 "98.5%"
INFER_MS = 12.3


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


class _StubTrainer:
    """替身：不加载 YOLO，只返回两个固定框（用时 ~10ms）。"""

    def __init__(self, project_dir):
        self.project_dir = project_dir

    def predict(self, path, weights=None, conf=0.25, iou=0.45):
        time.sleep(0.01)
        return [{"bbox": [1, 1, 10, 10], "confidence": SCORES[0],
                 "category": "0", "category_id": 0},
                {"bbox": [20, 1, 10, 10], "confidence": SCORES[1],
                 "category": "0", "category_id": 0}]


def window():
    global _APP, _WIN, _TMP
    if _WIN is not None:
        return _WIN
    from PyQt5.QtWidgets import QApplication
    import ui.task_queue as tq

    fd, p = tempfile.mkstemp(prefix="aitp_stats_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    _APP = QApplication.instance() or QApplication([])
    init_ui_invoker()
    _TMP = tempfile.mkdtemp(prefix="aitp_stats_proj_")
    _WIN = MainWindow()
    return _WIN


def wait_until(pred, timeout=30.0, desc="条件"):
    end = time.time() + timeout
    while time.time() < end:
        _APP.processEvents()
        time.sleep(0.02)
        if pred():
            return
    raise AssertionError("等待超时：%s" % desc)


def _png(path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (64, 32), (20, 20, 20)).save(path)


def make_project(with_model=True, n=1):
    """最小检测工程 + 一张图（可选放个假模型文件让 DetInfer 不报"模型不存在"）。"""
    w = window()
    name = "_stats_%d" % (len(_MADE) + 1)
    w.pm.create_project(name, task_type="目标检测", parent_dir=_TMP)
    _MADE.append(name)
    pd = str(w.pm.get_project_dir(name))
    if with_model:
        mp = os.path.join(pd, "models", "detection")
        os.makedirs(mp, exist_ok=True)
        with open(os.path.join(mp, "best.pt"), "wb") as f:
            f.write(b"stub")
    for i in range(n):
        _png(os.path.join(pd, "images", "img%d.png" % (i + 1)))
    return name, pd


def run_det_flow(pd, images):
    """跑一次真实的 build_det_flow（DetInfer 用桩替身）。"""
    flow = build_det_flow(pd, out_dir=os.path.join(pd, "outputs"), classes="0")
    frames = BatchRunner(flow, project_dir=pd).run(images)
    return frames


def out_json(pd, base):
    return os.path.join(pd, "outputs", base + "_det.json")


# ------------------------------------------------------------------ 落盘

def test_writer_persists_infer_ms_and_scores():
    """★ DetInfer 记耗时 → DetResultWriter 落盘（meta.infer_ms + 每框 score）。"""
    _name, pd = make_project()
    img = os.path.join(pd, "images", "img1.png")
    old = dt_mod.DetectionTrainer
    dt_mod.DetectionTrainer = _StubTrainer
    try:
        run_det_flow(pd, [img])
    finally:
        dt_mod.DetectionTrainer = old

    p = out_json(pd, "img1")
    assert os.path.exists(p), "没写出 %s" % p
    d = json.load(open(p, encoding="utf-8"))
    assert len(d["annotations"]) == 2, d
    assert all(a.get("score") is not None for a in d["annotations"]), d["annotations"]
    ms = (d.get("meta") or {}).get("infer_ms")
    assert ms is not None, "★ 输出 JSON 里没有 meta.infer_ms（重启后耗时必然空白）"
    assert 0 <= float(ms) < 5000, "耗时不合理：%r" % ms


def test_old_json_without_meta_still_reads_boxes():
    """老文件没有 meta：读框不受影响，只是没有耗时（不报错）。"""
    from detection.coco_io import load_coco_json, save_coco_json
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "x_det.json")
        save_coco_json(p, "x.png", 64, 32, [{"category": "0", "bbox": [1, 1, 5, 5]}], ["0"])
        d = json.load(open(p, encoding="utf-8"))
        assert "meta" not in d, "没传 meta 时不该凭空造一个 key"
        boxes, _ = load_coco_json(p)
        assert len(boxes) == 1, boxes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 读回

def test_loader_fills_conf_and_time():
    """★ `_load_output_stats_async` 从 outputs 读回：平均置信度 + 落盘耗时。"""
    w = window()
    _name, pd = make_project()
    img = os.path.join(pd, "images", "img1.png")
    old = dt_mod.DetectionTrainer
    dt_mod.DetectionTrainer = _StubTrainer
    try:
        run_det_flow(pd, [img])
    finally:
        dt_mod.DetectionTrainer = old

    w._img_conf, w._infer_times = {}, {}
    w._load_output_stats_async(pd)
    got_conf = (w._img_conf or {}).get("img1")
    assert got_conf is not None, "★ 分数没算出来（_img_conf 为空）"
    assert abs(got_conf - 0.985) < 1e-6, "平均置信度算错：%r" % got_conf
    assert (w._infer_times or {}).get("img1") is not None, "★ 耗时没读回来"
    assert (w._pred_count or {}).get("img1") == 2, w._pred_count


def test_loader_does_not_override_inmemory_time():
    """本会话刚跑出来的耗时优先：落盘的旧值不许盖掉它。"""
    w = window()
    _name, pd = make_project()
    img = os.path.join(pd, "images", "img1.png")
    old = dt_mod.DetectionTrainer
    dt_mod.DetectionTrainer = _StubTrainer
    try:
        run_det_flow(pd, [img])
    finally:
        dt_mod.DetectionTrainer = old

    w._infer_times = {"img1": 999.0}
    w._load_output_stats_async(pd)
    assert w._infer_times["img1"] == 999.0, \
        "★ 落盘值覆盖了内存里更可信的耗时：%r" % w._infer_times["img1"]


def test_list_columns_rendered():
    """★ 上屏：列表「分数」= "98.5%"，「耗时ms」= "12.3"。"""
    w = window()
    name, pd = make_project()
    img = os.path.join(pd, "images", "img1.png")
    old = dt_mod.DetectionTrainer
    dt_mod.DetectionTrainer = _StubTrainer
    try:
        run_det_flow(pd, [img])
    finally:
        dt_mod.DetectionTrainer = old

    w._img_conf, w._infer_times = {}, {}
    w._load_output_stats_async(pd)
    w._open_project_by_name(name)
    wait_until(lambda: len(getattr(w, "_filtered_data", []) or []) > 0, desc="列表就绪")
    row = 0
    sc = w.image_list_widget.item(row, 6).text()
    ms = w.image_list_widget.item(row, 7).text()
    assert sc.endswith("%") and abs(float(sc[:-1]) - 98.5) < 0.2, "★ 分数列不对：%r" % sc
    assert ms, "★ 耗时列空白"
    assert float(ms) < 5000, "耗时列不对：%r" % ms


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
        ("★ DetInfer 记耗时 + 落盘（meta.infer_ms / score）", test_writer_persists_infer_ms_and_scores),
        ("老文件没有 meta 也能读框", test_old_json_without_meta_still_reads_boxes),
        ("★ 读回：平均置信度 + 落盘耗时", test_loader_fills_conf_and_time),
        ("本会话耗时不被旧文件覆盖", test_loader_does_not_override_inmemory_time),
        ("★ 上屏：分数列与耗时列", test_list_columns_rendered),
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
