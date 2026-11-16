# -*- coding: utf-8 -*-
"""坏 `_det.json` 不许把"打开工程"顶掉，也不许给用户看原始 JSON 异常。

## 现象（复核实测）

`images/<base>_det.json` 内容坏了（不是合法 JSON）时，异常从
`BoxManager.load()` 一路冒到 `_open_project_by_name` 的兜底 `except`：

```
QMessageBox.warning(self, "Error", "Expecting property name enclosed in double quotes: line 1 column 3 (char 2)")
```

用户看到的是原始 JSON 解析错误，而且此刻工程处于**半开**状态
（画布有预览图、`current_project` 已设、列表是空的）。

## 现在的约定

- `BoxManager.load()` / `load_predictions()` **不抛**，坏文件按"0 个框/0 个预测"处理，
  原因记进 `load_errors`；
- 界面在 `_load_det_annotations()` 里打一行日志：
  `⚠️ 该图标注文件损坏，已按 0 个框显示：xxx.png（JSONDecodeError: ...）`。

运行：C:\\aitp_venv\\Scripts\\python.exe tests\\test_corrupt_det_json.py
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

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

from detection.box_manager import BoxManager  # noqa: E402
from detection.coco_io import save_coco_json  # noqa: E402

failures = []
_APP = None
_WIN = None
_TMP = None
_MADE = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def window():
    global _WIN, _TMP
    if _WIN is not None:
        return _WIN
    import ui.task_queue as tq

    fd, p = tempfile.mkstemp(prefix="aitp_badjson_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    app()
    init_ui_invoker()
    _TMP = tempfile.mkdtemp(prefix="aitp_badjson_proj_")
    _WIN = MainWindow()
    return _WIN


def wait_until(pred, timeout=30.0, desc="条件"):
    end = time.time() + timeout
    while time.time() < end:
        app().processEvents()
        time.sleep(0.02)
        if pred():
            return
    raise AssertionError("等待超时：%s" % desc)


def _png(path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (64, 32), (20, 20, 20)).save(path)


def _bad(path):
    """写一个**非法** JSON（复核那条报错就是 line 1 column 3）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ oops")


def make_project(corrupt_det=True):
    w = window()
    name = "_badjson_%d" % (len(_MADE) + 1)
    w.pm.create_project(name, task_type="目标检测", parent_dir=_TMP)
    _MADE.append(name)
    pd = str(w.pm.get_project_dir(name))
    img = os.path.join(pd, "images", "img1.png")
    _png(img)
    det = os.path.join(pd, "images", "img1_det.json")
    if corrupt_det:
        _bad(det)
    else:
        save_coco_json(det, img, 64, 32,
                       [{"category": "0", "bbox": [1, 1, 5, 5]},
                        {"category": "0", "bbox": [10, 1, 5, 5]},
                        {"category": "0", "bbox": [20, 1, 5, 5]}],
                       ["0"])
    return name, pd, img


# ------------------------------------------------------------------ 单元层

def test_load_corrupt_does_not_raise():
    """★ 坏文件：不抛、0 个框、原因记下来。"""
    tmp = tempfile.mkdtemp()
    try:
        img = os.path.join(tmp, "a.png")
        _png(img)
        _bad(os.path.join(tmp, "a_det.json"))
        bm = BoxManager()
        bm.load(img)                                  # 不许抛
        assert bm.count == 0, "坏文件应该按 0 个框算：%r" % bm.count
        err = bm.load_errors.get("annot")
        assert err, "★ 没有记录失败原因（界面就没法说明白）"
        assert "JSON" in err or "Expecting" in err, err
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_valid_file_still_works():
    tmp = tempfile.mkdtemp()
    try:
        img = os.path.join(tmp, "b.png")
        _png(img)
        save_coco_json(os.path.join(tmp, "b_det.json"), img, 64, 32,
                       [{"category": "0", "bbox": [1, 1, 5, 5]}], ["0"])
        bm = BoxManager()
        bm.load(img)
        assert bm.count == 1, bm.count
        assert not bm.load_errors.get("annot"), bm.load_errors
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_predictions_corrupt_does_not_raise():
    tmp = tempfile.mkdtemp()
    try:
        img = os.path.join(tmp, "c.png")
        _png(img)
        out = os.path.join(tmp, "outputs")
        os.makedirs(out)
        _bad(os.path.join(out, "c_det.json"))
        bm = BoxManager()
        bm.load(img)
        bm.load_predictions(out)                      # 不许抛
        assert bm.pred_boxes == [], bm.pred_boxes
        assert bm.load_errors.get("pred"), "★ 推理结果坏了没记录"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 界面层

def test_open_project_with_corrupt_det_json():
    """★ 带坏标注的工程要能正常打开：列表有行、不弹原始 JSON 异常。"""
    w = window()
    name, pd, _img = make_project(corrupt_det=True)

    # 吞掉所有弹窗并记录：这条用例就是"别再弹原始异常"
    popped = []
    from ui import main_window as mw
    orig_warn, orig_crit, orig_info = mw.QMessageBox, None, None
    orig_gw = QMessageBox.warning

    def spy(*a, **k):
        popped.append(a[2] if len(a) > 2 else "")
        return QMessageBox.Ok

    QMessageBox.warning = staticmethod(spy)
    try:
        w._open_project_by_name(name)
        wait_until(lambda: len(getattr(w, "_filtered_data", []) or []) > 0,
                   desc="列表填上（工程没半开）")
    finally:
        QMessageBox.warning = orig_gw

    bad_pop = [msg for msg in popped if "Expecting" in str(msg) or "JSON" in str(msg)]
    assert not bad_pop, "★ 仍然把原始 JSON 异常弹给用户了：%r" % bad_pop[:2]

    # 加载点：坏文件按 0 个框，且记下原因
    wait_until(lambda: getattr(w.canvas, "current_image_path", None), desc="首图已显示")
    msgs = []
    w.log_signal.connect(lambda m: msgs.append(m))
    w._load_det_annotations()
    assert w._box_manager is not None, "检测模式的 BoxManager 没建起来"
    assert w._box_manager.count == 0, "坏文件应该是 0 个框：%r" % w._box_manager.count
    assert (w._box_manager.load_errors or {}).get("annot"), "★ 没有记录坏文件原因"
    assert any("标注文件损坏" in str(m) for m in msgs), \
        "★ 日志里没说清楚（用户只会看到「框没了」）：%r" % msgs[-3:]


def test_open_project_with_good_det_json():
    """对照组：正常文件仍然是 3 个框（修坏文件别把好文件带偏）。"""
    w = window()
    name, _pd, _img = make_project(corrupt_det=False)
    w._open_project_by_name(name)
    wait_until(lambda: getattr(w.canvas, "current_image_path", None), desc="首图已显示")
    w._load_det_annotations()
    assert w._box_manager.count == 3, "正常文件应读到 3 个框：%r" % w._box_manager.count
    assert not (w._box_manager.load_errors or {}).get("annot"), w._box_manager.load_errors


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
        ("★ 坏 _det.json：load 不抛、0 个框、记原因", test_load_corrupt_does_not_raise),
        ("正常文件仍读到框", test_load_valid_file_still_works),
        ("坏推理结果：load_predictions 不抛", test_load_predictions_corrupt_does_not_raise),
        ("★ 带坏标注的工程能正常打开、不弹原始异常", test_open_project_with_corrupt_det_json),
        ("对照组：正常标注仍是 3 个框", test_open_project_with_good_det_json),
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
