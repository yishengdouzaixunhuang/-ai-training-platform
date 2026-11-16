# -*- coding: utf-8 -*-
"""右键「推理选中的图像」：只跑选中的那些图。

## 为什么要有这条

友商的图片列表右键有「推理选中的图像」；我们的批量推理只有"全部"，
想重跑几张（比如刚补的、或上次失败的那几张）只能整工程再跑一遍。

## 两条要钉的事

1. **选中的行 → 图片清单**：只包含选中的，顺序按行号，去重、跳过不存在的；
2. **清单真的传到了流水线**：`BatchRunner.run(images)` 收到的就是这几张 ——
   只测第 1 条会漏掉"传下去了但实现里又用 `list_images()` 覆盖"这种。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_infer_selected.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import types

import _testenv  # noqa: F401  无显示器 + 配置目录隔离（见 tests/_testenv.py）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:                                  # 与 main.py 一致：torch 先于 PyQt5
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

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
        from PyQt5.QtWidgets import QApplication
        _APP = QApplication.instance() or QApplication([])
    return _APP


def window():
    global _WIN, _TMP
    if _WIN is not None:
        return _WIN
    import ui.task_queue as tq

    fd, p = tempfile.mkstemp(prefix="aitp_infsel_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    app()
    init_ui_invoker()
    _TMP = tempfile.mkdtemp(prefix="aitp_infsel_proj_")
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


def _png(path, w=64, h=32):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), (20, 20, 20)).save(path)


def make_project(n=3, task="目标检测"):
    w = window()
    name = "_infsel_%d" % (len(_MADE) + 1)
    w.pm.create_project(name, task_type=task, parent_dir=_TMP)
    _MADE.append(name)
    pd = str(w.pm.get_project_dir(name))
    paths = []
    for i in range(n):
        p = os.path.join(pd, "images", "img%d.png" % i)
        _png(p)
        paths.append(p)
    # 检测批量推理要求模型文件存在（内容无所谓：BatchRunner 被替掉了）
    mdir = os.path.join(pd, "models", "detection")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "best.pt"), "wb") as f:
        f.write(b"stub")
    return name, pd, paths


def open_project(name, pd, n=3):
    """等列表真的渲染出 n 行 —— 条数要跟着工程走，别写死 3。"""
    w = window()
    w._open_project_by_name(name)
    wait_until(lambda: w._project_dir == pd and w.image_list_widget.rowCount() >= n,
               desc="列表就绪（%d 行）" % n)
    return w


class _MB2:
    """接住 QMessageBox：这些用例不该弹错误框（offscreen 下真弹会 segfault）。"""

    calls = []

    @staticmethod
    def _rec(kind, *a, **k):
        # ⚠️ 第 2 个参数是**标题**、第 3 个才是正文。存全文，断言才查得到"暂不支持"。
        _MB2.calls.append((kind, " | ".join(str(x) for x in a)))
        return 0

    @staticmethod
    def warning(*a, **k):
        return _MB2._rec("warning", *a, **k)

    @staticmethod
    def information(*a, **k):
        return _MB2._rec("information", *a, **k)

    @staticmethod
    def critical(*a, **k):
        return _MB2._rec("critical", *a, **k)

    @staticmethod
    def question(*a, **k):
        return _MB2._rec("question", *a, **k)


def _with_mb(fn):
    import ui.main_window as mw
    real = mw.QMessageBox
    _MB2.calls = []
    mw.QMessageBox = _MB2
    try:
        return fn()
    finally:
        mw.QMessageBox = real


class _StubRunner:
    """替掉 BatchRunner：只记录收到的图片清单，不碰 torch。"""

    seen = []

    def __init__(self, flow, project_dir=None, stop_check=None):
        self.flow = flow

    def __call__(self, *a, **k):            # 兼容别的构造习惯
        return self

    def run(self, images, on_progress=None):
        _StubRunner.seen.append(list(images))
        if on_progress:
            for i, p in enumerate(images, 1):
                frame = types.SimpleNamespace(result={"count": 1},
                                              meta={"infer_ms": 12.5})
                on_progress(i, len(images), p, frame, None)


# ------------------------------------------------------------------ 用例

def test_selected_rows_only():
    """★ 只把选中的行送进流水线（不是整工程）。"""
    name, pd, paths = make_project(3)
    w = open_project(name, pd)

    got = []
    w._enqueue_inference = lambda only=None: got.append(list(only or []))
    try:
        _with_mb(lambda: w._infer_selected_images({0, 2}, pd))
    finally:
        del w._enqueue_inference             # 恢复成类方法

    assert len(got) == 1, "★ 没发起推理：%r" % (got,)
    assert got[0] == [paths[0], paths[2]], \
        "★ 送进流水线的不是选中那两张：%r" % (got[0],)
    assert not _MB2.calls, "不该弹框：%r" % (_MB2.calls,)


def test_empty_selection_does_nothing():
    """没选中任何图片：提示一句，不发起推理。"""
    name, pd, paths = make_project(2)
    w = open_project(name, pd, 2)
    got = []
    w._enqueue_inference = lambda only=None: got.append(only)
    try:
        _with_mb(lambda: w._infer_selected_images(set(), pd))
    finally:
        del w._enqueue_inference
    assert not got, "★ 没选图也发起了推理"
    assert any(k == "information" for k, _ in _MB2.calls), "★ 没告诉用户为什么没反应"


def test_missing_files_are_skipped():
    """选中里混了已删除的图：跳过它，其余照跑。"""
    name, pd, paths = make_project(3)
    w = open_project(name, pd)
    os.remove(paths[1])
    got = []
    w._enqueue_inference = lambda only=None: got.append(list(only or []))
    try:
        _with_mb(lambda: w._infer_selected_images({0, 1, 2}, pd))
    finally:
        del w._enqueue_inference
    assert got and got[0] == [paths[0], paths[2]], "★ 缺失文件没被跳过：%r" % (got,)


def test_all_missing_does_nothing():
    """选中的全都不存在：只提示，不发起空任务。"""
    name, pd, paths = make_project(2)
    w = open_project(name, pd, 2)
    for p in paths:
        os.remove(p)
    got = []
    w._enqueue_inference = lambda only=None: got.append(only)
    try:
        _with_mb(lambda: w._infer_selected_images({0, 1}, pd))
    finally:
        del w._enqueue_inference
    assert not got, "★ 空清单也发起了推理"
    assert any(k == "information" for k, _ in _MB2.calls)


def test_subset_reaches_batch_runner():
    """★ 清单真的到 `BatchRunner.run(images)` 了（不是被 list_images 覆盖）。"""
    name, pd, paths = make_project(3)
    w = open_project(name, pd)

    import flow.batch as fb
    real = fb.BatchRunner
    _StubRunner.seen = []
    fb.BatchRunner = _StubRunner
    try:
        _with_mb(lambda: w._run_det_batch_inference(pd, only=[paths[1]]))
        wait_until(lambda: _StubRunner.seen, timeout=30, desc="BatchRunner 收到清单")
    finally:
        fb.BatchRunner = real

    got = _StubRunner.seen[0]
    assert got == [paths[1]], "★ 流水线收到的不是「只那一张」：%r" % (got,)
    assert not _MB2.calls, "不该弹框：%r" % (_MB2.calls,)


def test_dispatch_forwards_only():
    """★ 分派层要透传 only。

    上面那条直接调 `_run_det_batch_inference`，**绕过了 `_dispatch_batch_inference`** ——
    变异"分派时把 only 丢掉"在那里照样全绿（本轮实拍）。这条专门走分派层。
    """
    name, pd, paths = make_project(3)
    w = open_project(name, pd)
    assert w._detection_mode, "前提：这个工程应进入检测模式"

    import flow.batch as fb
    real = fb.BatchRunner
    _StubRunner.seen = []
    fb.BatchRunner = _StubRunner
    try:
        _with_mb(lambda: w._dispatch_batch_inference(pd, only=[paths[2]]))
        wait_until(lambda: _StubRunner.seen, timeout=30, desc="分派后 BatchRunner 收到清单")
    finally:
        fb.BatchRunner = real

    assert _StubRunner.seen[0] == [paths[2]], \
        "★ 分派层没把 only 传下去：%r" % (_StubRunner.seen[0],)


def test_cls_mode_refuses_politely():
    """分类工程：不假装支持（批处理按目录遍历，传子集会被忽略）。"""
    name, pd, paths = make_project(2, task="图像分类")
    w = open_project(name, pd, 2)
    got = []
    w._enqueue_inference = lambda only=None: got.append(only)
    try:
        _with_mb(lambda: w._infer_selected_images({0}, pd))
    finally:
        del w._enqueue_inference
    assert not got, "★ 分类工程也发起了「只跑选中」"
    msgs = [m for k, m in _MB2.calls if k == "information"]
    assert msgs and "暂不支持" in str(msgs[0]), "★ 没说明为什么不支持：%r" % (msgs,)


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
        ("★ 只把选中的行送进流水线", test_selected_rows_only),
        ("没选中任何图片 → 提示，不发任务", test_empty_selection_does_nothing),
        ("选中里混了已删除的图 → 跳过其余照跑", test_missing_files_are_skipped),
        ("选中的全不存在 → 只提示", test_all_missing_does_nothing),
        ("★ 清单真的到了 BatchRunner.run", test_subset_reaches_batch_runner),
        ("★ 分派层透传 only", test_dispatch_forwards_only),
        ("分类工程：明确说不支持", test_cls_mode_refuses_politely),
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
