# -*- coding: utf-8 -*-
"""Annot / Ignore / Pred 三个显示开关是**用户偏好**，换图不许动（2026-11-17 用户报）。

用户原话："Annot、Ignore、Pred 这三个参数只受快捷键和对应按钮的控制，
在列表用鼠标切图，不应该改变它们。"

## 根因（改前实测）

`AnnotationCanvas.load_image()` 里有一句 `self._show_prediction = False`
（2026-10-22 那次"换图清掉上一张推理结果"时顺手写进去的）。于是**每点一次列表**，
Pred 就被强制关掉；而 Annot / Ignore 在画布层本来就没被重置（探针实测：
切图前后 annot=False/ignore=False 原样保留，pred=True→False）。

这句的本意是"别让上一张的预测掩膜残影"，但残影是 `_prediction_mask` 的事
（那里已经在清了）——显示开关是用户的选择，不该替用户做决定。

## 修法与这条用例

`load_image()` 开头存下五个显示标志、结尾原样还原（含检测模式的
`_show_annotation_boxes` / `_show_pred_boxes`）。用例钉两件事：
① 画布层：分割/检测两套标志换图后一个都不许变；
② 窗口层：**走列表点击那条真路径**（`currentCellChanged` → `_load_image_by_index`），
   换到另一行后标志仍然不变。

运行：C:\\aitp_venv\\Scripts\\python.exe tests\\test_layer_flags_sticky.py
"""
import _testenv  # noqa: F401  无显示器 + 配置目录隔离（见 tests/_testenv.py）
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ⚠️ 与 main.py 一致：torch 必须先于 PyQt5（否则 WinError 1114 c10.dll）
try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

failures = []
_APP = None
_WIN = None

#: 用本机的小工程测（与 test_project_switch_view 同一套，不碰 D 盘大工程）
PROJ = "20260604"

FLAGS = ("_show_annotation", "_show_ignore", "_show_prediction",
         "_show_annotation_boxes", "_show_pred_boxes")


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


def _two_images():
    """两张临时图（不同颜色，避免"读的还是同一张"这种假绿）。"""
    d = tempfile.mkdtemp(prefix="aitp_flags_")
    a = os.path.join(d, "a.png")
    b = os.path.join(d, "b.png")
    from PIL import Image
    Image.new("RGB", (320, 240), (10, 20, 30)).save(a)
    Image.new("RGB", (320, 240), (30, 20, 10)).save(b)
    return d, a, b


def _canvas(det=False):
    from annotation.mask_editor import AnnotationCanvas
    app()
    d, a, b = _two_images()
    c = AnnotationCanvas()
    c.project_dir = d
    c.label_manager = None
    if det:
        c._det_overlay = object()
    c.load_image(a)
    return c, a, b


def _set_non_default(c, det):
    """把五个标志拨到**都不是默认值**，这样任何一处被重置都会被抓到。"""
    c._show_annotation = False
    c._show_ignore = False
    c._show_prediction = True
    c._show_annotation_boxes = False
    c._show_pred_boxes = False


def _snapshot(c):
    return {k: getattr(c, k) for k in FLAGS}


def test_seg_flags_survive_image_switch():
    """分割模式：换图后 annot/ignore/pred 原样保留，状态行也照旧。"""
    c, _a, b = _canvas(det=False)
    _set_non_default(c, det=False)
    before = _snapshot(c)
    c.load_image(b)
    after = _snapshot(c)
    assert after == before, "★ 换图把显示开关改了：%s → %s" % (before, after)
    s = c.status_info_text()
    assert "Annot:OFF" in s and "Ignore:OFF" in s and "Pred:ON" in s, \
        "★ 状态行没跟上（应当 Annot:OFF/Ignore:OFF/Pred:ON）：%s" % s


def test_det_flags_survive_image_switch():
    """检测模式：`_show_annotation_boxes` / `_show_pred_boxes` 也不许被换图重置。"""
    c, _a, b = _canvas(det=True)
    _set_non_default(c, det=True)
    before = _snapshot(c)
    c.load_image(b)
    after = _snapshot(c)
    assert after == before, "★ 检测模式换图把显示开关改了：%s → %s" % (before, after)
    s = c.status_info_text()
    assert "Annot:OFF" in s and "Pred:OFF" in s, "★ 检测模式状态行不对：%s" % s


def test_prediction_mask_still_cleared():
    """反向约束：显示开关要留，但**上一张的预测数据必须清**（防残影）。"""
    c, a, b = _canvas(det=False)
    import numpy as np
    c._prediction_mask = np.ones((240, 320), dtype=np.uint8)
    c.load_image(b)
    assert c._prediction_mask is None, "★ 上一张的预测掩膜没清 —— 会画出残影"


def window():
    global _WIN
    if _WIN is not None:
        return _WIN
    from PyQt5.QtWidgets import QApplication
    import ui.task_queue as tq
    fd, p = tempfile.mkstemp(prefix="aitp_flags_win_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p          # 别写真实的 .task_history.json
    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker
    app()
    init_ui_invoker()
    _WIN = MainWindow()
    return _WIN


def _wait(cond, timeout=40.0, desc="条件"):
    end = time.time() + timeout
    while time.time() < end:
        app().processEvents()
        time.sleep(0.02)
        if cond():
            return True
    raise AssertionError("等待超时：%s" % desc)


def test_list_click_keeps_flags():
    """★ 真路径：列表换一行（currentCellChanged）之后，五个标志一个都不许变。"""
    w = window()
    w._open_project_by_name(PROJ)
    _wait(lambda: w.image_list_widget.rowCount() >= 2
          and getattr(w.canvas, "current_image_path", None),
          desc="%s 就绪（至少两张图）" % PROJ)

    c = w.canvas
    _set_non_default(c, det=(getattr(c, "_det_overlay", None) is not None))
    before = _snapshot(c)
    old_path = str(c.current_image_path)

    w.image_list_widget.setCurrentCell(1, 0)      # = 用鼠标点第二行
    app().processEvents()

    assert str(c.current_image_path) != old_path, \
        "前提不成立：列表换行没触发换图（%s）" % old_path
    after = _snapshot(c)
    assert after == before, "★ 列表点击换图把显示开关改了：%s → %s" % (before, after)


def main():
    print("=== Annot / Ignore / Pred 换图保持 ===")
    check("分割模式：换图不动三个开关", test_seg_flags_survive_image_switch)
    check("检测模式：换图不动框开关", test_det_flags_survive_image_switch)
    check("反方向：预测掩膜仍要清", test_prediction_mask_still_cleared)
    check("列表点击路径：换行不动开关", test_list_click_keeps_flags)
    print()
    if failures:
        print("❌ %d/%d 项未通过" % (len(failures), 4))
        for n, e in failures:
            print("   - %s: %s" % (n, e))
        return 1
    print("✅ 全部通过  4/4 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
