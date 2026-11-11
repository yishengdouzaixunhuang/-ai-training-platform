# -*- coding: utf-8 -*-
"""双击切换工程后，画布/列表必须**立刻**跟着清空（2026-11-11 用户报的问题）。

## 为什么要有这条

用户原话："双击切换工程的时候，右边的窗口没有同步切换对应的工程的图片"。

实测根因不是"没切换"，而是**切换期间还挂着上一个工程的东西**：

- `canvas.set_project()` 只换 `project_dir` / `label_manager`，**不动
  `image` / `mask` / `current_image_path` / 叠加层**；
- 新工程的首图要等后台扫描 + 解码完成才加载 —— 实测 海目星二期 2001 张 **8.2 秒**、
  速博达 **9.6 秒**。这 8~10 秒里画布一直显示旧图，用户当然认为没切换；
- 更危险的是：`label_manager` 已经是**新工程**的，而 `current_image_path` 还是旧工程的
  图 —— 掩膜 PNG 的 500ms 防抖一旦触发，会把**旧图的名字**写进**新工程**的 annotations。

所以断言是"切换那一刻就清空"，不是"最终能切过去"（后者一直是对的，掩盖了问题）。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_project_switch_view.py
"""
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ⚠️ 与 main.py 一致：torch 必须先于 PyQt5（否则 WinError 1114 c10.dll）——
#    之前把 Qt 的 import 写在模块顶部忘了这条，整份用例全炸在这上面。
try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

from PyQt5.QtCore import Qt  # noqa: E402

failures = []
_APP = None
_WIN = None

#: 用本机的小工程测（不碰用户的 D 盘大工程，几秒内跑完）
PROJ_A = "Classify"
PROJ_B = "20260604"


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def wait_until(pred, timeout=30.0, desc="条件"):
    end = time.time() + timeout
    while time.time() < end:
        _APP.processEvents()
        time.sleep(0.02)
        if pred():
            return
    raise AssertionError("等待超时：%s" % desc)


def window():
    global _APP, _WIN
    if _WIN is not None:
        return _WIN
    try:
        import torch  # noqa: F401  必须先于 PyQt5
    except OSError:
        pass
    from PyQt5.QtWidgets import QApplication
    import ui.task_queue as tq

    fd, p = tempfile.mkstemp(prefix="aitp_sw_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p            # 别写真实的 .task_history.json

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    _APP = QApplication.instance() or QApplication([])
    init_ui_invoker()
    _WIN = MainWindow()
    return _WIN


def project_dir(name):
    return str(_WIN.pm.get_project_dir(name)).replace("\\", "/")


def wait_ready(name, timeout=40.0):
    """等到列表有行、且画布真的显示了这个工程里的图。"""
    d = project_dir(name)
    wait_until(lambda: _WIN.image_list_widget.rowCount() > 0
               and str(getattr(_WIN.canvas, "current_image_path", "") or ""
                       ).replace("\\", "/").startswith(d),
               timeout=timeout, desc="%s 就绪" % name)


def canvas_is_empty(w):
    c = w.canvas
    return (getattr(c, "current_image_path", None) is None
            and getattr(c, "image", None) is None
            and getattr(c, "mask", None) is None)


def test_switch_clears_view_immediately():
    """★ 切换工程**返回那一刻**：不许再挂着上一个工程的图/行，掩膜防抖必须停。

    约定（2026-11-11 起）：返回时画布**可以是空的，也可以已经显示了新工程的首图**
    （后者是首图预览，见 `_preview_first_project_image`），但**绝不能还是旧工程的图**。
    """
    w = window()
    w._open_project_by_name(PROJ_A)
    wait_ready(PROJ_A)
    assert not canvas_is_empty(w), "前置条件不成立：A 工程没把图画出来"
    old_cur = str(getattr(w.canvas, "current_image_path", "") or "")

    # 模拟"标注刚改过、掩膜防抖还在排队" —— 这是跨工程写错文件的那条路
    w._mask_png_timer.start()
    assert w._mask_png_timer.isActive(), "前置条件不成立：防抖定时器没起来"

    w._open_project_by_name(PROJ_B)        # 不等、直接检查
    d = project_dir(PROJ_B)
    cur = str(getattr(w.canvas, "current_image_path", "") or "").replace("\\", "/")

    assert cur != old_cur.replace("\\", "/"), "★ 画布还停在上一个工程的那张图"
    assert (not cur) or cur.startswith(d), "★ 切换后画布上的图不属于新工程：%r" % cur
    assert w.image_list_widget.rowCount() == 0, "★ 切换后列表还留着上一个工程的行"
    assert not w._all_images, "★ _all_images 还留着上一个工程的路径"
    assert not w._mask_png_timer.isActive(), \
        "★ 掩膜防抖没停 —— 它会把旧图的名字写进新工程的 annotations"


def test_first_image_shown_without_waiting_for_list():
    """★ 切换返回时就该看到新工程的首图（列表要 9 秒，图不该等列表）。

    这条钉的是 `_preview_first_project_image()`：实测 海目星二期 2001 张，
    列表就绪 9577ms、首图 10814ms —— 预览把"看到新工程的图"从 10.8s 压到 1.1s。
    """
    w = window()
    w._open_project_by_name(PROJ_A)
    wait_ready(PROJ_A)
    w._open_project_by_name(PROJ_B)        # 不等事件循环
    d = project_dir(PROJ_B)
    cur = str(getattr(w.canvas, "current_image_path", "") or "").replace("\\", "/")
    assert cur.startswith(d), \
        "切换返回时画布还没显示新工程的首图（预览没接上）：%r" % (cur or "(空)")
    assert w.image_list_widget.rowCount() == 0, \
        "这条用例的前提是列表还没建好，否则测不到「预览」（实际 %d 行）" % w.image_list_widget.rowCount()


def test_switch_eventually_loads_new_project():
    """最终仍要正确加载新工程的首图（清理不能把正常流程弄坏）。"""
    w = window()
    wait_ready(PROJ_B)
    d = project_dir(PROJ_B)
    cur = str(getattr(w.canvas, "current_image_path", "") or "").replace("\\", "/")
    assert cur.startswith(d), "新工程的首图没加载出来：%r" % cur
    assert w.image_list_widget.rowCount() > 0


def test_close_project_clears_view():
    """关工程也要清干净（同一个清理函数，别只修一半）。"""
    w = window()
    wait_ready(PROJ_B)
    w._close_project()
    _APP.processEvents()
    assert canvas_is_empty(w), "★ 关工程后画布还有图"
    assert w.image_list_widget.rowCount() == 0, "★ 关工程后列表还有行"


def test_switch_back_and_forth():
    """来回切 A→B→A 也不能串：每次都要清、每次都要加载对。"""
    w = window()
    prev = None
    for name in (PROJ_A, PROJ_B, PROJ_A):
        w._open_project_by_name(name)
        cur = str(getattr(w.canvas, "current_image_path", "") or "").replace("\\", "/")
        if prev is not None:
            assert cur != prev, "★ 切到 %s 后画布还停在上一个工程的图" % name
        assert (not cur) or cur.startswith(project_dir(name)), \
            "★ 切到 %s 后画布上的图不属于它：%r" % (name, cur)
        wait_ready(name)
        cur = str(getattr(w.canvas, "current_image_path", "") or "").replace("\\", "/")
        assert cur.startswith(project_dir(name)), \
            "切到 %s 后画布显示的是别的工程的图：%r" % (name, cur)
        prev = cur


# ---------------------------------------------------------------- 预览/中间态

def _mk_project(root, name, files, json_mt=None):
    """造一个最小工程：images/*.png（+ 对应 json），返回目录。"""
    from PIL import Image as _Img
    pd = os.path.join(root, name)
    img_dir = os.path.join(pd, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(pd, "annotations"), exist_ok=True)
    with open(os.path.join(pd, "project.json"), "w", encoding="utf-8") as f:
        f.write('{"name": "%s", "task_type": "语义分割", "classes": ["background"]}' % name)
    for fn in files:
        _Img.new("RGB", (8, 8), (20, 20, 20)).save(os.path.join(img_dir, fn))
        jp = os.path.join(img_dir, os.path.splitext(fn)[0] + ".json")
        with open(jp, "w", encoding="utf-8") as f:
            f.write('{"shapes": []}')
        if json_mt and os.path.splitext(fn)[0] in json_mt:
            t = json_mt[os.path.splitext(fn)[0]]
            os.utime(jp, (t, t))
    return pd


def test_preview_matches_list_first_row():
    """预览那张图必须就是列表第一行 —— 否则"图没变却跳了行"。

    列表默认顺序是 `sorted(os.listdir())`（见 `_load_image_list_async`），
    预览用 `min(名字)`，两者必须一致。
    """
    w = window()
    w._open_project_by_name(PROJ_A)
    wait_ready(PROJ_A)
    # 列表第一行的图片路径（UserRole 里存的是路径）
    it = w.image_list_widget.item(0, 0)
    first_row = str(it.data(Qt.UserRole) or "") if it is not None else ""
    cur = str(getattr(w.canvas, "current_image_path", "") or "")
    assert first_row, "列表第一行没拿到路径，用例前提不成立"
    assert os.path.normcase(cur) == os.path.normcase(first_row), \
        "预览/列表不一致：canvas=%r 第一行=%r" % (cur, first_row)


def test_preview_respects_recent_sort():
    """勾了「Recent」时列表按标注 JSON 的 mtime 降序，预览也必须按同一规则挑。"""
    tmp = tempfile.mkdtemp(prefix="sw_recent_")
    # aa.png 的 json 很旧，zz.png 的 json 很新：字典序最小是 aa，但 Recent 下首行是 zz
    pd = _mk_project(tmp, "p", ["aa.png", "zz.png"],
                     json_mt={"aa": 1000.0, "zz": 2000.0})
    w = window()
    w.canvas.set_project(pd)
    w._project_dir = pd

    w.recent_check.blockSignals(True)
    try:
        w.recent_check.setChecked(False)
        w._preview_first_image(pd)
        assert os.path.basename(w.canvas.current_image_path) == "aa.png", \
            "默认顺序应该取字典序最小的 aa.png，实际 %r" % w.canvas.current_image_path

        w.recent_check.setChecked(True)
        w._preview_first_image(pd)
        assert os.path.basename(w.canvas.current_image_path) == "zz.png", \
            "Recent 顺序应该取标注最新的 zz.png，实际 %r" % w.canvas.current_image_path
    finally:
        w.recent_check.setChecked(False)
        w.recent_check.blockSignals(False)


def test_autosave_during_preview_window_goes_to_new_project():
    """★ "有图无列表"的中间态：自动保存必须写进**新**工程，且文件名对得上。

    审核指出：预览期间列表还是空的（海目星二期有 9.6 秒），这窗口里用户可以画一笔
    触发 `_mask_png_timer`。此时 `label_manager` / `_project_dir` 都已是新工程，
    `current_image_path` 是预览那张图 —— 写出来的掩膜应该落在新工程、名字对得上。
    这里直接驱动"清理 → 预览 → 同步掩膜"三步，把中间态钉住。
    """
    import numpy as np

    tmp = tempfile.mkdtemp(prefix="sw_mid_")
    pd_old = _mk_project(tmp, "old", ["old1.png"])
    pd_new = _mk_project(tmp, "new", ["new1.png", "new2.png"])

    w = window()
    w.canvas.set_project(pd_old)
    w._project_dir = pd_old
    w._preview_first_image(pd_old)
    assert os.path.basename(w.canvas.current_image_path) == "old1.png"

    # 切换：清理 → 换目录 → 预览（此时列表还没建，正是中间态）
    w._reset_view_for_project_switch()
    w._project_dir = pd_new
    w.canvas.set_project(pd_new)
    w._preview_first_image(pd_new)
    assert w.image_list_widget.rowCount() == 0, "用例前提：列表还没建好"

    # 模拟"用户画了一笔 → 防抖触发"
    w.canvas.mask = np.zeros((8, 8), dtype=np.uint8)
    w._sync_mask_png_for_current()

    new_ann = os.path.join(pd_new, "annotations")
    old_ann = os.path.join(pd_old, "annotations")
    new_files = sorted(os.listdir(new_ann))
    old_files = sorted(os.listdir(old_ann))
    assert new_files == ["new1.png".replace(".png", "_mask.png")], \
        "新工程该只多出一张对应预览图的掩膜，实际 %r" % new_files
    assert old_files == [], "★ 旧工程的 annotations 被写了东西：%r" % old_files


def test_reset_actually_clears_canvas_fields():
    """★ 清理必须**真的执行**：往每个字段塞哨兵值，切一次后必须一个都不剩。

    这条是被一次事故逼出来的（2026-11-11 审核驳回）：第一版把清单写进
    `AnnotationCanvas.reset_view()`，而本类末尾**早就有一个** `reset_view()`
    （工具栏「刷新视图」，只做 `_fit_to_window()`）。类体里后定义的覆盖先定义的，
    于是那 58 行清理**一行都没跑**，而主窗口的 15 个 `setattr` 已经删了
    ⇒ 切换工程时一个字段都没清。

    原来的 `test_switch_clears_view_immediately` 抓不到，因为它只断言
    `current_image_path != 旧图` —— 而 `_open_project_by_name` 在清理之后紧接着
    调 `_preview_first_image()`，`load_image()` 顺手就把这个字段覆盖成新图了。
    **唯一被断言的字段，恰好被预览救了。** 所以这里直接对 `_reset_view_for_project_switch()`
    本身下断言，一个字段一个字段地看。
    """
    import numpy as np

    w = window()
    c = w.canvas
    # 塞哨兵：类型各不相同，避免"假清空"（比如只置 None 却留下空 dict）
    c.current_image_path = "SENTINEL/old_project/old.bmp"
    c._cached_qimage = "SENTINEL_QIMAGE"
    c._cached_gray = "SENTINEL_GRAY"
    c.prediction_overlay = "SENTINEL_OVERLAY"
    c._prediction_mask = "SENTINEL_MASK"
    c._pred_prob = "SENTINEL_PROB"
    c._pred_overlay_cache = {"stale": 1}
    c._pred_stats_cache = {"stale": 2}
    c._height_image = "SENTINEL_HEIGHT"
    c._height_path = "SENTINEL/old.tif"
    c._polygon_points = [(1, 2), (3, 4)]
    c._stroke_pts = [(5, 6)]
    c.image = np.zeros((4, 4, 3), dtype=np.uint8)
    c.zoom_level = 7.5

    w._reset_view_for_project_switch()

    leftovers = {
        "current_image_path": c.current_image_path,
        "_cached_qimage": c._cached_qimage,
        "_cached_gray": c._cached_gray,
        "prediction_overlay": c.prediction_overlay,
        "_prediction_mask": c._prediction_mask,
        "_pred_prob": c._pred_prob,
        "_pred_overlay_cache": c._pred_overlay_cache,
        "_pred_stats_cache": c._pred_stats_cache,
        "_height_image": c._height_image,
        "_height_path": c._height_path,
        "_polygon_points": c._polygon_points,
        "_stroke_pts": c._stroke_pts,
        "image": c.image,
    }
    def _cleared(v):
        """None / 空 dict / 空 list 都算清干净了（这三者的"空"形态不同，别只认 None）。"""
        return v is None or (isinstance(v, (dict, list, tuple)) and not v)

    dirty = sorted(k for k, v in leftovers.items() if not _cleared(v))
    assert not dirty, "★ 切换工程后这些字段没清（清理逻辑没执行？）：%s" % dirty
    assert c.zoom_level == 1.0, "★ 缩放没复位：%r" % c.zoom_level

    # 用户偏好**不该**被清
    c.brush_size = 17
    c.current_class_id = 3
    w._reset_view_for_project_switch()
    assert c.brush_size == 17, "笔刷大小是用户偏好，不该被清"
    assert c.current_class_id == 3, "当前类别是用户偏好，不该被清"


def test_reset_list_is_owned_by_canvas():
    """清字段的清单必须住在 canvas 里，不能又散回主窗口。

    ★ 审核建议：清单硬编码在 `MainWindow` 里，将来谁给 canvas 加一个 `_cached_xxx`
      就必然漏，症状和"切换后看到上一张图"一模一样。

    ⚠️ 方法名必须是 `reset_for_project_switch`，**不能**叫 `reset_view` ——
      那个名字已被本类的「刷新视图」占用，同名会被覆盖（这条断言就是那次事故的墓碑）。
    """
    import inspect
    from annotation.mask_editor import AnnotationCanvas
    from ui.main_window import MainWindow

    assert hasattr(AnnotationCanvas, "reset_for_project_switch"), \
        "canvas 没有 reset_for_project_switch()"
    assert "reset_view" not in dir(AnnotationCanvas) or \
        AnnotationCanvas.reset_view is not AnnotationCanvas.reset_for_project_switch, \
        "两个方法撞名了"
    src = inspect.getsource(MainWindow._reset_view_for_project_switch)
    assert "canvas.reset_for_project_switch()" in src, "主窗口没走 canvas.reset_for_project_switch()"
    for field in ("_cached_qimage", "_prediction_mask", "_pred_prob", "_cached_gray"):
        assert field not in src, "主窗口里又在直接清 %s —— 清单该回到 canvas" % field


def main():
    print("=== 切换工程后视图同步 回归测试 ===")
    checks = [
        ("★ 切换瞬间清掉旧工程的一切", test_switch_clears_view_immediately),
        ("★ 清理真的执行（哨兵值）", test_reset_actually_clears_canvas_fields),
        ("★ 不等列表就显示新工程首图", test_first_image_shown_without_waiting_for_list),
        ("最终加载新工程首图", test_switch_eventually_loads_new_project),
        ("关工程也清空", test_close_project_clears_view),
        ("来回切换不串工程", test_switch_back_and_forth),
        ("预览图 == 列表第一行", test_preview_matches_list_first_row),
        ("预览排序跟随「Recent」", test_preview_respects_recent_sort),
        ("★ 中间态自动保存写进新工程", test_autosave_during_preview_window_goes_to_new_project),
        ("★ 清字段清单住在 canvas 里", test_reset_list_is_owned_by_canvas),
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
