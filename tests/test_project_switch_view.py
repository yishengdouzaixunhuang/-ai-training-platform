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


def main():
    print("=== 切换工程后视图同步 回归测试 ===")
    checks = [
        ("★ 切换瞬间清掉旧工程的一切", test_switch_clears_view_immediately),
        ("★ 不等列表就显示新工程首图", test_first_image_shown_without_waiting_for_list),
        ("最终加载新工程首图", test_switch_eventually_loads_new_project),
        ("关工程也清空", test_close_project_clears_view),
        ("来回切换不串工程", test_switch_back_and_forth),
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
