# -*- coding: utf-8 -*-
"""closeEvent 回归测试。

背景：`ui/main_window.py` 曾有两份同名 `closeEvent`，Python 里后定义者胜，
那份只有 7 行的版本静默覆盖了前面 16 行的版本，于是这三件事**从未执行**：

    1. `_sys_monitor.stop()`        —— 关窗后系统监控采样线程残留
    2. `_save_cls_train_settings()` —— 分类训练参数关窗不存，重开就回滚
    3. `unregister_global_hotkey()` —— 全局热键不注销，下次启动注册可能失败

这个测试把四条行为钉住，防止同类"后定义者胜"再次退化。

运行（无显示器也能跑）：
    set QT_QPA_PLATFORM=offscreen
    python tests/test_close_event.py
"""
import _testenv  # noqa: F401  无显示器 + 配置目录隔离（见 tests/_testenv.py）
import json
import os
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ⚠️ 和 main.py 一样：torch 必须先于 PyQt5/cv2，否则 WinError 1114 c10.dll
try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print(f"[warn] torch 加载失败，跳过依赖它的检查：{_e}")

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def _make_window():
    from PyQt5.QtWidgets import QApplication

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    app = QApplication.instance() or QApplication(sys.argv)
    init_ui_invoker()
    w = MainWindow()
    # ★ 关窗时会自动保存训练参数（closeEvent → _save_panel_train_settings）。
    #   这里是测试窗口，面板值不是用户的设置 —— 指到临时文件，别改写真配置。
    #   （以前跑一遍测试，用户的 epochs / batch_size / tile_size 就被换成测试值了）
    w._train_settings_path = lambda: os.path.join(
        tempfile.mkdtemp(prefix="aitp_cfg_"), ".train_settings.json")
    return app, w


def test_close_event_is_unique():
    """类里 closeEvent 只能有一份——两份就是第二份在偷偷覆盖第一份。"""
    import ast
    from collections import Counter

    src = open(os.path.join(ROOT, "ui", "main_window.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    cls = [n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == "MainWindow"][0]
    names = Counter(n.name for n in cls.body if isinstance(n, ast.FunctionDef))
    assert names["closeEvent"] == 1, f"closeEvent 有 {names['closeEvent']} 份"


def test_close_runs_all_four_actions():
    """关窗时四个收尾动作都要跑，且「没有工程」时也要存面板参数。"""
    app, w = _make_window()
    hits = []

    w._sys_monitor.stop = lambda: hits.append("monitor")
    # closeEvent 用 getattr 取这两个方法，所以实例属性可以覆盖掉它们
    w._save_panel_train_settings = lambda: hits.append("panel")
    w._save_cls_train_settings = lambda: hits.append("cls")

    w._screen_ocr_global = True
    w._screen_ocr_hotkey_id = 0x5358
    w._project_dir = None                      # 刻意不给工程目录

    import ui.main_window as mw
    orig = mw.unregister_global_hotkey
    mw.unregister_global_hotkey = lambda hwnd, hid: hits.append("hotkey")
    try:
        w.close()
        app.processEvents()
    finally:
        mw.unregister_global_hotkey = orig

    for key in ("monitor", "panel", "cls", "hotkey"):
        assert key in hits, f"closeEvent 漏了 {key}，实际只调了 {hits}"


def test_cls_settings_actually_written():
    """有工程时，分类参数必须真的落到工程目录的 .cls_train_settings.json。"""
    app, w = _make_window()
    tmp = tempfile.mkdtemp(prefix="aitp_cls_")
    w._project_dir = tmp
    try:
        w.close()
        app.processEvents()
    finally:
        w._project_dir = None

    path = os.path.join(tmp, ".cls_train_settings.json")
    assert os.path.exists(path), "关窗后没有生成 .cls_train_settings.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict) and data, "分类参数文件写成了空 dict"
    assert "model" in data or "epochs" in data, \
        f"缺少训练参数键，实际只有 {list(data)[:8]}"


def test_no_project_still_saves_panel_settings():
    """没有工程也不能抛异常——首页关窗是最常见路径。"""
    app, w = _make_window()
    w._project_dir = None
    try:
        w.close()
        app.processEvents()
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"无工程时关窗抛异常：{exc}") from exc


def main():
    print("=== closeEvent 回归测试 ===")
    checks = [
        ("closeEvent 只有一份", test_close_event_is_unique),
        ("关窗四个动作都执行", test_close_runs_all_four_actions),
        ("分类参数真的落盘", test_cls_settings_actually_written),
        ("无工程关窗不抛异常", test_no_project_still_saves_panel_settings),
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
