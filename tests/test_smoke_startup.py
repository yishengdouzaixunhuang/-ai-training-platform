# -*- coding: utf-8 -*-
"""启动冒烟测试：验证 MainWindow 能构造出来、主线程转发器已就绪。

这不是完整的功能测试，只回答一个问题：
    「改完线程安全和 dtype 之后，程序还能不能起来？」

运行（无显示器也能跑）：
    set QT_QPA_PLATFORM=offscreen
    python tests/test_smoke_startup.py
"""
import _testenv  # noqa: F401  无显示器 + 配置目录隔离（见 tests/_testenv.py）
import os
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ⚠️ 和 main.py 一样：必须在任何可能引入冲突 DLL 的模块（cv2 / numpy 等）
#    之前先 import torch，否则 WinError 1114 "c10.dll 初始化例程失败"。
try:
    import torch  # noqa: F401
    _TORCH_OK = True
except OSError as _e:
    _TORCH_OK = False
    print(f"[warn] torch 加载失败，将跳过依赖它的检查：{_e}")

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def test_imports():
    """核心模块都能导入（包括新加的 qt_thread、imgio 工具）。"""
    import core.imgio  # noqa: F401
    import ui.qt_thread  # noqa: F401

    assert hasattr(core.imgio, "resize_mask_u")
    assert hasattr(core.imgio, "pil_from_array")
    assert hasattr(ui.qt_thread, "ui_call")
    assert hasattr(ui.qt_thread, "ui_call_sync")
    assert hasattr(ui.qt_thread, "run_bg")


def test_invoker_init():
    """在主线程初始化转发器，并验证 ui_call 在主线程是同步执行的。"""
    from ui.qt_thread import init_ui_invoker, ui_call

    init_ui_invoker()
    box = {}
    ui_call(lambda: box.setdefault("v", 42))
    assert box.get("v") == 42, "主线程内 ui_call 应当同步执行"


def test_mainwindow_constructs():
    """MainWindow 能构造出来（最容易在重构后炸的地方）。"""
    from PyQt5.QtWidgets import QApplication

    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    from ui.qt_thread import init_ui_invoker
    init_ui_invoker()

    w = MainWindow()
    # ★ 关窗时会自动保存训练参数（closeEvent → _save_panel_train_settings）。
    #   这里是测试窗口，面板值不是用户的设置 —— 指到临时文件，别改写真配置。
    #   （以前跑一遍测试，用户的 epochs / batch_size / tile_size 就被换成测试值了）
    w._train_settings_path = lambda: os.path.join(
        tempfile.mkdtemp(prefix="aitp_cfg_"), ".train_settings.json")
    assert w is not None
    # 关键控件存在性
    for attr in ("log_signal", "progress_signal", "refresh_list_signal"):
        assert hasattr(w, attr), f"缺少信号 {attr}"
    w.close()
    app.processEvents()


def test_ui_call_from_worker():
    """子线程调 ui_call：主线程能收到并执行。"""
    import threading
    import time

    from PyQt5.QtWidgets import QApplication

    from ui.qt_thread import init_ui_invoker, ui_call

    app = QApplication.instance() or QApplication(sys.argv)
    init_ui_invoker()

    box = {}

    def worker():
        ui_call(lambda: box.setdefault("done", True))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()

    # 抽几轮事件循环，让排队的调用真的被执行
    deadline = time.time() + 3.0
    while not box.get("done") and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert box.get("done") is True, "子线程的 ui_call 没有被主线程执行"


def main():
    print("=== 启动冒烟测试 ===")
    checks = [
        ("模块导入", test_imports),
        ("转发器初始化", test_invoker_init),
        ("MainWindow 构造", test_mainwindow_constructs),
        ("跨线程 ui_call", test_ui_call_from_worker),
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
