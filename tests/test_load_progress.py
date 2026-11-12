# -*- coding: utf-8 -*-
"""打开工程时的进度显示回归测试。

背景：大图工程「双击 → 数据齐全」实测约 11 秒（列表 3.3s + 后台统计标注 7.5s），
这段时间以前只有一句日志，用户不知道是卡住还是在干活。现在状态栏有进度条，
这个测试把三条行为钉住：

    1. 平时不显示（不能一直挂在状态栏上）
    2. 有总数时按 N/M 显示，值被夹在 [0, total] 内
    3. 加载结束必须隐藏（失败路径也不能让它一直转）

运行（无显示器也能跑）：
    set QT_QPA_PLATFORM=offscreen
    python tests/test_load_progress.py
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

# ⚠️ 和 main.py 一样：torch 必须先于 PyQt5
try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print(f"[warn] torch 加载失败：{_e}")

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


def test_hidden_by_default():
    app, w = _make_window()
    assert hasattr(w, "_load_bar"), "缺少 _load_bar"
    assert w._load_bar.isHidden(), "进度条初始应是隐藏的"
    assert hasattr(w, "load_progress_signal"), "缺少 load_progress_signal"
    w.close()
    app.processEvents()


def test_confirmed_progress():
    """给了总数就按 N/M 显示，且不能越界。"""
    app, w = _make_window()
    w._load_t0 = time.perf_counter() - 1.0      # 绕过 400ms 防闪烁阈值
    w._on_load_progress(120, 400, "正在统计标注 120/400")
    assert not w._load_bar.isHidden(), "超过阈值后应显示"
    assert w._load_bar.maximum() == 400
    assert w._load_bar.value() == 120
    assert "120/400" in w._load_bar.format() or w._load_bar.value() == 120

    w._on_load_progress(999, 400, "越界测试")
    assert w._load_bar.value() == 400, "超过 total 时应被夹到 total"
    w.close()
    app.processEvents()


def test_indeterminate_progress():
    """没给总数（total<=0）时用走马灯，不能除零或卡住。"""
    app, w = _make_window()
    w._load_t0 = time.perf_counter() - 1.0
    w._on_load_progress(0, 0, "正在扫描图片目录…")
    assert not w._load_bar.isHidden()
    assert w._load_bar.maximum() == 0, "走马灯模式下 maximum 应为 0"
    w.close()
    app.processEvents()


def test_hide_on_finish():
    app, w = _make_window()
    w._load_t0 = time.perf_counter() - 1.0
    w._on_load_progress(5, 10, "测试")
    assert not w._load_bar.isHidden()
    w._hide_load_progress()
    assert w._load_bar.isHidden(), "加载结束必须隐藏"
    assert w._load_bar.value() == 0, "隐藏时应把值复位"
    w.close()
    app.processEvents()


def main():
    print("=== 打开工程进度条回归测试 ===")
    checks = [
        ("初始隐藏且信号存在", test_hidden_by_default),
        ("确定进度 N/M", test_confirmed_progress),
        ("不确定进度（走马灯）", test_indeterminate_progress),
        ("结束后隐藏并复位", test_hide_on_finish),
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
