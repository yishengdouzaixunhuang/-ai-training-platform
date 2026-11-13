# -*- coding: utf-8 -*-
"""配置文件写入者的回归测试（2026-11-13 全量清点后补）。

## 为什么单独一个文件

`tests/test_settings.py` 管的是 `core/settings.py` 这个**引擎**；
这个文件管的是"**谁在写配置文件、写得对不对**"——清点时发现的问题都在这儿：

1. **`core/config.py` 能把平台搞成起不来**：
   `load_config()` 裸 `json.load` 没有 try，`MainWindow.__init__` 第 176 行
   `self.config = load_config()` 也没兜。文件半截（`save_config` 写到一半断电
   就会这样）⇒ 抛 JSONDecodeError ⇒ 双击图标毫无反应。
   **写完再读、两头都堵**：写改原子，读改容错。

2. **`.custom_projects.json` 写坏 = 外部工程全消失**：
   这份文件登记所有外部工程的路径，非原子写坏后 `_load_custom_dirs()` 静默返回 {}
   —— 文件还在磁盘上，但「工程管理」里一个都不显示。

3. **对话框里四个分块控件在撒谎**（审核 2026-11-13 提的产品一致性问题）：
   `tile` / `tile_size` / `tile_stride` / `tile_mode` 的值来自文件，
   但 `_save_and_accept()` 不写回它们 —— 用户改了点「确定」什么都没发生。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_config_files.py
"""
import io
import json
import os
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ⚠️ 与 main.py 一致：torch 必须先于 PyQt5（交接说明地雷 #2）
try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

import core.config as app_config  # noqa: E402

failures = []
_APP = None
_WIN = None
_TMP = None


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def _tmpdir():
    return tempfile.mkdtemp(prefix="cfgfiles_")


def _write(path, payload, raw=False):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        if raw:
            f.write(payload)
        else:
            json.dump(payload, f, ensure_ascii=False)


def _read(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _with_config_file(path):
    """临时把全局配置路径指到 path（进程内，不动用户真实文件）。"""
    old = app_config.CONFIG_FILE
    app_config.CONFIG_FILE = path
    return old


# ---------------------------------------------------------------- core/config.py

def test_load_config_survives_corrupt_file():
    """★ 配置半截时 `load_config()` 不许抛 —— 它是平台启动的第一步。

    改前它裸 `json.load`，`MainWindow.__init__` 也没兜，坏文件直接让平台起不来。
    """
    d = _tmpdir()
    p = os.path.join(d, ".ai_training_config.json")
    _write(p, '{"workspace": "C:/x", ', raw=True)          # 半个 JSON
    old = _with_config_file(p)
    try:
        got = app_config.load_config()
    finally:
        app_config.CONFIG_FILE = old

    assert isinstance(got, dict), got
    assert got.get("workspace") == app_config.DEFAULT_WORKSPACE, \
        "坏文件应退回默认 workspace，实际 %r" % (got.get("workspace"),)
    assert os.path.exists(p + ".bad"), "坏文件应改名留档（*.bad），不能静默删掉"
    assert not os.path.exists(p), "坏文件应被移走，否则下次启动还会读到它"


def test_load_config_fills_missing_keys():
    """老配置缺 `workspace` 时也不能 KeyError（`__init__` 直接下标取它）。"""
    d = _tmpdir()
    p = os.path.join(d, ".ai_training_config.json")
    _write(p, {"recent_projects": ["a", "b"]})
    old = _with_config_file(p)
    try:
        got = app_config.load_config()
    finally:
        app_config.CONFIG_FILE = old

    assert "workspace" in got and got["workspace"], "缺 workspace 时要补默认值：%r" % got
    assert got["recent_projects"] == ["a", "b"], "已有的键不该被默认值覆盖：%r" % got


def test_save_config_is_atomic():
    """写配置必须走 `os.replace` —— 它不是"好看"，是防止把启动文件写成半截。"""
    d = _tmpdir()
    p = os.path.join(d, ".ai_training_config.json")
    old = _with_config_file(p)
    calls = []
    orig_replace = os.replace

    def spy(src, dst):
        calls.append(dst)
        return orig_replace(src, dst)

    os.replace = spy
    try:
        app_config.save_config({"workspace": "C:/ws", "recent_projects": []})
    finally:
        os.replace = orig_replace
        app_config.CONFIG_FILE = old

    assert calls, "★ save_config 没走 os.replace —— 不是原子写"
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")], "留了临时文件"
    assert _read(p)["workspace"] == "C:/ws", _read(p)


def test_save_config_roundtrip():
    d = _tmpdir()
    p = os.path.join(d, ".ai_training_config.json")
    old = _with_config_file(p)
    try:
        app_config.save_config({"workspace": "D:/ws", "recent_projects": ["p1"],
                                "future_key": 1})
        got = app_config.load_config()
    finally:
        app_config.CONFIG_FILE = old
    assert got["workspace"] == "D:/ws", got
    assert got["recent_projects"] == ["p1"], got
    assert got.get("future_key") == 1, "未知键应原样保留：%r" % got


# ---------------------------------------------------------------- 端到端：起不来

def _window_with_corrupt_config():
    """用一份坏配置启动真实 MainWindow —— 这条是"平台起不来"的复现。"""
    global _APP, _WIN, _TMP
    if _WIN is not None:
        return _WIN
    from PyQt5.QtWidgets import QApplication
    import ui.task_queue as tq

    d = _tmpdir()
    bad = os.path.join(d, ".ai_training_config.json")
    _write(bad, '{"workspace": "C:/x", "recent_projects": [', raw=True)
    _TMP = d
    _with_config_file(bad)

    fd, hp = tempfile.mkstemp(prefix="aitp_cfghist_", suffix=".json")
    os.close(fd)
    os.unlink(hp)
    tq.DEFAULT_HISTORY_PATH = hp                 # 别写真实的 .task_history.json

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    _APP = QApplication.instance() or QApplication([])
    init_ui_invoker()
    _WIN = MainWindow()
    # 别让关窗时的自动保存改写用户真实的 .train_settings.json
    _WIN._train_settings_path = lambda: os.path.join(
        tempfile.mkdtemp(prefix="aitp_cfg_"), ".train_settings.json")
    return _WIN


def test_main_window_starts_with_corrupt_config():
    """★ 端到端：`.ai_training_config.json` 半截时，平台**必须**还能起来。

    改前实测：`MainWindow()` 直接抛
    `JSONDecodeError: Expecting property name enclosed in double quotes`。
    用户看到的就是"双击图标没反应"——没有任何界面告诉他配置坏了。
    """
    w = _window_with_corrupt_config()
    assert w is not None
    assert getattr(w, "config", None), "窗口没拿到 config"
    assert w.config.get("workspace"), "workspace 要有值（退默认），不能是空：%r" % w.config
    assert os.path.exists(os.path.join(_TMP, ".ai_training_config.json.bad")), \
        "坏配置应改名留档，方便事后查"


# ---------------------------------------------------------------- 外部工程注册表

def test_custom_dirs_write_is_atomic():
    """`.custom_projects.json` 必须原子写 —— 写坏等于所有外部工程从列表消失。"""
    from core.project_manager import ProjectManager

    d = _tmpdir()
    calls = []
    orig_replace = os.replace

    def spy(src, dst):
        calls.append(dst)
        return orig_replace(src, dst)

    pm = ProjectManager(d)
    pm._custom_dirs = {"工程A": os.path.join(d, "a"), "工程B": os.path.join(d, "b")}
    os.replace = spy
    try:
        pm._save_custom_dirs()
    finally:
        os.replace = orig_replace

    assert calls, "★ _save_custom_dirs 没走 os.replace —— 不是原子写"
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")], "留了临时文件"
    registry = pm._custom_registry_path()
    assert os.path.exists(registry), "注册表没写出来"
    assert _read(registry) == {"工程A": os.path.join(d, "a"),
                               "工程B": os.path.join(d, "b")}, _read(registry)


def test_custom_dirs_roundtrip():
    """写完能被新实例读回来 —— 外部工程列表不能丢。"""
    from core.project_manager import ProjectManager

    d = _tmpdir()
    pm = ProjectManager(d)
    pm._custom_dirs = {"X": os.path.join(d, "x")}
    pm._save_custom_dirs()

    pm2 = ProjectManager(d)
    assert pm2._custom_dirs == {"X": os.path.join(d, "x")}, pm2._custom_dirs


# ---------------------------------------------------------------- 对话框只读控件

def test_dialog_tile_controls_are_readonly():
    """★ 对话框不保存分块设置，那四个控件就不该让用户改（UI 不能撒谎）。

    审核 2026-11-13 提的产品一致性问题：用户看到 `tile_size=1024`，改成 512，
    点「确定」，什么都没保存 —— 下次打开还是 1024，只会来问"我改的分块怎么没了"。
    """
    from ui.widgets import TrainSettingsDialog

    w = _window_with_corrupt_config()
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": 200, "tile": True, "tile_size": 1024, "tile_stride": 75,
               "tile_mode": "slide"})
    orig_file = TrainSettingsDialog.SETTINGS_FILE
    TrainSettingsDialog.SETTINGS_FILE = p
    try:
        dlg = TrainSettingsDialog(w)
    finally:
        TrainSettingsDialog.SETTINGS_FILE = orig_file

    for name in ("tile_check", "tile_spin", "stride_spin", "tile_mode_combo"):
        ctl = getattr(dlg, name, None)
        assert ctl is not None, "对话框缺控件 %s" % name
        assert not ctl.isEnabled(), \
            "★ %s 还能改 —— 但改了不会保存，UI 在撒谎" % name
    hint = getattr(dlg, "tile_hint", None)
    assert hint is not None, "缺少说明文字（用户不知道去哪儿改）"
    assert "主界面" in hint.text(), "说明要指出去哪儿改：%r" % hint.text()

    # 值仍然来自文件（只读≠瞎显示）
    assert dlg.tile_spin.value() == 1024, dlg.tile_spin.value()


def main():
    print("=== 配置文件写入者回归测试 ===")
    checks = [
        ("★ 坏配置不让 load_config 抛", test_load_config_survives_corrupt_file),
        ("老配置缺 workspace 也能读", test_load_config_fills_missing_keys),
        ("★ save_config 走原子写", test_save_config_is_atomic),
        ("save_config 往返", test_save_config_roundtrip),
        ("★ 坏配置下平台仍能启动（端到端）", test_main_window_starts_with_corrupt_config),
        ("★ 外部工程注册表原子写", test_custom_dirs_write_is_atomic),
        ("外部工程注册表往返", test_custom_dirs_roundtrip),
        ("★ 对话框分块控件只读 + 去向说明", test_dialog_tile_controls_are_readonly),
    ]
    for name, fn in checks:
        check(name, fn)

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
