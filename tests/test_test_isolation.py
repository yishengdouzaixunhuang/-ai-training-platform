# -*- coding: utf-8 -*-
"""测试自身的隔离：不许把**用户的真实配置**写脏。

## 背景（第三次了）

跑完测试 `git status` 里冒出 `M .train_settings.json`：
`MainWindow.closeEvent()` → `_save_panel_train_settings()` 把仓库根的真实配置
改成了测试窗口的面板状态（实测 `flip: true → false`）。

前两次的修法是给个别测试 stub `_train_settings_path`，第三次又漏了。
所以现在改成**机制**：

1. `core.settings.cfg_path()` / `Scope.default_path()` 认 `AITP_TEST_CFG_DIR`；
2. `tests/_testenv.py` 把它指到临时目录；
3. 这个文件静态检查"建 MainWindow 的测试都 import 了 _testenv"。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_test_isolation.py
"""
import ast
import os
import shutil
import sys
import tempfile
import traceback

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


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def _test_files():
    d = os.path.join(ROOT, "tests")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".py"))


def test_every_window_test_imports_testenv():
    """★ 建 MainWindow 的测试必须 import _testenv（否则关窗就改真实配置）。"""
    missing = []
    for p in _test_files():
        name = os.path.basename(p)
        if name.startswith("_"):
            continue
        src = open(p, "r", encoding="utf-8-sig").read()
        if "MainWindow()" not in src and "MainWindow (" not in src:
            continue
        if "import _testenv" not in src:
            missing.append(name)
    assert not missing, "★ 这些测试建了主窗但没隔离配置目录：%r（在 docstring 后加 " \
                        "`import _testenv`）" % missing


def test_cfg_path_follows_env():
    """`cfg_path()` / `Scope.default_path()` 认 AITP_TEST_CFG_DIR。"""
    from core.settings import SCOPES, cfg_path
    tmp = tempfile.mkdtemp(prefix="aitp_cfgcheck_")
    old = os.environ.get("AITP_TEST_CFG_DIR")
    try:
        os.environ["AITP_TEST_CFG_DIR"] = tmp
        p = cfg_path(".train_settings.json")
        assert p == os.path.join(tmp, ".train_settings.json"), p
        scope_path = SCOPES["train"].default_path()
        assert os.path.dirname(scope_path) == tmp, \
            "★ Scope.default_path() 没走隔离目录：%r" % scope_path
    finally:
        if old is None:
            os.environ.pop("AITP_TEST_CFG_DIR", None)
        else:
            os.environ["AITP_TEST_CFG_DIR"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_cfg_path_defaults_to_repo_root():
    """没设环境变量时仍指向仓库根（别把产品行为改掉）。"""
    from core.settings import ROOT_DIR, cfg_path
    old = os.environ.pop("AITP_TEST_CFG_DIR", None)
    try:
        p = cfg_path(".train_settings.json")
        assert p == os.path.join(ROOT_DIR, ".train_settings.json"), p
    finally:
        if old is not None:
            os.environ["AITP_TEST_CFG_DIR"] = old


def test_window_settings_path_isolated():
    """★ 真窗口的 `_train_settings_path()` 必须落在隔离目录里。

    这条钉的是"有人把路径又手拼回去"——本轮之前它就是手拼 `__file__`，
    于是所有建窗测试的 closeEvent 都在写真实的 `.train_settings.json`。
    """
    global _APP, _WIN
    from PyQt5.QtWidgets import QApplication
    import ui.task_queue as tq

    fd, hp = tempfile.mkstemp(prefix="aitp_iso_", suffix=".json")
    os.close(fd)
    os.unlink(hp)
    tq.DEFAULT_HISTORY_PATH = hp

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    _APP = QApplication.instance() or QApplication([])
    init_ui_invoker()
    _WIN = MainWindow()

    cfg_dir = os.environ.get("AITP_TEST_CFG_DIR")
    assert cfg_dir, "前提：_testenv 应该设了 AITP_TEST_CFG_DIR"
    p = _WIN._train_settings_path()
    assert os.path.dirname(p) == cfg_dir, \
        "★ 主窗的配置路径没走隔离目录：%r（真实仓库根配置会被测试改掉）" % p


def cleanup():
    global _WIN
    if _WIN is not None:
        try:
            _WIN.close()
        except Exception:  # noqa: BLE001
            pass
        _WIN = None


def main():
    checks = [
        ("★ 建主窗的测试都 import 了 _testenv", test_every_window_test_imports_testenv),
        ("cfg_path / Scope.default_path 认隔离环境变量", test_cfg_path_follows_env),
        ("没设环境变量时仍指向仓库根", test_cfg_path_defaults_to_repo_root),
        ("★ 主窗配置路径落在隔离目录", test_window_settings_path_isolated),
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
