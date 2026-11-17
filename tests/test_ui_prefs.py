# -*- coding: utf-8 -*-
"""UI 偏好落盘（P0）：缺文件给默认值、写入原子、坏 JSON 不覆盖原文件。

运行：C:\\aitp_venv\\Scripts\\python.exe tests\\test_ui_prefs.py
"""
import json
import os
import sys
import tempfile

import _testenv  # noqa: F401  配置目录隔离（AITP_TEST_CFG_DIR）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

failures = []


def check(name, fn):
    try:
        fn()
        print("  [OK]   %s" % name)
    except Exception as e:
        failures.append((name, e))
        print("  [FAIL] %s: %s" % (name, e))


def _mod():
    try:
        from core import ui_prefs
    except Exception as e:
        raise AssertionError("core/ui_prefs.py 不存在或导入失败：%s" % e)
    return ui_prefs


def _tmp_path():
    fd, path = tempfile.mkstemp(suffix=".json", text=True)
    os.close(fd)
    os.remove(path)
    return path


def test_missing_file_defaults():
    """文件不存在：返回默认值，语言默认 zh。"""
    P = _mod()
    path = _tmp_path()
    d = P.load(path)
    assert d.get("lang") == "zh", "默认 lang 应为 zh，实际 %r" % d.get("lang")
    assert d.get("theme") == "dark"
    assert not os.path.exists(path), "只读 load 不应创建文件"


def test_save_and_reload():
    """写入 lang 后能读回，且不丢默认键。"""
    P = _mod()
    path = _tmp_path()
    P.save({"lang": "en"}, path)
    d = P.load(path)
    assert d["lang"] == "en"
    assert d["theme"] == "dark"


def test_unknown_key_preserved():
    """未知键不许被一次 save 抹掉。"""
    P = _mod()
    path = _tmp_path()
    P.save({"foo": 1, "lang": "zh"}, path)
    P.save({"lang": "en"}, path)
    d = P.load(path)
    assert d.get("foo") == 1, "未知键 foo 被抹掉了：%r" % d
    assert d["lang"] == "en"


def test_atomic_no_tmp_left():
    """原子写：落点旁不残留 .tmp。"""
    P = _mod()
    path = _tmp_path()
    P.save({"tiled": True}, path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp"), "残留 .tmp"


def test_corrupt_json_not_overwritten():
    """坏 JSON：load 返回默认值，但不改动磁盘原文件。"""
    P = _mod()
    path = _tmp_path()
    raw = b"{ this is broken "
    with open(path, "wb") as f:
        f.write(raw)
    d = P.load(path)
    assert d.get("lang") == "zh"
    with open(path, "rb") as f:
        assert f.read() == raw, "load 把坏文件覆盖了"


def test_default_path_honors_test_env():
    """默认落点必须认 AITP_TEST_CFG_DIR，且写它不能碰仓库根真实配置。"""
    P = _mod()
    cfg_dir = os.environ.get("AITP_TEST_CFG_DIR")
    assert cfg_dir, "AITP_TEST_CFG_DIR 未设置（tests/_testenv.py 没生效）"
    p = P.default_path()
    assert os.path.abspath(p) == os.path.abspath(
        os.path.join(cfg_dir, ".ui_settings.json")), \
        "默认路径没走测试配置目录：%r" % p

    repo_cfg = os.path.join(ROOT, ".ui_settings.json")
    before = None
    if os.path.exists(repo_cfg):
        with open(repo_cfg, "rb") as f:
            before = f.read()

    P.save({"probe_key": 123})
    assert os.path.isfile(p), "隔离目录里没写出配置文件"
    with open(p, "r", encoding="utf-8") as f:
        assert json.load(f)["probe_key"] == 123

    if before is not None:
        with open(repo_cfg, "rb") as f:
            assert f.read() == before, "★ ui_prefs 写了仓库根的真实 .ui_settings.json"


def main():
    print("=== UI 偏好落盘 ===")
    check("缺文件给默认值", test_missing_file_defaults)
    check("写入后可读回", test_save_and_reload)
    check("未知键保留", test_unknown_key_preserved)
    check("原子写无残留", test_atomic_no_tmp_left)
    check("坏 JSON 不覆盖", test_corrupt_json_not_overwritten)
    check("默认路径认测试隔离", test_default_path_honors_test_env)
    print()
    if failures:
        print("❌ %d/6 项未通过" % len(failures))
        return 1
    print("✅ 全部通过  6/6 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
