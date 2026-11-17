# -*- coding: utf-8 -*-
"""全量编译扫描：每个 .py 都要能编译成字节码（补「验收三件套」的盲区）。

为什么需要它：
    `test_duplicate_defs` / `test_undefined_names` 只用 `ast.parse()`，
    而 `from __future__` 位置、`return` 外语句这类错误**只有编译期才报**。
    真实事故（2026-11-17 审核）：`tests/perf_open_project.py` 把 `import _testenv`
    写在 `from __future__ import annotations` 之前 —— `ast.parse` 通过、
    `py_compile` 报错 ⇒ 门禁第一步「全量编译」永远是红的，而日常验收看不见。

覆盖范围与 `scripts/gate.py` 保持一致（业务目录 + tests + 根目录脚本），
排除 `vendor/` `_attic/` `dist2/` `wheels/` `installers/`（第三方与历史归档）。

⚠️ 编译产物写临时目录，不许污染项目的 `__pycache__`。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import py_compile
import tempfile

SKIP_DIRS = {".git", "vendor", "wheels", "installers", "__pycache__",
             "_attic", "dist2", "logs", ".obsidian"}
MIN_EXPECTED = 200      # 守卫：目录遍历要是坏了，扫出 0 个文件也会"全绿"


def iter_sources():
    for dp, dn, fs in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for fn in fs:
            if fn.endswith(".py"):
                yield os.path.join(dp, fn)


def test_oracle_catches_future_import_position():
    """自检：先证明这个判据抓得到那条真实事故（否则永远绿也没意义）。"""
    bad = "import os\nfrom __future__ import annotations\n"
    try:
        compile(bad, "<bad-future>", "exec")
    except SyntaxError:
        return
    raise AssertionError("compile() 居然接受了「__future__ 不在最前」的源码")


def test_all_files_compile():
    files = sorted(iter_sources())
    assert len(files) >= MIN_EXPECTED, f"只扫到 {len(files)} 个 .py，目录遍历可能坏了"

    tmp = tempfile.mkdtemp(prefix="aitp_compile_")
    cfile = os.path.join(tmp, "x.pyc")
    bad = []
    for f in files:
        try:
            py_compile.compile(f, doraise=True, cfile=cfile)
        except Exception as e:  # noqa: BLE001
            bad.append((os.path.relpath(f, ROOT), str(e).splitlines()[0][:160]))

    assert not bad, "%d 个文件编译失败：\n%s" % (
        len(bad), "\n".join("  %s -> %s" % (n, e) for n, e in bad[:10]))


def test_scan_covers_tests_and_entrypoints():
    """`tests/` 与根目录入口（main.py/launcher.py）必须在扫描范围内。"""
    files = [os.path.relpath(f, ROOT).replace("\\", "/") for f in iter_sources()]
    assert any(p.startswith("tests/") for p in files), "tests/ 不在扫描范围"
    assert "main.py" in files and "launcher.py" in files, "根目录入口不在扫描范围"


def main():
    tests = [
        ("判据自检：抓得到 __future__ 位置错误", test_oracle_catches_future_import_position),
        ("全量 py_compile 无失败", test_all_files_compile),
        ("范围：tests/ 与根目录入口被覆盖", test_scan_covers_tests_and_entrypoints),
    ]
    passed = 0
    print("=== 全量编译扫描 ===")
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print()
    if passed == len(tests):
        print(f"✅ 全部通过  {passed}/{len(tests)} 通过")
        return 0
    print(f"❌ {len(tests) - passed}/{len(tests)} 项未通过")
    return 1


if __name__ == "__main__":
    sys.exit(main())
