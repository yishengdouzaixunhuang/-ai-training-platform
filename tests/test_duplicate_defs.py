# -*- coding: utf-8 -*-
"""类内重名方法扫描：同一个类里定义两次同名方法 = 后一个静默覆盖前一个。

★ 为什么要有这个文件（2026-11-11 审核驳回那次）：

  把"切换工程时清画布"的字段清单从 `MainWindow` 搬进 `AnnotationCanvas.reset_view()`，
  结果 `AnnotationCanvas` 末尾**早就有一个** `reset_view()`（工具栏「刷新视图」，
  只做 `_fit_to_window()`）。Python 类体里后定义的覆盖先定义的 ⇒ 新写的 58 行清理
  **一行都没执行**，而主窗口那 15 个 `setattr` 已经删掉了。
  净效果：切换工程时**一个字段都没清**，比改之前更糟（改之前至少清 15 个）。

  这个错误没有任何运行时报错、没有警告、测试也全绿 —— 只能靠静态扫描发现。

用法：
    C:/aitp_venv/Scripts/python.exe -u tests/test_duplicate_defs.py

它是 `test_undefined_names.py` 的姊妹：那个查"名字不存在"，这个查"名字存在两份"。
"""
import ast
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SCAN_DIRS = ("ui", "core", "training", "inference", "flow", "annotation",
             "classification", "detection", "ocr", "ocv", "tools", "scripts", "tests")
SKIP_DIRS = {"__pycache__", "_attic", "vendor", "wheels", "installers", "dist2",
             ".git", ".obsidian", "node_modules"}

# 允许同名的装饰器：property 家族的 getter/setter/deleter、typing.overload、
# functools.singledispatch 的 register —— 这些是**语言特性**，不是覆盖。
_PROP_SUFFIX = (".setter", ".getter", ".deleter")


def _deco_name(node):
    """把装饰器表达式压成字符串，便于判断。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _deco_name(node.value)
        return (base + "." if base else "") + node.attr
    if isinstance(node, ast.Call):
        return _deco_name(node.func)
    return ""


def _is_legit_pair(funcs):
    """同一名字的多个定义，是否属于语言允许的情形。"""
    decos = []
    for f in funcs:
        decos.extend(_deco_name(d) for d in f.decorator_list)
    if any(d == "property" for d in decos):
        return True                                    # @property + @x.setter
    if any(d.endswith(_PROP_SUFFIX) for d in decos):
        return True
    if any(d == "overload" or d.endswith(".register") for d in decos):
        return True                                    # typing.overload / singledispatch
    return False


def scan_source(src, filename):
    """返回 [(类名, 方法名, [行号...])]，只报**真覆盖**。"""
    try:
        tree = ast.parse(src, filename)
    except SyntaxError:
        return []          # 解析不了的文件不属于"重名"范畴；语法错由 py_compile 负责

    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        groups = {}
        for item in node.body:                          # 只看类体直接子节点
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                groups.setdefault(item.name, []).append(item)
        for name, funcs in groups.items():
            if len(funcs) < 2:
                continue
            if _is_legit_pair(funcs):
                continue
            bad.append((node.name, name, [f.lineno for f in funcs]))
    return bad


def iter_files():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [n for n in dirnames if n not in SKIP_DIRS]
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    yield os.path.join(dirpath, fn)


def test_scanner_catches_real_duplicate():
    """自检 1：真有同名方法时必须报出来。

    ⚠️ 没有自检的静态检查最危险：写错了就返回空列表，测试永远是绿的。
    """
    src = (
        "class A:\n"
        "    def run(self):\n"
        "        return 1\n"
        "    def other(self):\n"
        "        return 2\n"
        "    def run(self):\n"
        "        return 3\n"
    )
    hits = scan_source(src, "<self-check-1>")
    assert len(hits) == 1, "自检失败：应当报 1 处重名，实际 %r" % (hits,)
    cls, name, lines = hits[0]
    assert (cls, name) == ("A", "run"), "自检失败：报错了对象 %r" % ((cls, name),)
    assert lines == [2, 6], "自检失败：行号不对 %r" % (lines,)


def test_scanner_ignores_property_and_overload():
    """自检 2：property 的 getter/setter 对、typing.overload 都不许误报。"""
    src = (
        "from typing import overload\n"
        "\n"
        "class B:\n"
        "    @property\n"
        "    def x(self):\n"
        "        return self._x\n"
        "\n"
        "    @x.setter\n"
        "    def x(self, v):\n"
        "        self._x = v\n"
        "\n"
        "    @overload\n"
        "    def f(self, a: int) -> int: ...\n"
        "    @overload\n"
        "    def f(self, a: str) -> str: ...\n"
        "    def f(self, a):\n"
        "        return a\n"
    )
    hits = scan_source(src, "<self-check-2>")
    assert not hits, "自检失败：把 property/overload 误报成覆盖：%r" % (hits,)


def test_no_real_duplicate_in_project():
    """全项目类内不许有真覆盖。"""
    offenders = []
    scanned = 0
    for path in iter_files():
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        if rel.startswith(("_probe_", "tests/test_duplicate_defs")):
            continue
        try:
            with open(path, "rb") as f:
                src = f.read()        # ★ 传 bytes：让 ast 自己按 coding cookie / BOM 解码。
        except OSError:               #   用 io.open(encoding="utf-8") 读会把 BOM 留成 U+FEFF，
            continue                  #   全项目一半文件都会"语法错误"（踩过）。
        scanned += 1
        for cls, name, lines in scan_source(src, rel):
            offenders.append("%s → class %s → def %s %r" % (rel, cls, name, lines))
    assert not offenders, (
        "★ 同一类里有同名方法，后定义会静默覆盖前一个（被覆盖的代码永不执行）：\n  "
        + "\n  ".join(sorted(set(offenders)))
        + "\n如果确实是 property/overload，请在 _is_legit_pair 里补规则并写清理由。"
    )
    print("     （扫了 %d 个 .py，无类内重名覆盖）" % scanned)


def main():
    print("=== 类内重名方法扫描 ===")
    checks = [
        ("扫描器自检：抓得到真重名", test_scanner_catches_real_duplicate),
        ("扫描器自检：不误报 property/overload", test_scanner_ignores_property_and_overload),
        ("全项目无类内重名覆盖", test_no_real_duplicate_in_project),
    ]
    failures = 0
    for name, fn in checks:
        try:
            fn()
            print("[PASS] %s" % name)
        except AssertionError as e:
            failures += 1
            print("[FAIL] %s\n        %s" % (name, e))
        except Exception as e:                      # noqa: BLE001
            failures += 1
            print("[FAIL] %s\n        %s: %s" % (name, type(e).__name__, e))
    print("\n%s  %d/%d 通过" % ("✅ 全部通过" if not failures else "❌ 有失败",
                               len(checks) - failures, len(checks)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
