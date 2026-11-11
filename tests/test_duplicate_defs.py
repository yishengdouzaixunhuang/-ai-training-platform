# -*- coding: utf-8 -*-
"""同名定义扫描：同一作用域里定义两次同名函数 = 后一个静默覆盖前一个。

覆盖**两个作用域**：
  · 类体   —— `class A:` 里两个 `def f`（2026-11-11 那次事故：`reset_view`）
  · 模块级 —— 文件里两个 `def f`（★ 2026-11-11 审核补充：测试文件里两个
              `def test_xxx` 会让**前一条用例静默消失**，数量对不上却没人发现，
              症状和类内重名一模一样）

★ 为什么要有这个文件（那两次事故）：

  ① 把"切换工程时清画布"的字段清单从 `MainWindow` 搬进 `AnnotationCanvas.reset_view()`，
     而本类末尾**早就有一个** `reset_view()`（工具栏「刷新视图」，只做 `_fit_to_window()`）。
     Python 类体里后定义的覆盖先定义的 ⇒ 新写的 58 行清理**一行都没执行**，
     主窗口那 15 个 `setattr` 又已经删掉了 ⇒ 切换工程时**一个字段都没清**（比改前更糟）。

  ② 审核指出：`tests/` 里两个同名 `def test_xxx` 的效果完全一样 ——
     前一条用例不会报错、不会警告，只是**再也不会被执行**。
     `test_project_switch_view.py` 已经 400 多行 10 条用例，以后只会更多。

  这两种错误都没有运行时异常、没有警告、测试也全绿 —— 只能靠静态扫描发现。

用法：
    C:/aitp_venv/Scripts/python.exe -u tests/test_duplicate_defs.py

与 `test_undefined_names.py` 互补：那个查"名字不存在"，这个查"名字存在两份"。
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
_LEGIT_NAMES = {"overload", "property"}
MODULE_SCOPE = "<module>"

# 顺便把"解析失败"探出来：解析不了的文件会被静默跳过，若大半个项目都解析失败，
# 扫描结果永远是 0 处 —— 那就是个假绿的扫描器（BOM 那次就踩在这上面）。
PARSE_FAILURES = []


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
    if any(d in _LEGIT_NAMES for d in decos):
        return True                                    # @property + @x.setter
    if any(d.endswith(_PROP_SUFFIX) for d in decos):
        return True
    if any(d == "overload" or d.endswith(".register") for d in decos):
        return True                                    # typing.overload / singledispatch
    return False


def _group(body):
    """把一段 body 里**直接**的函数定义按名字分组。

    只看直接子节点：`if sys.platform == ...: def f()` / `try: ... except: def f()`
    这类条件定义在 If/Try 节点里，不会进这里 —— 它们是合法的多份定义。
    """
    groups = {}
    for item in body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            groups.setdefault(item.name, []).append(item)
    return groups


def _collect(scope_name, body, bad):
    for name, funcs in _group(body).items():
        if len(funcs) < 2 or _is_legit_pair(funcs):
            continue
        bad.append((scope_name, name, [f.lineno for f in funcs]))


def scan_source(src, filename):
    """返回 [(作用域名, 函数名, [行号...])]，只报**真覆盖**。

    作用域名是类名，或 `<module>`（模块级同名函数）。
    """
    try:
        tree = ast.parse(src, filename)
    except SyntaxError as e:
        PARSE_FAILURES.append((filename, getattr(e, "lineno", 0), str(e)))
        return []

    bad = []
    _collect(MODULE_SCOPE, tree.body, bad)             # ★ 模块级
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _collect(node.name, node.body, bad)        # ★ 类体内
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


def _scan_all():
    PARSE_FAILURES.clear()          # 每次重扫都从零开始，别把上次的累积进来
    hits = []
    scanned = 0
    for path in iter_files():
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        if rel.startswith("_probe_"):
            continue
        try:
            with open(path, "rb") as f:
                src = f.read()      # ★ 传 bytes：让 ast 自己按 coding cookie / BOM 解码。
        except OSError:             #   用 io.open(encoding="utf-8") 读会把 BOM 留成 U+FEFF，
            continue                #   全项目一半文件都会"语法错误"（踩过）。
        scanned += 1
        for scope, name, lines in scan_source(src, rel):
            hits.append((rel, scope, name, lines))
    return hits, scanned


# ---------------------------------------------------------------- 自检

def test_scanner_catches_class_duplicate():
    """自检 1：类体内真重名必须报出来。

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
    scope, name, lines = hits[0]
    assert (scope, name) == ("A", "run"), "自检失败：报错了对象 %r" % ((scope, name),)
    assert lines == [2, 6], "自检失败：行号不对 %r" % (lines,)


def test_scanner_catches_module_duplicate():
    """自检 2：**模块级**同名函数也必须报 —— 就是测试文件两个 `def test_xxx` 那种。

    ★ 审核 2026-11-11 指出的场景：前一条用例会被静默覆盖，条数变少但没人发现。
      第一版扫描器只走 `ClassDef`，这类根本抓不到（哪怕 tests/ 在扫描范围内）。
    """
    src = (
        "def test_a():\n"
        "    return 1\n"
        "\n"
        "def test_b():\n"
        "    return 2\n"
        "\n"
        "def test_a():\n"
        "    return 3\n"
    )
    hits = scan_source(src, "<self-check-2>")
    assert len(hits) == 1, "自检失败：模块级重名没报出来，实际 %r" % (hits,)
    scope, name, lines = hits[0]
    assert scope == MODULE_SCOPE, "自检失败：作用域应标成 <module>，实际 %r" % (scope,)
    assert name == "test_a", "自检失败：报错了对象 %r" % (name,)
    assert lines == [1, 7], "自检失败：行号不对 %r" % (lines,)


def test_scanner_ignores_property_and_overload():
    """自检 3：property 的 getter/setter 对、typing.overload 都不许误报（模块级也要放过）。"""
    src = (
        "from typing import overload\n"
        "\n"
        "@overload\n"
        "def g(a: int) -> int: ...\n"
        "@overload\n"
        "def g(a: str) -> str: ...\n"
        "def g(a):\n"
        "    return a\n"
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
    hits = scan_source(src, "<self-check-3>")
    assert not hits, "自检失败：把 property/overload 误报成覆盖：%r" % (hits,)


def test_scanner_ignores_conditional_definitions():
    """自检 4：`if 平台:` / `try-except` 里的同名 def 是合法的，不许误报。"""
    src = (
        "import sys\n"
        "if sys.platform == 'win32':\n"
        "    def open_path(p):\n"
        "        return 1\n"
        "else:\n"
        "    def open_path(p):\n"
        "        return 2\n"
    )
    hits = scan_source(src, "<self-check-4>")
    assert not hits, "自检失败：条件定义被误报：%r" % (hits,)


# ---------------------------------------------------------------- 全项目

def test_tests_dir_is_scanned():
    """★ `tests/` 必须在扫描范围内 —— 否则测试文件里的重名永远查不到。

    ★ 审核建议把 tests/ 纳入；查下来它本来就在 `SCAN_DIRS` 里（19 个文件），
      但**第一版扫描器只走 ClassDef**，所以"在范围内"也没用。
      这条用例同时钉住目录范围，防止将来谁把 tests/ 从 SCAN_DIRS 里删掉。
    """
    files = list(iter_files())
    under_tests = [p for p in files if os.sep + "tests" + os.sep in p]
    assert len(under_tests) >= 10, \
        "tests/ 下只扫到 %d 个 .py —— 扫描范围被改小了？" % len(under_tests)
    names = {os.path.basename(p) for p in under_tests}
    for must in ("test_project_switch_view.py", "test_task_isolation.py",
                 "test_undefined_names.py"):
        assert must in names, "tests/%s 不在扫描范围内" % must


def test_source_parse_failures_are_zero():
    """扫描前所有文件都必须能解析 —— 否则"0 处重名"可能只是"一个文件都没读进来"。

    ★ 这条是血的教训：第一版用 `io.open(encoding="utf-8")` 读源码，本项目一半文件带
      UTF-8 BOM（读成 str 后是 U+FEFF）⇒ 全部报 SyntaxError ⇒ 扫描器静默返回 0 处 ⇒
      测试照样绿。
    """
    hits, scanned = _scan_all()
    assert scanned > 150, "只扫到 %d 个文件，范围异常" % scanned
    assert not PARSE_FAILURES, (
        "有 %d 个文件解析不了，扫描结果不可信：\n  " % len(PARSE_FAILURES)
        + "\n  ".join("%s:%s %s" % (f, ln, msg[:70]) for f, ln, msg in PARSE_FAILURES[:10])
    )
    return hits


def test_no_real_duplicate_in_project():
    """全项目（模块级 + 类内）不许有真覆盖。"""
    hits, scanned = _scan_all()
    assert not hits, (
        "★ 同一作用域里有同名函数，后定义的静默覆盖前一个（被覆盖的那份永不执行）：\n  "
        + "\n  ".join(sorted("%s → %s → def %s %r" % h for h in hits))
        + "\n如果确实是 property/overload/条件定义，请在 _is_legit_pair 里补规则并写清理由。"
    )
    print("     （扫了 %d 个 .py，模块级与类内均无重名覆盖）" % scanned)


def main():
    print("=== 同名定义扫描（模块级 + 类内）===")
    checks = [
        ("扫描器自检：抓得到类内重名", test_scanner_catches_class_duplicate),
        ("扫描器自检：抓得到模块级重名", test_scanner_catches_module_duplicate),
        ("扫描器自检：不误报 property/overload", test_scanner_ignores_property_and_overload),
        ("扫描器自检：不误报条件定义", test_scanner_ignores_conditional_definitions),
        ("tests/ 在扫描范围内", test_tests_dir_is_scanned),
        ("所有文件都能解析", test_source_parse_failures_are_zero),
        ("全项目无同名覆盖", test_no_real_duplicate_in_project),
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
