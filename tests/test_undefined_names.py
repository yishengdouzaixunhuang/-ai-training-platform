# -*- coding: utf-8 -*-
"""「引用了但任何作用域都没定义」的名字扫描（stdlib `symtable`，不 import 被测代码）。

## 为什么要有这条

2026-11-11 那轮把推理的停止判据从 `_active_stop()` 换成"每条路径自带令牌"，
在 `ui/ocr_ocv_mixin.py` 里改了**两处** `stop_check=stop`，却只给其中一个方法加了
`stop = self._infer_stop(stop)` —— `_ocv_batch_inspect` 里的 `stop` 从未定义。

后果：**OCV 批量检测一点就 NameError**，而且是在 `run()` 子线程里抛，
被线程 excepthook 吞掉，用户只看到状态栏卡在 "Batch inspecting..."。

87 项测试全绿没抓到它：队列层/标志层的用例一条都不走 OCV 批量路径。
而上一轮新加的"分派方法必须存在"护栏能扫出**方法名**不存在，扫不出**变量名**不存在 ——
同一个错误换了个形态。所以补这一条：**按作用域链检查所有名字**。

## 原理

`symtable` 给出每个作用域的符号表。某个名字若"被引用、但不是局部/参数/import/闭包变量"，
它就是个**全局查找** —— 那它必须出现在模块级名字里，或者 builtins 里。
两边都没有 ⇒ 运行到那一行必 `NameError`。

## 已知边界（写在用例里，别让假阳性把这条测试关掉）

- `import *` 的模块跳过（名字来源静态看不到）；
- `_attic/`、`vendor/`、`wheels/`、`installers/`、`dist2/` 不扫；
- 模块级动态注入的名字（`globals()["x"] = ...`）会被误报 —— 真遇到就加进 `ALLOW`，
  但**加之前先确认它真的是动态注入**，别拿它当消音器。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_undefined_names.py
"""
import builtins
import os
import symtable
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

#: 代码目录（相对 ROOT）。刻意只扫我们自己的代码。
SCAN_DIRS = ["ui", "core", "training", "inference", "flow", "annotation",
             "classification", "detection", "ocr", "ocv", "tools", "scripts", "tests"]
SKIP_DIRS = {"__pycache__", "_attic", "vendor", "wheels", "installers", "dist2", ".git"}

#: 模块级注入的名字（`python -m` / 解释器给的那几个）
INJECTED = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
            "__loader__", "__builtins__", "__class__", "__cached__"}

#: 允许的例外：{相对路径: {作用域: {名字, ...}}}。加之前先确认真是动态注入。
ALLOW = {}

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def scan_source(src, filename):
    """返回 [(作用域名, 名字)] —— 被引用但任何作用域都没定义的。"""
    top = symtable.symtable(src, filename, "exec")
    module_names = {s.get_name() for s in top.get_symbols()}
    if "*" in module_names:            # import *：名字来源静态看不到，跳过整个文件
        return []
    builtin_names = set(dir(builtins)) | INJECTED
    hits = []

    def walk(tbl):
        if tbl.get_type() != "module":
            for sym in tbl.get_symbols():
                nm = sym.get_name()
                if not sym.is_referenced():
                    continue
                if sym.is_local() or sym.is_parameter() or sym.is_imported() or sym.is_free():
                    continue
                if nm in builtin_names or nm in module_names:
                    continue
                hits.append((tbl.get_name(), nm))
        for child in tbl.get_children():
            walk(child)

    walk(top)
    return hits


def scan_file(path, rel):
    with open(path, "rb") as f:
        src = f.read()
    try:
        hits = scan_source(src, path)
    except SyntaxError as e:
        return [("<syntax>", str(e))]
    allow = ALLOW.get(rel.replace("\\", "/"), {})
    out = []
    for scope, nm in hits:
        if nm in allow.get(scope, set()):
            continue
        out.append((scope, nm))
    return out


def test_scanner_catches_undefined():
    """★ 扫描器自检：它必须能抓住"变量名未定义"这件事，否则这整条测试是空的。"""
    bad = (
        "def outer():\n"
        "    def run():\n"
        "        print(stop_check=stop)\n"
        "    return run\n"
    )
    hits = scan_source(bad, "<bad>")
    names = {nm for _scope, nm in hits}
    assert "stop" in names, "扫描器没抓到未定义的名字：%r" % (hits,)

    good = (
        "def outer(stop=None):\n"
        "    def run():\n"
        "        print(stop)\n"
        "    return run\n"
    )
    assert not scan_source(good, "<good>"), "参数/闭包变量被误报了"


def test_no_undefined_names_in_project():
    """全项目不许有"引用了但没定义"的名字。"""
    offenders = []
    scanned = 0
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, ROOT)
                scanned += 1
                for scope, nm in scan_file(p, rel):
                    offenders.append("%s → %s → %s" % (rel, scope, nm))
    offenders = sorted(set(offenders))     # symtable 偶尔会把同一处报两次，去重后更好读
    assert scanned > 100, "只扫到 %d 个文件，扫描范围不对" % scanned
    assert not offenders, "发现 %d 处未定义名字：\n    %s" % (
        len(offenders), "\n    ".join(offenders[:20]))


def main():
    print("=== 未定义名字扫描 ===")
    checks = [
        ("扫描器自检（能抓到/不误报）", test_scanner_catches_undefined),
        ("全项目无未定义名字", test_no_undefined_names_in_project),
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
