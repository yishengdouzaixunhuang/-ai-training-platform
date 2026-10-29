# -*- coding: utf-8 -*-
"""门禁（Gate）：任何改动合并前必须全绿。

## 为什么需要它

多 AI 协作时，最大的风险不是"改得慢"，而是**"子 AI 说改好了，但其实没好"**。
LLM 自述的"已完成"没有信息量 —— 只有退出码有。

这个脚本就是那条唯一的裁判线：
    编译过 → 单测过 → 冒烟过 → 性能不退 → 没偷改测试
全绿才允许合并，任一红就打回重派。

## 用法

    python scripts/gate.py              # 常规检查（编译 + 单测 + 冒烟 + diff 守卫）
    python scripts/gate.py --fast       # 只查编译，2 秒内出结果（改完随手跑）
    python scripts/gate.py --perf       # 额外跑性能基线（需要 .gate.json 配置工程）
    python scripts/gate.py --allow-protected
                                        # 确实需要动测试/门禁本身时，显式放行

退出码：0 = 全绿；1 = 有红；2 = 环境问题（比如解释器不对）。

## 设计取舍

- **不做并行**：几项加起来不到 30 秒，串行更容易看清哪一步挂。
- **每项独立超时**：一项卡死不能拖住整个门禁（子 AI 很爱写出死循环）。
- **失败只打尾部**：完整日志太长，取最后 25 行足够定位。
- **性能用中位数**：单次会被磁盘抖动带偏（实测同一台机器 1.3s/1.7s 都出现过）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# 业务代码目录（不含 _attic / build / __pycache__）
SRC_DIRS = ["ui", "core", "annotation", "training", "inference", "detection",
            "classification", "flow", "ocr", "ocv", "tools", "interop", "tests"]

# 保护区：子 AI 不许碰。Coder 改了测试就能"改到通过"，那门禁就失去意义了。
PROTECTED = ("tests/", "scripts/gate.py", ".gate.json")

STEP_TIMEOUT = 240


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _run(cmd, timeout=STEP_TIMEOUT, env_extra=None):
    """跑子进程，返回 (ok, output, returncode)。输出合并 stdout/stderr。"""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout, env=env)
        return p.returncode == 0, (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired:
        return False, f"超时（>{timeout}s）", -999
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}", -998


def _tail(text, n=25):
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def _load_cfg():
    p = os.path.join(ROOT, ".gate.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------

def check_compile():
    """① 全量编译：所有业务 py 文件都要能过语法。"""
    import py_compile

    targets = []
    for d in SRC_DIRS:
        targets += glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True)
    targets += glob.glob(os.path.join(ROOT, "*.py"))
    targets = [t for t in targets if "__pycache__" not in t]

    bad = []
    # 编译产物丢临时目录，不污染项目的 __pycache__
    tmp = tempfile.mkdtemp()
    cfile = os.path.join(tmp, "x.pyc")
    for f in targets:
        try:
            py_compile.compile(f, doraise=True, cfile=cfile)
        except Exception as e:  # noqa: BLE001
            bad.append((os.path.relpath(f, ROOT), str(e).splitlines()[0][:120]))

    if bad:
        detail = "\n".join(f"{f} -> {e}" for f, e in bad[:10])
        return "FAIL", f"{len(bad)} 个文件编译失败", detail
    return "PASS", f"{len(targets)} 文件", ""


def check_unit_tests():
    """② 单元测试：tests/test_*.py 全部要绿。"""
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    # perf_open_project.py 是探针不是测试，跳过
    files = [f for f in files if not os.path.basename(f).startswith("perf_")]
    if not files:
        return "SKIP", "没找到测试", ""

    env = {"QT_QPA_PLATFORM": "offscreen"}
    failed, passed = [], 0
    for f in files:
        ok, out, _rc = _run([PY, f], env_extra=env)
        name = os.path.basename(f)
        if ok:
            passed += 1
        else:
            failed.append((name, _tail(out, 12)))

    if failed:
        detail = "\n\n".join(f"--- {n} ---\n{o}" for n, o in failed)
        return "FAIL", f"{len(failed)}/{len(files)} 套挂了", detail
    return "PASS", f"{len(files)} 套", ""


def check_smoke():
    """③ 启动冒烟：MainWindow 能不能起来，跨线程转发器能不能用。"""
    f = os.path.join(ROOT, "tests", "test_smoke_startup.py")
    if not os.path.exists(f):
        return "SKIP", "无冒烟脚本", ""
    ok, out, _rc = _run([PY, f], env_extra={"QT_QPA_PLATFORM": "offscreen"})
    if ok:
        # 脚本自己会打 "4/4 项通过"
        hit = [l for l in out.splitlines() if "项通过" in l]
        return "PASS", (hit[-1].strip() if hit else "通过"), ""
    return "FAIL", "冒烟未通过", _tail(out, 25)


def check_perf(cfg):
    """④ 性能基线：打开工程耗时不得突破门限。"""
    perf_cfg = (cfg or {}).get("perf") or {}
    project = perf_cfg.get("project")
    if not project:
        return "SKIP", "未配置（.gate.json 里填 perf.project）", ""

    max_open = float(perf_cfg.get("max_open_ms", 2000))
    max_filter = float(perf_cfg.get("max_filter_ms", 900))
    repeat = int(perf_cfg.get("repeat", 3))

    ok, out, rc = _run([PY, os.path.join(ROOT, "tests", "perf_open_project.py"),
                        "--project", project, "--repeat", str(repeat)],
                       timeout=300, env_extra={"QT_QPA_PLATFORM": "offscreen"})
    if not ok:
        return "FAIL", f"探针执行失败（退出码 {rc}）", _tail(out, 20)

    try:
        data = json.loads([l for l in out.strip().splitlines() if l.startswith("{")][-1])
    except Exception:
        return "FAIL", "探针输出不是 JSON", _tail(out, 20)

    o, fl = data["open_ms"], data["filter_ms"]
    detail = (f"打开 {o:.0f}ms（限 {max_open:.0f}）/ 列表 {fl:.0f}ms（限 {max_filter:.0f}）"
              f" / {data.get('rows', '?')} 行 / 样本 {data.get('samples')}")
    over = []
    if o > max_open:
        over.append(f"打开超限 {o - max_open:.0f}ms")
    if fl > max_filter:
        over.append(f"列表超限 {fl - max_filter:.0f}ms")
    if over:
        return "FAIL", "；".join(over), detail
    return "PASS", f"打开 {o/1000:.2f}s", detail


def check_protected(allow):
    """⑤ diff 守卫：Coder 不许改测试 / 门禁本身。"""
    if allow:
        return "SKIP", "已显式放行（--allow-protected）", ""

    ok, out, _rc = _run(["git", "status", "--porcelain"])
    if not ok:
        return "SKIP", "不是 git 仓库或 git 不可用", ""

    touched = []
    for line in out.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        # 「新增」不算越界 —— Tester 加新测试本来就是它的活儿。
        # 真正危险的是**修改/删除**已有测试：那才是"Coder 把测试改成通过"。
        if status.strip() in ("A", "??"):
            continue
        # 格式：XY path  或  XY old -> new
        path = line[3:].split("->")[-1].strip().strip('"')
        norm = path.replace("\\", "/")
        for p in PROTECTED:
            if norm == p or norm.startswith(p):
                touched.append(norm)
                break

    if touched:
        uniq = sorted(set(touched))
        return ("FAIL", f"修改了保护区 {len(uniq)} 个文件",
                "以下文件属于保护区，不允许被修改/删除：\n  "
                + "\n  ".join(uniq)
                + "\n\nCoder 改测试 = 门禁失效。若确有必要（如修测试自身 bug），"
                  "显式加 --allow-protected 重跑并在提交信息里说明。")
    return "PASS", "未触碰保护区", ""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

BAR = "─" * 62


def main():
    ap = argparse.ArgumentParser(description="门禁：改动合并前必须全绿")
    ap.add_argument("--fast", action="store_true", help="只跑编译（秒级）")
    ap.add_argument("--perf", action="store_true", help="加跑性能基线")
    ap.add_argument("--allow-protected", action="store_true",
                    help="允许改动测试/门禁本身")
    args = ap.parse_args()

    cfg = _load_cfg()

    steps = []
    # ★ 性能必须**第一个跑**：前面的编译/单测/冒烟会占满磁盘缓存和内存，
    #   实测同一份未改动的代码，跑在测试之后是 2921ms、跑在最初是 1774ms ——
    #   差 1.1 秒全是干扰，这样的门禁会天天误报。
    if not args.fast and args.perf:
        steps.append(("性能基线", lambda: check_perf(cfg)))

    steps.append(("全量编译", lambda: check_compile()))

    if not args.fast:
        steps += [
            ("单元测试", lambda: check_unit_tests()),
            ("启动冒烟", lambda: check_smoke()),
        ]
        if not args.perf:
            steps.append(("性能基线", lambda: ("SKIP", "未启用（加 --perf）", "")))
    # diff 守卫放最后，且 --fast 也跑：它只读 git status，0.0s，
    # 但能在"改完随手一跑"时就发现动了测试，不用等整套跑完。
    steps.append(("diff 守卫", lambda: check_protected(args.allow_protected)))

    print()
    print(f" 门禁 gate  ·  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(BAR)

    results = []
    t_all = time.perf_counter()
    for i, (name, fn) in enumerate(steps, 1):
        t0 = time.perf_counter()
        try:
            status, brief, detail = fn()
        except Exception as e:  # noqa: BLE001
            status, brief, detail = "FAIL", f"检查本身出错：{type(e).__name__}: {e}", ""
        dt = time.perf_counter() - t0

        dots = "." * max(2, 20 - len(name) * 2)
        print(f" [{i}/{len(steps)}] {name} {dots} {status:<5} {brief}  {dt:.1f}s")
        if detail and status != "PASS":
            for line in detail.splitlines():
                print(f"        {line}")
            print()
        results.append({"name": name, "status": status, "brief": brief, "ms": dt * 1000})

    total = time.perf_counter() - t_all
    print(BAR)

    fails = [r for r in results if r["status"] == "FAIL"]
    skips = [r for r in results if r["status"] == "SKIP"]

    # 机器可读的一行，方便调度者解析而不必读整屏
    print("GATE_RESULT " + json.dumps({
        "ok": not fails,
        "failed": [r["name"] for r in fails],
        "skipped": [r["name"] for r in skips],
        "total_ms": round(total * 1000),
    }, ensure_ascii=False))

    if fails:
        print(f" 结果：{len(fails)} 项未通过（{', '.join(r['name'] for r in fails)}）—— 打回重派")
        print()
        return 1

    note = f"，{len(skips)} 项跳过" if skips else ""
    print(f" 结果：全绿{note}，用时 {total:.1f}s —— 可以合并")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
