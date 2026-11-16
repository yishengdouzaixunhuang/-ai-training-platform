# -*- coding: utf-8 -*-
"""性能探针：用真实工程测「打开工程」耗时，输出 JSON 给门禁判红绿。

为什么单独一个进程：
    它要 import torch + 建 MainWindow（几百 MB、几秒），
    如果塞进门禁进程里跑，会把后面的检查全部拖慢，
    而且 Qt 全局状态一旦建立就不好干净退出。
    独立进程跑完即退，互不影响。

用法：
    python tests/perf_open_project.py --project "工程名" [--repeat 3]
输出（stdout 最后一行）：
    {"open_ms": 1350, "filter_ms": 843, "rows": 1690, ...}
失败时退出码非 0，并在 stderr 说明原因。
"""
import _testenv  # noqa: F401  无显示器 + 配置目录隔离（见 tests/_testenv.py）
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="工程名（ProjectManager 能打开的）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="重复次数。⚠️ 默认 1：实测同一个进程里连续建第 2/3 个 MainWindow "
                         "会逐次变慢（1.8s → 6.1s → 3.9s），内存和 GC 压力累积，"
                         "多测反而失真。要对比请分多次独立进程跑。")
    ap.add_argument("--timeout", type=float, default=120.0, help="单次等待上限（秒）")
    args = ap.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    os.chdir(root)

    # ⚠️ 必须最先 import torch（见 main.py 顶部注释：否则 WinError 1114 c10.dll）
    try:
        import torch  # noqa: F401
    except OSError as e:
        print(f"torch 加载失败：{e}", file=sys.stderr)
        return 2

    from PyQt5.QtWidgets import QApplication

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    app = QApplication.instance() or QApplication(sys.argv)
    init_ui_invoker()

    results = []
    for i in range(max(1, args.repeat)):
        w = MainWindow()
        # 每次都要一个新的 gen，避免上次的缓存把这次测快了
        t0 = time.perf_counter()
        try:
            w._open_project_by_name(args.project)
        except Exception as e:
            print(f"打开工程失败：{type(e).__name__}: {e}", file=sys.stderr)
            return 3

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            app.processEvents()
            time.sleep(0.005)
            if getattr(w, "_last_filter_ms", None) is not None:
                break
        else:
            print("等待列表超时", file=sys.stderr)
            return 4

        open_ms = (time.perf_counter() - t0) * 1000.0
        filter_ms = float(getattr(w, "_last_filter_ms", 0) or 0)
        rows = int(getattr(w, "_last_filter_rows", 0) or 0)
        results.append({"open_ms": open_ms, "filter_ms": filter_ms, "rows": rows})

        w.close()
        app.processEvents()
        # 尽量把上一个窗口的内存还掉，否则下一次测量会被 GC 拖慢
        del w
        import gc
        gc.collect()

    out = {
        "project": args.project,
        "repeat": len(results),
        # 用**最小值**判红绿：这台机器上同一份代码在空闲时 1.8s、
        # 有人开着软件在跑时 3.0s，中位数/均值会被负载带偏而天天误报。
        # 最小值代表"这台机器能做到多快"，正好用来拦"回到 6 秒/64 秒"那种真退化。
        "open_ms": round(min(r["open_ms"] for r in results), 1),
        "filter_ms": round(min(r["filter_ms"] for r in results), 1),
        "open_median": round(statistics.median(r["open_ms"] for r in results), 1),
        "rows": results[0]["rows"],
        "samples": [round(r["open_ms"], 1) for r in results],
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
