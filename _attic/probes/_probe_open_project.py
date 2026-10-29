# -*- coding: utf-8 -*-
"""拆解「双击打开工程」的耗时（2026-10-28）。

用户反馈：工程列表里双击打开工程很卡。实测打开 489 张图的工程要 **25+ 秒**。

做法：给每个可疑方法套一层计时包装，跑一次真实打开，把各段耗时按顺序打出来。
只读：不保存标注、不改工程注册表。
"""
import time

import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication([])

from ui.main_window import MainWindow  # noqa: E402

PROJ = "验证_极耳层数_Flow66_Tao_20261020"

LOG = []


def wrap(win, name, label=None):
    """给 win.<name> 套计时（实例级覆盖）。"""
    orig = getattr(win, name, None)
    if orig is None:
        return
    label = label or name

    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            if dt > 1.0:            # 只记 >1ms 的，避免刷屏
                LOG.append((label, dt))

    setattr(win, name, timed)


def pump(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.003)


def main():
    print("构造主窗口 …")
    t0 = time.perf_counter()
    win = MainWindow()
    t_ctor = (time.perf_counter() - t0) * 1000
    win.resize(1600, 980)
    win.show()
    pump(0.5)
    print("  MainWindow() 构造 + 首屏 %.0f ms" % t_ctor)
    if getattr(win, "current_project", None):
        print("  ⚠ 构造时自动打开了工程：%s" % win.current_project.get("name"))

    # 先关掉可能已打开的工程，保证下面测的是"从零打开"
    try:
        win._close_project()
        pump(0.3)
    except Exception as e:
        print("  关闭工程失败:", e)

    for n in ("open_project", "_load_classes", "_set_task", "_load_image_list_async",
              "_populate_image_list", "_filter_images", "_render_page",
              "_refresh_image_info", "_refresh_annotation_views", "_refresh_versions",
              "_load_output_stats_async", "show_right_tab", "show_left_tab",
              "_fit_list_height", "_schedule_fit_list_height", "_update_accordion_counts"):
        wrap(win, n)
    # ProjectManager 上的也包一层
    wrap(win.pm, "open_project", "pm.open_project")

    print("\n双击打开工程（%s）…" % PROJ)
    LOG.clear()
    t0 = time.perf_counter()
    win._open_project_by_name(PROJ)

    # 等到图片列表真的填满（异步扫描完成的标志）
    deadline = time.time() + 90
    rows = 0
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
        try:
            rows = win.image_list_widget.rowCount()
        except Exception:
            rows = 0
        if rows > 0 and time.time() - t0 > 1.0:
            # 再多等 1.5 秒，把随后的刷新定时器/统计跑完
            pump(1.5)
            break
    total = (time.perf_counter() - t0) * 1000

    print("\n各段耗时（>1ms 才记，按发生顺序）")
    print("  %-34s %10s" % ("步骤", "ms"))
    print("  " + "-" * 46)
    for label, dt in LOG:
        print("  %-34s %10.1f" % (label, dt))
    s = sum(d for _, d in LOG)
    print("  " + "-" * 46)
    print("  %-34s %10.1f" % ("小计（重叠部分会重复计）", s))
    print("\n  端到端（到图片列表出现）: %.0f ms = %.1f s  列表 %d 行"
          % (total, total / 1000.0, rows))


if __name__ == "__main__":
    main()
