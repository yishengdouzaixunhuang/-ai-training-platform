# -*- coding: utf-8 -*-
"""真实工程切图耗时拆解（只读，不写任何文件）。

用法：
    "C:/aitp_venv/Scripts/python.exe" -u _probe_switch_perf.py ["工程目录"] ["大图路径"]
默认取 `速博达` 工程里那张 4096×9320 的 BMP。

把「切一张图」拆成用户能感知的几段分别计时：
  ① PIL 解码 + convert("RGB")
  ② 显示图降采样 + QImage 缓存
  ③ 读标注掩膜（load_mask_from_json）
  ④ 读推理结果叠加（_load_prediction_overlay）
  ⑤ Annotation 表（_update_ann_stats）
  ⑥ Result 表（_update_stats）
"""
import cProfile
import io
import os
import pstats
import sys
import time

import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_PD = (r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程"
              r"\训练工程_20261021\05-TrainingProject\速博达")
DEFAULT_IMG = os.path.join(DEFAULT_PD, "images", "No_13_2026-10-20-17-14-48-808.bmp")

PD = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PD
IMG = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMG


def main():
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    print("工程: %s" % PD)
    print("大图: %s  (%.1f MB)" % (os.path.basename(IMG), os.path.getsize(IMG) / 1e6))

    from PIL import Image
    print("尺寸: %s 模式=%s" % (Image.open(IMG).size, Image.open(IMG).mode))

    from ui.main_window import MainWindow
    win = MainWindow()
    win.canvas.project_dir = PD
    win._cached_classes = ["background", "Fu", "Zheng"]
    try:
        win.canvas.label_manager.load()
    except Exception:
        pass

    # ── 分段计时：给几个方法套一层计时器 ──────────────────────
    buckets = {}

    def wrap(obj, name, label):
        orig = getattr(obj, name)

        def inner(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                buckets[label] = buckets.get(label, 0.0) + (time.perf_counter() - t)
        setattr(obj, name, inner)

    import annotation.mask_editor as ME
    wrap(win.canvas, "_cache_qimage", "② 显示图降采样+QImage")
    wrap(win.canvas, "_load_prediction_overlay", "④ 读推理结果叠加")
    _orig_lmj = ME.load_mask_from_json

    def _lmj(*a, **k):
        t = time.perf_counter()
        try:
            return _orig_lmj(*a, **k)
        finally:
            buckets["③ 读标注掩膜"] = buckets.get("③ 读标注掩膜", 0.0) + (time.perf_counter() - t)
    ME.load_mask_from_json = _lmj

    # ① 单测解码
    t = time.perf_counter()
    pim = Image.open(IMG)
    _ = pim.convert("RGB")
    dec = time.perf_counter() - t
    print("\n① 解码+convert(RGB)：%.3fs" % dec)

    # 整段 load_image
    t = time.perf_counter()
    win.canvas.load_image(IMG)
    total_load = time.perf_counter() - t
    print("load_image 总计   ：%.3fs" % total_load)
    for k in sorted(buckets):
        print("   %s：%.3fs" % (k, buckets[k]))

    # ⑤⑥ 两张统计表
    t = time.perf_counter()
    win._update_ann_stats()
    t_ann = time.perf_counter() - t
    t = time.perf_counter()
    try:
        win._update_stats()
    except Exception as e:
        print("   (_update_stats 抛错: %s: %s)" % (type(e).__name__, e))
    t_res = time.perf_counter() - t
    print("\n⑤ Annotation 表   ：%.3fs（行数 %d）" % (t_ann, win.ann_table.rowCount()))
    print("⑥ Result 表       ：%.3fs（行数 %d）" % (t_res, win.stats_table.rowCount()))
    print("\n合计（用户感知）  ：%.3fs" % (t_ann + t_res + total_load))

    # 第二轮（缓存已热）——用户连续切图时就是这一档
    t = time.perf_counter()
    win.canvas.load_image(IMG)
    win._update_ann_stats()
    try:
        win._update_stats()
    except Exception:
        pass
    print("第二轮（热缓存）  ：%.3fs" % (time.perf_counter() - t))

    # cProfile：谁最贵
    pr = cProfile.Profile()
    pr.enable()
    win.canvas.load_image(IMG)
    try:
        win._update_stats()
    except Exception:
        pass
    win._update_ann_stats()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
    txt = s.getvalue()
    print("\n" + "=" * 70)
    print("热点（按自身耗时）")
    print("=" * 70)
    for line in txt.splitlines():
        if line.strip() and ("{" in line or ".py:" in line or "ncalls" in line):
            print(line)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
