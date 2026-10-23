# -*- coding: utf-8 -*-
"""生成「平均灰度自校验」合成数据集：背景灰度固定 255，前景平均灰度覆盖 [0,255]。

用途：训练/推理后，平台算出的「平均灰度」应当与这里的目标值**逐张精确相等**，
      差多少就说明灰度口径有问题（缩放？加权？取整？用了缩略图？）。

每张图：
  · 512×512，整幅背景 = 255
  · 中央 200×200 矩形区域 = 前景（类别 Fore，掩膜值 1）
  · 前景像素用 base / base+1 的**棋盘抖动**拼出精确均值 v
      —— 故意不做成纯色块：纯色块任何口径都会算对，抖动才能暴露取整/加权错误
  · v 从 0 到 255 均匀取点，含 0 / 255 两个边界，并含 .5 这类半像素值

产物（可直接用平台「打开工程」加载）：
  <out>/project.json              类别 ["background", "Fore"]
  <out>/images/gray_XXX.png       图像
  <out>/images/gray_XXX.json      labelme 标注（一个矩形 Fore）
  <out>/annotations/gray_XXX_mask.png  类别掩膜（1 = Fore）
  <out>/manifest.csv              filename, target_mean   ← 真值表
  <out>/train_test_split.json     全部 train（8 张划出去做 val 也行，脚本里给全 train）

用法：
    python _gen_gray_testset.py [输出目录]
"""
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

# 目标均值序列：0..255 均匀 26 档 + 255 边界 + 两个半像素值（考验取整）
# 去重：0..255 步长 10.2 已经含 255，再显式加一次会撞同名文件
TARGETS = list(dict.fromkeys([round(i * 255 / 25.0, 1) for i in range(26)]
                             + [255.0, 127.5, 63.5]))

W, H = 512, 512
BOX = 200                      # 前景矩形边长
X0, Y0 = (W - BOX) // 2, (H - BOX) // 2

DEFAULT_OUT = (r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程"
               r"\训练工程_20261021\05-TrainingProject\灰度校验_合成_Arthur_20261022")


def make_patch(mean_v, size=BOX):
    """造一块均值精确等于 mean_v 的方块（base/base+1 棋盘抖动）。"""
    base = int(np.floor(mean_v))
    frac = float(mean_v) - base
    n = size * size
    n_hi = int(round(n * frac))
    if base >= 255:
        return np.full((size, size), 255, np.uint8)
    if n_hi <= 0:
        return np.full((size, size), base, np.uint8)
    if n_hi >= n:
        return np.full((size, size), min(255, base + 1), np.uint8)
    # ⚠️ 不能用棋盘：棋盘只有一半像素能抬 1，frac 最大只能到 0.5
    #    （实测 183.6 被挤成 183.5）。改成"固定种子随机置换后取前 n_hi 个"，
    #    40000 个像素 ⇒ 分辨率 1/40000 = 0.000025，任意两位小数都能精确表示。
    rng = np.random.default_rng(20261022)
    idx = rng.permutation(n)
    patch = np.full(n, base, np.uint8)
    patch[idx[:n_hi]] = base + 1
    return patch.reshape(size, size)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "annotations"), exist_ok=True)

    with open(os.path.join(out, "project.json"), "w", encoding="utf-8") as f:
        json.dump({"name": os.path.basename(out), "task_type": "语义分割",
                   "created": "2026-10-22T21:05:00",
                   "classes": ["background", "Fore"],
                   "parent_dir": os.path.dirname(out)}, f, ensure_ascii=False, indent=2)

    rows = []
    for i, v in enumerate(TARGETS):
        name = "gray_%03d" % int(round(v * 10))     # 127.5 -> gray_1275
        img = np.full((H, W), 255, np.uint8)
        patch = make_patch(v)
        img[Y0:Y0 + BOX, X0:X0 + BOX] = patch
        # 落盘：单通道 PNG（平台会 convert("RGB")，三通道等值 ⇒ 等权平均仍是 v）
        Image.fromarray(img, mode="L").save(os.path.join(out, "images", name + ".png"))

        # labelme 标注（一个矩形）
        # ⚠️ 右下角必须写 BOX-1：写 BOX 会让 fillPoly 多含一圈边线（201×201），
        #    与 200×200 的掩膜不一致 —— 实测就是它导致"标注表比真值高 2.5"。
        shapes = [{"label": "Fore", "points": [[X0, Y0], [X0 + BOX - 1, Y0],
                                              [X0 + BOX - 1, Y0 + BOX - 1], [X0, Y0 + BOX - 1]],
                   "group_id": None, "shape_type": "polygon", "flags": {}}]
        with open(os.path.join(out, "images", name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"version": "5.0.1", "flags": {}, "shapes": shapes,
                       "imagePath": name + ".png", "imageData": None,
                       "imageHeight": H, "imageWidth": W}, f, ensure_ascii=False)

        # 类别掩膜：**用平台自己的 shapes_to_mask 生成**，保证与 JSON 同源同源
        from annotation.labelme_io import shapes_to_mask
        mask = shapes_to_mask(shapes, ["background", "Fore"], (H, W))
        Image.fromarray(mask.astype(np.uint8), mode="L").save(
            os.path.join(out, "annotations", name + "_mask.png"))

        # 自检：落盘后重新读回来算一遍，确认真值成立
        _im = np.array(Image.open(os.path.join(out, "images", name + ".png")),
                       dtype=np.float64)
        real = float(_im[mask == 1].mean())
        if int((mask == 1).sum()) != BOX * BOX:
            print("  !! %s 掩膜面积 %d != %d" % (name, int((mask == 1).sum()), BOX * BOX))
        rows.append((name, v, round(real, 4), round(abs(real - v), 6)))
        if abs(real - v) > 0.02:
            print("  !! %s 目标 %.2f 实际 %.4f" % (name, v, real))

    with open(os.path.join(out, "manifest.csv"), "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "target_mean", "real_mean", "abs_err"])
        wr.writerows(rows)

    # 划分：前 24 张 train，末尾 3 张 val（含 255 与两个半像素值，专测取整）
    split = {r[0]: ("train" if i < len(rows) - 3 else "val") for i, r in enumerate(rows)}
    with open(os.path.join(out, "train_test_split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print("生成完成：%s" % out)
    print("  图像 %d 张  前景 %dx%d  背景灰度 255" % (len(rows), BOX, BOX))
    print("  目标均值覆盖 %.1f ~ %.1f（含半像素值 127.5 / 63.5）" % (min(TARGETS), max(TARGETS)))
    worst = max(rows, key=lambda r: r[3])
    print("  落盘后真值最大偏差 %.6f（%s）" % (worst[3], worst[0]))


if __name__ == "__main__":
    main()
