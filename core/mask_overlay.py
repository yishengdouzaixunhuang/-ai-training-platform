# -*- coding: utf-8 -*-
"""掩膜叠加层（标注可视化）——**主画布与多视图参考窗共用这一份实现**。

为什么必须共用：参考窗只拿到"仅掩膜"的透明层（各自画自己的视图原图），
主画布拿到的是"原图 + 掩膜"的合成图。两边各写一套配色/纹理，
就会出现"同一个标注在窗口 1 和窗口 2 长得不一样"（已踩过）。

叠加层为什么不能只用斜纹（本模块存在的直接原因）：
    早期版本只画 25% 密度的斜纹（12px 周期、3px 宽）。缩放到 fit（约 0.12 倍）时，
    屏幕上斜纹周期只剩 1.5px —— 直接变成摩尔纹/消隐。实测同一个忽略区：
      · 窗口宽度差 1px（146 vs 147），叠加层可见像素从 23291 掉到 1103；
      · IGNORE_CLASS_COLOR=(128,128,128) 与形状图1 的忽略区底色 130 几乎同色，
        灰度对灰度直接"隐身"。
    所以现在改成 **低透明度纯色底 + 高透明度斜纹 + 轮廓线**：
    纯色底是"平铺"的，缩放/相位都不会让它消失；斜纹负责放大时的质感。
    另外忽略区改用**彩色**（青蓝）——三个视图都是灰阶图（实测忽略区平均色
    140/130/3，R=G=B），灰底上只有"色相差异"才在所有亮度下都看得见。
"""
import numpy as np

from core.config import CLASS_COLORS, IGNORE_CLASS_ID

# 忽略区叠加色：不用 (128,128,128) 灰 —— 形状图本身就是灰阶，灰上加灰看不见。
# 选亮青蓝：与五个缺陷类别色（绿/橄榄/藏青/紫/青绿）都拉得开，且在灰阶上永不隐身。
IGNORE_OVERLAY_COLOR = (90, 190, 255)

# 叠加层参数
HATCH_PERIOD = 24        # 斜纹周期（图像像素）：比 12 粗，缩小后还能分辨
HATCH_WIDTH = 7          # 斜纹线宽
TINT_ALPHA = 80          # 纯色底透明度：保证任何缩放下都看得见（96 略重，80 是观感折中）
HATCH_ALPHA = 215        # 斜纹透明度：放大时有质感
CONTOUR_ALPHA = 255      # 轮廓线


def build_overlay_rgba(mask, show_classes=True, show_ignore=True,
                       ignore_color=IGNORE_OVERLAY_COLOR,
                       hatch_period=HATCH_PERIOD, hatch_width=HATCH_WIDTH,
                       tint_alpha=TINT_ALPHA, hatch_alpha=HATCH_ALPHA,
                       contours=True):
    """掩膜 -> 仅掩膜的 RGBA 叠加层（背景全透明）。

    show_classes / show_ignore 分别控制"缺陷类别"和"忽略区(__ignore__, 255)"是否上色，
    对应 Ctrl+Space 和 Ctrl+Alt 两个独立的显示开关。

    返回 (H, W, 4) uint8；mask 为 None 时返回 None。
    """
    if mask is None:
        return None
    try:
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        hatch = ((xx + yy) % max(2, int(hatch_period)) < max(1, int(hatch_width)))
        for cid in np.unique(mask):
            cid_i = int(cid)
            if cid_i == 0:
                continue
            is_ignore = (cid_i == IGNORE_CLASS_ID)
            if is_ignore and not show_ignore:
                continue
            if (not is_ignore) and not show_classes:
                continue
            region = (mask == cid)
            if not region.any():
                continue
            if is_ignore:
                color = ignore_color
            else:
                color = CLASS_COLORS[cid_i % len(CLASS_COLORS)]
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            # ① 纯色底（低透明度）：平铺，缩放/相位无关，保证"一定看得见"
            rgba[region, 0] = r
            rgba[region, 1] = g
            rgba[region, 2] = b
            rgba[region, 3] = tint_alpha
            # ② 斜纹（高透明度）：放大后提供方向质感
            sel = region & hatch
            rgba[sel, 3] = hatch_alpha
            # ③ 轮廓线：把区域边界勾出来
            if contours:
                try:
                    import cv2
                    binary = region.astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(rgba, cnts, -1, (r, g, b, CONTOUR_ALPHA), 1)
                except Exception:
                    pass
        return rgba
    except Exception:
        return None


def composite_over_image(img_rgb, rgba):
    """把 RGBA 叠加层合成到 RGB 图像上（给主画布用：它显示的是"原图+掩膜"）。"""
    if rgba is None:
        return img_rgb
    h, w = rgba.shape[:2]
    if img_rgb.shape[:2] != (h, w):
        return img_rgb
    a = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    out = img_rgb.astype(np.float32) * (1.0 - a) + rgba[:, :, :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)
