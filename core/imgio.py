# -*- coding: utf-8 -*-
"""Unicode 安全的 OpenCV 读写。

为什么需要它：cv2.imread / cv2.imwrite 使用 ANSI 路径，
现场文件名大量含中文与全角括号（`-0908-6号机（压伤）-大面-漏检1_te_0.jpg`），
会**静默失败**，只在日志里留一句 `can't open/read file`，很难发现。

统一改用 np.fromfile/np.tofile + cv2.imdecode/imencode，即可完全支持 Unicode 路径。
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """等价 cv2.imread，但支持中文/任意 Unicode 路径。失败返回 None。"""
    if not path:
        return None
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def imwrite_u(path, img, params=None):
    """等价 cv2.imwrite，但支持中文/任意 Unicode 路径。返回 bool。"""
    if img is None or not path:
        return False
    path = str(path)
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params if params is not None else [])
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def imread_gray_u(path):
    return imread_u(path, cv2.IMREAD_GRAYSCALE)


def imread_unchanged_u(path):
    return imread_u(path, cv2.IMREAD_UNCHANGED)


# ---------------------------------------------------------------------------
# ndarray → uint8 规范化
#
# 为什么需要它：PIL 的 Image.fromarray 只认有限的 dtype/通道组合，
# 而 numpy 里 mask / 滑窗推理结果 / 统计中间量最常见的就是 **int64**。
# 一旦把 int64 直接喂进去就会抛：
#     TypeError: Cannot handle this data type: (1, 1), '<i8'
# crash_log.txt 里 2026-10-22 当天连崩 3 次，全是这个原因；
# 更糟的是它发生在 paintEvent 里 —— 每次重绘都可能炸，直接弹崩溃框。
#
# 所以：**任何**把 ndarray 交给 Image.fromarray 的地方，都必须先过这里。
# ---------------------------------------------------------------------------

def ensure_uint8(arr, clip=True, normalize_float=True):
    """把任意数值 ndarray 规范成 uint8，可安全交给 Image.fromarray。

    - bool            → 0 / 255
    - 浮点            → 若 `normalize_float` 且最大值 <= 1.0，视为 0~1 归一化图，乘 255
    - 整数/其他       → 裁剪到 0~255 后转 uint8
    - 已是 uint8      → 原地返回（不复制，避免大图多余拷贝）

    ⚠️ 处理**类别掩膜**时必须 `normalize_float=False`：
       掩膜里的数值就是类别 id，不能被当成亮度做归一化拉伸。

    非 ndarray 输入（如 None）原样返回 None，便于调用点做短路判断。
    """
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.bool_:
        return a.astype(np.uint8) * 255
    if np.issubdtype(a.dtype, np.floating):
        if normalize_float:
            # 0~1 的概率图/归一化图是常见输入，不能一律当 0~255 处理
            if a.size and float(np.nanmax(a)) <= 1.0:
                a = a * 255.0
            a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
        else:
            a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
    if clip:
        a = np.clip(a, 0, 255)
    return a.astype(np.uint8)


def pil_from_array(arr, mode=None):
    """安全版 Image.fromarray：先规范 dtype，再交给 PIL。失败返回 None。"""
    from PIL import Image

    a = ensure_uint8(arr)
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    try:
        return Image.fromarray(a, mode=mode) if mode else Image.fromarray(a)
    except (TypeError, ValueError):
        # 兜底：通道数异常时按灰度降维再试一次
        if a.ndim == 3 and a.shape[2] in (2, 3, 4):
            return Image.fromarray(np.ascontiguousarray(a[..., 0]))
        raise


def resize_mask_u(mask, size, mode=None):
    """把**类别掩膜**缩放到 (w, h)，全程 uint8，NEAREST 插值保住类别值。

    专治 `np.array(Image.fromarray(mask).resize(...))` 这种写法：
    mask 是 int64 时直接崩（crash_log.txt 2026-10-22 连崩 3 次）。

    与 ensure_uint8 的区别：这里**不做 0~1 归一化**。掩膜数值是类别 id，
    乘 255 会把「1 类」变成「255 类」，统计和彩色映射全错。
    """
    from PIL import Image

    if mask is None:
        return None
    a = np.asarray(mask)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    a = np.ascontiguousarray(a)
    w, h = int(size[0]), int(size[1])
    if a.shape[:2] == (h, w):
        return a
    # ND 灰度掩膜足够；若误传了多通道图，也照样能缩
    img = Image.fromarray(a if a.ndim == 2 else a[..., 0])
    return np.array(img.resize((w, h), mode or Image.NEAREST))
