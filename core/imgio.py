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
