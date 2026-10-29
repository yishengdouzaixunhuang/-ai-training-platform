# -*- coding: utf-8 -*-
"""dtype 规范化回归测试（对应 crash_log.txt 2026-10-22 的连环崩溃）。

背景：
    `Image.fromarray(int64_mask)` 抛
        TypeError: Cannot handle this data type: (1, 1), <i8>
    该调用出现在 paintEvent 链路上，等于每次重绘都可能弹崩溃框。
    修复方式：所有 ndarray → PIL 的地方统一走 core.imgio。

运行：
    python tests/test_imgio_dtype.py
或：
    python -m pytest tests/test_imgio_dtype.py -q
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.imgio import ensure_uint8, pil_from_array, resize_mask_u  # noqa: E402


def test_int64_mask_crashes_naively():
    """先证明原生写法确实会崩 —— 这是修复的动机。"""
    mask = np.zeros((8, 8), dtype=np.int64)
    try:
        Image.fromarray(mask)
    except TypeError as e:
        assert "data type" in str(e).lower()
        return
    raise AssertionError("预期 Image.fromarray(int64) 抛 TypeError，实际没有")


def test_resize_mask_u_handles_int64():
    """int64 掩膜缩放不应抛异常，且类别值必须原样保留。"""
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[10:20, 10:20] = 3      # 一个类别 3 的方块
    mask[50:60, 50:60] = 7      # 一个类别 7 的方块

    out = resize_mask_u(mask, (50, 50))

    assert out is not None
    assert out.shape == (50, 50), out.shape
    assert out.dtype == np.uint8, out.dtype
    # 关键：类别 id 不能被拉伸成别的值
    assert set(np.unique(out)).issubset({0, 3, 7}), np.unique(out)


def test_resize_mask_u_same_size_returns_as_is():
    mask = np.ones((10, 10), dtype=np.int32)
    out = resize_mask_u(mask, (10, 10))
    assert out.shape == (10, 10)
    assert np.array_equal(out, np.ones((10, 10), dtype=np.uint8))


def test_ensure_uint8_float_probability_map():
    """0~1 概率图应映射到 0~255，而不是被压成 0/1。"""
    prob = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    out = ensure_uint8(prob)
    assert out.dtype == np.uint8
    assert out.max() == 255, out
    assert out[0, 1] == 127 or out[0, 1] == 128, out


def test_ensure_uint8_mask_no_normalize():
    """掩膜模式不得做归一化：0/1 的掩膜不能被放大成 0/255。"""
    mask = np.array([[0, 1], [1, 0]], dtype=np.int64)
    out = ensure_uint8(mask, normalize_float=False)
    assert np.array_equal(out, np.array([[0, 1], [1, 0]], dtype=np.uint8)), out


def test_ensure_uint8_bool_and_clip():
    b = np.array([[True, False]], dtype=np.bool_)
    assert np.array_equal(ensure_uint8(b), np.array([[255, 0]], dtype=np.uint8))

    over = np.array([-5, 0, 128, 300], dtype=np.int32)
    assert np.array_equal(ensure_uint8(over), np.array([0, 0, 128, 255], dtype=np.uint8))


def test_ensure_uint8_none_passthrough():
    assert ensure_uint8(None) is None
    assert pil_from_array(None) is None


def test_rgb_uint8_untouched():
    """已是 uint8 的 RGB 图应零拷贝返回，避免大图多余复制。"""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    out = ensure_uint8(img)
    assert out is img


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
