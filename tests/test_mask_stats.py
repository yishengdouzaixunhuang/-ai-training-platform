# -*- coding: utf-8 -*-
"""掩膜统计函数回归测试（对应 2026-10-29「打开工程卡顿」的性能改造）。

`_mask_stats_of` 的返回值直接决定：
    - 列表「标注」列显示的数字
    - 「仅看已标注 / 仅看未标注」筛选是否命中

所以性能优化（np.unique → PIL.getcolors）**不能改变任何语义**。
这个测试就是拿它跟旧实现逐项对比。

运行：
    python tests/test_mask_stats.py
"""
import os
import sys
import tempfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.main_window import _mask_stats_of  # noqa: E402


def _old_impl(mask_path):
    """旧实现（np.unique 版），作为语义基准。"""
    arr = np.array(Image.open(mask_path))
    u = np.unique(arr)
    return len([x for x in u if x not in (0, 255)]), bool(arr.any())


def _save(arr, tag):
    d = tempfile.mkdtemp()
    p = os.path.join(d, f"mask_{tag}.png")
    Image.fromarray(arr).save(p)
    return p


def test_empty_mask():
    """全 0 掩膜：没有类别，也没有内容。"""
    p = _save(np.zeros((64, 64), dtype=np.uint8), "empty")
    assert _mask_stats_of(p) == (0, False)
    assert _mask_stats_of(p) == _old_impl(p)


def test_multi_class():
    """3 个类别 → 3 类、有内容；与旧实现一致。"""
    m = np.zeros((64, 64), dtype=np.uint8)
    m[0:10, 0:10] = 1
    m[20:30, 20:30] = 2
    m[40:50, 40:50] = 3
    p = _save(m, "multi")
    assert _mask_stats_of(p) == (3, True)
    assert _mask_stats_of(p) == _old_impl(p)


def test_ignore_value_255_excluded():
    """255 是 ignore，不计入类别数，但**算**有内容。"""
    m = np.zeros((32, 32), dtype=np.uint8)
    m[0:16, :] = 255
    p = _save(m, "ignore")
    classes, has = _mask_stats_of(p)
    assert classes == 0, f"255 应被排除，实际 classes={classes}"
    assert has is True, "255 属于非 0 像素，应算有内容"
    assert (classes, has) == _old_impl(p)


def test_ignore_plus_real_class():
    """255(ignore) + 1(真类别) → 只算 1 个类别。"""
    m = np.zeros((32, 32), dtype=np.uint8)
    m[0:10, :] = 255
    m[20:30, :] = 1
    p = _save(m, "mixed")
    assert _mask_stats_of(p) == (1, True)
    assert _mask_stats_of(p) == _old_impl(p)


def test_equivalence_random():
    """随机掩膜上逐项比对，确保没有任何取值组合跑偏。"""
    rng = np.random.default_rng(20261029)
    for i in range(6):
        m = rng.integers(0, 5, size=(48, 48), dtype=np.uint8)
        if i % 2 == 0:
            m[m == 4] = 255
        p = _save(m, f"rnd{i}")
        assert _mask_stats_of(p) == _old_impl(p), f"第 {i} 组不一致"


def test_getcolors_is_faster():
    """记录性能收益：大掩膜上 getcolors 应快于 np.unique（宽松断言，防机器差异）。"""
    m = np.zeros((1500, 1500), dtype=np.uint8)
    m[100:300, 200:400] = 1
    m[800:900, 1000:1100] = 2
    p = _save(m, "big")

    n = 12
    t = time.perf_counter()
    for _ in range(n):
        _old_impl(p)
    old_ms = (time.perf_counter() - t) / n * 1000

    t = time.perf_counter()
    for _ in range(n):
        _mask_stats_of(p)
    new_ms = (time.perf_counter() - t) / n * 1000

    print(f"        1500² 掩膜：旧 {old_ms:.2f} ms → 新 {new_ms:.2f} ms"
          f"（{(old_ms / new_ms - 1) * 100:.0f}% 提升）")
    # 只要不倒退即可（不同机器/负载下幅度会变，别写死）
    assert new_ms <= old_ms * 1.05, f"新实现反而更慢：{new_ms:.2f} vs {old_ms:.2f}"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
