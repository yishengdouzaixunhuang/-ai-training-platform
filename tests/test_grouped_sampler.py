# -*- coding: utf-8 -*-
"""分组采样器回归测试（对应 2026-10-29 "训练 4.2 小时/epoch" 的修复）。

问题：4096² 的图 48MB/张，dataset 图片缓存只有 1.2GB，而 `shuffle=True`
      会把 10 万个 tile 完全打散 ⇒ 同一张图的 484 块被摊到整个 epoch ⇒
      缓存全程失效 ⇒ 每块重新解码一张大图（实测 150ms）⇒ 4.2 小时/epoch。
      实测对比：取 320 个 tile，随机顺序解码 275 次/60.3 秒，
               分组顺序解码 1 次/0.76 秒（79 倍）。

这里钉住四条语义，防止以后有人把 sampler 改回全量随机：
    ① 长度与 dataset 一致（epoch 不能少块）
    ② 每个索引恰好出现一次（不能重也不能漏）
    ③ 同一张图的块**必须连续**（这是缓存命中的前提）
    ④ 每个 epoch 图的顺序不同（随机性还在，不能变成固定顺序）

运行：python tests/test_grouped_sampler.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import GroupedShuffleSampler  # noqa: E402


def _groups(sizes):
    """按给定每组大小造一个 groups。"""
    g, s = [], 0
    for n in sizes:
        g.append((s, s + n))
        s += n
    return g


def test_length_matches():
    """总长度 = 所有组大小之和（drop 的块会让训练少数据）。"""
    groups = _groups([484, 484, 121, 1])
    smp = GroupedShuffleSampler(groups, shuffle=True)
    assert len(smp) == 1090, len(smp)
    assert len(list(smp)) == 1090


def test_covers_each_index_once():
    """每个索引恰好出现一次。"""
    groups = _groups([5, 3, 10, 1])
    total = sum(b - a for a, b in groups)
    got = list(GroupedShuffleSampler(groups, shuffle=True))
    assert sorted(got) == list(range(total)), got


def test_same_image_contiguous():
    """⚠️ 核心：同一张图的块必须连续出现（否则缓存命中不了）。

    验证方式：对每个 group，它包含的块在**输出序列里的位置**必须是一段
    连续整数。若被随机打散，这些位置就会散落各处。
    """
    groups = _groups([6, 4, 8])
    seq = list(GroupedShuffleSampler(groups, shuffle=True))
    pos = {v: i for i, v in enumerate(seq)}

    for a, b in groups:
        positions = sorted(pos[v] for v in range(a, b))
        n = b - a
        assert positions == list(range(positions[0], positions[0] + n)), (
            f"group ({a},{b}) 的块被打散了：位置 {positions}")


def test_order_varies_between_epochs():
    """每个 epoch 图的顺序应当不同（随机性还在）。"""
    groups = _groups([2] * 30)
    a = list(GroupedShuffleSampler(groups, shuffle=True))
    b = list(GroupedShuffleSampler(groups, shuffle=True))
    assert a != b, "两个 epoch 的顺序完全一样，说明随机没生效"


def test_shuffle_false_is_sequential():
    """shuffle=False 时严格按原顺序（调试/评测时要用）。"""
    groups = _groups([3, 2, 4])
    assert list(GroupedShuffleSampler(groups, shuffle=False)) == list(range(9))


def test_single_tile_groups():
    """整图训练（每组 1 块）时也要正确工作。"""
    groups = _groups([1] * 50)
    got = list(GroupedShuffleSampler(groups, shuffle=True))
    assert sorted(got) == list(range(50))


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
