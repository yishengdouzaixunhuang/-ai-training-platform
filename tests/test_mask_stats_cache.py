# -*- coding: utf-8 -*-
"""`core.mask_stats_cache` 的回归测试。

这份缓存决定"打开工程时要不要重新解码 4096² 掩膜"，所以两条底线必须钉住：

    1. 坏了必须退化成全量重算，不能抛异常把工程打开流程卡死
    2. 写入必须原子 —— 断电后不能留半个 JSON

运行：
    python tests/test_mask_stats_cache.py
"""
import json
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import mask_stats_cache as msc  # noqa: E402

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def test_roundtrip():
    d = tempfile.mkdtemp(prefix="msc_")
    src = {
        "annotations/a_mask.png": (1700000000123456789, 4096, 3, True),
        "annotations/b_mask.png": (1700000000987654321, 1024, 0, False),
    }
    assert msc.save(d, src).ok is True
    got = msc.load(d).value
    assert got == src, f"往返不一致: {got}"


def test_missing_file():
    d = tempfile.mkdtemp(prefix="msc_")
    r = msc.load(d)
    assert r.value is None, "失败时 value 应为 None（取值请用 unwrap_or）"
    assert r.unwrap_or({}) == {}, "unwrap_or 应给出空 dict"
    assert r.code == "not_found", "文件不存在应给 not_found（正常路径，不该报警）：%r" % r.code
    assert r.benign is True, "not_found 必须算 benign"


def test_corrupted_json():
    d = tempfile.mkdtemp(prefix="msc_")
    with open(msc.cache_path(d), "w", encoding="utf-8") as f:
        f.write('{"version": 1, "entries": { oops')
    r = msc.load(d)
    assert r.unwrap_or({}) == {}, "坏 JSON 必须退化成空，不能抛"
    assert r.code == "corrupt", "坏 JSON 应给 corrupt：%r" % r.code
    assert r.benign is False, "corrupt 不该被当成\"本来就没有\""


def test_wrong_version():
    d = tempfile.mkdtemp(prefix="msc_")
    with open(msc.cache_path(d), "w", encoding="utf-8") as f:
        json.dump({"version": 999, "entries": {"a": [1, 2, 3, True]}}, f)
    r = msc.load(d)
    assert r.unwrap_or({}) == {}, "版本不符应整体作废"
    assert r.code == "version", "版本不符应给 version：%r" % r.code


def test_partial_garbage():
    """单条坏掉只丢那一条，其余照用。"""
    d = tempfile.mkdtemp(prefix="msc_")
    payload = {
        "version": msc.VERSION,
        "entries": {
            "good_mask.png": [1, 2, 3, True],
            "bad_mask.png": [1, 2],           # 长度不对
            "worse_mask.png": "notalist",
        },
    }
    with open(msc.cache_path(d), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    r = msc.load(d)
    got = r.value
    assert list(got) == ["good_mask.png"], f"应只保留好条目，实际 {got}"
    assert "丢弃" in r.message, "丢了坏条目就该在 message 里说一声：%r" % r.message
    assert got["good_mask.png"] == (1, 2, 3, True)


def test_atomic_no_temp_left():
    """写完目录里不能留 .tmp 残渣。"""
    d = tempfile.mkdtemp(prefix="msc_")
    msc.save(d, {"x_mask.png": (1, 2, 0, False)})
    leftovers = [f for f in os.listdir(d) if f.endswith(".tmp")]
    assert not leftovers, f"残留临时文件: {leftovers}"


def test_save_into_missing_dir():
    """工程目录不可写时返回 False，不能抛。"""
    d = os.path.join(tempfile.gettempdir(), "definitely_not_exist_xyz", "proj")
    r = msc.save(d, {"a": (1, 2, 3, True)})
    assert r.ok is False
    assert r.code == "io", "写盘失败应给 io：%r" % r.code


def test_stat_of():
    d = tempfile.mkdtemp(prefix="msc_")
    p = os.path.join(d, "x.bin")
    with open(p, "wb") as f:
        f.write(b"0123456789")
    ns, sz = msc.stat_of(p)
    assert sz == 10 and ns > 0
    assert msc.stat_of(os.path.join(d, "nope.bin")) == (0, -1)


def main():
    print("=== mask_stats_cache 回归测试 ===")
    checks = [
        ("存取往返", test_roundtrip),
        ("文件不存在", test_missing_file),
        ("JSON 损坏", test_corrupted_json),
        ("版本不符", test_wrong_version),
        ("单条损坏只丢该条", test_partial_garbage),
        ("写入不留临时文件", test_atomic_no_temp_left),
        ("目录不可写返回 False", test_save_into_missing_dir),
        ("stat_of", test_stat_of),
    ]
    for name, fn in checks:
        check(name, fn)

    print()
    if failures:
        print(f"❌ {len(failures)}/{len(checks)} 项失败")
        return 1
    print(f"✅ {len(checks)}/{len(checks)} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
