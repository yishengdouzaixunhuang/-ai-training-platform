# -*- coding: utf-8 -*-
"""`core.result` 的回归测试（重构清单 P0-1）。

这个模块的价值全在"失败**可判定**"上，所以用例盯的是：

1. `ok` 与 `code` 不能自相矛盾（`ok=True` 却带 IO 码这种事必须构造不出来）
2. `bool(result)` 与 `ok` 一致 —— 老代码 `if save(...)` 才能不改就跑
3. `guard` 不能把"函数本来就返回 Result"再包一层
4. 至少有一个**真实模块**真的用上了错误码（`core.mask_stats_cache`）——
   否则这个模块就是个空壳

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_result.py
"""
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from core.result import ErrCode, Result, ResultError, guard  # noqa: E402

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def test_success():
    r = Result.success({"a": 1})
    assert r.ok is True and bool(r) is True
    assert r.code == ErrCode.OK and r.value == {"a": 1}
    assert r.code_text == "ok"


def test_fail_fields():
    r = Result.fail(ErrCode.CORRUPT, "JSON 坏了", where="load")
    assert r.ok is False and bool(r) is False
    assert r.value is None
    assert r.code_text == "corrupt / load: JSON 坏了", r.code_text


def test_fail_with_ok_is_coerced():
    """`fail(OK)` 一定是笔误 —— 必须纠成 UNKNOWN，不能构造出自相矛盾的结果。"""
    r = Result.fail(ErrCode.OK, "手滑")
    assert r.ok is False, "fail(OK) 造出了 ok=True 的对象"
    assert r.code == ErrCode.UNKNOWN
    assert r.message == "手滑"


def test_benign():
    assert Result.fail(ErrCode.NOT_FOUND).benign is True
    assert Result.success().benign is True
    assert Result.fail(ErrCode.CORRUPT).benign is False
    assert Result.fail(ErrCode.IO).benign is False


def test_unwrap_or_and_unwrap():
    assert Result.success(7).unwrap_or(0) == 7
    assert Result.fail(ErrCode.IO).unwrap_or(0) == 0
    assert Result.success(7).unwrap() == 7
    try:
        Result.fail(ErrCode.IO, "磁盘满").unwrap()
    except ResultError as e:
        assert e.code == ErrCode.IO, e.code
        assert "磁盘满" in str(e)
    else:
        raise AssertionError("失败时 unwrap 必须抛 ResultError")


def test_raise_if_failed():
    ok = Result.success(1)
    assert ok.raise_if_failed() is ok
    try:
        Result.fail(ErrCode.VERSION).raise_if_failed()
    except ResultError as e:
        assert e.code == ErrCode.VERSION
    else:
        raise AssertionError("失败时 raise_if_failed 必须抛")


def test_guard_converts_exception():
    @guard(ErrCode.IO, where="demo.save")
    def boom():
        raise OSError("disk full")

    r = boom()
    assert r.ok is False and r.code == ErrCode.IO
    assert r.where == "demo.save"
    assert "OSError" in r.message and "disk full" in r.message, r.message


def test_guard_wraps_plain_value():
    @guard(ErrCode.IO)
    def fine():
        return 42

    r = fine()
    assert r.ok is True and r.value == 42


def test_guard_passthrough_result():
    """函数本来就返回 Result 时不能包第二层（否则取值要多问一次 ok）。"""
    @guard(ErrCode.IO, where="outer")
    def inner_fails():
        return Result.fail(ErrCode.CORRUPT, "坏的", where="inner")

    r = inner_fails()
    assert r.code == ErrCode.CORRUPT, "被 guard 覆盖成了别的码：%r" % r.code
    assert r.where == "inner", "where 被外层覆盖了：%r" % r.where
    assert not isinstance(r.value, Result), "包了两层 Result"


def test_guard_uses_fn_name_as_where():
    @guard(ErrCode.IO)
    def save_thing():
        raise ValueError("x")

    assert save_thing().where == "save_thing"


def test_used_by_real_module():
    """★ 验收标准：至少有一个真实模块真的用上了错误码，而不是只加了个空壳模块。"""
    from core import mask_stats_cache as msc

    d = tempfile.mkdtemp(prefix="res_")
    r = msc.load(d)
    assert isinstance(r, Result), "mask_stats_cache.load 还没迁到 Result"
    assert r.code == ErrCode.NOT_FOUND and r.benign

    msc.save(d, {"a_mask.png": (1, 2, 3, True)})
    assert msc.load(d).ok is True

    with open(msc.cache_path(d), "w", encoding="utf-8") as f:
        f.write('{"version": 1, "entries": { oops')
    r2 = msc.load(d)
    assert r2.code == ErrCode.CORRUPT, "坏文件应给 corrupt：%r" % r2.code
    assert r2.benign is False, "corrupt 不该被当成「本来就没有」"


def main():
    print("=== core.result 回归测试（P0-1）===")
    checks = [
        ("success", test_success),
        ("fail 字段与 code_text", test_fail_fields),
        ("fail(OK) 纠成 UNKNOWN", test_fail_with_ok_is_coerced),
        ("benign 判定", test_benign),
        ("unwrap_or / unwrap", test_unwrap_or_and_unwrap),
        ("raise_if_failed", test_raise_if_failed),
        ("guard 把异常变错误码", test_guard_converts_exception),
        ("guard 包普通返回值", test_guard_wraps_plain_value),
        ("guard 透传已有 Result", test_guard_passthrough_result),
        ("guard 默认 where", test_guard_uses_fn_name_as_where),
        ("★ 真实模块已用上（mask_stats_cache）", test_used_by_real_module),
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
