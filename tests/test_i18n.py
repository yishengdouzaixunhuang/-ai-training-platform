# -*- coding: utf-8 -*-
"""i18n 核心行为（P0）：中文恒等、英文查表、缺 key 回退、坏字典不崩。

运行：C:\\aitp_venv\\Scripts\\python.exe tests\\test_i18n.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

failures = []


def check(name, fn):
    try:
        fn()
        print("  [OK]   %s" % name)
    except Exception as e:
        failures.append((name, e))
        print("  [FAIL] %s: %s" % (name, e))


def _mod():
    try:
        from core import i18n
    except Exception as e:
        raise AssertionError("core/i18n.py 不存在或导入失败：%s" % e)
    return i18n


def _dict_file(mapping):
    fd, path = tempfile.mkstemp(suffix=".json", text=True)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    return path


def test_default_zh_identity():
    """默认中文：任何源串原样返回，不查表。"""
    I = _mod()
    I.reset_for_tests()
    assert I.get_language() == "zh", "默认语言应为 zh，实际 %r" % I.get_language()
    assert I.tr("开始训练") == "开始训练"
    assert I.tr("View(&V)") == "View(&V)"


def test_en_lookup():
    """英文模式：命中字典返回英文。"""
    I = _mod()
    path = _dict_file({"开始训练": "Start Training", "文件(&F)": "File(&F)"})
    I.reset_for_tests()
    I.load_dictionary(path)
    I.set_language("en")
    assert I.tr("开始训练") == "Start Training"
    assert I.tr("文件(&F)") == "File(&F)"


def test_missing_key_falls_back():
    """字典缺 key：回退中文源串，不抛异常、不返回空串。"""
    I = _mod()
    path = _dict_file({"开始训练": "Start Training"})
    I.reset_for_tests()
    I.load_dictionary(path)
    I.set_language("en")
    assert I.tr("保存参数") == "保存参数"


def test_bad_language_falls_back_zh():
    """非法语言值按 zh 处理。"""
    I = _mod()
    I.reset_for_tests()
    I.set_language("fr")
    assert I.get_language() == "zh"
    assert I.tr("标注") == "标注"


def test_bad_dictionary_does_not_crash():
    """坏 JSON / 非 dict：load_dictionary 返回 0，tr 仍恒等。"""
    I = _mod()
    fd, path = tempfile.mkstemp(suffix=".json", text=True)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ not json ")
    I.reset_for_tests()
    assert I.load_dictionary(path) == 0
    I.set_language("en")
    assert I.tr("开始训练") == "开始训练"


def main():
    print("=== i18n 核心行为 ===")
    check("默认中文恒等", test_default_zh_identity)
    check("英文查表", test_en_lookup)
    check("缺 key 回退源串", test_missing_key_falls_back)
    check("非法语言回退 zh", test_bad_language_falls_back_zh)
    check("坏字典不崩", test_bad_dictionary_does_not_crash)
    print()
    if failures:
        print("❌ %d/5 项未通过" % len(failures))
        return 1
    print("✅ 全部通过  5/5 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
