# -*- coding: utf-8 -*-
"""i18n 核心：一张 `中→英` 字典 + 运行期语言状态。

约定（与《i18n 落地方案》一致）：

- **key 就是中文源串本身**，不做 strip / 归一化；Qt 文案里的空格、省略号、
  制表符都要原样参与匹配。
- `zh` 模式恒等返回；`en` 模式查 `resources/i18n/en.json`，缺 key 回退中文。
- `tr()` 永不抛异常：UI 文案查不到最多是中文，不能让界面崩。
- 不做 `%` / `{}` 格式化；动态文案是第二批的事。

运行期只改内存；语言偏好由 `core/ui_prefs.py` 落盘，启动时读。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_PATH = os.path.join(ROOT, "resources", "i18n", "en.json")

LANG_ZH = "zh"
LANG_EN = "en"

_LANG = LANG_ZH
_EN = {}
_TRANSLATOR = None


def reset_for_tests():
    """把模块状态恢复默认（仅测试用）。"""
    global _LANG, _EN, _TRANSLATOR
    _LANG = LANG_ZH
    _EN = {}
    _TRANSLATOR = None


def load_dictionary(path=None):
    """读字典，返回条数；文件缺失/坏 JSON/非 dict 一律返回 0，不抛异常。"""
    global _EN
    p = path or DICT_PATH
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        _EN = {}
        return 0
    if not isinstance(data, dict):
        _EN = {}
        return 0
    _EN = {str(k): str(v) for k, v in data.items()}
    return len(_EN)


def set_language(code):
    """只接受 'zh' / 'en'；其它值按 zh 处理。"""
    global _LANG
    _LANG = LANG_EN if code == LANG_EN else LANG_ZH


def get_language():
    return _LANG


def tr(text):
    """中文恒等；英文查表，缺 key 回退源串。"""
    if not isinstance(text, str):
        return text
    if _LANG == LANG_EN:
        return _EN.get(text, text)
    return text


def install_qt_translator(app, code):
    """中文模式装 qt_zh_CN.qm；英文模式不装（Qt 默认英文）。

    返回 QTranslator 并保存在模块级，避免被 GC 后 Qt 内置按钮退回英文。
    """
    global _TRANSLATOR
    if _TRANSLATOR is not None:
        try:
            app.removeTranslator(_TRANSLATOR)
        except Exception:
            pass
        _TRANSLATOR = None
    if code != LANG_ZH:
        return None
    try:
        from PyQt5.QtCore import QLibraryInfo, QTranslator
        qm = os.path.join(QLibraryInfo.location(QLibraryInfo.TranslationsPath),
                          "qt_zh_CN.qm")
        if not os.path.isfile(qm):
            return None
        t = QTranslator(app)
        if not t.load(qm):
            return None
        app.installTranslator(t)
        _TRANSLATOR = t
        return t
    except Exception:
        return None
