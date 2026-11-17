# -*- coding: utf-8 -*-
"""`.ui_settings.json` 的唯一读写入口（原子、保留未知键）。

修的是历史问题：`MainWindow._save_ui_setting` 只改内存、从不落盘，
`_load_ui_setting` 也没有从 JSON 载入的路径 —— `tiled` / `cls_tta` /
面板可见性全是「本次会话内有效」。语言（`lang`）要跨重启，必须先有这条链。

字段：
    theme                 深色/浅色
    lang                  zh / en
    tiled                 训练是否分块
    cls_tta               分类 TTA
    *_panel_visible       左右栏/日志栏
    tile_size             分块尺寸

未知键一律保留（前向兼容）；坏 JSON 只回退默认值，**不覆盖原文件**，
等下一次显式 save 再重写。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, ".ui_settings.json")

DEFAULTS = {
    "theme": "dark",
    "lang": "zh",
    "tiled": False,
    "cls_tta": False,
    "left_panel_visible": True,
    "right_panel_visible": True,
    "log_panel_visible": True,
}


def _path(path):
    return path or DEFAULT_PATH


def _read(path):
    """只读磁盘；坏文件/非 dict 返回 {}，绝不写盘。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load(path=None):
    """默认值 + 磁盘值。"""
    p = _path(path)
    d = dict(DEFAULTS)
    d.update(_read(p))
    return d


def get(key, default=None, path=None):
    """单键读取；文件不存在/坏文件时返回 default。"""
    p = _path(path)
    d = load(p)
    return d.get(key, default)


def save(patch, path=None):
    """读-改-写 + 原子替换；未知键保留。"""
    p = _path(path)
    cur = _read(p)
    cur.update(patch)
    parent = os.path.dirname(p)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    from core.file_ops import _save_json
    _save_json(p, cur)
