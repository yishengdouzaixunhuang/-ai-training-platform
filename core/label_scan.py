# -*- coding: utf-8 -*-
"""labelme JSON 的「只取类别」快速扫描 + 进程级缓存。

## 为什么需要这个模块（2026-10-28 实测）

打开一个 489 张图的工程要 **25~34 秒**，profile 出来两个真凶：
  1. **984 次 `json.load`，共 7.4 秒** —— 同一批 JSON 被 `discover_labels()`
     和 `ClassTableView._compute_counts()` **各扫了一遍**；
  2. 这些 JSON 里头有 **71 个 14~18MB 的巨型文件**（自动生成的掩膜轮廓，
     shape 的 points 上万个顶点），全工程合计 **1044 MB**。

而这两处调用方**只关心"这张图有哪些类别、有几个 shape"** —— 完整解析把上亿个
坐标点建成 Python 对象纯属浪费（内存还要飙到上百 MB）。

实测（8 个 14MB 文件）：`json.load` 340ms → 只抽 label 字段 114ms → bytes 正则更快。

## 用法

    from core.label_scan import scan_labels
    info = scan_labels(path)          # -> (labels_tuple, n_shapes) 或 None
    info[0]                           # 该图出现过的全部 label（含 ignore / background）
    info[1]                           # shape 个数（= "label" 出现次数）

⚠️ 只用于"发现类别/统计个数"。要**真正的坐标数据**（重建掩膜、导入导出）时，
   仍然必须走 `json.load` —— 这个模块不解析 points。
"""
import json
import os
import re

# labelme 的 shapes 元素形如 {"label": "Cs", "points": [...], ...}
# 在 bytes 上直接匹配：跳过 14MB 的 UTF-8 解码（实测那步本身就要 0.9s）
_LABEL_RE = re.compile(rb'"label"\s*:\s*"((?:[^"\\]|\\.)*)"')

#: path -> (签名, labels 元组, shape 数)
_CACHE = {}
#: 缓存上限：超了整体清空（长时间运行的会话里换过很多工程时兜底）
_MAX_ENTRIES = 6000


def _sig(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _decode(raw):
    """把正则抓到的字节串还原成字符串（处理 \\u 之类的转义）。"""
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        s = raw.decode("utf-8", "replace")
    if "\\" in s:
        try:
            return json.loads('"' + s + '"')
        except Exception:
            return s
    return s


def scan_labels(path):
    """扫描单个 labelme JSON，返回 `(labels_tuple, n_shapes)`；读不了返回 None。

    labels 保留**原始顺序**且**含重复**（每个 shape 一项），
    调用方自己去重/过滤（例如 `class_discovery.is_real_class`）。
    """
    p = str(path)
    sig = _sig(p)
    if sig is None:
        _CACHE.pop(p, None)
        return None
    hit = _CACHE.get(p)
    if hit is not None and hit[0] == sig:
        return hit[1], hit[2]
    try:
        with open(p, "rb") as f:
            data = f.read()
    except OSError:
        return None
    labels = tuple(_decode(m.group(1)) for m in _LABEL_RE.finditer(data))
    n = len(labels)
    if len(_CACHE) > _MAX_ENTRIES:
        _CACHE.clear()
    _CACHE[p] = (sig, labels, n)
    return labels, n


def cached_signature(path):
    """该文件当前缓存用的签名（调试/探针用）。"""
    hit = _CACHE.get(str(path))
    return hit[0] if hit else None


def clear_cache():
    """换工程/强制刷新时清空。"""
    _CACHE.clear()
