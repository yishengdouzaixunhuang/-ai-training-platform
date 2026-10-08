# -*- coding: utf-8 -*-
"""InnoVision 全局变量表 <-> 我们平台。

设计思路：InnoVision 的 Global.lua 是"数据字典"（235 个变量：类型/序号/名字/默认值/中文注释），
格式： --[[BOOL]] [[0]] bTriggerFlag1 = false --[上表面触发拍照]--
我们把它读成一份 globals.json，并按**序号**提供读写 —— 因为流程脚本里全是按序号引用的
（Lua_GetBusinessParam("HhGLOBAL","HhGLOBAL","HhGLOBAL",119,...)），
按序号读写才能让原脚本不改一个字就跑起来。
"""
from __future__ import annotations
import base64, json, os, re

PAT = re.compile(r'--\[\[(\w+)\]\]\s*\[\[(\d+)\]\]\s*([A-Za-z_]\w*)\s*=(.*?)--\[(.*?)--')


def decode_lua(path):
    """InnoVision 的 .lua 一律 base64 编码，这里统一解码。"""
    raw = open(path, 'rb').read()
    try:
        t = base64.b64decode(raw, validate=False)
    except Exception:
        return None
    for enc in ('utf-8', 'gbk'):
        try:
            return t.decode(enc)
        except Exception:
            pass
    return t.decode('utf-8', 'replace')


def parse_global_lua(path):
    """-> [ {idx,type,name,value,comment}, ... ]（按序号排序）"""
    txt = decode_lua(path) if path.lower().endswith('.lua') else open(path, encoding='utf-8').read()
    out = []
    for line in (txt or '').splitlines():
        m = PAT.match(line.strip())
        if not m:
            continue
        typ, idx, name, val, cmt = m.groups()
        out.append(dict(idx=int(idx), type=typ, name=name,
                        value=val.strip(), comment=cmt.strip()))
    out.sort(key=lambda d: d['idx'])
    return out


def to_platform_json(vars_, out_path):
    """写成我们平台用的 globals.json（按序号 + 按名字两条索引，方便脚本与界面各取所需）。"""
    data = {
        "source": "InnoVision Global.lua",
        "count": len(vars_),
        "by_index": {str(v['idx']): v for v in vars_},
        "by_name": {v['name']: v['idx'] for v in vars_},
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return out_path


class GlobalVars:
    """运行期全局变量：按序号存值，类型做轻量转换（BOOL/INT/DOUBLE/STRING）。"""

    def __init__(self, vars_):
        self.meta = {v['idx']: v for v in vars_}
        self.vals = {}
        for v in vars_:
            self.vals[v['idx']] = self._cast(v['type'], v['value'])

    @staticmethod
    def _cast(typ, val):
        val = (val or '').strip()
        if typ.startswith('BOOL'):
            return val.lower() == 'true'
        if typ.startswith('INT'):
            try: return int(float(val))
            except Exception: return 0
        if typ.startswith('DOUBLE') or typ.startswith('FLOAT'):
            try: return float(val)
            except Exception: return 0.0
        return val.strip("'").strip('"')

    # ---- 按序号（脚本用的方式）----
    def get(self, idx):
        return self.vals.get(int(idx))

    def set(self, idx, value):
        self.vals[int(idx)] = value
        return value

    # ---- 按名字（界面/人用的方式）----
    def get_by_name(self, name):
        i = self.meta_index(name)
        return None if i is None else self.vals.get(i)

    def meta_index(self, name):
        for i, m in self.meta.items():
            if m['name'] == name:
                return i
        return None

    def changed(self, baseline):
        """对比基线，返回被改过的变量（调试/对拍用）。"""
        return {i: (baseline.get(i), self.vals.get(i))
                for i in self.vals if baseline.get(i) != self.vals.get(i)}
