# -*- coding: utf-8 -*-
"""InnoVision 脚本兼容层：用真 Lua 跑他们**原封不动**的计算器脚本。

核心思路
--------
InnoVision 的计算器脚本里只有两类调用：
  读： Lua_GetBusinessParam("流程","类别","名称",序号,类型,项号,标志,0)
  写： Lua_SetBusinessParam("HhGLOBAL","HhGLOBAL","HhGLOBAL",序号,类型,0,值)
  回传：Lua_UpdateCalculator(ANS|STR)("流程","类别","名称",值,序号,0)
只要在 Lua 运行时里把这几条 **照语义实现**（接到我们自己的全局变量表 + 分割结果上），
他们的脚本就能**一个字不改**地跑起来 —— 这就是"逻辑平移"的做法。

数据来源用 DataSource 抽象：本项目里接我们的分割结果/全局变量；对拍时接"录制的输入"。
"""
from __future__ import annotations
import json, os


class DataSource:
    """脚本能看到的外部世界：全局变量 + 各分割算子的结果 + 各计算器的输出。"""

    def __init__(self, globals_obj):
        self.g = globals_obj                      # GlobalVars
        self.seg = {}                             # {(算子名, 序号): 值 或 [逐项列表]}
        self.calc = {}                            # {(流程, 计算器名): {序号: 值}}
        self.missing = []                         # 脚本要了但没给的数据（用于补映射）
        self.tasks = []                           # 被调用的算子（Lua_BusinessToolTask）

    def seg_get(self, name, idx, i):
        key = (name, idx)
        if key not in self.seg:
            self.missing.append(('seg', name, idx, i, None))
            return 0
        v = self.seg[key]
        if isinstance(v, (list, tuple)):
            return v[i] if 0 <= i < len(v) else 0
        return v

    def calc_get(self, flow, name, idx, typ):
        d = self.calc.get((flow, name), {})
        if idx not in d:
            self.missing.append(('calc', flow, name, idx, typ))
            return False if typ == 1 else ('' if typ == 3 else 0)
        return d[idx]


def run_calculator(script_path, ds, flow, name, n_load_num=1, timeout_s=5):
    """在真 Lua 里执行一个计算器脚本，返回它的输出 {序号: 值}。"""
    from lupa import LuaRuntime
    lua = LuaRuntime(unpack_returned_tuples=True)
    out = {}

    def GetBusinessParam(f, kind, nm, idx, typ, i=0, flag=False, n=0):
        f, kind, nm = str(f), str(kind), str(nm)
        if f == 'HhGLOBAL':
            v = ds.g.get(idx)
            if typ == 1:
                if isinstance(v, str):
                    return v.strip().upper() == 'OK'      # 平台把 'OK'/'NG' 当布尔用
                return bool(v)
            return v
        if 'CALCULATOR' in kind:
            return ds.calc_get(f, nm, int(idx), int(typ))
        return ds.seg_get(nm, int(idx), int(i))

    def SetBusinessParam(f, kind, nm, idx, typ, a=0, value=None):
        f = str(f)
        if f == 'HhGLOBAL':
            if typ == 1 and isinstance(value, str):
                value = value.strip().upper() == 'OK'
            ds.g.set(idx, value)
        else:
            ds.calc.setdefault((f, str(nm)), {})[int(idx)] = value

    def UpdateCalc(kind):
        def _u(f, cat, nm, value, idx, flag=0):
            out[int(idx)] = value
            ds.calc.setdefault((str(f), str(nm)), {})[int(idx)] = value
        return _u

    g = lua.globals()
    g.Lua_GetBusinessParam = GetBusinessParam
    g.Lua_SetBusinessParam = SetBusinessParam
    g.Lua_UpdateCalculatorANS = UpdateCalc('ANS')
    g.Lua_UpdateCalculatorSTR = UpdateCalc('STR')
    g.Lua_BusinessToolTask = lambda *a: ds.tasks.append(a) or True
    g.Lua_GetPCInfo = lambda *a: 0.5          # 磁盘占用率
    g.Lua_Delay = lambda *a: True
    g.nLoadNum = n_load_num                    # 脚本里用来区分 A/B 电池

    src = open(script_path, encoding='utf-8').read()
    lua.execute(src)
    return out


def load_calculator(path, ds, flow, name, n_load_num=1):
    return run_calculator(path, ds, flow, name, n_load_num)


def load_globals(json_path):
    from interop.global_vars import GlobalVars
    data = json.load(open(json_path, encoding='utf-8'))
    vars_ = list(data['by_index'].values())
    return GlobalVars(vars_)
