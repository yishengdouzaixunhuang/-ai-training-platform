# -*- coding: utf-8 -*-
"""计算器脚本执行器（InnoVision Lua 兼容）。

为什么需要"修补"：从流程文件里切出来的计算器块**不是独立可编译的** ——
块首可能带着上一个构造的收尾 `end`，块尾又缺自己的 `end`。
这里用两步修好：① 按"块收支"找回最后一个平衡起点；② 让 Lua 编译器驱动补 end
（缺一个补一个，直到能编译）—— 比人工数块可靠。
"""
from __future__ import annotations
import re, time
from lupa import LuaRuntime, LuaSyntaxError

_OPEN = re.compile(r'\b(for|while|if|function)\b|^\s*do\s*$', re.M)
_CLOSE = re.compile(r'\bend\b')


def find_balanced_start(src):
    lines = src.splitlines(); bal = 0; start = 0
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith('else') and bal == 0:
            bal = 0; start = i + 1; continue
        bal += len(_OPEN.findall(l)) - len(_CLOSE.findall(l))
        if bal < 0:
            bal = 0; start = i + 1
    return '\n'.join(lines[start:]), start


def compile_with_repair(body, lua, max_extra=8):
    for extra in range(max_extra + 1):
        src = body + '\n' + 'end\n' * extra
        try:
            lua.execute('local _ = function()\n' + src + '\nend')
            return src, extra
        except LuaSyntaxError as e:
            if 'near <eof>' in str(e):
                continue
            raise
    raise RuntimeError('补 end 超过 %d 次仍不能编译' % max_extra)


def run_script(path, ds, n_load_num=1, loop_guard=1000, max_calls=300000, deadline_s=8.0):
    """在真 Lua 里执行一个计算器脚本 -> (输出{序号:值}, 修补信息)。"""
    lua = LuaRuntime(unpack_returned_tuples=True)
    body, off = find_balanced_start(open(path, encoding='utf-8').read())
    src, extra = compile_with_repair(body, lua)
    out, st = {}, {'d': 0}

    def Get(f, kind, nm, idx, typ, i=0, flag=False, k=0):
        f, kind, nm, idx, typ, i = str(f), str(kind), str(nm), int(idx), int(typ), int(i)
        if nm == 'error' and typ == 1:
            return True                       # 平台"运行中"标志
        if f == 'HhGLOBAL':
            v = ds.g.get(idx)
            if typ == 1:
                return (v.strip().upper() == 'OK') if isinstance(v, str) else bool(v)
            return v
        if 'CALCULATOR' in kind:
            return ds.calc_get(f, nm, idx, typ)
        return ds.seg_get(nm, idx, i)

    def Set(f, kind, nm, idx, typ, a=0, value=None):
        if str(f) == 'HhGLOBAL':
            if typ == 1 and isinstance(value, str):
                value = value.strip().upper() == 'OK'
            ds.g.set(idx, value)

    def Upd(f, cat, nm, value, idx, flag=0):
        out[int(idx)] = value
        ds.calc.setdefault((str(f), str(nm)), {})[int(idx)] = value

    g = lua.globals()
    g.Lua_GetBusinessParam = Get
    g.Lua_SetBusinessParam = Set
    g.Lua_UpdateCalculatorANS = Upd
    g.Lua_UpdateCalculatorSTR = Upd
    g.Lua_BusinessToolTask = lambda *a: ds.tasks.append(a) or True
    g.Lua_GetPCInfo = lambda *a: 0.5
    def _delay(*a):
        st['d'] += 1
        if st['d'] > loop_guard:
            raise RuntimeError('循环保护触发（疑似死循环）')
        return True
    g.Lua_Delay = _delay
    g.nLoadNum = n_load_num
    lua.execute(src)
    return out, {'skipped_lines': off, 'added_ends': extra}
