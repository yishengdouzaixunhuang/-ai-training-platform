# -*- coding: utf-8 -*-
"""输入需求探针（子进程隔离版）。

为什么用子进程：脚本里存在 `while` 恒真的**死循环**（平台式的等待循环），
在同一个进程里跑会把探针卡死（已踩）。这里每个脚本跑在独立子进程里、超时即杀，
**探针永远不会被单个坏脚本拖死** —— 这一点和平台运行时一样重要。

用法：
  python interop/probe_inputs.py             # 全部脚本
  python interop/probe_inputs.py --one X.lua # 子进程模式（内部用）
"""
from __future__ import annotations
import json, os, subprocess, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def run_one(script_path):
    from interop.lv_lua import DataSource, load_globals
    from interop.calc_runner import run_script
    gv = load_globals(os.path.join(HERE, 'globals.json'))
    ds = DataSource(gv)
    try:
        run_script(script_path, ds, n_load_num=1)
        err = ''
    except Exception as e:
        err = '%s: %s' % (type(e).__name__, str(e)[:80])
    print('@@RESULT@@' + json.dumps({'missing': ds.missing, 'err': err}, ensure_ascii=False))


def main():
    SD = r'D:\01_Arthur\孚能\InnoVision脚本提取'
    REP = r'D:\01_Arthur\孚能\InnoVision输入需求表.md'
    files = sorted(f for f in os.listdir(SD) if f.endswith('.lua'))
    need = collections.defaultdict(lambda: collections.defaultdict(set))
    ok = bad = tmo = 0
    errs = []
    for i, fn in enumerate(files, 1):
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), '--one', os.path.join(SD, fn)],
                               capture_output=True, text=True, encoding='utf-8', timeout=12)
            payload = None
            for line in (r.stdout or '').splitlines():
                if line.startswith('@@RESULT@@'):
                    payload = json.loads(line[len('@@RESULT@@'):])
            if payload is None:
                bad += 1; errs.append((fn, '子进程无结果')); continue
            if payload['err']:
                bad += 1; errs.append((fn, payload['err']))
            else:
                ok += 1
            for rec in payload['missing']:
                if rec[0] == 'seg':                      # ('seg', 算子名, 下标, 项号, None)
                    _, nm, idx, _it, _ = rec
                    need[nm][(int(idx), 0)].add(fn)
                else:                                    # ('calc', 流程, 计算器名, 下标, 类型)
                    _, _flow, nm, idx, typ = rec
                    need['计算器:' + str(nm)][(int(idx), int(typ or 0))].add(fn)
        except subprocess.TimeoutExpired:
            tmo += 1; errs.append((fn, '超时被杀（疑似死循环）'))
        if i % 40 == 0:
            print('   进度 %d/%d  ok=%d 失败=%d 超时=%d' % (i, len(files), ok, bad, tmo), flush=True)

    print('  完成：成功 %d，失败 %d，超时 %d' % (ok, bad, tmo))
    seg = {k: v for k, v in need.items() if not k.startswith('计算器:')}
    calc = {k: v for k, v in need.items() if k.startswith('计算器:')}
    L = ['# InnoVision 脚本"输入需求"表（探针实测）\n',
         '做法：193 个计算器脚本逐一在**独立子进程**里空跑（超时 12s 即杀），',
         '记录每一次 `Lua_GetBusinessParam` 的索要 —— 这就是脚本对平台的输入契约。\n',
         '\n## 一、分割/图像类算子需要的下标\n']
    for k in sorted(seg):
        L.append('\n### %s\n\n| 下标 | 类型 | 需要的脚本数 |\n|---|---|---|\n' % k)
        for (idx, typ), fs in sorted(seg[k].items()):
            L.append('| %d | %d | %d |\n' % (idx, typ, len(fs)))
    L.append('\n## 二、计算器之间的依赖\n')
    for k in sorted(calc):
        L.append('\n### %s\n' % k)
        for (idx, typ), fs in sorted(calc[k].items()):
            L.append('- 下标 %d（类型 %d）← %s\n' % (idx, typ, '、'.join(sorted(x.replace('_flow.lua','') for x in fs))[:150]))
    if errs:
        L.append('\n## 三、失败/超时清单\n')
        for fn, e in errs: L.append('- %s：%s\n' % (fn, e))
    open(REP, 'w', encoding='utf-8').write(''.join(L))
    print('  报告:', REP)
    print()
    for k in sorted(seg):
        print('  [%s] %d 下标: %s' % (k, len(seg[k]),
              ' '.join('%d/t%d' % (i, t) for (i, t) in sorted(seg[k]))))
    print()
    for k in sorted(calc)[:6]:
        print('  %-24s ← %s' % (k, ' '.join('idx%d' % i for (i, _) in sorted(calc[k]))))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--one':
        run_one(sys.argv[2])
    else:
        main()
