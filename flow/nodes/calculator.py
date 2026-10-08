# -*- coding: utf-8 -*-
"""计算器节点：在我们平台的流程里跑 InnoVision 的计算器脚本（原样兼容）。

输入：上游分割节点的结果（或 data_json 回放数据）
输出：计算器输出 {序号: 值} + 被改动的全局变量 + 整图判据
"""
from __future__ import annotations
import json, os
from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class Calculator(Node):
    """执行一个 InnoVision 计算器脚本（Lua），把输出挂到 frame.result。"""

    node_type = "calculator"
    display_name = "计算器·Lua (Calculator)"
    category = "逻辑"
    PARAMS = [
        NodeParam("script", "脚本文件", ptype="str", default="",
                  tooltip="计算器脚本路径（D:\\...\\InnoVision脚本提取\\流程__STR计算器X.lua）"),
        NodeParam("flow_name", "流程名", ptype="str", default="", tooltip="脚本所属流程，用于回读日志"),
        NodeParam("calc_name", "计算器名", ptype="str", default="", tooltip="如 ANS计算器4"),
        NodeParam("globals_json", "全局变量表", ptype="str", default="interop/globals.json",
                  tooltip="Global.lua 解析产物；留空则用平台默认那份"),
        NodeParam("data_json", "回放数据(JSON)", ptype="str", default="",
                  tooltip='留空则用上游结果；对拍时填 {"seg":{"分割(2)1":{"3":["Ad"],"4":[200],"18":["125,180,7"]}}}'),
        NodeParam("n_load_num", "nLoadNum", ptype="int", default=1, tooltip="1=A 电池，2=B 电池"),
    ]

    def run(self, frame: Frame) -> Frame:
        from interop.lv_lua import DataSource
        from interop.global_vars import GlobalVars
        from interop.calc_runner import run_script

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gj = self.params.get('globals_json') or os.path.join(root, 'interop', 'globals.json')
        if not os.path.isabs(gj):
            gj = os.path.join(root, gj)
        data = json.load(open(gj, encoding='utf-8'))
        gv = GlobalVars(list(data['by_index'].values()))
        base = dict(gv.vals)

        ds = DataSource(gv)
        # ① 回放数据（对拍/离线验证）
        replay = self.params.get('data_json') or ''
        if replay:
            j = json.loads(replay) if replay.strip().startswith('{') else json.load(open(replay, encoding='utf-8'))
            for seg_name, fields in (j.get('seg') or {}).items():
                for idx, val in fields.items():
                    ds.seg[(seg_name, int(idx))] = val
            for calc_name, fields in (j.get('calc') or {}).items():
                for idx, val in fields.items():
                    ds.calc.setdefault(('', calc_name), {})[int(idx)] = val
        # ② 用上游节点的结果
        res = frame.result or {}
        for seg_name, fields in (res.get('seg_fields') or {}).items():
            for idx, val in fields.items():
                ds.seg[(seg_name, int(idx))] = val
        if res.get('calc'):
            for calc_name, outs in res['calc'].items():
                for idx, val in outs.items():
                    ds.calc.setdefault(('', calc_name), {})[int(idx)] = val

        script = self.params.get('script') or ''
        if not script or not os.path.exists(script):
            raise FileNotFoundError('计算器脚本不存在: %s' % script)
        outs, fix = run_script(script, ds, n_load_num=int(self.params.get('n_load_num') or 1))

        name = self.params.get('calc_name') or os.path.splitext(os.path.basename(script))[0]
        result = dict(res)
        result.setdefault('calc', {})[name] = outs
        changed = {i: (base.get(i), v) for i, v in gv.vals.items() if base.get(i) != v}
        result['globals_changed'] = changed
        result['ng'] = 1 if (gv.get_by_name('HhGLOBALNG1') or 0) else 0
        result['calc_fix'] = fix
        result['calc_missing'] = list(ds.missing)[:20]
        frame.result = result
        return frame
