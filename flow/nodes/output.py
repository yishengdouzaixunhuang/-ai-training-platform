# -*- coding: utf-8 -*-
"""输出节点：结果写入 JSON / Excel（占位）。"""
from __future__ import annotations

import json
import os

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class ResultToJson(Node):
    """把 Frame.result 追加写入 JSON Lines 文件（每行一条）。"""

    node_type = "result_to_json"
    display_name = "结果写JSON (ResultToJson)"
    category = "输出"
    PARAMS = [
        NodeParam("out_path", "输出文件", ptype="str", default="results.jsonl"),
    ]

    def run(self, frame: Frame) -> Frame:
        out_path = self.params["out_path"]
        if not os.path.isabs(out_path):
            out_path = os.path.join(frame.meta.get("project_dir", ""), out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        record = {
            "path": frame.path,
            "base": frame.base,
            "result": frame.result,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return frame


@register_node
class ResultToExcel(Node):
    """结果写入 Excel（占位，后续复用 openpyxl 实现多工作表）。"""

    node_type = "result_to_excel"
    display_name = "结果写Excel (ResultToExcel)"
    category = "输出"
    PARAMS = [
        NodeParam("out_path", "输出文件", ptype="str", default="results.xlsx"),
    ]

    def run(self, frame: Frame) -> Frame:
        raise NotImplementedError("ResultToExcel 尚未实现")