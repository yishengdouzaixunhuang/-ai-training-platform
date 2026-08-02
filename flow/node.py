# -*- coding: utf-8 -*-
"""Node —— 算子节点基类与参数 Schema。

每个节点声明：
- node_type: 唯一类型名（注册/反序列化用）
- display_name: 展示名（流程编辑器用）
- category: 分组（采集/预处理/算法/可视化/判定/输出）
- PARAMS: 参数声明列表，供编辑器自动生成参数面板
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .frame import Frame


class NodeParam:
    """单个参数声明。"""

    def __init__(
        self,
        name: str,
        label: str = "",
        ptype: str = "float",       # int/float/str/bool/choice
        default: Any = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        step: Optional[float] = None,
        choices: Optional[List[str]] = None,
        unit: str = "",
        tooltip: str = "",
    ):
        self.name = name
        self.label = label or name
        self.ptype = ptype
        self.default = default
        self.min = min
        self.max = max
        self.step = step
        self.choices = choices or []
        self.unit = unit
        self.tooltip = tooltip

    def coerce(self, value: Any) -> Any:
        """把外部值转成参数类型（JSON 反序列化时保证类型正确）。"""
        if value is None:
            return self.default
        try:
            if self.ptype == "int":
                return int(value)
            if self.ptype == "float":
                return float(value)
            if self.ptype == "bool":
                return bool(value)
            if self.ptype == "choice":
                return str(value)
        except (TypeError, ValueError):
            return self.default
        return value

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "label": self.label,
            "type": self.ptype,
            "default": self.default,
            "unit": self.unit,
            "tooltip": self.tooltip,
        }
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.step is not None:
            d["step"] = self.step
        if self.choices:
            d["choices"] = self.choices
        return d


class Node:
    """算子节点基类。子类需设置 node_type / display_name / category 并实现 run()。"""

    node_type: str = "base"
    display_name: str = "Base Node"
    category: str = "other"
    PARAMS: List[NodeParam] = []

    def __init__(self, node_id: str = "", **params):
        self.node_id = node_id or self.node_type
        self.params: Dict[str, Any] = {}
        for p in self.PARAMS:
            val = params.get(p.name, p.default)
            self.params[p.name] = p.coerce(val)
        # 兼容未声明的额外参数（宽容处理）
        for k, v in params.items():
            if k not in self.params:
                self.params[k] = v

    # ---- 子类实现 ----
    def run(self, frame: Frame) -> Frame:
        """执行节点逻辑，返回（可能被修改的）Frame。"""
        raise NotImplementedError

    # ---- 序列化 ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        return cls(node_id=d.get("id", ""), **d.get("params", {}))

    def param_schema(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.PARAMS]

    def __repr__(self):
        return f"<{self.node_type} id={self.node_id}>"