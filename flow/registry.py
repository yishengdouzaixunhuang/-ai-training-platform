# -*- coding: utf-8 -*-
"""NODE_REGISTRY —— 节点类型注册表。"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from .node import Node


NODE_REGISTRY: Dict[str, Type[Node]] = {}


def register_node(cls: Type[Node]) -> Type[Node]:
    """类装饰器：把节点类型注册进全局注册表。"""
    NODE_REGISTRY[cls.node_type] = cls
    return cls


def create_node(node_type: str, node_id: str = "", **params) -> Node:
    """按类型名实例化节点。"""
    cls = NODE_REGISTRY.get(node_type)
    if cls is None:
        raise ValueError(f"Unknown node type: {node_type}. Registered: {sorted(NODE_REGISTRY)}")
    return cls(node_id=node_id, **params)


def list_nodes(category: Optional[str] = None) -> List[Dict[str, object]]:
    """列出节点元信息（编辑器用）。"""
    out = []
    for cls in NODE_REGISTRY.values():
        if category and cls.category != category:
            continue
        out.append({
            "type": cls.node_type,
            "name": cls.display_name,
            "category": cls.category,
            "params": [p.to_dict() for p in cls.PARAMS],
        })
    return sorted(out, key=lambda x: (x["category"], x["type"]))