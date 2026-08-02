# -*- coding: utf-8 -*-
"""Flow / Runner —— 流程定义与 DAG 执行引擎。

Flow：节点列表 + 有向边（from_id -> to_id），可 JSON 序列化。
Runner：按拓扑序执行，支持停止标志；节点间通过 Frame 传递数据。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from .frame import Frame
from .node import Node
from .registry import NODE_REGISTRY, create_node


class Flow:
    """一组节点与连接边。"""

    def __init__(self, nodes: Optional[List[Node]] = None, edges: Optional[List[Tuple[str, str]]] = None,
                 name: str = "unnamed"):
        self.name = name
        self.nodes: List[Node] = nodes or []
        self.edges: List[Tuple[str, str]] = edges or []  # (from_id, to_id)

    def add_node(self, node: Node) -> "Flow":
        self.nodes.append(node)
        return self

    def add_edge(self, from_id: str, to_id: str) -> "Flow":
        self.edges.append((from_id, to_id))
        return self

    def get_node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(f"Node not found: {node_id}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": 1,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [{"from": f, "to": t} for f, t in self.edges],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Flow":
        flow = cls(name=d.get("name", "unnamed"))
        for nd in d.get("nodes", []):
            flow.nodes.append(create_node(nd["type"], node_id=nd.get("id", ""), **nd.get("params", {})))
        for e in d.get("edges", []):
            flow.edges.append((e["from"], e["to"]))
        return flow

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Flow":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class Runner:
    """按拓扑序执行 Flow。"""

    def __init__(self, flow: Flow):
        self.flow = flow
        self._stop = False

    def stop(self):
        self._stop = True

    def _topo_order(self) -> List[Node]:
        """Kahn 拓扑排序；检测环。"""
        node_map = {n.node_id: n for n in self.flow.nodes}
        indeg: Dict[str, int] = {n.node_id: 0 for n in self.flow.nodes}
        adj: Dict[str, List[str]] = {n.node_id: [] for n in self.flow.nodes}
        for f, t in self.flow.edges:
            if f in node_map and t in node_map:
                adj[f].append(t)
                indeg[t] += 1
        queue = [nid for nid, d in indeg.items() if d == 0]
        order: List[Node] = []
        while queue:
            nid = queue.pop(0)
            order.append(node_map[nid])
            for nxt in adj[nid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self.flow.nodes):
            raise ValueError("Flow contains a cycle or dangling nodes")
        return order

    def run(self, input_frame: Optional[Frame] = None,
            on_node: Optional[callable] = None) -> Frame:
        """执行整个流程；返回最后一个节点输出的 Frame。"""
        self._stop = False
        order = self._topo_order()
        frame = input_frame or Frame()
        for node in order:
            if self._stop:
                break
            if on_node:
                on_node(node, frame)
            frame = node.run(frame)
        return frame