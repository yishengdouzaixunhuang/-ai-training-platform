# -*- coding: utf-8 -*-
"""OCV 字符质检节点：复用 ocv/inspector.py 的特征嵌入质检。"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class OCVInfer(Node):
    """OCV 字符质检，输出 OK/NG 与异常分。"""

    node_type = "ocv_infer"
    display_name = "OCV 字符质检 (OCVInfer)"
    category = "算法"
    PARAMS = [
        NodeParam("model_path", "模型文件", ptype="str", default="models/ocv_model.pkl",
                  tooltip="绝对路径，或相对项目目录"),
        NodeParam("threshold", "判定阈值", ptype="float", default=3.0, min=0.1, max=50.0, step=0.5),
    ]

    def __init__(self, node_id: str = "", **params):
        super().__init__(node_id=node_id, **params)
        self._inspector = None
        self._model_path = None

    def _ensure_inspector(self, model_path: str):
        if self._inspector is None or self._model_path != model_path:
            from ocv.inspector import OCVInspector
            inspector = OCVInspector()
            inspector.load(model_path)
            self._inspector = inspector
            self._model_path = model_path
        return self._inspector

    def run(self, frame: Frame) -> Frame:
        model_path = self.params["model_path"]
        if not os.path.isabs(model_path):
            base = frame.meta.get("project_dir", "") or os.getcwd()
            model_path = os.path.join(base, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OCVInfer: 模型不存在 {model_path}")
        inspector = self._ensure_inspector(model_path)
        inspector.threshold = float(self.params["threshold"])

        image = frame.image
        if image is None and frame.path:
            image = np.array(Image.open(frame.path).convert("RGB"))
        if image is None:
            raise ValueError("OCVInfer: 输入无图像")
        r = inspector.inspect(image)
        nf = frame.clone_with(result={
            "ok": bool(r["ok"]),
            "score": float(r["score"]),
            "threshold": float(r["threshold"]),
            "defect_type": r.get("defect_type", ""),
        })
        return nf
