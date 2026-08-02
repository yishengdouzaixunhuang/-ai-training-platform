# -*- coding: utf-8 -*-
"""OCR 文字识别节点：复用 ocr/engine.py 的引擎单例。"""
from __future__ import annotations

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class OCRInfer(Node):
    """OCR 文字检测 + 识别，输出文本区域列表到 result["regions"]。"""

    node_type = "ocr_infer"
    display_name = "OCR 文字识别 (OCRInfer)"
    category = "算法"
    PARAMS = [
        NodeParam("lang", "语言", ptype="choice", choices=["ch", "en", "ch_en"], default="ch"),
        NodeParam("top_k", "最多结果", ptype="int", default=0, min=0, max=100, tooltip="0=全部"),
    ]

    def __init__(self, node_id: str = "", **params):
        super().__init__(node_id=node_id, **params)
        self._engine = None

    def run(self, frame: Frame) -> Frame:
        if self._engine is None:
            from ocr.engine import get_ocr_engine
            self._engine = get_ocr_engine()
        source = frame.path or frame.image
        if source is None:
            raise ValueError("OCRInfer: 输入无图像")
        regions = self._engine.detect_and_recognize(source)
        top_k = int(self.params["top_k"])
        if top_k > 0:
            regions = regions[:top_k]
        text = regions[0].get("text", "") if regions else ""
        score = regions[0].get("score", 0.0) if regions else 0.0
        nf = frame.clone_with(result={
            "regions": regions,
            "count": len(regions),
            "text": text,
            "confidence": score,
        })
        return nf
