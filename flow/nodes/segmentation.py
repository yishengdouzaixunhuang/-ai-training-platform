# -*- coding: utf-8 -*-
"""语义分割推理节点：复用 inference/predictor.py 的多后端 Predictor。"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class SegInfer(Node):
    """语义分割推理（PyTorch/ONNX/TensorRT），输出 mask 与 overlay。"""

    node_type = "seg_infer"
    display_name = "语义分割推理 (SegInfer)"
    category = "算法"
    PARAMS = [
        NodeParam("project_dir", "项目目录", ptype="str", default="", tooltip="可留空，批处理时自动从 Frame.meta 注入"),
        NodeParam("model_name", "模型文件", ptype="str", default="best_model.pth"),
        NodeParam("backend", "推理后端", ptype="choice",
                  choices=["pytorch", "onnx", "trt", "compiled", "torchscript"], default="pytorch"),
        NodeParam("tiled", "分块推理", ptype="bool", default=True, tooltip="大图自动分块，避免显存不足"),
        NodeParam("scale", "缩放比例", ptype="float", default=1.0, min=0.1, max=1.0, step=0.05),
        # ★ 2026-10-22：默认**不合成 overlay**。批量推理每张要在 20MP 上合成一次
        #   RGB 叠加图（实测 0.084s/张），而批量场景里基本没人看这张 JPG ——
        #   画布是按 `_pred.png` 现合成叠加层的。需要 JPG 时把这个开关打开。
        NodeParam("return_overlay", "合成叠加图(批量不建议)", ptype="bool", default=False,
                  tooltip="勾上才会生成 overlay 图像（批量时每张多 ~80ms）；预览不依赖它"),
    ]

    def __init__(self, node_id: str = "", **params):
        super().__init__(node_id=node_id, **params)
        self._predictor = None
        self._predictor_key = None

    def _ensure_predictor(self, project_dir: str, model_name: str, backend: str):
        key = (project_dir, model_name, backend)
        if self._predictor is None or self._predictor_key != key:
            from inference.predictor import Predictor
            self._predictor = Predictor(project_dir, model_name, backend=backend)
            self._predictor_key = key
        return self._predictor

    def run(self, frame: Frame) -> Frame:
        project_dir = self.params["project_dir"] or frame.meta.get("project_dir", "")
        if not project_dir or not os.path.isdir(project_dir):
            raise ValueError("SegInfer: project_dir 未设置或不存在")
        model_name = self.params["model_name"]
        model_path = os.path.join(project_dir, "models", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SegInfer: 模型不存在 {model_path}")
        predictor = self._ensure_predictor(project_dir, model_name, self.params["backend"])

        image = frame.image
        if image is None and frame.path:
            image = np.array(Image.open(frame.path).convert("RGB"))
        if image is None:
            raise ValueError("SegInfer: 输入无图像")
        # ★ 纯推理计时：**只包住 predict()**
        #   读图（上面那行 Image.open）和存盘都在外面 → 这里出来的数
        #   才是和友商"2ms"同一个口径，能直接比。
        import time as _time
        _t0 = _time.time()
        _want_ov = bool(self.params.get("return_overlay", False))
        if _want_ov:
            pred, overlay = predictor.predict(
                image, return_overlay=True,
                tiled=bool(self.params["tiled"]),
                scale=float(self.params["scale"]),
            )
        else:
            pred = predictor.predict(
                image, return_overlay=False,
                tiled=bool(self.params["tiled"]),
                scale=float(self.params["scale"]),
            )
            overlay = None
        _infer_ms = (_time.time() - _t0) * 1000.0
        nf = frame.clone_with(result={
            "mask": pred,
            "overlay": overlay,
            "classes": list(predictor.classes),
            # 置信度图：给 SegFilter 的"分数"过滤用（后端没提供时为 None）
            "prob": getattr(predictor, "last_prob", None),
        })
        try:
            nf.meta["infer_ms"] = round(_infer_ms, 1)
        except Exception:
            pass
        return nf
