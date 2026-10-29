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
        # ★★ 2026-10-29：`tiled_auto` 默认 True ⇒ 忽略下面那个 `tiled`，
        #   由 Predictor 按**训练模式**自动决定要不要滑窗。界面上的 Tiled 勾选已去掉。
        #   为什么要多一个开关而不是让 tiled=None：Flow 的 NodeParam 会用 default
        #   把 None 覆盖成 True（实测 create_node(tiled=None).params["tiled"] == True），
        #   而 `bool(None)` 会变成 False —— 对分块训练的模型等于强制整图，尺度直接错位。
        NodeParam("tiled_auto", "自动匹配训练模式", ptype="bool", default=True,
                  tooltip="开=按训练模式自动选择（推荐）；关=用手动的「分块推理」"),
        NodeParam("tiled", "分块推理", ptype="bool", default=True,
                  tooltip="仅在「自动匹配训练模式」关闭时生效"),
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
        # 自动模式传 None 给 Predictor（按训练模式决定）；手动模式才用 tiled 的值
        _tiled = (None if self.params.get("tiled_auto", True)
                  else bool(self.params.get("tiled", True)))
        if _want_ov:
            pred, overlay = predictor.predict(
                image, return_overlay=True,
                tiled=_tiled,
                scale=float(self.params["scale"]),
            )
        else:
            pred = predictor.predict(
                image, return_overlay=False,
                tiled=_tiled,
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
