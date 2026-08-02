# -*- coding: utf-8 -*-
"""目标检测推理节点：复用 detection/trainer.py 的 DetectionTrainer.predict。"""
from __future__ import annotations

import os
import tempfile

from PIL import Image

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class DetInfer(Node):
    """目标检测推理（YOLO 系），输出检测框列表到 result["boxes"]。"""

    node_type = "det_infer"
    display_name = "目标检测推理 (DetInfer)"
    category = "算法"
    PARAMS = [
        NodeParam("project_dir", "项目目录", ptype="str", default="", tooltip="可留空，批处理时自动从 Frame.meta 注入"),
        NodeParam("model_name", "模型文件", ptype="str", default="best.pt", tooltip="models/detection/ 下的权重名"),
        NodeParam("conf", "置信度阈值", ptype="float", default=0.25, min=0.01, max=1.0, step=0.05),
        NodeParam("iou", "IoU 阈值", ptype="float", default=0.45, min=0.01, max=1.0, step=0.05),
    ]

    def __init__(self, node_id: str = "", **params):
        super().__init__(node_id=node_id, **params)
        self._trainer = None
        self._trainer_key = None

    def _ensure_trainer(self, project_dir: str, model_name: str):
        key = (project_dir, model_name)
        if self._trainer is None or self._trainer_key != key:
            from detection.trainer import DetectionTrainer
            self._trainer = DetectionTrainer(project_dir)
            self._trainer_key = key
        return self._trainer

    def run(self, frame: Frame) -> Frame:
        project_dir = self.params["project_dir"] or frame.meta.get("project_dir", "")
        if not project_dir or not os.path.isdir(project_dir):
            raise ValueError("DetInfer: project_dir 未设置或不存在")
        model_name = self.params["model_name"]
        model_path = os.path.join(project_dir, "models", "detection", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DetInfer: 模型不存在 {model_path}")
        trainer = self._ensure_trainer(project_dir, model_name)
        conf = float(self.params["conf"])
        iou = float(self.params["iou"])

        path = frame.path
        tmp = None
        if not path and frame.image is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            Image.fromarray(frame.image).save(tmp.name)
            path = tmp.name
        if not path:
            raise ValueError("DetInfer: 输入无图像路径")
        try:
            boxes = trainer.predict(path, weights=model_path, conf=conf, iou=iou)
        finally:
            if tmp:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
        nf = frame.clone_with(result={
            "boxes": boxes,
            "count": len(boxes),
        })
        return nf
