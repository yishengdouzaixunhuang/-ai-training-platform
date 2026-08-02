# -*- coding: utf-8 -*-
"""分类推理节点：复用 classification/trainer.py 的 predict_single。"""
from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class ClsInfer(Node):
    """图像分类推理（ResNet/EfficientNet/MobileNet/ViT），输出 top-k 预测。"""

    node_type = "cls_infer"
    display_name = "分类推理 (ClsInfer)"
    category = "算法"
    PARAMS = [
        NodeParam("project_dir", "项目目录", ptype="str", default="", tooltip="可留空，批处理时自动从 Frame.meta 注入"),
        NodeParam("model_name", "模型名", ptype="str", default="best_model.pth"),
        NodeParam("top_k", "Top-K", ptype="int", default=1, min=1, max=10),
    ]

    def __init__(self, node_id: str = "", **params):
        super().__init__(node_id=node_id, **params)
        self._trainer = None
        self._trainer_key = None

    def _ensure_trainer(self, project_dir: str, model_name: str):
        key = (project_dir, model_name)
        if self._trainer is None or self._trainer_key != key:
            from classification.trainer import ClassificationTrainer
            ct = ClassificationTrainer(project_dir)
            model_path = os.path.join(project_dir, "models", "classification", model_name)
            ct.load_model(model_path)
            self._trainer = ct
            self._trainer_key = key
        return self._trainer

    def run(self, frame: Frame) -> Frame:
        project_dir = self.params["project_dir"] or frame.meta.get("project_dir", "")
        if not project_dir or not os.path.isdir(project_dir):
            raise ValueError("ClsInfer: project_dir 未设置或不存在")
        model_name = self.params["model_name"]
        if not model_name.endswith(".pth"):
            model_name = "best_model.pth"
        model_path = os.path.join(project_dir, "models", "classification", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ClsInfer: 模型不存在 {model_path}")
        top_k = int(self.params["top_k"])
        image = frame.image
        if image is None and frame.path:
            image = np.array(Image.open(frame.path).convert("RGB"))
        if image is None:
            raise ValueError("ClsInfer: 输入无图像")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        Image.fromarray(image).save(tmp.name)
        try:
            res = self._ensure_trainer(project_dir, model_name).predict_single(tmp.name, model_path, top_k=top_k)
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        if not res or not res.get("predictions"):
            raise RuntimeError("ClsInfer: 无预测结果")
        preds = res["predictions"]
        top = preds[0]
        nf = frame.clone_with(result={
            "class": top.get("class", ""),
            "class_id": top.get("class_id", -1),
            "confidence": top.get("confidence", 0.0),
            "top_k": preds,
        })
        return nf
