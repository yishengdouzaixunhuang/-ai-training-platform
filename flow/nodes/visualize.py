# -*- coding: utf-8 -*-
"""可视化节点：Grad-CAM 热力图叠加。"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class HeatmapOverlay(Node):
    """分类 Grad-CAM 热力图叠加：把 frame.image 换成热力图叠加图。

    需要先有分类模型（models/classification/*.pth）与分类结果（可选，用于指定类别）。
    """

    node_type = "heatmap_overlay"
    display_name = "热力图叠加 (Grad-CAM)"
    category = "可视化"
    PARAMS = [
        NodeParam("project_dir", "项目目录", ptype="str", default="", tooltip="可留空，批处理时自动从 Frame.meta 注入"),
        NodeParam("model_name", "模型文件", ptype="str", default="", tooltip="留空=取 models/classification 下最新 .pth"),
        NodeParam("target_class", "目标类别", ptype="str", default="", tooltip="留空=自动取预测最高类别"),
        NodeParam("alpha", "叠加透明度", ptype="float", default=0.5, min=0.0, max=1.0, step=0.05),
    ]

    def __init__(self, node_id: str = "", **params):
        super().__init__(node_id=node_id, **params)
        self._trainer = None
        self._trainer_key = None

    def _ensure_trainer(self, project_dir: str, model_file: str):
        key = (project_dir, model_file)
        if self._trainer is None or self._trainer_key != key:
            from classification.trainer import ClassificationTrainer
            ct = ClassificationTrainer(project_dir)
            ct.load_model(model_file)
            self._trainer = ct
            self._trainer_key = key
        return self._trainer

    def run(self, frame: Frame) -> Frame:
        from classification.gradcam import generate_gradcam_heatmap

        project_dir = self.params["project_dir"] or frame.meta.get("project_dir", "")
        if not project_dir or not os.path.isdir(project_dir):
            raise ValueError("HeatmapOverlay: project_dir 未设置或不存在")

        models_dir = os.path.join(project_dir, "models", "classification")
        model_file = self.params["model_name"]
        if not model_file:
            if not os.path.isdir(models_dir):
                raise FileNotFoundError(f"HeatmapOverlay: 无模型目录 {models_dir}")
            pths = sorted(f for f in os.listdir(models_dir) if f.lower().endswith(".pth"))
            if not pths:
                raise FileNotFoundError(f"HeatmapOverlay: models/classification 下无 .pth 模型")
            model_file = pths[-1]  # 最新模型（与 UI 行为一致）

        ct = self._ensure_trainer(project_dir, model_file)
        if ct.model is None:
            raise RuntimeError("HeatmapOverlay: 模型未加载")

        image = frame.image
        if image is None and frame.path:
            image = np.array(Image.open(frame.path).convert("RGB"))
        if image is None:
            raise ValueError("HeatmapOverlay: 输入无图像")

        # 目标类别：参数指定类别名，或取 frame.result 中已预测的类别，否则自动
        target_class = None
        target_name = str(self.params["target_class"] or "").strip()
        if not target_name:
            pred = frame.result.get("class", "")
            target_name = str(pred)
        if target_name and hasattr(ct, "class_names") and ct.class_names:
            try:
                target_class = ct.class_names.index(target_name)
            except ValueError:
                target_class = None

        overlay = generate_gradcam_heatmap(
            ct.model, image,
            model_name=getattr(ct, "model_name", "resnet18"),
            target_class=target_class,
            image_size=getattr(ct, "_image_size", None),
        )
        if overlay is None:
            raise RuntimeError("HeatmapOverlay: 模型不支持热力图（ViT 等）或生成失败")

        alpha = float(self.params["alpha"])
        if alpha < 1.0:
            overlay = (overlay * alpha + image * (1.0 - alpha)).astype(np.uint8)

        nf = frame.clone_with(image=overlay)
        nf.result["heatmap"] = overlay
        nf.result["heatmap_model"] = model_file
        return nf
