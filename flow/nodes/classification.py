# -*- coding: utf-8 -*-
"""分类推理节点：复用 classification/trainer.py 的 predict_single。"""
from __future__ import annotations

import os

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
        NodeParam("project_dir", "项目目录", ptype="str", default=""),
        NodeParam("model_name", "模型名", ptype="str", default="best_model.pth"),
        NodeParam("top_k", "Top-K", ptype="int", default=1, min=1, max=10),
    ]

    def run(self, frame: Frame) -> Frame:
        from classification.trainer import ClassificationTrainer
        project_dir = self.params["project_dir"] or frame.meta.get("project_dir", "")
        if not project_dir or not os.path.isdir(project_dir):
            raise ValueError("ClsInfer: project_dir 未设置或不存在")
        model_name = self.params["model_name"]
        if not model_name.endswith(".pth"):
            model_name = "best_model.pth"
        top_k = int(self.params["top_k"])
        if frame.image is None:
            raise ValueError("ClsInfer: 输入无图像")
        # predict_single 接受文件路径；暂存为临时文件
        import tempfile
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        Image.fromarray(frame.image).save(tmp.name)
        try:
            ct = ClassificationTrainer(project_dir)
            ct.load_model(os.path.join(project_dir, "models", "classification", model_name))
            res = ct.predict_single(tmp.name, os.path.join(project_dir, "models", "classification", model_name),
                                    top_k=top_k)
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