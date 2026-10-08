# -*- coding: utf-8 -*-
"""语义分割后处理节点：按「推理参数」过滤连通域。

对应友商 iSense「推理参数」面板里的那一组过滤：面积 / 宽度 / 高度 / 角度 / 灰度 / 分数
的 min~max 区间，外加孔洞填充、中心半径、排序、最大检测个数。

参数从 `<工程>/inference_params.json` 读（GUI 的「推理参数」面板改完就写这个文件），
所以**单张推理和批处理用的是同一份参数**，不会出现"单张看着对、批量结果不一样"。
"""
from __future__ import annotations

import os

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class SegFilter(Node):
    """按推理参数过滤分割结果的连通域。"""

    node_type = "seg_filter"
    display_name = "分割后处理·推理参数过滤 (SegFilter)"
    category = "算法"
    PARAMS = [
        NodeParam("project_dir", "项目目录", ptype="str", default="",
                  tooltip="留空则用 Frame.meta 里的 project_dir；参数文件在该目录下"),
        NodeParam("enabled", "启用", ptype="bool", default=True),
    ]

    def run(self, frame: Frame) -> Frame:
        if not self.params.get("enabled", True):
            return frame
        project_dir = self.params.get("project_dir") or frame.meta.get("project_dir", "")
        res = frame.result or {}
        pred = res.get("mask")
        if pred is None:
            return frame
        try:
            from inference.params import filter_prediction, load_params, summarize
            classes = list(res.get("classes") or [])
            params = load_params(project_dir, classes)
            gray = None
            try:
                import numpy as np
                from PIL import Image
                img = frame.image
                if img is None and frame.path:
                    img = np.array(Image.open(frame.path).convert("RGB"))
                if img is not None:
                    gray = np.asarray(Image.fromarray(img).convert("L"), dtype=np.float32)
            except Exception:
                gray = None
            out, report = filter_prediction(pred, classes=classes, params=params,
                                            gray=gray, prob=res.get("prob"),
                                            image=frame.image)
            nf = frame.clone_with()
            nf.result = dict(res)
            nf.result["mask"] = out
            nf.result["filter_report"] = summarize(report)
            return nf
        except Exception as e:
            # 过滤失败不能把整批推理带崩：原样透传，并把原因记在结果里
            nf = frame.clone_with()
            nf.result = dict(res)
            nf.result["filter_report"] = "过滤失败: %s: %s" % (type(e).__name__, e)
            return nf
