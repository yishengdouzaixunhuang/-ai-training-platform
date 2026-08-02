# -*- coding: utf-8 -*-
"""输出节点：结果写入 JSON / Excel（占位）。"""
from __future__ import annotations

import json
import os

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class ResultToJson(Node):
    """把 Frame.result 追加写入 JSON Lines 文件（每行一条）。"""

    node_type = "result_to_json"
    display_name = "结果写JSON (ResultToJson)"
    category = "输出"
    PARAMS = [
        NodeParam("out_path", "输出文件", ptype="str", default="results.jsonl"),
    ]

    def run(self, frame: Frame) -> Frame:
        out_path = self.params["out_path"]
        if not os.path.isabs(out_path):
            out_path = os.path.join(frame.meta.get("project_dir", ""), out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        record = {
            "path": frame.path,
            "base": frame.base,
            "result": frame.result,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return frame


@register_node
class ResultToExcel(Node):
    """结果写入 Excel（占位，后续复用 openpyxl 实现多工作表）。"""

    node_type = "result_to_excel"
    display_name = "结果写Excel (ResultToExcel)"
    category = "输出"
    PARAMS = [
        NodeParam("out_path", "输出文件", ptype="str", default="results.xlsx"),
    ]

    def run(self, frame: Frame) -> Frame:
        raise NotImplementedError("ResultToExcel 尚未实现")
def _resolve_path(value: str, frame: Frame) -> str:
    """相对路径基于项目目录解析。"""
    if os.path.isabs(value):
        return value
    base = frame.meta.get("project_dir", "") or os.getcwd()
    return os.path.join(base, value)


def _sanitize(obj):
    """把 numpy 等不可序列化对象转成可 JSON 序列化的值。"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@register_node
class SegResultWriter(Node):
    """把分割 mask / overlay 写入输出目录（批处理用）。"""

    node_type = "seg_result_writer"
    display_name = "分割结果保存 (SegWriter)"
    category = "输出"
    PARAMS = [
        NodeParam("out_dir", "输出目录", ptype="str", default="outputs"),
        NodeParam("save_mask", "保存Mask", ptype="bool", default=True),
        NodeParam("save_overlay", "保存Overlay", ptype="bool", default=True),
    ]

    def run(self, frame: Frame) -> Frame:
        import numpy as np
        from PIL import Image
        out_dir = _resolve_path(self.params["out_dir"], frame)
        os.makedirs(out_dir, exist_ok=True)
        base = frame.base or "image"
        mask = frame.result.get("mask")
        overlay = frame.result.get("overlay")
        if mask is not None and self.params["save_mask"]:
            Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(os.path.join(out_dir, base + "_pred.png"))
        if overlay is not None and self.params["save_overlay"]:
            Image.fromarray(np.asarray(overlay)).save(os.path.join(out_dir, base + "_overlay.jpg"))
        return frame


@register_node
class DetResultWriter(Node):
    """把检测框写入 COCO JSON（批处理用，与旧版 _det.json 一致）。"""

    node_type = "det_result_writer"
    display_name = "检测结果保存 (DetWriter)"
    category = "输出"
    PARAMS = [
        NodeParam("out_dir", "输出目录", ptype="str", default="outputs"),
        NodeParam("classes", "类别列表(逗号分隔)", ptype="str", default="", tooltip="留空则自动从检测框类别收集"),
    ]

    def run(self, frame: Frame) -> Frame:
        from PIL import Image
        out_dir = _resolve_path(self.params["out_dir"], frame)
        os.makedirs(out_dir, exist_ok=True)
        boxes = frame.result.get("boxes", [])
        classes = [c.strip() for c in str(self.params["classes"]).split(",") if c.strip()]
        if not classes:
            seen = []
            for b in boxes:
                cat = b.get("category", "unknown")
                if cat not in seen:
                    seen.append(cat)
            classes = seen or ["background"]
        # 与旧版行为一致：按项目类别顺序重映射 category_id
        for b in boxes:
            cat = b.get("category", "unknown")
            b["category_id"] = classes.index(cat) if cat in classes else 0
            b["score"] = b.get("score", b.get("confidence"))
        img_w = img_h = 0
        if frame.image is not None:
            img_h, img_w = frame.image.shape[:2]
        elif frame.path:
            try:
                img_w, img_h = Image.open(frame.path).size
            except Exception:
                pass
        from detection.coco_io import save_coco_json
        json_path = os.path.join(out_dir, frame.base + "_det.json")
        save_coco_json(json_path, frame.path or frame.base, img_w, img_h, boxes, classes, bbox_type="hbb")
        return frame


@register_node
class OCRResultWriter(Node):
    """把 OCR 结果写入逐图 JSON 与叠加图（批处理用）。"""

    node_type = "ocr_result_writer"
    display_name = "OCR 结果保存 (OCRWriter)"
    category = "输出"
    PARAMS = [
        NodeParam("out_dir", "输出目录", ptype="str", default="outputs/ocr"),
        NodeParam("save_overlay", "保存叠加图", ptype="bool", default=True),
    ]

    def run(self, frame: Frame) -> Frame:
        import numpy as np
        from PIL import Image
        out_dir = _resolve_path(self.params["out_dir"], frame)
        os.makedirs(out_dir, exist_ok=True)
        regions = frame.result.get("regions", [])
        base = frame.base or "image"
        record = {"path": frame.path, "base": base, "result": _sanitize(frame.result)}
        with open(os.path.join(out_dir, base + ".json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        if self.params["save_overlay"]:
            from ocr.pipeline import draw_ocr_overlay
            image = frame.image
            if image is None and frame.path:
                image = Image.open(frame.path).convert("RGB")
            if image is not None:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                overlay_dir = os.path.join(out_dir, "overlays")
                os.makedirs(overlay_dir, exist_ok=True)
                overlay = draw_ocr_overlay(image.copy(), regions)
                overlay.save(os.path.join(overlay_dir, base + "_ocr.jpg"), quality=90)
        return frame
