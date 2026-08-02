# -*- coding: utf-8 -*-
"""批量推理执行器：把「图片列表 + Flow」变成可取消、带进度回调的批量服务。

P1 目标：批量推理逻辑从 ui/main_window.py 迁出。UI 只负责：
- 构建对应任务的 Flow（本模块提供 build_*_flow 模板）
- 启动线程调用 BatchRunner.run()
- 通过 on_progress 回调刷新进度/日志/列表

说明：
- 同一 Flow 节点实例在批处理期间复用，节点内模型引擎缓存生效（模型只加载一次）。
- 停止检查：外部可传 stop_check 回调（如 lambda: self._stop_flag）。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from .frame import Frame
from .registry import create_node
from .runner import Flow, Runner


class BatchRunner:
    """对一组图片顺序执行 Flow。"""

    def __init__(self, flow: Flow, project_dir: str = "", stop_check: Optional[Callable[[], bool]] = None):
        self.flow = flow
        self.project_dir = project_dir
        self._stop = False
        self._stop_check = stop_check

    def stop(self):
        self._stop = True

    @property
    def stopped(self) -> bool:
        if self._stop:
            return True
        if self._stop_check is not None:
            try:
                return bool(self._stop_check())
            except Exception:
                return False
        return False

    def run(self, images: List[str],
            on_progress: Optional[Callable[[int, int, str, Frame, Optional[Exception]], None]] = None,
            ) -> List[Tuple[str, Frame, Optional[Exception]]]:
        """顺序执行全部图片。

        Args:
            images: 图片路径列表
            on_progress: (index, total, path, frame, error) 回调，index 从 1 开始

        Returns:
            [(path, out_frame, error_or_None), ...]
        """
        total = len(images)
        meta = {"project_dir": self.project_dir}
        outputs: List[Tuple[str, Frame, Optional[Exception]]] = []
        for i, path in enumerate(images):
            if self.stopped:
                break
            frame = Frame(path=path, meta=dict(meta))
            err: Optional[Exception] = None
            try:
                runner = Runner(self.flow)
                frame = runner.run(frame)
            except Exception as e:  # noqa: BLE001 - 单张失败不中断批次
                err = e
            outputs.append((path, frame, err))
            if on_progress is not None:
                try:
                    on_progress(i + 1, total, path, frame, err)
                except Exception:
                    pass
        return outputs


# ============ 任务 Flow 模板 ============

def build_cls_flow(project_dir: str = "", model_name: str = "best_model.pth",
                   top_k: int = 1, out_path: str = "") -> Flow:
    """图像分类：ClsInfer -> (可选) ResultToJson。"""
    cls = create_node("cls_infer", node_id="cls",
                      project_dir=project_dir, model_name=model_name, top_k=top_k)
    flow = Flow(name="图像分类推理").add_node(cls)
    if out_path:
        out = create_node("result_to_json", node_id="out", out_path=out_path)
        flow.add_node(out).add_edge("cls", "out")
    return flow


def build_seg_flow(project_dir: str = "", model_name: str = "best_model.pth",
                   out_dir: str = "outputs", backend: str = "pytorch",
                   tiled: bool = True, scale: float = 1.0) -> Flow:
    """语义分割：SegInfer -> SegResultWriter。"""
    seg = create_node("seg_infer", node_id="seg",
                      project_dir=project_dir, model_name=model_name,
                      backend=backend, tiled=tiled, scale=scale)
    wr = create_node("seg_result_writer", node_id="out", out_dir=out_dir)
    return (Flow(name="语义分割推理")
            .add_node(seg).add_node(wr)
            .add_edge("seg", "out"))


def build_det_flow(project_dir: str = "", model_name: str = "best.pt",
                   out_dir: str = "outputs", conf: float = 0.25,
                   iou: float = 0.45, classes: str = "") -> Flow:
    """目标检测：DetInfer -> DetResultWriter。"""
    det = create_node("det_infer", node_id="det",
                      project_dir=project_dir, model_name=model_name,
                      conf=conf, iou=iou)
    wr = create_node("det_result_writer", node_id="out",
                     out_dir=out_dir, classes=classes)
    return (Flow(name="目标检测推理")
            .add_node(det).add_node(wr)
            .add_edge("det", "out"))


def build_ocr_flow(out_dir: str = "outputs/ocr") -> Flow:
    """OCR 文字识别：OCRInfer -> OCRResultWriter。"""
    ocr = create_node("ocr_infer", node_id="ocr")
    wr = create_node("ocr_result_writer", node_id="out", out_dir=out_dir)
    return (Flow(name="OCR 文字识别")
            .add_node(ocr).add_node(wr)
            .add_edge("ocr", "out"))


def build_ocv_flow(model_path: str = "models/ocv_model.pkl",
                   threshold: float = 3.0,
                   out_path: str = "outputs/ocv_results.jsonl") -> Flow:
    """OCV 字符质检：OCVInfer -> ResultToJson。"""
    ocv = create_node("ocv_infer", node_id="ocv",
                      model_path=model_path, threshold=threshold)
    out = create_node("result_to_json", node_id="out", out_path=out_path)
    return (Flow(name="OCV 字符质检")
            .add_node(ocv).add_node(out)
            .add_edge("ocv", "out"))


def list_images(project_dir: str) -> List[str]:
    """列出项目 images/ 下的常规图像。"""
    img_dir = os.path.join(project_dir, "images")
    ext = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if not os.path.isdir(img_dir):
        return []
    return sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in ext
    )
