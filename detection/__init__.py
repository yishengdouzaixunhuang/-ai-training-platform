# -*- coding: utf-8 -*-
"""Detection package.

⚠️ `DetectionOverlay` 用**懒加载**导出（PEP 562），不在包初始化时硬导入：
   `coco_io` / `dataset` / `eval` / `labelme_rect` 是纯数据与文件逻辑，命令行工具
   （`tools/labelme_rect_to_det.py`）和测试只想 import 它们。包初始化一旦把
   PyQt5 拽进来：
     · 无显示环境直接 ImportError；
     · 而且 PyQt5 先于 torch 导入会触发 WinError 1114（c10.dll，交接说明地雷 #2）。
   有 Qt 时照旧 `from detection import DetectionOverlay`，第一次访问才真正导入。
"""
from detection.box_manager import BoxManager
from detection.coco_io import load_coco_json, save_coco_json, get_det_json_path
from detection.dataset import build_yolo_dataset, get_annotated_count
from detection.eval import evaluate_predictions
from detection.labelme_rect import convert_one, convert_items
from detection.trainer import DetectionTrainer, DET_MODELS


def __getattr__(name):
    """只有 box_overlay 依赖 PyQt5 —— 等真的要用画布叠加层时再导入。"""
    if name == "DetectionOverlay":
        from detection.box_overlay import DetectionOverlay
        return DetectionOverlay
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
