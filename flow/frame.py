# -*- coding: utf-8 -*-
"""Frame —— 算子流统一数据载体。

约定：
- image: numpy 数组，RGB 顺序，uint8，HxWx3（无图时为 None）
- height_map: numpy 数组，float32，HxW（高度图专属，无则 None）
- roi: (x, y, w, h) 像素坐标，None 表示整图
- result: dict，算法输出（分类/检测/分割/OCR 统一放这里）
- meta: dict，任意元数据（文件名、时间戳、工程信息）
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


class Frame:
    def __init__(
        self,
        image: Optional[np.ndarray] = None,
        path: str = "",
        roi: Optional[Tuple[int, int, int, int]] = None,
        height_map: Optional[np.ndarray] = None,
        result: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.image = image
        self.path = path
        self.roi = roi
        self.height_map = height_map
        self.result = result if result is not None else {}
        self.meta = meta if meta is not None else {}

    # ---- 便捷属性 ----
    @property
    def width(self) -> int:
        if self.image is None:
            return 0
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        if self.image is None:
            return 0
        return int(self.image.shape[0])

    @property
    def basename(self) -> str:
        return os.path.basename(self.path or "")

    @property
    def base(self) -> str:
        """无扩展名文件名，用于关联结果/热力图等。"""
        return os.path.splitext(self.basename)[0]

    # ---- 工具 ----
    def copy(self) -> "Frame":
        """浅拷贝 Frame（image/height_map 共享引用，节点内只读场景足够）。"""
        return Frame(
            image=self.image,
            path=self.path,
            roi=self.roi,
            height_map=self.height_map,
            result=dict(self.result),
            meta=dict(self.meta),
        )

    def clone_with(self, **kwargs) -> "Frame":
        """复制并覆盖指定字段。"""
        f = self.copy()
        for k, v in kwargs.items():
            if not hasattr(f, k):
                raise AttributeError(f"Frame has no field: {k}")
            setattr(f, k, v)
        return f

    def to_summary(self) -> Dict[str, Any]:
        """精简摘要，用于日志/调试。"""
        return {
            "path": self.path,
            "size": f"{self.width}x{self.height}" if self.image is not None else None,
            "result_keys": list(self.result.keys()),
            "meta_keys": list(self.meta.keys()),
        }