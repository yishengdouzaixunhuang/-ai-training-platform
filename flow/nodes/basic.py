# -*- coding: utf-8 -*-
"""基础节点：文件采集 / 缩放 / 灰度 / 高度图彩虹映射。"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ..frame import Frame
from ..node import Node, NodeParam
from ..registry import register_node


@register_node
class FileSource(Node):
    """从本地文件读取图像。tif/tiff 自动识别为高度图。"""

    node_type = "file_source"
    display_name = "文件采集 (File Source)"
    category = "采集"
    PARAMS = [
        NodeParam("path", "图像路径", ptype="str", default="", tooltip="支持 bmp/png/jpg/jpeg/tif"),
    ]

    def run(self, frame: Frame) -> Frame:
        path = self.params["path"] or frame.path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        pil = Image.open(path)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".tif", ".tiff"):
            arr = np.array(pil)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            height_map = arr.astype(np.float32)
            # 灰度显示图：percentile 拉伸
            vmin, vmax = np.percentile(height_map, (0.5, 99.5))
            if vmax <= vmin:
                vmin, vmax = float(height_map.min()), float(height_map.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
            gray = np.clip((height_map - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            image = np.stack([gray] * 3, axis=-1)
            return Frame(image=image, path=path, height_map=height_map,
                         meta={**frame.meta, "height_vmin": float(vmin), "height_vmax": float(vmax)})
        image = np.array(pil.convert("RGB"))
        return Frame(image=image, path=path, meta={**frame.meta})


@register_node
class Resize(Node):
    """按宽/高缩放比例缩放图像（0.1~1.0），与训练面板 Pre-Scale 一致。"""

    node_type = "resize"
    display_name = "缩放 (Resize)"
    category = "预处理"
    PARAMS = [
        NodeParam("scale_w", "宽度比例", ptype="float", default=1.0, min=0.1, max=1.0, step=0.05, unit="×"),
        NodeParam("scale_h", "高度比例", ptype="float", default=1.0, min=0.1, max=1.0, step=0.05, unit="×"),
    ]

    def run(self, frame: Frame) -> Frame:
        if frame.image is None:
            return frame
        sw = float(self.params["scale_w"])
        sh = float(self.params["scale_h"])
        h, w = frame.image.shape[:2]
        nw = max(1, int(w * sw))
        nh = max(1, int(h * sh))
        img = Image.fromarray(frame.image).resize((nw, nh), Image.BILINEAR)
        nf = frame.clone_with(image=np.array(img))
        if frame.height_map is not None:
            hm = Image.fromarray(frame.height_map).resize((nw, nh), Image.BILINEAR)
            nf.height_map = np.array(hm, dtype=np.float32)
        return nf


@register_node
class GrayConvert(Node):
    """转灰度图（输出 HxW uint8 单通道）。"""

    node_type = "gray_convert"
    display_name = "转灰度 (Gray)"
    category = "预处理"
    PARAMS = []

    def run(self, frame: Frame) -> Frame:
        if frame.image is None:
            return frame
        gray = np.asarray(Image.fromarray(frame.image).convert("L"))
        return frame.clone_with(image=gray)


_JET_LUT = None


def _jet_lut() -> np.ndarray:
    """生成 256 色 jet 查找表（RGB，0~255）。"""
    global _JET_LUT
    if _JET_LUT is None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.cm import jet
        _JET_LUT = (np.array(jet(np.linspace(0, 1, 256))[:, :3]) * 255).astype(np.uint8)
    return _JET_LUT


@register_node
class HeightMapRainbow(Node):
    """高度图彩虹伪彩映射：vmin->蓝, vmax->红，低于/高于量程裁剪。

    数据约定：高度值在 frame.height_map（float32）。NaN 渲染为黑色。
    """

    node_type = "heightmap_rainbow"
    display_name = "高度图彩虹 (Rainbow)"
    category = "可视化"
    PARAMS = [
        NodeParam("vmin", "量程下限", ptype="float", default=None, unit="mm", tooltip="留空=自动 0.5~99.5 百分位"),
        NodeParam("vmax", "量程上限", ptype="float", default=None, unit="mm"),
        NodeParam("auto_range", "自动量程", ptype="bool", default=True),
    ]

    def run(self, frame: Frame) -> Frame:
        if frame.height_map is None:
            # 允许输入灰度图
            if frame.image is None:
                return frame
            if frame.image.ndim == 3:
                hm = np.asarray(Image.fromarray(frame.image).convert("L")).astype(np.float32)
            else:
                hm = frame.image.astype(np.float32)
        else:
            hm = frame.height_map.astype(np.float32)

        if self.params["auto_range"]:
            vmin = frame.meta.get("height_vmin")
            vmax = frame.meta.get("height_vmax")
            if vmin is None or vmax is None:
                vmin, vmax = np.percentile(hm, (0.5, 99.5))
        else:
            vmin = self.params["vmin"]
            vmax = self.params["vmax"]
        if vmin is None or vmax is None:
            vmin, vmax = np.percentile(hm, (0.5, 99.5))
        if vmax <= vmin:
            vmin, vmax = float(hm.min()), float(hm.max())
        if vmax <= vmin:
            vmax = vmin + 1.0

        lut = _jet_lut()
        idx = np.clip((hm - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        rgb = lut[idx]  # HxWx3
        # NaN -> 黑色
        nan_mask = np.isnan(hm)
        rgb[nan_mask] = 0
        nf = frame.clone_with(image=rgb)
        nf.meta["height_vmin"] = float(vmin)
        nf.meta["height_vmax"] = float(vmax)
        return nf