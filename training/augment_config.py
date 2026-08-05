# -*- coding: utf-8 -*-
"""Global augment config."""
import json
import os
import random
from typing import Callable, Dict, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
AUGMENT_FLAG_LABELS = [
    ("zoom_out", "缩小"),
    ("zoom_in", "放大"),
    ("translate", "平移"),
    ("aspect", "长宽比"),
    ("shear", "剪切"),
    ("rotate", "旋转"),
    ("brightness", "增强亮度"),
    ("contrast", "对比度"),
    ("invert", "对比度反转"),
    ("noise", "高斯噪声"),
    ("flip", "翻转"),
]
FLAG_KEYS = [k for k, _ in AUGMENT_FLAG_LABELS]
DEFAULT_AUGMENT_FLAGS = {
    "enabled": True,
    "zoom_out": False,
    "zoom_in": False,
    "translate": False,
    "aspect": False,
    "shear": False,
    "rotate": False,
    "brightness": True,
    "contrast": True,
    "invert": False,
    "noise": False,
    "flip": True,
}
def _settings_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".train_settings.json")
def merge_flags(flags):
    merged = dict(DEFAULT_AUGMENT_FLAGS)
    if isinstance(flags, dict):
        for k in merged:
            if k in flags:
                merged[k] = bool(flags[k])
    return merged
def preset_to_flags(level):
    flags = dict(DEFAULT_AUGMENT_FLAGS)
    level = (level or "none").lower()
    if level in ("", "none", "off", "disabled", "0"):
        flags["enabled"] = False
        return flags
    if level == "light":
        on = {"brightness", "contrast", "flip"}
    elif level == "strong":
        on = {"brightness", "contrast", "flip", "noise", "rotate", "translate", "shear"}
    elif level == "heavy":
        on = set(FLAG_KEYS)
    else:
        on = {"brightness", "contrast", "flip", "noise"}
    for k in FLAG_KEYS:
        flags[k] = k in on
    flags["enabled"] = True
    return flags
def flags_to_level(flags):
    if not flags.get("enabled", True):
        return "none"
    on = {k for k in FLAG_KEYS if flags.get(k)}
    if on == {"brightness", "contrast", "flip"}:
        return "light"
    if on == {"brightness", "contrast", "flip", "noise"}:
        return "medium"
    if on == {"brightness", "contrast", "flip", "noise", "rotate", "translate", "shear"}:
        return "strong"
    if len(on) >= 9:
        return "heavy"
    return "custom"
def load_augment_flags():
    try:
        with open(_settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        flags = data.get("augment_flags")
        if isinstance(flags, dict):
            return merge_flags(flags)
        preset = data.get("augment")
        if preset:
            return preset_to_flags(str(preset))
    except Exception:
        pass
    return dict(DEFAULT_AUGMENT_FLAGS)
def save_augment_flags(flags):
    flags = merge_flags(flags)
    path = _settings_path()
    try:
        data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data["augment_flags"] = flags
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
def _to_pil(img):
    if isinstance(img, Image.Image):
        return img
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
def _mask_to_pil(mask_np):
    return Image.fromarray(np.clip(mask_np, 0, 255).astype(np.uint8))
def _mask_to_np(mask_pil):
    return np.array(mask_pil).astype(np.int64)


def _fill_color(img):
    """PIL 单通道图（L/1/I/F）fillcolor 需要 int，多通道用 RGB 元组。"""
    if img.mode in ("L", "1", "P", "I", "F"):
        return 0
    return (0, 0, 0)
def _flip(img, mask_np, rng):
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if mask_np is not None:
            mask_np = np.fliplr(mask_np).copy()
    if rng.random() < 0.3:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if mask_np is not None:
            mask_np = np.flipud(mask_np).copy()
    return img, mask_np
def _rotate(img, mask_np, rng):
    angle = rng.uniform(-15, 15)
    if abs(angle) < 0.5:
        return img, mask_np
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=_fill_color(img))
    if mask_np is not None:
        mask_pil = _mask_to_pil(mask_np).rotate(angle, resample=Image.NEAREST, fillcolor=0)
        mask_np = _mask_to_np(mask_pil)
    return img, mask_np
def _affine_coeffs(w, h, flags, rng):
    a, b, c = 1.0, 0.0, 0.0
    d, e, f = 0.0, 1.0, 0.0
    sx, sy = 1.0, 1.0
    if flags.get("translate") and rng.random() < 0.5:
        c += rng.uniform(-0.12, 0.12) * w
        f += rng.uniform(-0.12, 0.12) * h
    if flags.get("zoom_out") and rng.random() < 0.5:
        s = rng.uniform(0.75, 0.95)
        sx, sy = s, s
    if flags.get("zoom_in") and rng.random() < 0.5:
        s = rng.uniform(1.05, 1.30)
        sx, sy = s, s
    if flags.get("aspect") and rng.random() < 0.5:
        r = rng.uniform(0.9, 1.1)
        sx *= r
        sy *= 1.0 / r
    if flags.get("shear") and rng.random() < 0.5:
        k = rng.uniform(-0.15, 0.15)
        b += k
        d += k
    c += w * (1 - sx) / 2
    f += h * (1 - sy) / 2
    return (a, b, c, d, e, f)
def _affine(img, mask_np, flags, rng):
    if not (flags.get("translate") or flags.get("zoom_out") or flags.get("zoom_in")
            or flags.get("aspect") or flags.get("shear")):
        return img, mask_np
    w, h = img.size
    coeffs = _affine_coeffs(w, h, flags, rng)
    img = img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR,
                        fillcolor=_fill_color(img))
    if mask_np is not None:
        mask_pil = _mask_to_pil(mask_np).transform((w, h), Image.AFFINE, coeffs, resample=Image.NEAREST, fillcolor=0)
        mask_np = _mask_to_np(mask_pil)
    return img, mask_np
def _photometric(img, flags, rng):
    if flags.get("brightness") and rng.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    if flags.get("contrast") and rng.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    if flags.get("invert") and rng.random() < 0.5:
        img = ImageOps.invert(img.convert("RGB"))
    if flags.get("noise") and rng.random() < 0.5:
        # 用 rng 派生种子，保证灰度图+高度图噪声一致且不污染全局随机状态
        nstate = np.random.RandomState(rng.randrange(2**32))
        arr = np.asarray(img).astype(np.float32)
        noise = nstate.normal(0.0, nstate.uniform(3.0, 12.0), arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return img
def _augment_image(img, flags, rng):
    mask_np = None
    if flags.get("flip"):
        img, mask_np = _flip(img, mask_np, rng)
    if flags.get("rotate"):
        img, mask_np = _rotate(img, mask_np, rng)
    img, mask_np = _affine(img, mask_np, flags, rng)
    return _photometric(img, flags, rng)
def apply_image_augment(img, flags=None, rng=None):
    if not flags or not flags.get("enabled", True):
        return img
    img = _to_pil(img)
    orig_mode = img.mode
    out = _augment_image(img, flags, rng or random.Random())
    if out.mode != orig_mode:
        out = out.convert(orig_mode)
    return out
def build_augment_fn(flags):
    if not flags or not flags.get("enabled", True):
        return None
    def augment(image, mask):
        img = _to_pil(image)
        mask_np = np.asarray(mask).astype(np.int64)
        if flags.get("flip"):
            img, mask_np = _flip(img, mask_np, random.Random())
        if flags.get("rotate"):
            img, mask_np = _rotate(img, mask_np, random.Random())
        img, mask_np = _affine(img, mask_np, flags, random.Random())
        return _photometric(img, flags, random.Random()), mask_np
    return augment
def flags_to_yolo_args(flags):
    if not flags or not flags.get("enabled", True):
        return {}
    args = {}
    if flags.get("rotate"):
        args["degrees"] = 15.0
    if flags.get("translate"):
        args["translate"] = 0.2
    if flags.get("shear"):
        args["shear"] = 10.0
    if flags.get("zoom_out") or flags.get("zoom_in"):
        args["scale"] = 0.5
    if flags.get("flip"):
        args["fliplr"] = 0.5
        args["flipud"] = 0.3
    if flags.get("brightness"):
        args["hsv_v"] = 0.3
    if flags.get("contrast"):
        args["hsv_s"] = 0.5
    return args
