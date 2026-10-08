# -*- coding: utf-8 -*-
"""推理参数（对标友商 iSense 的「推理参数」面板）。

友商的参数面板是**按类别**给每个类一组过滤区间：
    分数 / 面积 / 宽度 / 高度 / 角度 / 灰度   —— 每项 min ~ max
外加全局项：孔洞填充 / 中心半径 / 排序 / 最大检测个数。

我们的推理是**语义分割**（输出像素级类别图），所以这些参数对应到
"对每个类别的**连通域**做后处理过滤"：
    面积   = 连通域像素数
    宽度/高度 = 最小外接矩形 (cv2.minAreaRect) 的 w / h
    角度   = 最小外接矩形的角度（°）
    灰度   = 原图灰度在该连通域内的均值
    分数   = 该连通域内的平均置信度（**需要概率图**；当前后端没提供时该项自动跳过并说明）
    孔洞填充 = 填掉面积 ≤ N 的内部空洞（0=关）
    中心半径 = 连通域的等效半径 sqrt(面积/π) 下限（0=关）
    排序/最大检测个数 = 结果表里块的排序方式与每类保留上限

默认值刻意与友商一致：min 填 0 / 5 / 1 / 1，max 填**负数表示"不限"**。
这样默认状态下唯一真正生效的是"面积 ≥ 5 px" —— 恰好是我们最需要的那条
（实测：缺陷中位尺寸 11×12 px，67.7% 小于 32 px，碎点必须能滤掉）。
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

PARAMS_FILENAME = "inference_params.json"

# 每个类别的过滤区间：key -> (中文名, 默认 [min, max])
RANGE_KEYS = [
    ("score", "分数", [0.0, -1.0]),
    ("area", "面积", [5.0, -1e9]),
    ("width", "宽度", [1.0, -1e9]),
    ("height", "高度", [1.0, -1e9]),
    ("angle", "角度", [-180.0, -180.0]),
    ("gray", "灰度", [0.0, -255.0]),
]
RANGE_LABELS = {k: n for k, n, _ in RANGE_KEYS}

SORT_CHOICES = ["面积降序", "面积升序", "分数降序", "宽度降序", "无"]
GLOBAL_DEFAULTS = {
    "hole_fill": 0,        # 孔洞填充：填掉面积 <= N 的内部空洞
    "center_radius": 0,    # 中心半径：等效半径下限（0=关）
    "sort": "面积降序",
    "max_det": 99999,      # 每类最大保留个数
}


def default_class_params():
    return {k: list(v) for k, _, v in RANGE_KEYS}


def default_params(classes=None):
    """整份默认参数：classes 为 None 时只给 global（类别按需补）。"""
    p = {"version": 1, "global": dict(GLOBAL_DEFAULTS), "classes": {}}
    for c in (classes or []):
        if c == "background":
            continue
        p["classes"][c] = default_class_params()
    return p


def params_path(project_dir):
    return os.path.join(project_dir, PARAMS_FILENAME)


def load_params(project_dir, classes=None):
    """读工程参数；缺失/损坏时给默认值（**并把缺的类别补齐**）。"""
    p = default_params(classes)
    path = params_path(project_dir)
    if project_dir and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                p["global"].update({k: v for k, v in (d.get("global") or {}).items()
                                    if k in GLOBAL_DEFAULTS})
                for c, rng in (d.get("classes") or {}).items():
                    base = default_class_params()
                    if isinstance(rng, dict):
                        for k, _n, _d in RANGE_KEYS:
                            v = rng.get(k)
                            if isinstance(v, (list, tuple)) and len(v) == 2:
                                base[k] = [float(v[0]), float(v[1])]
                    p["classes"][c] = base
        except Exception:
            pass
    for c in (classes or []):
        if c != "background" and c not in p["classes"]:
            p["classes"][c] = default_class_params()
    return p


def save_params(project_dir, params):
    if not project_dir:
        return False
    try:
        with open(params_path(project_dir), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


# ───────────────────────── 过滤实现 ─────────────────────────

def _range_ok(value, rng):
    """区间判定：max < 0 视为"不限"；角度 min==max==-180 也视为"不限"。"""
    if rng is None:
        return True, ""
    try:
        lo, hi = float(rng[0]), float(rng[1])
    except (TypeError, IndexError):
        return True, ""
    if value < lo:
        return False, "小于下限"
    if hi >= 0 and value > hi:
        return False, "超过上限"
    return True, ""


def _angle_unset(rng):
    try:
        return abs(float(rng[0]) + 180.0) < 1e-6 and abs(float(rng[1]) + 180.0) < 1e-6
    except (TypeError, IndexError):
        return True


def fill_holes(binary, max_area):
    """填掉面积 <= max_area 的内部空洞（不与图像边界相连的背景）。"""
    if max_area <= 0:
        return binary
    import cv2
    inv = (1 - binary).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    h, w = binary.shape
    out = binary.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # 贴着图像边界的一定是外部背景，不是空洞
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        if area <= max_area:
            out[lab == i] = 1
    return out


def component_props(binary, gray=None, prob=None, min_center_radius=0):
    """把二值图拆成连通域属性列表（面积/宽高/角度/灰度/分数/半径）。"""
    import cv2
    props = []
    if not binary.any():
        return props
    n, lab, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area <= 0:
            continue
        region = (lab == i)
        radius = float(np.sqrt(area / np.pi))
        if min_center_radius and radius < float(min_center_radius):
            props.append(dict(idx=i, area=area, width=bw, height=bh, angle=0.0,
                              gray=0.0, score=-1.0, radius=radius,
                              bbox=(x, y, bw, bh), drop="中心半径不足"))
            continue
        # 宽度/高度用**外接矩形 (bbox)**，不用 minAreaRect：
        #   minAreaRect 对细长块会把长边当 width（5×10 的块会报 width=10），
        #   用户按"宽度 8"过滤时看着结果表里的"宽度 10"却还在，纯属自找麻烦。
        #   bbox 与结果表里显示的宽度/高度是同一个口径。角度仍用 minAreaRect。
        cnts, _ = cv2.findContours(region.astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ang = 0.0
        if cnts:
            _rect, _wh, ang_ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
            ang = float(ang_)
            # 归一到 (-45, 45]：0 = 横平竖直，±45 = 45° 斜。
            # OpenCV 各版本 minAreaRect 的角度约定不一致（(0,90] / [-90,0) 都见过，
            # 实测本机是 [-90,0)），不归一的话同一块在不同版本上算出的角度能差 90°。
            while ang <= -45.0:
                ang += 90.0
            while ang > 45.0:
                ang -= 90.0
        g = float(gray[region].mean()) if (gray is not None and gray.shape == binary.shape) else 0.0
        sc = float(prob[region].mean() if (prob is not None and prob.shape == binary.shape) else -1.0)
        props.append(dict(idx=i, area=area, width=float(bw), height=float(bh), angle=ang,
                          gray=g, score=sc, radius=radius, bbox=(x, y, bw, bh), drop=""))
    return props


def _sort_props(props, how):
    if how == "面积升序":
        return sorted(props, key=lambda p: p["area"])
    if how == "分数降序":
        return sorted(props, key=lambda p: -p.get("score", -1.0))
    if how == "宽度降序":
        return sorted(props, key=lambda p: -p["width"])
    if how == "无":
        return list(props)
    return sorted(props, key=lambda p: -p["area"])       # 面积降序（默认）


def filter_prediction(pred, classes=None, params=None, gray=None, prob=None,
                      image=None):
    """按推理参数过滤分割结果。

    返回 (filtered_mask, report)：
        filtered_mask —— 只保留通过过滤的连通域（原来的类别标号不变）
        report        —— dict(kept=[每类保留数], dropped=[每类丢弃数],
                              reasons=Counter, kept_props=[(类名, 属性)...],
                              score_skipped=bool, no_angle=bool)
    """
    report = dict(kept={}, dropped={}, reasons=Counter(), kept_props=[],
                  score_skipped=False, score_checked=False)
    if pred is None:
        return pred, report
    params = params or default_params(classes)
    gcfg = dict(GLOBAL_DEFAULTS)
    gcfg.update(params.get("global") or {})
    cls_cfg = params.get("classes") or {}

    if gray is None and image is not None:
        try:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
        except Exception:
            gray = None
    if gray is not None and getattr(gray, "shape", None) != pred.shape:
        gray = None
    if prob is not None and getattr(prob, "shape", None) != pred.shape:
        prob = None

    names = list(classes or [])
    out = np.zeros_like(pred)
    max_det = int(gcfg.get("max_det", 99999) or 99999)
    hole_fill = int(gcfg.get("hole_fill", 0) or 0)
    center_r = int(gcfg.get("center_radius", 0) or 0)

    for cid in sorted(set(np.unique(pred).tolist()) - {0, 255}):
        cname = names[cid] if cid < len(names) else "Class%d" % cid
        rng = cls_cfg.get(cname) or default_class_params()
        binary = (pred == cid)
        if hole_fill > 0:
            binary = fill_holes(binary, hole_fill)
        props = component_props(binary, gray=gray, prob=prob, min_center_radius=center_r)
        if not props:
            report["kept"][cname] = 0
            report["dropped"][cname] = 0
            continue

        use_score = any(p["score"] >= 0 for p in props)
        if use_score:
            report["score_checked"] = True
        elif float(rng.get("score", [0, -1])[0]) > 0 or float(rng.get("score", [0, -1])[1]) >= 0:
            # 参数里要求按分数过滤，但这次没有概率图 -> 明确记下来，别装作过滤过了
            report["score_skipped"] = True

        kept, dropped = [], 0
        for p in props:
            if p.get("drop"):
                dropped += 1
                report["reasons"][p["drop"]] += 1
                continue
            bad = ""
            for key in ("area", "width", "height", "gray"):
                ok, why = _range_ok(p[key], rng.get(key))
                if not ok:
                    bad = "%s%s" % (RANGE_LABELS[key], why)
                    break
            if not bad and (_range_ok(p["score"], rng.get("score"))[0] is False) and use_score:
                bad = "分数不在区间"
            if not bad and not _angle_unset(rng.get("angle")):
                ok, why = _range_ok(p["angle"], rng.get("angle"))
                if not ok:
                    bad = "角度%s" % why
            if bad:
                dropped += 1
                report["reasons"][bad] += 1
            else:
                kept.append(p)

        kept = _sort_props(kept, gcfg.get("sort"))
        if len(kept) > max_det:
            report["reasons"]["超过最大检测个数"] += len(kept) - max_det
            dropped += len(kept) - max_det
            kept = kept[:max_det]
        keep_idx = {p["idx"] for p in kept}
        # 用原图二值 + 保留的连通域重建（孔洞填充后的形状一起保留）
        n, lab = cv2_connected(binary)
        for p in kept:
            out[lab == p["idx"]] = cid
            report["kept_props"].append((cname, p))
        report["kept"][cname] = len(kept)
        report["dropped"][cname] = dropped
    return out, report


def cv2_connected(binary):
    import cv2
    return cv2.connectedComponents(binary.astype(np.uint8), 8)


def summarize(report, limit=4):
    """把过滤结果压成一行日志。"""
    if not report:
        return ""
    parts = []
    for cname, n in report.get("kept", {}).items():
        d = report.get("dropped", {}).get(cname, 0)
        if d:
            parts.append("%s 保留%d/丢%d" % (cname, n, d))
        elif n:
            parts.append("%s %d" % (cname, n))
    why = "、".join("%s×%d" % (k, v) for k, v in report.get("reasons", Counter()).most_common(limit))
    txt = "；".join(parts[:6])
    if why:
        txt += "（丢弃原因：%s）" % why
    if report.get("score_skipped"):
        txt += " ⚠ 分数过滤已跳过：当前后端没有概率图"
    return txt
