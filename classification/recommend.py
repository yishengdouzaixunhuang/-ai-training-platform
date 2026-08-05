# -*- coding: utf-8 -*-
"""分类训练参数智能推荐器。

输入：项目目录（数据集统计）+ 硬件信息 + 可选目标推理耗时(ms)
输出：推荐训练参数字典 + 每条参数的理由 + 环境摘要

规则参考工业视觉经验：
- 训练尺寸：最大图边长 x0.6 对齐常用档位（公司软件逻辑）
- 模型：推理预算优先，类别数/数据量修正
- batch：显存驱动；epochs：数据量驱动；loss：类别不平衡驱动
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

import numpy as np

# 与训练面板一致的模型池（按复杂度升序）
MODEL_POOL = ["mobilenet_v3_small", "mobilenet_v2", "mobilenet_v3_large",
              "resnet18", "resnet34", "efficientnet_b0",
              "resnet50", "efficientnet_b3", "resnet101", "vit_b_16", "vit_b_32"]

SIZE_TIERS = [128, 192, 256, 384, 512, 640, 768]


def _nearest_tier(value: float, direction: str = "nearest") -> int:
    """把目标边长对齐到常用档位。direction: nearest/up/down"""
    if direction == "up":
        for t in SIZE_TIERS:
            if t >= value:
                return t
        return SIZE_TIERS[-1]
    if direction == "down":
        best = SIZE_TIERS[0]
        for t in SIZE_TIERS:
            if t <= value:
                best = t
        return best
    return min(SIZE_TIERS, key=lambda t: abs(t - value))


# ============ 输入采集 ============

def collect_dataset_stats(project_dir: str, max_scan: int = 300) -> Dict:
    """统计数据集：样本数、类别数、每类样本数、图片尺寸（抽样）。"""
    stats = {
        "total": 0, "num_classes": 0, "per_class": [], "class_names": [],
        "imbalance_ratio": 1.0, "max_w": 0, "max_h": 0, "avg_edge": 0.0,
        "scanned": 0, "has_labels": False,
    }
    # 类别统计
    label_path = os.path.join(project_dir, "annotations", "class_labels.json")
    mapping = {}
    if os.path.exists(label_path):
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            labels = data.get("labels", [])
            mapping = data.get("mapping", {})
            stats["class_names"] = list(labels)
            stats["num_classes"] = len(labels)
            counts = [0] * len(labels)
            for v in mapping.values():
                try:
                    counts[int(v)] += 1
                except (TypeError, ValueError, IndexError):
                    pass
            stats["per_class"] = counts
            stats["total"] = sum(counts)
            stats["has_labels"] = stats["total"] > 0
            pos = [c for c in counts if c > 0]
            if pos:
                stats["imbalance_ratio"] = max(pos) / max(1.0, min(pos))
        except Exception:
            pass
    # 图片尺寸抽样
    img_dir = os.path.join(project_dir, "images")
    if os.path.isdir(img_dir):
        files = sorted(
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))
        )
        sample = files[:max_scan]
        edges = []
        from PIL import Image
        for p in sample:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                stats["max_w"] = max(stats["max_w"], w)
                stats["max_h"] = max(stats["max_h"], h)
                edges.append((w + h) / 2.0)
            except Exception:
                continue
        stats["scanned"] = len(edges)
        if edges:
            stats["avg_edge"] = float(np.mean(edges))
    if stats["max_w"] == 0 and stats["max_h"] == 0:
        stats["max_w"] = stats["max_h"] = 512
    return stats


def collect_hardware() -> Dict:
    """采集 GPU 型号/显存、CPU、内存。优先 pynvml，兜底 torch.cuda。"""
    hw = {
        "gpu_name": "", "gpu_mem_gb": 0.0, "gpu_available": False,
        "amp_supported": False, "cpu_cores": os.cpu_count() or 4,
        "ram_gb": 0.0, "device_label": "CPU",
    }
    try:
        import psutil
        hw["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        hw["gpu_name"] = name.decode() if isinstance(name, bytes) else str(name)
        hw["gpu_mem_gb"] = round(mem.total / 1024 ** 3, 1)
        hw["gpu_available"] = True
        hw["amp_supported"] = True
        hw["device_label"] = f"GPU {hw['gpu_name']} ({hw['gpu_mem_gb']:.1f}GB)"
        return hw
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            hw["gpu_name"] = p.name
            hw["gpu_mem_gb"] = round(p.total_memory / 1024 ** 3, 1)
            hw["gpu_available"] = True
            hw["amp_supported"] = True
            hw["device_label"] = f"GPU {p.name} ({hw['gpu_mem_gb']:.1f}GB)"
    except Exception:
        pass
    return hw


# ============ 推荐逻辑 ============

def recommend(project_dir: str = "",
              dataset: Optional[Dict] = None,
              hardware: Optional[Dict] = None,
              target_latency_ms: float = 0.0) -> Dict:
    """生成推荐训练参数。

    Args:
        project_dir: 项目目录（dataset 为 None 时自动采集）
        dataset: 数据集统计（可选）
        hardware: 硬件信息（可选）
        target_latency_ms: 目标推理耗时(ms)，0 = 不约束

    Returns:
        {"params": {...}, "reasons": {...}, "summary": {...}}
    """
    dataset = dataset or (collect_dataset_stats(project_dir) if project_dir else collect_dataset_stats(""))
    hw = hardware or collect_hardware()
    params: Dict = {}
    reasons: Dict[str, str] = {}

    total = max(1, dataset["total"])
    ncls = max(1, dataset["num_classes"])
    per_class = [c for c in dataset.get("per_class", []) if c > 0] or [total]
    min_per_class = min(per_class)
    avg_per_class = total / ncls
    imbalance = dataset.get("imbalance_ratio", 0.0) or 0.0
    if imbalance <= 0.0 and per_class:
        _pos = [c for c in per_class if c > 0]
        if len(_pos) >= 2:
            imbalance = max(_pos) / max(1.0, min(_pos))
        else:
            imbalance = 1.0
    max_edge = max(dataset.get("max_w", 0), dataset.get("max_h", 0))
    gpu = hw.get("gpu_available", False)
    gpu_mem = hw.get("gpu_mem_gb", 0.0)
    latency = max(0.0, target_latency_ms or 0.0)

    # ---- 训练尺寸：最大图 x0.6 对齐档位 ----
    # 类别少的二分类任务对缺陷细节更敏感：档位取高，且缩放保底 0.8，
    # 避免推荐出过低的训练分辨率导致细节丢失。
    raw_target = max_edge * 0.6
    tier_dir = "up" if (ncls >= 8 or ncls <= 2) else "nearest"
    target = _nearest_tier(raw_target, tier_dir)
    target = max(128, min(768, target))
    if max_edge > 0 and target < max_edge:
        scale = round(target / max_edge, 2)
        scale = max(0.1, min(1.0, scale))
        if ncls <= 2:
            scale = max(0.8, scale)
    else:
        scale = 1.0
    params["scale_w"] = params["scale_h"] = scale
    if ncls >= 8:
        scale_note = "类别多,取高档"
    elif ncls <= 2:
        scale_note = "类别少,取高档,缩放保底0.8"
    else:
        scale_note = "就近取档"
    reasons["scale"] = (f"最大图边长 {max_edge}px × 0.6 → 目标 {target}px"
                        f"（{scale_note}）")

    # ---- 模型：推理预算优先，类别数/数据量修正 ----
    if latency > 0:
        if latency <= 10:
            idx = MODEL_POOL.index("mobilenet_v3_small")
            lat_desc = f"推理预算 ≤10ms"
        elif latency <= 30:
            idx = MODEL_POOL.index("resnet18")
            lat_desc = f"推理预算 ≤30ms"
        elif latency <= 60:
            idx = MODEL_POOL.index("resnet34")
            lat_desc = f"推理预算 ≤60ms"
        elif latency <= 120:
            idx = MODEL_POOL.index("resnet50")
            lat_desc = f"推理预算 ≤120ms"
        else:
            idx = MODEL_POOL.index("efficientnet_b0")
            lat_desc = f"推理预算 {latency:.0f}ms"
    else:
        idx = MODEL_POOL.index("resnet18")
        lat_desc = "无推理预算约束"
    if ncls >= 10:
        idx = min(idx + 1, len(MODEL_POOL) - 3)  # 类别多升一档
    if total < 300:
        idx = max(idx - 1, 0)  # 数据少降一档
    if not gpu:
        # CPU-only 封顶 resnet18（类别极多时 efficientnet_b0）
        cap = MODEL_POOL.index("resnet18") if ncls < 10 else MODEL_POOL.index("efficientnet_b0")
        idx = min(idx, cap)
    if gpu and gpu_mem < 4.0:
        idx = min(idx, MODEL_POOL.index("resnet18"))
    model = MODEL_POOL[idx]
    params["model"] = model
    reasons["model"] = (f"{lat_desc}；类别数 {ncls}、样本 {total}"
                        f"{'，升档' if ncls >= 10 else ''}{'，降档(数据少)' if total < 300 else ''}"
                        f"{'，CPU 封顶' if not gpu else ''} → {model}")

    # ---- batch：显存驱动 ----
    if not gpu:
        batch = 8
        batch_desc = "CPU 训练，小 batch"
    elif gpu_mem < 4:
        batch = 16; batch_desc = f"显存 {gpu_mem:.1f}GB"
    elif gpu_mem < 8:
        batch = 32; batch_desc = f"显存 {gpu_mem:.1f}GB"
    elif gpu_mem < 16:
        batch = 64; batch_desc = f"显存 {gpu_mem:.1f}GB"
    else:
        batch = 128; batch_desc = f"显存 {gpu_mem:.1f}GB"
    heavy = MODEL_POOL.index(model) >= MODEL_POOL.index("resnet101")
    if gpu and (heavy or target >= 640):
        batch = max(4, batch // 2)
        batch_desc += "，模型大/输入大减半"
    elif gpu and target >= 512:
        batch = max(4, int(batch * 0.75))
    params["batch_size"] = batch
    reasons["batch_size"] = f"{batch_desc} → {batch}"

    # ---- epochs / 增强：数据量驱动 ----
    if total < 300:
        epochs = 100
    elif total < 1000:
        epochs = 120
    elif total < 5000:
        epochs = 150
    else:
        epochs = 200
    if min_per_class < 20:
        epochs = min(epochs, 120)
        reasons["epochs"] = (f"总样本 {total}，且每类最少 {min_per_class} 张（少样本防过拟合）→ {epochs}")
    else:
        reasons["epochs"] = f"总样本 {total} → {epochs}"

    if min_per_class < 30 or total < 300:
        augment = "heavy"
    elif avg_per_class < 100:
        augment = "medium"
    else:
        augment = "light"
    reasons["augment"] = f"每类平均 {avg_per_class:.0f} 张 → {augment} 增强"
    params.update({"epochs": epochs, "augment": augment})

    # ---- loss：类别不平衡 ----
    if imbalance >= 3.0:
        loss = "focal"
        reasons["loss"] = f"类别不平衡 {imbalance:.1f}:1 → focal loss（聚焦难样本）"
    else:
        loss = "label_smoothing"
        reasons["loss"] = f"类别较均衡（{imbalance:.1f}:1）→ label smoothing 防过拟合"
    params["loss"] = loss
    params["label_smoothing"] = 0.1
    params["focal_gamma"] = 2.0

    # ---- 类平衡权重：类别不平衡时建议开启 ----
    if imbalance >= 2.0:
        params["class_balanced"] = True
        reasons["class_balanced"] = f"类别不平衡 {imbalance:.1f}:1 → 建议开启类平衡权重"
    else:
        params["class_balanced"] = False
        reasons["class_balanced"] = f"类别较均衡（{imbalance:.1f}:1）→ 无需类平衡权重"

    # ---- lr / optimizer / scheduler ----
    lr = 0.0005 if total < 300 else 0.001
    params["lr"] = lr
    reasons["lr"] = f"数据{'少(下调)' if total < 300 else '正常'} → lr={lr}"
    params["optimizer"] = "AdamW"
    params["weight_decay"] = 0.0001
    params["momentum"] = 0.9
    params["lr_scheduler"] = "cosine"
    params["warmup_epochs"] = 5
    params["early_stop_patience"] = 30 if total < 1000 else 20
    params["dropout"] = 0.1 if model.startswith("vit") else 0.0
    params["resume"] = False

    # ---- AMP ----
    params["use_amp"] = bool(gpu and gpu_mem >= 4.0)
    reasons["use_amp"] = "GPU 显存充足 → 开启 AMP 混合精度" if params["use_amp"] else "CPU 或无独立显存 → 关闭 AMP"

    # ---- K-Fold ----
    if total < 200:
        params["k_folds"] = 3
        reasons["k_folds"] = f"样本仅 {total} 张 → 建议 3 折交叉验证"
    else:
        params["k_folds"] = 1
        reasons["k_folds"] = f"样本 {total} 张 → 单次训练+验证集"

    # ---- EMA / TTA：训练与推理侧稳定化 ----
    params["ema"] = True
    reasons["ema"] = "训练时自动启用 EMA 权重平均，验证与最优模型更稳（无额外开销）"
    params["tta"] = True
    reasons["tta"] = "单张推理默认开启 TTA 多裁剪投票，提升单图判定稳定性"

    summary = {
        "dataset": {
            "total": dataset.get("total", 0),
            "num_classes": ncls,
            "min_per_class": min_per_class,
            "imbalance_ratio": round(imbalance, 2),
            "max_edge": max_edge,
            "avg_edge": round(dataset.get("avg_edge", 0.0), 1),
            "scanned": dataset.get("scanned", 0),
        },
        "hardware": {
            "device": hw.get("device_label", "CPU"),
            "gpu_name": hw.get("gpu_name", ""),
            "gpu_mem_gb": hw.get("gpu_mem_gb", 0.0),
            "cpu_cores": hw.get("cpu_cores", 0),
            "ram_gb": hw.get("ram_gb", 0.0),
        },
        "target_latency_ms": latency,
    }
    return {"params": params, "reasons": reasons, "summary": summary}


# 面板控件可套用的字段顺序（用于弹窗展示）
PARAM_ORDER = [
    ("model", "模型"), ("scale_w", "宽缩放"), ("scale_h", "高缩放"),
    ("epochs", "训练轮数"), ("batch_size", "Batch"), ("lr", "学习率"),
    ("optimizer", "优化器"), ("loss", "损失函数"), ("class_balanced", "类平衡权重"), ("augment", "数据增强"),
    ("use_amp", "AMP"), ("k_folds", "K-Fold"), ("weight_decay", "权重衰减"),
    ("momentum", "动量"), ("label_smoothing", "Label Smooth"),
    ("focal_gamma", "Focal Gamma"), ("dropout", "Dropout"),
    ("lr_scheduler", "LR 调度"), ("warmup_epochs", "Warmup"),
    ("early_stop_patience", "早停耐心"), ("resume", "断点续训"),
    ("ema", "EMA 权重平均"), ("tta", "推理 TTA"),
]
