# -*- coding: utf-8 -*-
"""Classification decision-threshold optimization — OK/NG 阈值调优 (过杀率/漏检率平衡)

工业 OK/NG 分类里，argmax 往往不是最优判定。真正决定产线表现的是**判定阈值**：
    NG概率 >= 阈值  ->  判 NG
阈值调高 -> 过杀下降但漏检上升；调低 -> 反之。本模块在验证集上扫阈值，
按产线约束（例如 漏检率 <= 0.2%）挑出**过杀率最小**的阈值。

用法（独立运行）:
    python -m classification.threshold <project_dir> [--split val] [--pos-class NG]
用法（代码调用）:
    from classification.threshold import collect_probs, binary_report, optimize_threshold
    probs, labels, names = collect_probs(proj, split="val", tta=True)
    rows = binary_report(probs, labels, pos_idx)
    best = optimize_threshold(probs, labels, pos_idx, max_miss_rate=0.002)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ClassificationDataset


# ────────────────────────────── 概率收集 ──────────────────────────────
@torch.inference_mode()
def collect_probs(project_dir, split="val", model_path="best_model.pth",
                  batch_size=16, image_size=224, device=None,
                  tta=False, log_callback=None):
    """跑一遍数据集，返回 (probs[N,C], labels[N], class_names)。tta=True 时用水平翻转平均。"""
    project_dir = Path(project_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    from .trainer import _create_classification_model, resolve_model_path
    ckpt_path = resolve_model_path(project_dir, model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    class_names = ckpt.get("class_names", [f"class_{i}" for i in range(num_classes)])
    model_name = ckpt.get("model_name", "resnet18")
    keep_aspect = bool(ckpt.get("keep_aspect", False))

    model = _create_classification_model(model_name, num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = ClassificationDataset(str(project_dir), split=split, image_size=image_size,
                               keep_aspect=keep_aspect)
    if len(ds) == 0:
        raise ValueError(f"split='{split}' 没有样本")

    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False,
                        num_workers=0, drop_last=False)

    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        p = torch.softmax(model(images), dim=1)
        if tta:
            # 水平翻转 + 垂直翻转 平均（多视图投票）
            for flip in (2, 3):  # torch.flip dims: 2=H, 3=W
                p = p + torch.softmax(model(torch.flip(images, dims=[flip])), dim=1)
            p = p / 3.0
        all_probs.append(p.cpu().numpy())
        all_labels.append(labels.numpy())
        if log_callback:
            log_callback(f"已处理 {len(all_labels) * batch_size} 张")
    probs = np.concatenate(all_probs, 0)
    labels = np.concatenate(all_labels, 0).astype(int)
    return probs, labels, list(class_names)


# ────────────────────────────── 阈值扫描 ──────────────────────────────
def binary_report(probs, labels, pos_idx, thresholds=None):
    """对二分类（正类=pos_idx，通常是 NG/缺陷类）扫阈值，返回每个阈值的指标。

    漏检率(miss_rate) = FN / (TP+FN)   真NG 中被放过的比例  —— 越低越好
    过杀率(overkill_rate) = FP / (FP+TN)  真OK 中被误判NG的比例 —— 越低越好
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.01), 2)
    p = np.asarray(probs)[:, pos_idx]
    y = (np.asarray(labels) == pos_idx).astype(int)
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        miss = fn / (tp + fn) if (tp + fn) else 0.0
        over = fp / (fp + tn) if (fp + tn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "threshold": round(float(t), 4),
            "miss_rate": round(miss, 5),        # 漏检率
            "overkill_rate": round(over, 5),    # 过杀率
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "youden_j": round(rec - over, 4),
            "accuracy": round((tp + tn) / max(1, tp + fn + fp + tn), 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        })
    return rows


def optimize_threshold(probs, labels, pos_idx,
                       max_miss_rate=0.002, objective="min_overkill"):
    """挑最优阈值。

    objective:
      - "min_overkill": 在 miss_rate <= max_miss_rate 的前提下，过杀率最小（产线常用）
      - "youden":       最大化 Youden's J (= recall - 过杀率)
      - "f1":           最大化 F1
    返回 dict（含 threshold 与对应指标）。
    """
    rows = binary_report(probs, labels, pos_idx)
    if objective == "min_overkill":
        feas = [r for r in rows if r["miss_rate"] <= max_miss_rate]
        if not feas:
            # 无可行解：退化为漏检最小
            best = min(rows, key=lambda r: (r["miss_rate"], r["overkill_rate"]))
            best["fallback"] = "无阈值满足漏检约束，已退化为漏检最小"
            return best
        best = min(feas, key=lambda r: (r["overkill_rate"], r["miss_rate"]))
    elif objective == "youden":
        best = max(rows, key=lambda r: r["youden_j"])
    elif objective == "f1":
        best = max(rows, key=lambda r: r["f1"])
    else:
        raise ValueError(f"未知 objective: {objective}")
    best["objective"] = objective
    return best


def pick_pos_idx(class_names, pos_class=None):
    """确定"正类"（NG/缺陷类）索引。未指定时按名称猜测，猜不到用少数类。"""
    if pos_class:
        if pos_class in class_names:
            return class_names.index(pos_class)
    for i, n in enumerate(class_names):
        t = str(n).lower()
        if any(k in str(n) for k in ("NG", "不良", "缺陷", "异常")) or "ng" in t or "defect" in t:
            return i
    return len(class_names) - 1


def save_threshold_result(project_dir, result, pos_class_name, rows=None, out_name="threshold_report.json"):
    """把阈值结果写入项目 models/classification/ 下，便于推理端读取。"""
    out_dir = Path(project_dir) / "models" / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"pos_class": pos_class_name, "best": result}
    if rows is not None:
        payload["sweep"] = rows
    p = out_dir / out_name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(p)


# ────────────────────────────── CLI ──────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="分类阈值调优 (过杀率/漏检率)")
    ap.add_argument("project_dir")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--pos-class", default=None, help="正类(NG)名称，默认自动识别")
    ap.add_argument("--max-miss", type=float, default=0.002, help="允许的最大漏检率(默认0.002=0.2%%)")
    ap.add_argument("--objective", default="min_overkill", choices=["min_overkill", "youden", "f1"])
    ap.add_argument("--tta", action="store_true", help="启用多视图TTA")
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    probs, labels, names = collect_probs(args.project_dir, split=args.split, model_path=args.model,
                                         image_size=args.image_size, tta=args.tta,
                                         log_callback=lambda m: None)
    pos = pick_pos_idx(names, args.pos_class)
    rows = binary_report(probs, labels, pos)
    best = optimize_threshold(probs, labels, pos, max_miss_rate=args.max_miss,
                              objective=args.objective)
    out = save_threshold_result(args.project_dir, best, names[pos], rows)
    print(f"类别: {names}   正类(NG): {names[pos]}   样本: {len(labels)}")
    print(f"最优阈值: {best['threshold']}   漏检率: {best['miss_rate']*100:.3f}%   "
          f"过杀率: {best['overkill_rate']*100:.3f}%   F1: {best['f1']}   "
          f"(TP{best['tp']} FN{best['fn']} FP{best['fp']} TN{best['tn']})")
    print(f"报告已写: {out}")


if __name__ == "__main__":
    main()
