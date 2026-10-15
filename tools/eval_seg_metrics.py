# -*- coding: utf-8 -*-
"""语义分割评估 —— 10 项指标 + HTML 报表

为什么需要它：`training/trainer.py::validate()` 目前只算 mIoU 与逐类 IoU，
类别不平衡（背景占 95%+）时 mIoU 会被"大而简单"的类主导，看不出真实缺陷的漏检/过杀。
本工具补齐链路：像素级 -> 类别级 -> 边界级 -> 目标级，共 10 项指标。

十项指标
    像素级  1 Pixel Accuracy (PA)            整体正确率，最容易被背景稀释
            2 Cohen's Kappa (κ)             去掉"瞎猜也有的分"
            3 FWIoU                          按真值频率加权的 IoU
    类别级  4 mIoU / 逐类 IoU              分割任务的事实标准
            5 Macro Precision               对的里面有多少是对的（对应"过杀"）
            6 Macro Recall                  该找的找到了多少（对应"漏检"）
            7 Macro Dice (F1)               不平衡数据下比 IoU 更敏感
    边界级  8 Boundary F1 (BF Score)        轮廓贴合度，与容差 tol 有关
            9 HD95 (95% Hausdorff Distance) 最坏情况下的轮廓偏差（像素）
    目标级 10 Region P/R/F1                 连通域级别：直接对应产线的"漏检/过杀"

两种输入方式
    1) 掩膜目录对（零 torch 依赖，最快）
       python tools/eval_seg_metrics.py --gt-dir GT --pred-dir PRED --class-names background,Atd
    2) DL Studio 项目（加载 best_model.pth，走和部署一致的推理路径）
       python tools/eval_seg_metrics.py --project D:/proj --split val --html report.html

约定（重要）
    * 掩膜取值 = 类别 id（0,1,2,...），255 视为 ignore 像素，不参与任何指标。
    * 只有"前景类"（名字不等于 background/背景）参与宏平均 —— 这是本项目的铁律，
      否则背景会稀释掉所有差异。逐类明细里仍会给出背景类。
    * 距离类指标（HD95）在"某一类 GT 为空"时无定义 -> 记 NaN 并单独计数，不偷偷当 0。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from scipy import ndimage as ndi

IGNORE_VALUE = 255
BACKGROUND_NAMES = {"background", "bg", "背景", "ok", "good", "正常", "none", "empty"}

METRIC_ORDER = [
    ("pa",            "像素准确率 PA",        "像素级"),
    ("kappa",         "Cohen's Kappa",       "像素级"),
    ("fwiou",         "FWIoU",               "像素级"),
    ("miou",          "mIoU（含背景）",       "类别级"),
    ("miou_fg",       "mIoU（仅前景）",       "类别级"),
    ("macro_precision", "Macro Precision",   "类别级"),
    ("macro_recall",  "Macro Recall",        "类别级"),
    ("macro_dice",    "Macro Dice / F1",     "类别级"),
    ("bf",            "Boundary F1",         "边界级"),
    ("hd95",          "HD95 (px)",           "边界级"),
    ("region_f1",     "Region F1",           "目标级"),
]


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def to_label(m):
    """把任意输入（PIL / ndarray / 多通道）统一成 2D uint8 标签图。"""
    a = np.asarray(m)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[2] == 1 else np.argmax(a, axis=2)
    if a.dtype == bool:
        return a.astype(np.uint8)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8) if a.max() <= 255 else np.clip(a, 0, 255).astype(np.uint8)
    return a


def is_foreground(name: str) -> bool:
    return name.strip().lower() not in BACKGROUND_NAMES and not name.startswith("__")


def safe_div(a, b, default=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(a, b)
    if np.isscalar(out):
        return default if (b == 0 or np.isnan(out)) else float(out)
    return out


def nanmean(values):
    """对列表求均值，自动跳过 NaN；全 NaN 时返回 NaN。"""
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


# --------------------------------------------------------------------------
# 1. 混淆矩阵 -> 像素级 / 类别级
# --------------------------------------------------------------------------
def confusion_matrix(gt, pred, num_classes, ignore_value=IGNORE_VALUE):
    """累计混淆矩阵，行=真值，列=预测。ignore 像素整体剔除。"""
    valid = (gt != ignore_value) & (gt < num_classes) & (pred < num_classes)
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    idx = g * num_classes + p
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(cm, class_names):
    """由混淆矩阵导出像素级 + 类别级指标。"""
    num_classes = len(class_names)
    total = cm.sum()
    if total == 0:
        return {}, {}, {}

    tp = np.diag(cm).astype(np.float64)
    gt_sum = cm.sum(axis=1).astype(np.float64)   # 真值像素数
    pr_sum = cm.sum(axis=0).astype(np.float64)   # 预测像素数
    fp = pr_sum - tp
    fn = gt_sum - tp
    tn = total - tp - fp - fn

    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)

    pa = float(tp.sum() / total)
    pe = float((gt_sum * pr_sum).sum() / (total * total))
    kappa = float((pa - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else float("nan")
    fwiou = float(np.nansum((gt_sum / total) * iou))

    fg_ids = [i for i, n in enumerate(class_names) if is_foreground(n)]
    if not fg_ids:
        fg_ids = list(range(num_classes))

    pixel = {"pa": pa, "kappa": kappa, "fwiou": fwiou}
    cls = {
        "miou": nanmean(iou),
        "miou_fg": nanmean(iou[fg_ids]),
        "macro_precision": nanmean(prec[fg_ids]),
        "macro_recall": nanmean(rec[fg_ids]),
        "macro_dice": nanmean(dice[fg_ids]),
        "macro_specificity": nanmean(spec[fg_ids]),
    }
    per_class = {
        name: {
            "iou": float(iou[i]), "dice": float(dice[i]), "precision": float(prec[i]),
            "recall": float(rec[i]), "specificity": float(spec[i]),
            "tp": float(tp[i]), "fp": float(fp[i]), "fn": float(fn[i]),
            "gt_pixels": float(gt_sum[i]), "pred_pixels": float(pr_sum[i]),
            "gt_ratio": float(gt_sum[i] / total),
        }
        for i, name in enumerate(class_names)
    }
    return pixel, cls, per_class


# --------------------------------------------------------------------------
# 2. 边界级：BF Score / HD95
# --------------------------------------------------------------------------
def boundary_of(binary):
    """3x3 形态学腐蚀求内边界（1 像素宽）。

    坑：必须显式把画面外的像素当作背景（borderValue=0）。cv2.erode 的默认
    borderValue 是"+inf"，图像最外圈的像素因此腐蚀不掉 -> 贴着画面边缘的缺陷
    轮廓会被整条漏计（自测里 32x32 全前景图算出空边界，就是这么暴露的）。
    """
    b = np.ascontiguousarray(binary.astype(np.uint8))
    if cv2 is not None:
        er = cv2.erode(b, np.ones((3, 3), np.uint8), iterations=1,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    else:  # pragma: no cover
        er = ndi.binary_erosion(b, np.ones((3, 3), bool), border_value=0).astype(np.uint8)
    return (b & (1 - er)).astype(bool)


def _crop_pair(gt_b, pr_b, margin=64):
    """裁剪到两者并集的 bbox（留 margin），大幅降低 EDT 开销。"""
    union = gt_b | pr_b
    if not union.any():
        return gt_b, pr_b
    ys, xs = np.where(union)
    y0, y1 = max(0, ys.min() - margin), min(gt_b.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(gt_b.shape[1], xs.max() + margin + 1)
    return gt_b[y0:y1, x0:x1], pr_b[y0:y1, x0:x1]


def boundary_metrics(gt_bin, pred_bin, tol=2.0):
    """轮廓贴合度。返回 dict：

        precision/recall/f1/hd95 + n_gt/n_pred/hit_gt/hit_pred（后四项供全数据集累计）

    * 双方轮廓皆空 -> 视为完全一致（f1=1, hd95=0）；
    * 一方有、一方无 -> 轮廓完全对不上，f1=0、hd95 无定义(NaN)；
    * 同时返回命中计数，调用方可以跨图累加后再算 P/R —— 等价于"把所有边界点
      拼起来算"，比逐图平均更稳，也不会重复算距离场。
    """
    gt_b, pr_b = boundary_of(gt_bin), boundary_of(pred_bin)
    n_gt, n_pr = float(gt_b.sum()), float(pr_b.sum())
    base = {"n_gt": n_gt, "n_pred": n_pr, "hit_gt": 0.0, "hit_pred": 0.0}
    if n_gt == 0 and n_pr == 0:
        return dict(base, precision=1.0, recall=1.0, f1=1.0, hd95=0.0)
    if n_gt == 0 or n_pr == 0:
        return dict(base, precision=0.0, recall=0.0, f1=0.0, hd95=float("nan"))

    c_gt, c_pr = _crop_pair(gt_b, pr_b)
    dt_gt = ndi.distance_transform_edt(~c_gt)   # 每个像素到 GT 轮廓的距离
    dt_pr = ndi.distance_transform_edt(~c_pr)
    hit_pred = float((dt_gt[c_pr] <= tol).sum())   # 预测轮廓有多少落在容差带内
    hit_gt = float((dt_pr[c_gt] <= tol).sum())     # GT 轮廓有多少被覆盖

    precision = hit_pred / n_pr
    recall = hit_gt / n_gt
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    d_pred2gt = float(np.percentile(dt_gt[c_pr], 95))
    d_gt2pred = float(np.percentile(dt_pr[c_gt], 95))
    return {"precision": precision, "recall": recall, "f1": f1,
            "hd95": max(d_pred2gt, d_gt2pred),
            "n_gt": n_gt, "n_pred": n_pr, "hit_gt": hit_gt, "hit_pred": hit_pred}


# --------------------------------------------------------------------------
# 3. 目标级：连通域匹配
# --------------------------------------------------------------------------
def _regions(binary):
    """返回 (标签图, 区域数)。8 邻域连通。"""
    if not binary.any():
        return np.zeros(binary.shape, np.int32), 0
    if cv2 is not None:
        n, lab = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return lab, n - 1
    lab, n = ndi.label(binary, structure=np.ones((3, 3), bool))
    return lab, n


def match_regions(gt_bin, pred_bin, iou_thr=0.5, min_area=0):
    """贪心按 IoU 降序匹配连通域，返回 (n_gt, n_pred, n_match)。

    min_area > 0 时，面积小于该值的碎块不计入统计（既不判漏检也不判过杀）。
    工业上对应"最小可识别面积"口径（客户 KPI 里的 2mm² 折算成像素就是它），
    但阈值必须由人拍板，所以这里只提供开关，默认 0 = 不过滤。

    内存：早期版本分配 (n_gt+1)*(n_pred+1) 的交集矩阵，一张图有几千个碎块时
    会直接吃掉几百 MB。现在改用 np.unique 只在"真实重叠的配对"上展开。
    """
    gt_lab, n_gt_raw = _regions(gt_bin)
    pr_lab, n_pred_raw = _regions(pred_bin)
    area_gt = (np.bincount(gt_lab.ravel(), minlength=n_gt_raw + 1)
               if n_gt_raw else np.zeros(1, np.int64))
    area_pr = (np.bincount(pr_lab.ravel(), minlength=n_pred_raw + 1)
               if n_pred_raw else np.zeros(1, np.int64))

    gt_ids = np.arange(1, n_gt_raw + 1)
    pr_ids = np.arange(1, n_pred_raw + 1)
    if min_area > 0:
        gt_ids = gt_ids[area_gt[1:] >= min_area]
        pr_ids = pr_ids[area_pr[1:] >= min_area]
    n_gt, n_pred = len(gt_ids), len(pr_ids)
    if n_gt == 0 or n_pred == 0:
        return n_gt, n_pred, 0

    both = (gt_lab > 0) & (pr_lab > 0)
    if not both.any():
        return n_gt, n_pred, 0

    stride = n_pred_raw + 1
    flat = gt_lab[both].astype(np.int64) * stride + pr_lab[both].astype(np.int64)
    pair, cnt = np.unique(flat, return_counts=True)
    g_all = pair // stride
    p_all = pair % stride
    keep = np.isin(g_all, gt_ids) & np.isin(p_all, pr_ids)
    if not keep.any():
        return n_gt, n_pred, 0
    g_all, p_all = g_all[keep], p_all[keep]
    ia = cnt[keep].astype(np.float64)
    ua = area_gt[g_all] + area_pr[p_all] - ia
    iou = ia / np.maximum(ua, 1e-9)

    ok = iou >= iou_thr
    if not ok.any():
        return n_gt, n_pred, 0
    g_all, p_all, iou = g_all[ok], p_all[ok], iou[ok]

    # 贪心：IoU 高的优先，已匹配的 GT / Pred 不再参与
    order = np.argsort(-iou)
    used_g, used_p, n_match = set(), set(), 0
    for k in order:
        g, p = int(g_all[k]), int(p_all[k])
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p); n_match += 1
    return n_gt, n_pred, n_match


# --------------------------------------------------------------------------
# 4. 数据集级聚合
# --------------------------------------------------------------------------
def evaluate_pairs(pairs, class_names, tol=2.0, region_iou=0.5, ignore_value=IGNORE_VALUE,
                   min_area=0, log=None, verbose=True):
    """pairs: [(name, gt_mask, pred_mask), ...]，逐图累计。

    像素/类别级用全局累计混淆矩阵（不受单图大小影响）；
    边界级用累计命中计数；HD95 只能逐图算 -> 取均值与最差值；
    目标级用累计区域计数（等价于按目标数汇总，就是产线口径）。
    """
    num_classes = len(class_names)
    fg_ids = [i for i, n in enumerate(class_names) if is_foreground(n)] or list(range(num_classes))

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    bf_hit_pr = bf_hit_gt = bf_cnt_pr = bf_cnt_gt = 0.0
    hd95_values, hd95_undef = [], 0
    n_gt_reg = n_pr_reg = n_match_reg = 0
    region_per_class = {i: [0, 0, 0] for i in fg_ids}
    per_image, skipped = [], []

    for idx, (name, gt_raw, pred_raw) in enumerate(pairs):
        gt = to_label(gt_raw)
        pred = to_label(pred_raw)
        if gt.shape != pred.shape:
            skipped.append((name, "shape mismatch %s vs %s" % (gt.shape, pred.shape)))
            continue

        cm_total += confusion_matrix(gt, pred, num_classes, ignore_value)
        # 有效像素：GT 标 ignore 的不要；预测若吐出 255 这类非法类别同样不算
        valid = (gt != ignore_value) & (pred != ignore_value)

        img_hd95 = []
        for c in fg_ids:
            gt_c = (gt == c) & valid
            pr_c = (pred == c) & valid
            # 边界级：逐图算 HD95；命中计数跨图累计（等价于把所有轮廓点拼起来算 P/R）
            bm = boundary_metrics(gt_c, pr_c, tol=tol)
            bf_hit_pr += bm["hit_pred"]; bf_cnt_pr += bm["n_pred"]
            bf_hit_gt += bm["hit_gt"];   bf_cnt_gt += bm["n_gt"]
            if np.isnan(bm["hd95"]):
                hd95_undef += 1
            else:
                hd95_values.append(bm["hd95"]); img_hd95.append(bm["hd95"])
            # 目标级
            a, b, m = match_regions(gt_c, pr_c, iou_thr=region_iou, min_area=min_area)
            n_gt_reg += a; n_pr_reg += b; n_match_reg += m
            region_per_class[c][0] += a; region_per_class[c][1] += b; region_per_class[c][2] += m

        per_image.append({
            "name": name,
            "gt_fg_ratio": float(((gt > 0) & valid).mean()) if valid.any() else 0.0,
            "hd95": float(np.mean(img_hd95)) if img_hd95 else float("nan"),
        })
        if verbose and log:
            log("  [%d/%d] %s" % (idx + 1, len(pairs), name))

    pixel, cls, per_class = metrics_from_confusion(cm_total, class_names)

    bf_precision = safe_div(bf_hit_pr, bf_cnt_pr, 0.0)
    bf_recall = safe_div(bf_hit_gt, bf_cnt_gt, 0.0)
    bf = (2 * bf_precision * bf_recall / (bf_precision + bf_recall)
          if (bf_precision + bf_recall) > 0 else 0.0)

    region_precision = safe_div(n_match_reg, n_pr_reg, 0.0)
    region_recall = safe_div(n_match_reg, n_gt_reg, 0.0)
    region_f1 = (2 * region_precision * region_recall / (region_precision + region_recall)
                 if (region_precision + region_recall) > 0 else 0.0)

    result = {
        "n_images": len(per_image),
        "skipped": skipped,
        "pixel": pixel,
        "class": cls,
        "per_class": per_class,
        "boundary": {
            "precision": bf_precision, "recall": bf_recall, "bf": bf,
            "hd95": nanmean(hd95_values),
            "hd95_worst": float(np.max(hd95_values)) if hd95_values else float("nan"),
            "hd95_undefined": hd95_undef,
        },
        "region": {
            "precision": region_precision, "recall": region_recall, "f1": region_f1,
            "n_gt": n_gt_reg, "n_pred": n_pr_reg, "n_match": n_match_reg,
            "n_miss": n_gt_reg - n_match_reg,      # 漏检目标数
            "n_false": n_pr_reg - n_match_reg,     # 过杀目标数
            "per_class": {
                class_names[c]: {
                    "n_gt": v[0], "n_pred": v[1], "n_match": v[2],
                    "recall": safe_div(v[2], v[0], 0.0),
                    "precision": safe_div(v[2], v[1], 0.0),
                } for c, v in region_per_class.items()
            },
        },
        "confusion": cm_total.tolist(),
        "per_image": per_image,
        "config": {"tol": tol, "region_iou": region_iou, "min_area": min_area,
                   "class_names": class_names,
                   "fg_classes": [class_names[i] for i in fg_ids]},
    }
    result["metrics"] = flat_metrics(result)
    return result


def flat_metrics(res):
    """把结果摊平成 {metric_key: 数值}，顺序与 METRIC_ORDER 一致。"""
    return {
        "pa": res["pixel"]["pa"],
        "kappa": res["pixel"]["kappa"],
        "fwiou": res["pixel"]["fwiou"],
        "miou": res["class"]["miou"],
        "miou_fg": res["class"]["miou_fg"],
        "macro_precision": res["class"]["macro_precision"],
        "macro_recall": res["class"]["macro_recall"],
        "macro_dice": res["class"]["macro_dice"],
        "bf": res["boundary"]["bf"],
        "hd95": res["boundary"]["hd95"],
        "region_f1": res["region"]["f1"],
    }


# --------------------------------------------------------------------------
# 5. 报表输出
# --------------------------------------------------------------------------
def print_report(res, stream=sys.stdout):
    m = res["metrics"]
    print("=" * 74, file=stream)
    print("语义分割评估报告  |  图像 %d 张  |  容差 tol=%spx  区域 IoU 阈值=%s  最小面积=%s"
          % (res["n_images"], res["config"]["tol"], res["config"]["region_iou"],
             res["config"].get("min_area") or "不过滤"), file=stream)
    print("前景类: %s" % ", ".join(res["config"]["fg_classes"]), file=stream)
    print("=" * 74, file=stream)
    last_group = None
    for key, label, group in METRIC_ORDER:
        if group != last_group:
            print("-" * 74, file=stream)
            print("  [%s]" % group, file=stream)
            last_group = group
        v = m.get(key)
        txt = "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (
            "%.4f" % v if key != "hd95" else "%.2f" % v)
        print("    %-24s %s" % (label, txt), file=stream)
    print("-" * 74, file=stream)
    print("  逐类明细")
    for name, d in res["per_class"].items():
        print("    %-14s IoU=%.4f Dice=%.4f P=%.4f R=%.4f  真值像素占比=%.4f%%"
              % (name, d["iou"], d["dice"], d["precision"], d["recall"], d["gt_ratio"] * 100),
              file=stream)
    r = res["region"]
    print("-" * 74, file=stream)
    print("  目标级（连通域）: 真值 %d 个 / 预测 %d 个 / 命中 %d 个  -> 漏检 %d, 过杀 %d"
          % (r["n_gt"], r["n_pred"], r["n_match"], r["n_miss"], r["n_false"]), file=stream)
    print("    区域 Recall(不漏)=%.4f  区域 Precision(不误)=%.4f  区域 F1=%.4f"
          % (r["recall"], r["precision"], r["f1"]), file=stream)
    if res["boundary"]["hd95_undefined"]:
        print("  注意: 有 %d 个类的 HD95 无定义（某一方为空），已跳过未计入均值。"
              % res["boundary"]["hd95_undefined"], file=stream)
    if res["skipped"]:
        print("  跳过 %d 张（尺寸不一致等）: %s" % (len(res["skipped"]), res["skipped"][:3]), file=stream)
    print("=" * 74, file=stream)


def _fmt(v, digits=4):
    if v is None:
        return "n/a"
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return ("%%.%df" % digits) % v


def _judge(key, v):
    """给出红/黄/绿判定，用于报表着色。判据是工程经验值，不是论文标准。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "na", "无定义"
    if key == "hd95":
        if v <= 2: return "good", "优秀 ≤2px"
        if v <= 5: return "warn", "可接受 ≤5px"
        return "bad", "需改进 >5px"
    if key in ("pa", "kappa", "fwiou"):
        if v >= 0.99: return "good", "优秀"
        if v >= 0.95: return "warn", "可接受"
        return "bad", "偏低"
    if key in ("macro_recall", "macro_precision"):
        if v >= 0.90: return "good", "优秀"
        if v >= 0.75: return "warn", "可接受"
        return "bad", "偏低"
    if v >= 0.85: return "good", "优秀"
    if v >= 0.65: return "warn", "可接受"
    return "bad", "偏低"


def _fmtcount(v):
    """大数友好显示：1234 / 20.5k / 20.53M。"""
    if v < 10000:
        return "%d" % int(v)
    if v < 1e6:
        return "%.1fk" % (v / 1000.0)
    return "%.2fM" % (v / 1e6)


def render_html(res, meta, out_path):
    """生成自包含 HTML 报表（离线可看，无外部依赖）。"""
    m = res["metrics"]
    groups = {}
    for key, label, group in METRIC_ORDER:
        groups.setdefault(group, []).append((key, label))

    cards = []
    for group, items in groups.items():
        cards.append('<h3 class="grp">%s</h3><div class="grid">' % group)
        for key, label in items:
            v = m.get(key)
            cls, note = _judge(key, v)
            txt = _fmt(v, 2 if key == "hd95" else 4)
            unit = " px" if key == "hd95" else ""
            cards.append(
                '<div class="card %s"><div class="k">%s</div>'
                '<div class="v">%s<span class="u">%s</span></div>'
                '<div class="n">%s</div></div>' % (cls, label, txt, unit, note))
        cards.append("</div>")

    pc_rows = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (name, _fmt(d["iou"]), _fmt(d["dice"]), _fmt(d["precision"]), _fmt(d["recall"]),
           _fmt(d["specificity"]), "%.4f%%" % (d["gt_ratio"] * 100))
        for name, d in res["per_class"].items())

    cm = np.asarray(res["confusion"], dtype=float)
    names = res["config"]["class_names"]
    cm_max = cm.max() if cm.size and cm.max() > 0 else 1
    cm_head = "".join("<th>%s</th>" % n for n in names)
    cm_body = ""
    for i, row in enumerate(cm):
        cells = "".join(
            '<td style="background:rgba(%d,102,255,%.3f);color:%s">%s</td>'
            % (66, 0.10 + 0.75 * (val / cm_max), "#fff" if val / cm_max > 0.45 else "#1A2233",
               _fmtcount(val))
            for val in row)
        cm_body += "<tr><th>%s</th>%s</tr>" % (names[i], cells)

    reg = res["region"]
    rm = '<div class="region">共 %d 个真值目标，预测出 %d 个，命中 %d 个；' \
         '<b class="bad">漏检 %d</b> 个，<b class="warn">过杀 %d</b> 个。</div>' % (
             reg["n_gt"], reg["n_pred"], reg["n_match"], reg["n_miss"], reg["n_false"])

    meta_rows = "".join("<tr><th>%s</th><td>%s</td></tr>" % (k, v) for k, v in meta.items())

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>语义分割评估报告</title>
<style>
:root{--bg:#F5F7FA;--panel:#FFF;--line:#E3E8EF;--ink:#1A2233;--mut:#5B6B84;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;padding:28px 32px 56px;}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 10px;padding-left:10px;border-left:4px solid #4285F4}
h3.grp{font-size:13px;color:var(--mut);margin:18px 0 8px;letter-spacing:.06em}
.sub{color:var(--mut);font-size:13px;margin-bottom:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
 box-shadow:0 1px 2px rgba(16,24,40,.04)}
.card .k{font-size:12px;color:var(--mut);margin-bottom:6px}
.card .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}
.card .v .u{font-size:12px;font-weight:400;color:var(--mut);margin-left:2px}
.card .n{font-size:11px;color:var(--mut);margin-top:4px}
.card.good{border-left:4px solid #1E8E3E}.card.good .v{color:#1E8E3E}
.card.warn{border-left:4px solid #E8A33D}.card.warn .v{color:#B87413}
.card.bad{border-left:4px solid #D93025}.card.bad .v{color:#D93025}
.card.na{border-left:4px solid #9AA5B4}
table{border-collapse:collapse;width:100%%;background:var(--panel);font-size:13px;
 border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:center}
th{background:#EEF2F7;font-weight:600;color:#33415C}
td:first-child,th:first-child{text-align:left}
.meta{max-width:640px}
.meta th{width:120px;text-align:left}
.region{background:#FFF7E6;border:1px solid #F5D08A;border-radius:8px;padding:10px 14px;font-size:13px}
.bad{color:#D93025}.warn{color:#B87413}
.foot{color:var(--mut);font-size:12px;margin-top:24px;border-top:1px solid var(--line);padding-top:10px}
</style></head><body>
<h1>语义分割评估报告</h1>
<div class="sub">DL Studio · 10 项指标 · 生成于 %(time)s</div>

<h2>十项指标总览</h2>
%(cards)s

<h2>目标级结果（产线口径）</h2>
%(region)s

<h2>逐类明细</h2>
<table><thead><tr><th>类别</th><th>IoU</th><th>Dice</th><th>Precision</th>
<th>Recall</th><th>Specificity</th><th>真值像素占比</th></tr></thead><tbody>%(pc)s</tbody></table>

<h2>混淆矩阵（像素数）</h2>
<table><thead><tr><th>真值 ↓ / 预测 →</th>%(cmh)s</tr></thead><tbody>%(cmb)s</tbody></table>

<h2>运行配置</h2>
<table class="meta"><tbody>%(meta)s</tbody></table>

<div class="foot">宏平均仅统计前景类（排除背景），避免背景稀释。HD95 单位：像素，越小越好。
判据为工程经验阈值，非论文标准；请结合产线过杀/漏检 KPI 综合判断。</div>
</body></html>""" % {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cards": "".join(cards), "pc": pc_rows, "cmh": cm_head, "cmb": cm_body,
        "meta": meta_rows, "region": rm,
    }
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# 6. 输入：掩膜目录对 / DL Studio 项目
# --------------------------------------------------------------------------
MASK_EXT = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_dir_pairs(gt_dir, pred_dir):
    from PIL import Image
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    pairs, missing = [], []
    for f in sorted(gt_dir.iterdir()):
        if f.suffix.lower() not in MASK_EXT:
            continue
        p = pred_dir / f.name
        if not p.exists():  # 允许预测用 .png 而 GT 用 .bmp
            cands = [pred_dir / (f.stem + e) for e in MASK_EXT]
            p = next((c for c in cands if c.exists()), None)
        if p is None:
            missing.append(f.name)
            continue
        pairs.append((f.name, np.array(Image.open(f)), np.array(Image.open(p))))
    return pairs, missing


def load_project_pairs(project_dir, split="val", model_file="best_model.pth",
                       batch_log=None, limit=None):
    """走 DL Studio 自己的推理路径：Predictor 把预测 resize 回原分辨率。"""
    project_dir = Path(project_dir).resolve()
    sys.path.insert(0, str(project_dir))
    from PIL import Image
    from annotation.labelme_io import load_labelme_json, shapes_to_mask
    from training.dataset import SegmentationDataset
    from inference.predictor import Predictor

    ds = SegmentationDataset(str(project_dir), split=split)
    class_names = list(getattr(ds, "classes", ["background"]))
    predictor = Predictor(str(project_dir))
    predictor.load(model_file)

    pairs = []
    for i in range(len(ds)):
        img_path, json_path, mask_path = ds._image_samples[i]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        # GT：优先用 labelme json 生成原分辨率掩膜（保证与标注一致）
        if json_path and Path(json_path).exists():
            shapes, _ = load_labelme_json(json_path)
            gt = shapes_to_mask(shapes, class_names, (h, w)) if shapes else np.zeros((h, w), np.uint8)
        elif mask_path and Path(mask_path).exists():
            gt = to_label(np.array(Image.open(mask_path)))
            if gt.shape != (h, w):
                gt = to_label(np.array(Image.open(mask_path).resize((w, h), Image.NEAREST)))
        else:
            continue
        pred = predictor.predict(img)   # Predictor.predict 返回原分辨率标签图
        pairs.append((Path(img_path).name, gt, to_label(pred)))
        if batch_log:
            batch_log("  [%d/%d] %s" % (i + 1, len(ds), Path(img_path).name))
        if limit and len(pairs) >= limit:
            break
    return pairs, class_names


# --------------------------------------------------------------------------
# 7. CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="语义分割评估：10 项指标 + HTML 报表",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", help="真值掩膜目录（取值=类别 id，255=ignore）")
    ap.add_argument("--pred-dir", help="预测掩膜目录（文件名与 GT 对应）")
    ap.add_argument("--project", help="DL Studio 项目目录（走 Predictor 推理）")
    ap.add_argument("--split", default="val", choices=["train", "val", "all"])
    ap.add_argument("--model-file", default="best_model.pth")
    ap.add_argument("--class-names", default=None, help="逗号分隔，如 background,Atd")
    ap.add_argument("--tol", type=float, default=2.0, help="边界 F1 容差（像素），默认 2")
    ap.add_argument("--region-iou", type=float, default=0.5, help="连通域匹配 IoU 阈值，默认 0.5")
    ap.add_argument("--min-area", type=int, default=0,
                    help="区域级最小面积（像素）：小于它的碎块不计入漏检/过杀。"
                         "对应客户口径的\"最小可识别面积\"，阈值需人工拍板，默认 0=不过滤")
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 张（调试用）")
    ap.add_argument("--html", default=None, help="输出 HTML 报表路径")
    ap.add_argument("--json", dest="json_out", default=None, help="输出 JSON 结果路径")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    def log(msg):
        if not args.quiet:
            print(msg)

    if args.project:
        log("加载项目: %s（split=%s）" % (args.project, args.split))
        pairs, class_names = load_project_pairs(
            args.project, args.split, args.model_file, batch_log=log, limit=args.limit)
        meta = {"项目": str(args.project), "数据划分": args.split, "模型": args.model_file,
                "推理口径": "Predictor（tiled/full，预测上采样回原分辨率）"}
        src = "project"
    elif args.gt_dir and args.pred_dir:
        pairs, missing = load_dir_pairs(args.gt_dir, args.pred_dir)
        if missing:
            log("警告: %d 个 GT 掩膜找不到对应预测: %s" % (len(missing), missing[:5]))
        if args.class_names:
            class_names = [s.strip() for s in args.class_names.split(",") if s.strip()]
        else:
            max_id = max((int(p[1].max()) for p in pairs), default=1)
            class_names = ["background"] + ["class_%d" % i for i in range(1, max_id + 1)]
        meta = {"真值目录": str(args.gt_dir), "预测目录": str(args.pred_dir),
                "类别": ", ".join(class_names)}
        src = "dirs"
    else:
        ap.error("需要 --gt-dir/--pred-dir 或 --project")
        return 2

    if not pairs:
        print("没有找到可比对的样本。", file=sys.stderr)
        return 2

    log("共 %d 张图像参与评估，类别 %d 个：%s" % (len(pairs), len(class_names), ", ".join(class_names)))
    t0 = time.time()
    res = evaluate_pairs(pairs, class_names, tol=args.tol, region_iou=args.region_iou,
                         min_area=args.min_area, log=log, verbose=not args.quiet)
    res["source"] = src
    res["elapsed_sec"] = round(time.time() - t0, 2)
    meta["耗时"] = "%.2f s" % res["elapsed_sec"]
    meta["边界容差 tol"] = "%s px" % args.tol
    meta["区域匹配 IoU"] = args.region_iou
    meta["区域最小面积"] = ("%d px（过滤碎块）" % args.min_area) if args.min_area else "不过滤"

    if not args.quiet:
        print_report(res)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        log("JSON 已写出: %s" % args.json_out)
    if args.html:
        render_html(res, meta, args.html)
        log("HTML 报表已写出: %s" % args.html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
