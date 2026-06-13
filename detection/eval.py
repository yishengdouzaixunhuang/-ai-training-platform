# -*- coding: utf-8 -*-
"""Detection Evaluation — mAP, precision, recall metrics."""
import json, os
from pathlib import Path
import numpy as np


def evaluate_predictions(gt_json, pred_boxes, iou_threshold=0.5):
    """Evaluate predictions against ground truth COCO JSON.
    
    Args:
        gt_json: Path to ground truth _det.json
        pred_boxes: List of pred dicts [{bbox, category, confidence, bbox_type}, ...]
        iou_threshold: IoU threshold for positive match
    
    Returns:
        dict with per-class and overall precision, recall, F1, AP@threshold
    """
    from detection.coco_io import load_coco_json
    
    gt_boxes, info = load_coco_json(gt_json)
    if gt_boxes is None:
        gt_boxes = []
    
    categories = {b["category"] for b in gt_boxes}
    categories |= {b["category"] for b in pred_boxes}
    
    results = {}
    all_tp, all_fp, all_fn = 0, 0, 0
    
    for cat in categories:
        gt_cat = [b for b in gt_boxes if b["category"] == cat]
        pred_cat = sorted(
            [b for b in pred_boxes if b["category"] == cat],
            key=lambda x: x.get("confidence", 1.0),
            reverse=True
        )
        
        matched = set()
        tp, fp = 0, 0
        for pred in pred_cat:
            best_iou = 0.0
            best_gt = None
            for i, gt in enumerate(gt_cat):
                if i in matched:
                    continue
                iou = _compute_iou(pred["bbox"], gt["bbox"],
                                   pred.get("bbox_type", "hbb"),
                                   gt.get("bbox_type", "hbb"))
                if iou > best_iou:
                    best_iou = iou
                    best_gt = i
            if best_iou >= iou_threshold and best_gt is not None:
                tp += 1
                matched.add(best_gt)
            else:
                fp += 1
        
        fn = len(gt_cat) - len(matched)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
        all_tp += tp
        all_fp += fp
        all_fn += fn
    
    total_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    total_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    total_f1 = (2 * total_precision * total_recall / (total_precision + total_recall)
                if (total_precision + total_recall) > 0 else 0.0)
    
    results["__overall__"] = {
        "precision": round(total_precision, 4),
        "recall": round(total_recall, 4),
        "f1": round(total_f1, 4),
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
    }
    
    return results


def _compute_iou(bbox_a, bbox_b, type_a="hbb", type_b="hbb"):
    """Compute IoU between two bounding boxes (HBB only for now)."""
    if type_a == "obb" or type_b == "obb":
        return _compute_iou_obb(bbox_a, bbox_b)
    # HBB: [x, y, w, h] or [x1, y1, x2, y2]
    if len(bbox_a) >= 4 and len(bbox_b) >= 4:
        ax1, ay1 = bbox_a[0], bbox_a[1]
        ax2, ay2 = ax1 + bbox_a[2], ay1 + bbox_a[3]
        bx1, by1 = bbox_b[0], bbox_b[1]
        bx2, by2 = bx1 + bbox_b[2], by1 + bbox_b[3]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
    return 0.0


def _compute_iou_obb(bbox_a, bbox_b):
    """IoU for OBB via simplified corner approximation."""
    # Use shapely if available, else approximate with box area ratio
    try:
        from shapely.geometry import Polygon
        def obb_to_poly(bbox):
            import math
            cx, cy, w, h, angle = bbox[:5]
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            hw, hh = w / 2, h / 2
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            pts = []
            for lx, ly in corners:
                pts.append((cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a))
            return Polygon(pts)
        poly_a = obb_to_poly(bbox_a)
        poly_b = obb_to_poly(bbox_b)
        inter = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return inter / union if union > 0 else 0.0
    except ImportError:
        # Fallback: use axis-aligned bounds
        return _compute_iou(bbox_a[:4], bbox_b[:4])
