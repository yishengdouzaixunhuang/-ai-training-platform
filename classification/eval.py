# -*- coding: utf-8 -*-
"""Classification Evaluation — Top-1/Top-5 accuracy + confusion matrix + per-class metrics."""

import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ClassificationDataset, IMAGENET_MEAN, IMAGENET_STD


@torch.inference_mode()
def evaluate_classification(
    project_dir,
    split="val",
    top_k=(1, 5),
    model_path="best_model.pth",
    batch_size=32,
    image_size=224,
    device=None,
    progress_callback=None,
    log_callback=None,
    compute_confusion=False,
):
    """Evaluate a trained classification model and return metrics dict.

    Args:
        project_dir: Project root directory
        split: 'train', 'val', or 'all'
        top_k: Tuple of k values for Top-k accuracy
        model_path: Relative path to model checkpoint within models/
        batch_size: Inference batch size
        image_size: Input image size
        device: torch device
        progress_callback: (current, total) → None
        log_callback: (msg) → None
        compute_confusion: If True, compute and include confusion matrix

    Returns:
        dict with keys: top1_acc, top5_acc, per_class_acc, num_samples,
                        (optional) confusion_matrix, class_names
    """
    project_dir = Path(project_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    from .trainer import ClassificationTrainer
    ckpt_path = project_dir / "models" / model_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    class_names = ckpt.get("class_names", [f"class_{i}" for i in range(num_classes)])
    model_name = ckpt.get("model_name", "resnet18")

    # Create and load model directly (avoids circular import)
    from .trainer import _create_classification_model
    model = _create_classification_model(model_name, num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build dataset
    dataset = ClassificationDataset(
        str(project_dir), split=split, image_size=image_size
    )

    if len(dataset) == 0:
        if log_callback:
            log_callback(f"No samples found for split='{split}'")
        return {
            "top1_acc": 0.0,
            f"top{max(top_k)}_acc": 0.0,
            "per_class_acc": {},
            "num_samples": 0,
            "class_names": class_names,
        }

    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    loader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False
    )

    # Tracking
    max_k = max(top_k)
    total = 0
    correct_topk = {k: 0 for k in top_k}  # top-k correct count
    all_preds = []
    all_labels = []

    with tqdm(total=len(loader), desc="Evaluating", ncols=80) as pbar:
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels_np = labels.numpy()

            outputs = model(images)
            probs = outputs.softmax(1)

            # Top-k accuracy
            _, topk_idx = probs.topk(max_k, dim=1)
            labels_t = labels.to(device).unsqueeze(1)
            correct = topk_idx.eq(labels_t)

            for k in top_k:
                correct_topk[k] += correct[:, :k].sum().item()

            # Store for confusion matrix
            if compute_confusion:
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels_np.tolist())

            total += labels_np.size
            pbar.update(1)

            if progress_callback:
                progress_callback(i + 1, len(loader))

    # Compute per-class accuracy
    per_class_acc = {}
    if compute_confusion and len(all_preds) > 0:
        from collections import defaultdict
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        for pred, label in zip(all_preds, all_labels):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

        for i, name in enumerate(class_names):
            per_class_acc[name] = (
                class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            )

    result = {
        "num_samples": total,
        "class_names": class_names,
        "top_k_results": {f"top{k}_acc": correct_topk[k] / max(total, 1) for k in top_k},
        "per_class_acc": per_class_acc,
    }

    # Compute confusion matrix
    if compute_confusion and len(all_preds) > 0:
        from sklearn.metrics import confusion_matrix
        try:
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
            result["confusion_matrix"] = cm.tolist()
        except Exception:
            result["confusion_matrix"] = None

    # Log summary
    if log_callback:
        top1 = result["top_k_results"].get("top1_acc", 0)
        top5 = result["top_k_results"].get("top5_acc", top1)
        log_callback(f"Evaluation ({total} samples) — Top-1: {top1:.4f}, Top-5: {top5:.4f}")

    return result


def compute_class_weights(project_dir, split="train"):
    """Compute balanced class weights for imbalanced datasets.

    Returns a tensor of weights for nn.CrossEntropyLoss(weight=...).
    """
    dataset = ClassificationDataset(str(Path(project_dir)), split=split)
    if len(dataset) == 0:
        return torch.ones(1)

    labels = [pid for _, _, pid in dataset._pairs]
    if not labels:
        return torch.ones(1)

    from collections import Counter
    counts = Counter(labels)
    num_classes = max(counts.keys()) + 1
    weights = torch.ones(num_classes)

    max_count = max(counts.values())
    for cls_id in range(num_classes):
        cnt = counts.get(cls_id, 0)
        if cnt > 0:
            weights[cls_id] = max_count / cnt

    return weights