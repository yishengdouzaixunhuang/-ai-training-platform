"""
Extended loss functions for semantic segmentation.
Pure PyTorch, zero extra dependencies.
References: mmsegmentation, "The Lovasz-Softmax loss" (Berman et al., 2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ?? Lov?sz-Softmax Loss ??????????????????????????????????????????????
# Optimizes the Jaccard index (IoU) directly. Particularly effective for
# small objects and class-imbalanced datasets like defect detection.

def _lovasz_grad(gt_sorted):
    """Compute Lov?sz gradient: g(m) = ?Jaccard"""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten_probas(probas, labels, ignore=None):
    """Flatten predictions and labels, remove ignore class.
    Supports both (C, H, W) and (B, C, H, W) input shapes.
    """
    if probas.dim() == 4:
        # B, C, H, W -> C, B*H*W
        B, C, H, W = probas.size()
        probas = probas.permute(1, 0, 2, 3).reshape(C, -1)
        labels = labels.view(-1)
    elif probas.dim() == 3:
        C, H, W = probas.size()
        probas = probas.view(C, -1)
        labels = labels.view(-1)
    if ignore is not None:
        valid = labels != ignore
        probas = probas[:, valid]
        labels = labels[valid]
    return probas, labels


def lovasz_softmax_flat(probas, labels, classes="present"):
    """Lov?sz-Softmax loss for flat (pre-flattened) predictions."""
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(0)
    losses = []
    class_to_sum = list(range(C)) if classes == "all" else labels.unique().tolist()
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (probas[c] - fg).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        loss = torch.dot(errors_sorted, grad)
        losses.append(loss)
    if len(losses) == 0:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax_loss(logits, labels, ignore=255, classes="present"):
    """
    Multi-class Lov?sz-Softmax loss.
    
    Args:
        logits: (B, C, H, W) raw logits
        labels: (B, H, W) integer labels
        ignore: label index to ignore
        classes: "present" (only classes in batch) or "all"
    """
    probas = F.softmax(logits, dim=1)
    probas, labels = _flatten_probas(probas, labels, ignore=ignore)
    return lovasz_softmax_flat(probas, labels, classes=classes)


# ?? Dice Loss ????????????????????????????????????????????????????????
# Directly optimizes the Dice coefficient (F1 at pixel level).

def dice_loss(logits, labels, ignore=255, smooth=1.0):
    """
    Multi-class Dice loss.
    
    Args:
        logits: (B, C, H, W) raw logits
        labels: (B, H, W) integer labels
        ignore: label index to ignore
        smooth: smoothing factor to avoid division by zero
    """
    probas = F.softmax(logits, dim=1)
    C = probas.size(1)
    
    # One-hot encode labels
    labels_onehot = F.one_hot(labels.clamp(0, C - 1), num_classes=C).permute(0, 3, 1, 2).float()
    
    # Ignore mask
    if ignore is not None:
        mask = (labels != ignore).unsqueeze(1).float()
        probas = probas * mask
        labels_onehot = labels_onehot * mask
    
    intersection = (probas * labels_onehot).sum(dim=(0, 2, 3))
    union = probas.sum(dim=(0, 2, 3)) + labels_onehot.sum(dim=(0, 2, 3))
    
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    # Exclude background (class 0)
    return 1.0 - dice_per_class[1:].mean()


# ?? Focal Loss ???????????????????????????????????????????????????????
# Down-weights easy examples, focuses on hard ones. Good for class imbalance.

def focal_loss(logits, labels, ignore=255, alpha=0.25, gamma=2.0):
    """
    Multi-class Focal Loss.
    
    Args:
        logits: (B, C, H, W) raw logits
        labels: (B, H, W) integer labels
        ignore: label index to ignore
        alpha: class balance weight (scalar or per-class tensor)
        gamma: focusing parameter (higher = more focus on hard examples)
    """
    C = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    
    # Gather log probs for target classes
    labels_flat = labels.view(labels.size(0), -1).clamp(0, C - 1)
    log_probs_flat = log_probs.view(logits.size(0), C, -1)
    log_pt = log_probs_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    
    # Focal weight: (1 - pt)^gamma
    focal_weight = (1 - pt).pow(gamma)
    
    loss = -alpha * focal_weight * log_pt
    
    if ignore is not None:
        mask = (labels_flat != ignore).float()
        loss = loss * mask
        loss = loss.sum() / mask.sum().clamp(min=1)
    else:
        loss = loss.mean()
    
    return loss


# ?? Combined Loss ????????????????????????????????????????????????????
# Wrapper to combine multiple losses with weights.

class CombinedLoss(nn.Module):
    """Combine multiple loss functions with configurable weights."""
    
    def __init__(self, losses_config, num_classes, device="cuda"):
        """
        Args:
            losses_config: list of (name, weight) tuples, e.g.
                [("cross_entropy", 1.0), ("lovasz", 0.5)]
            num_classes: number of classes
            device: device for weight tensors
        """
        super().__init__()
        self.config = losses_config
        self.num_classes = num_classes
        self.device = device
        self._ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    
    def set_ce_weights(self, weights):
        """Update CrossEntropyLoss class weights (for class imbalance)."""
        if weights is not None:
            self._ce_loss = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32).to(self.device),
                ignore_index=255)
    
    def forward(self, logits, labels):
        total = 0.0
        for name, weight in self.config:
            if weight == 0:
                continue
            if name == "cross_entropy":
                total += weight * self._ce_loss(logits, labels)
            elif name == "lovasz":
                total += weight * lovasz_softmax_loss(logits, labels)
            elif name == "dice":
                total += weight * dice_loss(logits, labels)
            elif name == "focal":
                total += weight * focal_loss(logits, labels)
        return total


# ?? Builder ??????????????????????????????????????????????????????????

def build_loss(loss_name, num_classes, device="cuda", class_weights=None):
    """
    Build a loss function by name.
    
    Supported names:
        "cross_entropy"  - standard CrossEntropyLoss
        "lovasz"         - Lov?sz-Softmax (IoU optimization)
        "dice"           - Dice loss
        "focal"           - Focal loss
        "ce+lovasz"      - CrossEntropy + Lov?sz (recommended)
        "ce+dice"        - CrossEntropy + Dice
    """
    if loss_name == "cross_entropy":
        kwargs = {"ignore_index": 255}
        if class_weights is not None:
            kwargs["weight"] = torch.tensor(class_weights, dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(**kwargs)
    
    elif loss_name == "lovasz":
        def lovasz_fn(logits, labels):
            return lovasz_softmax_loss(logits, labels)
        return lovasz_fn
    
    elif loss_name == "dice":
        def dice_fn(logits, labels):
            return dice_loss(logits, labels)
        return dice_fn
    
    elif loss_name == "focal":
        def focal_fn(logits, labels):
            return focal_loss(logits, labels)
        return focal_fn
    
    elif loss_name == "ce+lovasz":
        loss = CombinedLoss([("cross_entropy", 1.0), ("lovasz", 0.5)], num_classes, device)
        loss.set_ce_weights(class_weights)
        return loss
    
    elif loss_name == "ce+dice":
        loss = CombinedLoss([("cross_entropy", 1.0), ("dice", 0.5)], num_classes, device)
        loss.set_ce_weights(class_weights)
        return loss
    
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


LOSS_NAMES = ["cross_entropy", "lovasz", "dice", "focal", "ce+lovasz", "ce+dice"]
