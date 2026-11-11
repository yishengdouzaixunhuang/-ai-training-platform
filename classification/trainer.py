# -*- coding: utf-8 -*-
"""Classification Trainer — image classification training wrapper.

Supports:
- ResNet / EfficientNet / MobileNet / ViT architectures
- K-fold cross-validation with stratified splits
- AMP (automatic mixed precision) on CUDA
- Checkpoint resume (last_model.pth)
- Progress / log / stop / plot callbacks
- Model export to TorchScript and ONNX
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm

from .dataset import (
    ClassificationDataset,
    auto_split,
    get_dataset_stats,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# ── Available model configs ─────────────────────────────────────────────
CLS_MODELS = {
    "resnet18":   {"name": "ResNet-18",       "family": "resnet",  "params": "11.7M"},
    "resnet34":   {"name": "ResNet-34",       "family": "resnet",  "params": "21.8M"},
    "resnet50":   {"name": "ResNet-50",       "family": "resnet",  "params": "25.6M"},
    "resnet101":  {"name": "ResNet-101",      "family": "resnet",  "params": "44.5M"},
    "resnet152":  {"name": "ResNet-152",      "family": "resnet",  "params": "60.2M"},
    "efficientnet_b0": {"name": "EfficientNet-B0", "family": "efficientnet", "params": "5.3M"},
    "efficientnet_b3": {"name": "EfficientNet-B3", "family": "efficientnet", "params": "12.2M"},
    "efficientnet_b7": {"name": "EfficientNet-B7", "family": "efficientnet", "params": "66.3M"},
    "mobilenet_v2":    {"name": "MobileNet V2",    "family": "mobilenet",    "params": "3.5M"},
    "mobilenet_v3_small": {"name": "MobileNet V3 Small", "family": "mobilenet3", "params": "2.5M"},
    "mobilenet_v3_large": {"name": "MobileNet V3 Large", "family": "mobilenet3", "params": "5.5M"},
    "vit_b_16":         {"name": "ViT-B/16",     "family": "vit",      "params": "86.6M"},
    "vit_b_32":         {"name": "ViT-B/32",     "family": "vit",      "params": "88.2M"},
    "convnext_tiny":    {"name": "ConvNeXt-Tiny",     "family": "convnext",       "params": "28.6M"},
    "convnext_base":    {"name": "ConvNeXt-Base",     "family": "convnext",       "params": "88.6M"},
    "efficientnet_v2_s": {"name": "EfficientNet-V2-S", "family": "efficientnet_v2", "params": "21.5M"},
    "efficientnet_v2_m": {"name": "EfficientNet-V2-M", "family": "efficientnet_v2", "params": "54.1M"},
    "efficientnet_v2_l": {"name": "EfficientNet-V2-L", "family": "efficientnet_v2", "params": "119.5M"},
    "swin_t":           {"name": "Swin-T",            "family": "swin",           "params": "28.3M"},
    "swin_s":           {"name": "Swin-S",            "family": "swin",           "params": "49.6M"},
    "swin_b":           {"name": "Swin-B",            "family": "swin",           "params": "87.8M"},
    "densenet121":      {"name": "DenseNet-121",      "family": "densenet",       "params": "8.0M"},
    "densenet161":      {"name": "DenseNet-161",      "family": "densenet",       "params": "28.7M"},
    "densenet201":      {"name": "DenseNet-201",      "family": "densenet",       "params": "20.0M"},
}


# ── 设备判断 ─────────────────────────────────────────────────────────────
def _device_is_cuda(dev):
    """本项目的 device 是字符串（"cuda"/"cpu"）也可能是 torch.device，统一判断。"""
    return str(getattr(dev, "type", dev)).startswith("cuda")


# ── Focal Loss ──────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance. gamma=2, alpha=None by default."""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss = self.alpha * focal_loss
        return focal_loss.mean()


class ModelEMA:
    """Exponential moving average of model weights.

    A shadow copy is updated after each training step with an adaptive decay,
    so short runs (few hundred steps) still converge the shadow toward the
    trained weights instead of staying frozen near the initial weights.
    Validation and best-model saving use the smoothed weights.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    def update(self, model):
        self.updates += 1
        # TF-style adaptive decay: snap toward the current weights at the
        # start of training, then asymptotically approach self.decay.
        d = min(self.decay, (self.updates + 1) / (self.updates + 10))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow:
                    self.shadow[k] = v.detach().clone()
                elif v.is_floating_point():
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
                else:
                    self.shadow[k].copy_(v.detach())

    def apply_shadow(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None


# ── 验证指标：混淆矩阵 / 逐类 P/R/F1 / 漏检 / 过杀 ─────────────────────
# 缺陷分类里「准确率」会骗人：NG 79 / OK 78 时，全部预测成 OK 也有 ~49.7% 准确率。
# 产线真正关心的是 NG 召回（漏检）与 OK 过杀，所以每次验证一并算出。
NG_NAME_KEYS = ("ng", "不良", "缺陷", "defect", "fail")
OK_NAME_KEYS = ("ok", "良品", "合格", "good", "pass", "fine")


def _match_class_index(class_names, keys):
    """按类别名匹配索引（先精确、后包含），匹配不到返回 None。"""
    lowered = [str(n).strip().lower() for n in class_names]
    for i, n in enumerate(lowered):
        if n in keys:
            return i
    for i, n in enumerate(lowered):
        if any(k in n for k in keys):
            return i
    return None


def confusion_metrics(y_true, y_pred, class_names, num_classes):
    """混淆矩阵 + 逐类 Precision/Recall/F1 + macro-F1 / balanced-acc。

    纯 numpy 实现（不依赖 sklearn，便于在只有 torch 的环境里跑）。
    另按类别名找出 NG / OK，给出「NG 漏检率」与「OK 过杀率」。
    """
    n = max(1, int(num_classes))
    cm = np.zeros((n, n), dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    ok = (y_true >= 0) & (y_true < n) & (y_pred >= 0) & (y_pred < n)
    if ok.any():
        idx = y_true[ok] * n + y_pred[ok]
        cm += np.bincount(idx, minlength=n * n).reshape(n, n)

    per_class = {}
    f1s, recalls = [], []
    for c in range(n):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        support = float(cm[c, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        name = str(class_names[c]) if c < len(class_names) else f"class_{c}"
        per_class[name] = {"precision": prec, "recall": rec, "f1": f1,
                           "support": int(support)}
        f1s.append(f1)
        recalls.append(rec)

    out = {
        "confusion": cm.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "balanced_acc": float(np.mean(recalls)) if recalls else 0.0,
    }
    ng_i = _match_class_index(class_names, NG_NAME_KEYS)
    ok_i = _match_class_index(class_names, OK_NAME_KEYS)
    if ng_i is not None and str(class_names[ng_i]) in per_class:
        out["ng_class"] = str(class_names[ng_i])
        out["ng_recall"] = per_class[str(class_names[ng_i])]["recall"]
        out["ng_miss_rate"] = 1.0 - out["ng_recall"]
    if ok_i is not None and str(class_names[ok_i]) in per_class:
        n_ok = float(cm[ok_i, :].sum())
        err_ok = n_ok - float(cm[ok_i, ok_i])      # 真 OK 被判成别的 = 过杀
        out["ok_class"] = str(class_names[ok_i])
        out["ok_recall"] = per_class[str(class_names[ok_i])]["recall"]
        out["ok_overkill_rate"] = (err_ok / n_ok) if n_ok > 0 else 0.0
    return out


def metric_for_selection(val_acc, metrics, select_metric="acc"):
    """按 select_metric 取「选最优模型」用的标量；未知名称回退到 acc。"""
    name = str(select_metric or "acc").strip().lower()
    if name in ("macro_f1", "f1"):
        return float(metrics.get("macro_f1", 0.0))
    if name in ("balanced_acc", "bal_acc", "uar"):
        return float(metrics.get("balanced_acc", 0.0))
    if name in ("ng_recall", "recall"):
        v = metrics.get("ng_recall")
        return float(v) if v is not None else float(val_acc)
    return float(val_acc)


def pretrained_weight_status(model_name):
    """查预训练权重是否已在本地缓存 → (bool|None, 文件名|None)。

    `pretrained=True` 时 torchvision 会去 download.pytorch.org 取权重；
    工厂现场网络受限，这个下载可能卡很久甚至失败，所以训练前先报出来。
    """
    try:
        import glob as _glob
        d = os.path.join(torch.hub.get_dir(), "checkpoints")
        hits = _glob.glob(os.path.join(d, f"{model_name}*.pth"))
        return bool(hits), (os.path.basename(hits[0]) if hits else None)
    except Exception:
        return None, None


# ── Model factory ───────────────────────────────────────────────────────
def _replace_last_linear(model, num_classes, dropout=0.0):
    """递归找到并替换最后一个 nn.Linear 为分类头（兼容 ConvNeXt/Swin/DenseNet 等）。"""
    path = [None]

    def find(m, prefix=""):
        for name, child in m.named_children():
            p = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                path[0] = p
            find(child, p)

    find(model)
    if path[0] is None:
        raise ValueError(f"模型中没有找到 Linear 分类头: {type(model).__name__}")
    parts = path[0].split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    in_features = getattr(parent, parts[-1]).in_features
    if dropout > 0:
        new_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:
        new_head = nn.Linear(in_features, num_classes)
    setattr(parent, parts[-1], new_head)
    return in_features


def _create_classification_model(model_name, num_classes, pretrained=True, dropout=0.0):
    """Create a classification model by name."""
    cfg = CLS_MODELS.get(model_name, CLS_MODELS["resnet18"])
    family = cfg["family"]

    if family == "resnet":
        import torchvision.models as models
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            model_fn = models.resnet18
        model = model_fn(weights="DEFAULT" if pretrained else None)
        # Replace final FC
        in_features = model.fc.in_features
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)

    elif family == "efficientnet":
        import torchvision.models as models
        import re
        m = re.match(r"efficientnet_b(\d)", model_name)
        if m:
            ver = int(m.group(1))
            fn_name = f"efficientnet_b{ver}"
            model_fn = getattr(models, fn_name, models.efficientnet_b0)
        else:
            model_fn = models.efficientnet_b0
        model = model_fn(weights="DEFAULT" if pretrained else None)
        in_features = model.classifier[1].in_features
        if dropout > 0:
            model.classifier[1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.classifier[1] = nn.Linear(in_features, num_classes)

    elif family == "mobilenet":
        import torchvision.models as models
        model_fn = getattr(models, "mobilenet_v2", None)
        if model_fn is None:
            model_fn = models.mobilenet_v2
        model = model_fn(weights="DEFAULT" if pretrained else None)
        in_features = model.classifier[1].in_features
        if dropout > 0:
            model.classifier[1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.classifier[1] = nn.Linear(in_features, num_classes)

    elif family == "mobilenet3":
        import torchvision.models as models
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            model_fn = models.mobilenet_v3_large
        model = model_fn(weights="DEFAULT" if pretrained else None)
        in_features = model.classifier[3].in_features
        if dropout > 0:
            model.classifier[3] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.classifier[3] = nn.Linear(in_features, num_classes)

    elif family == "vit":
        import torchvision.models as models
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            model_fn = models.vit_b_16
        model = model_fn(weights="DEFAULT" if pretrained else None)
        in_features = model.heads.head.in_features
        if dropout > 0:
            model.heads.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.heads.head = nn.Linear(in_features, num_classes)

    elif family == "convnext":
        import torchvision.models as models
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            model_fn = models.convnext_tiny
        model = model_fn(weights="DEFAULT" if pretrained else None)
        _replace_last_linear(model, num_classes, dropout)

    elif family == "efficientnet_v2":
        import torchvision.models as models
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            model_fn = models.efficientnet_v2_s
        model = model_fn(weights="DEFAULT" if pretrained else None)
        in_features = model.classifier[1].in_features
        if dropout > 0:
            model.classifier[1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.classifier[1] = nn.Linear(in_features, num_classes)

    elif family == "swin":
        import torchvision.models as models
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            model_fn = models.swin_t
        model = model_fn(weights="DEFAULT" if pretrained else None)
        _replace_last_linear(model, num_classes, dropout)

    elif family == "densenet":
        import torchvision.models as models
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            model_fn = models.densenet121
        model = model_fn(weights="DEFAULT" if pretrained else None)
        _replace_last_linear(model, num_classes, dropout)

    else:
        raise ValueError(f"Unknown model family: {family}")

    return model


# ── Trainer ─────────────────────────────────────────────────────────────
def _resolve_cls_augment(augment):
    """分类增强入参统一解析：dict 开关 / str 预设 / bool / None -> flags dict 或 None。"""
    if isinstance(augment, dict):
        from training.augment_config import merge_flags
        return merge_flags(augment)
    if isinstance(augment, str):
        from training.augment_config import preset_to_flags
        return preset_to_flags(augment)
    if augment:
        from training.augment_config import DEFAULT_AUGMENT_FLAGS
        return dict(DEFAULT_AUGMENT_FLAGS)
    return None


# 模型版本管理
def resolve_model_path(project_dir, model_ref):
    """把模型下拉项（旧 .pth 文件名 或 V{n} 版本名）解析为 checkpoint 绝对路径。"""
    models_dir = Path(project_dir) / "models" / "classification"
    ref = str(model_ref or "").strip()
    if ref.startswith("classification/"):
        ref = ref[len("classification/"):]
    if re.fullmatch(r"V\d+", ref, re.IGNORECASE):
        ref = f"V{int(ref[1:])}"  # 统一为 V{n} 规范写法
        p = models_dir / "versions" / ref / "model.pth"
        if p.exists():
            return p
    p = models_dir / ref
    if p.exists() and p.is_file():
        return p
    p2 = models_dir / "versions" / ref / "model.pth"
    if p2.exists():
        return p2
    return p


def next_version_number(project_dir):
    """扫描 versions/ 目录，返回下一个版本号。"""
    versions_dir = Path(project_dir) / "models" / "classification" / "versions"
    nums = []
    if versions_dir.is_dir():
        for d in versions_dir.iterdir():
            if d.is_dir() and re.fullmatch(r"V\d+", d.name, re.IGNORECASE):
                try:
                    nums.append(int(d.name[1:]))
                except ValueError:
                    pass
    return (max(nums) + 1) if nums else 1


def write_version_archive(project_dir, version_num, state_dict, optimizer_state, meta, history):
    """把模型权重 + meta + history 归档到 models/classification/versions/V{n}/。"""
    vdir = Path(project_dir) / "models" / "classification" / "versions" / f"V{version_num}"
    vdir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer_state or {},
        "num_classes": meta.get("num_classes"),
        "class_names": meta.get("class_names"),
        "model_name": meta.get("model_name"),
        "image_size": meta.get("image_size"),
        "scale_factor": meta.get("scale_factor"),
        "target_size": meta.get("target_size"),
        "history": history,
        "version": version_num,
    }, vdir / "model.pth")
    meta = dict(meta)
    meta["version"] = version_num
    meta.setdefault("created", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _jsonable(o):
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    (vdir / "meta.json").write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    (vdir / "history.json").write_text(json.dumps(_jsonable(history), ensure_ascii=False, indent=2), encoding="utf-8")
    return vdir


class ClassificationTrainer:
    """Wraps classification training with callbacks for Qt UI integration."""

    def __init__(self, project_dir, model_name="resnet18", device=None):
        self.project_dir = Path(project_dir)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Read project meta for class info
        with open(self.project_dir / "project.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # Determine num_classes from class_labels.json or dataset scan
        stats = get_dataset_stats(str(project_dir))
        self.num_classes = stats.get("num_classes", 2)
        self.class_names = stats.get("class_names", [f"class_{i}" for i in range(self.num_classes)])

        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        # AMP 只在「有 CUDA + 用户勾了 AMP」时启用，见 train(use_amp=...)。
        # 旧实现无论 UI 怎么勾都用「cuda 是否可用」决定，等于 use_amp 是个死开关。
        self.scaler = None
        self._amp_enabled = False
        self._sched = None            # LR 调度器实例
        self._sched_step_with_metric = False
        self._select_metric = "acc"
        self._num_workers = 0
        self._drop_last = False
        self._stop_flag = False
        self._stop_check_fn = None
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self._image_size = 224
        self._ema = None

    # ── Public API ──────────────────────────────────────────────────

    @staticmethod
    def _pad_collate(batch):
        """Collate function that pads images to max H/W in the batch."""
        import torch.nn.functional as F
        images, labels = zip(*batch)
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded = []
        for img in images:
            if img.shape[1] < max_h or img.shape[2] < max_w:
                img = F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))
            padded.append(img)
        return torch.stack(padded), torch.tensor(labels)

    def train(self, epochs=50, batch_size=32, image_size=224, scale_factor=1.0,
              loss_func="cross_entropy", optimizer="adam", lr=1e-4,
              pretrained=True, resume=False, k_folds=1,
              augment="none", use_amp=True,
              weight_decay=1e-4, momentum=0.9,
              label_smoothing=0.1, focal_gamma=2.0,
              dropout=0.0,
              lr_scheduler="none", warmup_epochs=0,
              early_stop_patience=0,
              ema=True, ema_decay=0.999, class_balanced=False,
              progress_callback=None, log_callback=None,
              stop_check=None, plot_callback=None, batch_callback=None,
              keep_aspect=False, select_metric="acc"):
        """Train classification model.

        Args:
            epochs: Training epochs
            batch_size: Batch size
            image_size: Input image size (int → square)
            loss_func: Loss function name (cross_entropy, label_smoothing, focal)
            optimizer: Optimizer name (adam, adamw, sgd)
            lr: Learning rate
            pretrained: Use pretrained weights
            resume: Resume from checkpoint
            k_folds: K-fold cross-validation (1 = no CV)
            augment: 增强开关 dict / 预设名（none/light/medium/heavy）
            use_amp: 是否用 AMP（仅 CUDA 生效；CPU 上自动忽略）
            lr_scheduler: none / step / cosine / plateau（真正生效，见 _build_scheduler）
            warmup_epochs: 线性 warmup 轮数（0=关闭）
            select_metric: 选最优模型的指标 acc / macro_f1 / balanced_acc / ng_recall
            progress_callback: (epoch, total) → None
            log_callback: (msg) → None
            stop_check: () → bool
            plot_callback: (history) → None
            batch_callback: (batch, total, avg_loss) → None
        """
        self._image_size = image_size
        self._scale_factor = scale_factor
        self._keep_aspect = keep_aspect
        self._select_metric = str(select_metric or "acc").strip().lower()
        self._stop_flag = False
        self._stop_check_fn = stop_check

        # K-fold branch
        if k_folds > 1 and not resume:
            return self._train_kfold(
                k_folds, epochs, batch_size, image_size,
                loss_func, optimizer, lr, pretrained,
                weight_decay, momentum, label_smoothing, focal_gamma,
                dropout, lr_scheduler, warmup_epochs, early_stop_patience,
                progress_callback, log_callback, stop_check,
                plot_callback, batch_callback,
                ema=ema, ema_decay=ema_decay, class_balanced=class_balanced,
                keep_aspect=keep_aspect, use_amp=use_amp,
                select_metric=self._select_metric
            )

        # Ensure train/val split exists
        split_file = self.project_dir / "train_test_split.json"
        if not split_file.exists():
            auto_split(str(self.project_dir), val_split=0.2)
            if log_callback:
                log_callback("Auto-generated train/val split")

        # Resume logic
        start_epoch = 0
        if resume:
            ckpt_path = self.project_dir / "models" / "classification" / "last_model.pth"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.num_classes = ckpt["num_classes"]
                self.class_names = ckpt.get("class_names", [f"class_{i}" for i in range(self.num_classes)])
                self.model_name = ckpt.get("model_name", self.model_name)
                self._image_size = ckpt.get("image_size", self._image_size)
                self._scale_factor = ckpt.get("scale_factor", self._scale_factor)
                self._keep_aspect = ckpt.get("keep_aspect", False)
                self._target_size = ckpt.get("target_size", None)
                self.history = ckpt.get("history", {"train_loss": [], "val_loss": [], "val_acc": []})
                start_epoch = len(self.history.get("train_loss", []))

                self.model = _create_classification_model(
                    self.model_name, self.num_classes, pretrained=False, dropout=dropout
                ).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.train()
                self._build_optimizer(optimizer, lr, weight_decay, momentum)
                if "optimizer_state_dict" in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception:
                        if log_callback:
                            log_callback("Optimizer state mismatch, using fresh optimizer")

                if log_callback:
                    log_callback(f"Resumed from epoch {start_epoch} (best acc: {max(self.history.get('val_acc', [0])):.4f})")
            else:
                if log_callback:
                    log_callback("No checkpoint found, training from scratch")
                resume = False

        if not resume:
            self.model = _create_classification_model(
                self.model_name, self.num_classes, pretrained=pretrained, dropout=dropout
            ).to(self.device)
            self._build_optimizer(optimizer, lr, weight_decay, momentum)
            start_epoch = 0

        # Build datasets and dataloaders
        self._prepare_data(batch_size, image_size, scale_factor, augment=augment,
                           keep_aspect=keep_aspect)

        # Build loss function
        self._build_loss(loss_func, label_smoothing, focal_gamma, class_balanced=class_balanced)

        # ── AMP：只有「CUDA 可用 且 用户勾了 use_amp」才开 ───────────
        self._amp_enabled = bool(use_amp) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self._amp_enabled else None
        if torch.cuda.is_available() and not use_amp:
            torch.backends.cudnn.benchmark = True   # 固定输入尺寸下免费提速
        elif self._amp_enabled:
            torch.backends.cudnn.benchmark = True

        # ── LR 调度器：真正建出来（旧版这里是死开关）─────────────────
        self._build_scheduler(lr_scheduler, epochs, warmup_epochs, start_epoch=start_epoch)

        # EMA: smoothed weights for validation & best-model saving (auto-enabled)
        self._ema = ModelEMA(self.model, decay=ema_decay) if ema else None

        self._log_training_context(lr, optimizer, loss_func, lr_scheduler, warmup_epochs,
                                   batch_size, image_size, keep_aspect, augment,
                                   ema, class_balanced, dropout, label_smoothing,
                                   focal_gamma, pretrained, start_epoch, epochs,
                                   log_callback)

        # Run epochs
        epochs_total = start_epoch + epochs
        best_acc = max(self.history.get("val_acc") or [0.0])
        _hist_sel = self.history.get("val_select") or []
        best_metric = max(_hist_sel) if _hist_sel else best_acc
        no_improve = 0

        for epoch in range(start_epoch, epochs_total):
            if self._stop_flag or (stop_check and stop_check()):
                self._stop_flag = True
                if log_callback:
                    log_callback("Training stopped by user")
                break

            train_loss = self._train_epoch(batch_callback)
            if self._ema is not None:
                self._ema.apply_shadow(self.model)
            try:
                val_loss, val_acc, vmet = self._validate()
            finally:
                if self._ema is not None:
                    self._ema.restore(self.model)

            cur_metric = metric_for_selection(val_acc, vmet, self._select_metric)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history.setdefault("val_select", []).append(cur_metric)
            self.history.setdefault("lr", []).append(self._current_lr())
            # 逐类指标：即使某轮拿不到也补 NaN，保证所有曲线等长（绘图按长度对齐）
            for k in ("macro_f1", "balanced_acc", "ng_recall", "ok_overkill_rate"):
                self.history.setdefault("val_" + k, []).append(
                    float(vmet[k]) if (vmet and k in vmet) else float("nan"))

            # 「本轮是否变好」只判一次：旧实现先更新 best_acc 再判 val_acc > best_acc，
            # 该条件恒为 False ⇒ no_improve 每轮必 +1 ⇒ 早停退化成「固定跑 N 轮」。
            improved = cur_metric > best_metric
            if improved:
                best_metric = cur_metric
            if val_acc > best_acc:
                best_acc = val_acc

            msg = (f"Epoch {epoch+1}/{epochs_total} | Train Loss: {train_loss:.4f} | "
                   f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | lr {self._current_lr():.2e}")
            if vmet:
                bits = []
                if "ng_recall" in vmet:
                    bits.append(f"{vmet.get('ng_class', 'NG')}召回 {vmet['ng_recall']:.3f}")
                if "ok_overkill_rate" in vmet:
                    bits.append(f"{vmet.get('ok_class', 'OK')}过杀 {vmet['ok_overkill_rate']:.3f}")
                if bits:
                    msg += " | " + " ".join(bits)
            if log_callback:
                log_callback(msg)

            if improved:
                self._save_model("best_model.pth", ema=(self._ema is not None))
                if log_callback:
                    log_callback(f"  → Best saved ({self._select_metric}: {best_metric:.4f}, Acc: {val_acc:.4f})")

            self._save_model("last_model.pth")

            # 调度器推进（plateau 用验证指标）
            if self._sched is not None:
                try:
                    if self._sched_step_with_metric:
                        self._sched.step(cur_metric)
                    else:
                        self._sched.step()
                except Exception as e:
                    if log_callback:
                        log_callback(f"LR scheduler step failed (已禁用调度): {e}")
                    self._sched = None

            if improved:
                no_improve = 0
            elif early_stop_patience and early_stop_patience > 0:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    if log_callback:
                        log_callback(f"Early stop at epoch {epoch+1}: no val improvement for {early_stop_patience} epochs (best {best_metric:.4f})")
                    break

            if progress_callback:
                progress_callback(epoch + 1 - start_epoch, epochs_total - start_epoch)
            if plot_callback:
                plot_callback(self.history)

        if log_callback:
            log_callback(f"Training complete! Best {self._select_metric}: {best_metric:.4f} | Best Acc: {best_acc:.4f}")

        # Roll back to the best-validated checkpoint before export/archive
        try:
            self.load_model("best_model.pth")
            self._ema = None
            if log_callback:
                log_callback(f"Rolled back to best_model.pth (Best Acc: {best_acc:.4f})")
        except Exception as e:
            if log_callback:
                log_callback(f"Warning: best-model rollback failed: {e}")

        # Export for inference
        try:
            self._export_for_inference()
            if log_callback:
                log_callback("Exported TorchScript/ONNX models")
        except Exception:
            pass

        # 模型版本归档
        try:
            vdir = self.save_version(config={
                "epochs": epochs, "batch_size": batch_size, "lr": lr,
                "optimizer": optimizer, "loss": loss_func,
                "image_size": image_size,
                "scale_factor": list(scale_factor) if isinstance(scale_factor, (tuple, list)) else scale_factor,
                "k_folds": k_folds, "resume": resume,
                "weight_decay": weight_decay, "momentum": momentum,
                "label_smoothing": label_smoothing, "focal_gamma": focal_gamma,
                "dropout": dropout, "lr_scheduler": lr_scheduler,
                "warmup_epochs": warmup_epochs, "early_stop_patience": early_stop_patience,
                "ema": ema, "ema_decay": ema_decay, "class_balanced": class_balanced,
                "select_metric": str(getattr(self, "_select_metric", "acc")),
                "augment": dict(augment) if isinstance(augment, dict) else (augment if augment else None),
            })
            if log_callback:
                log_callback(f"模型版本 {vdir.name} 已归档")
        except Exception as e:
            if log_callback:
                log_callback(f"模型版本归档失败: {e}")

        return {"success": True, "best_acc": float(best_acc), "history": self.history}

    def stop(self):
        self._stop_flag = True

    # ── Internal methods ────────────────────────────────────────────
    def _build_optimizer(self, optimizer_name, lr, weight_decay=1e-4, momentum=0.9):
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── LR 调度器 ────────────────────────────────────────────────────
    # 旧实现：lr_scheduler / warmup_epochs 只是「传进来、写进版本 config」，
    # 训练循环里从未创建 scheduler ⇒ UI 选 cosine、智能推荐推 cosine，
    # 实际都按恒定 lr 跑（死开关）。这里补上真正的实现。
    def _build_scheduler(self, sched_name, total_epochs, warmup_epochs=0,
                         start_epoch=0, min_lr_ratio=0.01):
        """构建 LR 调度器；返回 (scheduler, step_with_metric)。

        - 名称 none/''/off → (None, False)
        - 用 LambdaLR 表达 step/cosine，把「第几个 epoch」写进 lambda，
          因此断点续训（resume）时只要传 start_epoch，曲线能自动对齐，
          不需要额外存取调度器状态。
        - plateau 用 ReduceLROnPlateau，按验证指标调整（step_with_metric=True）。
        """
        self._sched = None
        self._sched_step_with_metric = False
        name = str(sched_name or "none").strip().lower()
        if name in ("", "none", "off", "null"):
            return None, False
        total_epochs = max(1, int(total_epochs))
        wu = max(0, int(warmup_epochs or 0))

        if name == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5,
                patience=max(2, total_epochs // 10),
                min_lr=self.optimizer.param_groups[0]["lr"] * 0.01,
            )
            self._sched = sched
            self._sched_step_with_metric = True
            return sched, True

        def lr_at(ep):
            ep = int(ep)
            if wu > 0 and ep < wu:                      # 线性 warmup
                return float(ep + 1) / float(wu)
            p = (ep - wu) / max(1, total_epochs - wu)   # 剩余进度 0..1
            p = min(max(p, 0.0), 1.0)
            if name == "step":
                # 每 1/3 训练进度掉一档，γ=0.1：1.0 → 0.1 → 0.01
                return 0.1 ** min(2, int(p * 3))
            if name == "cosine":
                import math
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))
            # 未知名称：按恒定 lr 处理（并在日志里说明）
            return 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_at, last_epoch=int(start_epoch) - 1)
        self._sched = sched
        self._sched_step_with_metric = False
        return sched, False

    def _current_lr(self):
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return float("nan")

    def _log_training_context(self, lr, optimizer, loss_func, lr_scheduler, warmup_epochs,
                              batch_size, image_size, keep_aspect, augment, ema,
                              class_balanced, dropout, label_smoothing, focal_gamma,
                              pretrained, start_epoch, epochs, log_callback):
        """训练前把「这次到底在跑什么」打全。

        判「跑的是哪套」只能看运行实例自述，不能看源码默认值 ——
        这个问题在本项目已经吃过一次（阈值唯一真相源）。
        """
        log = log_callback or (lambda _m: None)
        dev = self.device
        if _device_is_cuda(dev):
            try:
                dev_desc = f"CUDA:{torch.cuda.get_device_name(dev)}"
            except Exception:
                dev_desc = "CUDA"
        else:
            dev_desc = "CPU（未检测到可用 CUDA 设备）"
        try:
            n_threads = torch.get_num_threads()
        except Exception:
            n_threads = -1

        n_tr = len(self.train_loader.dataset)
        n_va = len(self.val_loader.dataset)
        used = n_tr
        if self._drop_last and batch_size > 0:
            used = n_tr - (n_tr % batch_size)
        try:
            from collections import Counter
            cnt = Counter(pid for _, _, pid in self.train_loader.dataset._pairs)
            dist = " / ".join(f"{self.class_names[i]} {cnt.get(i, 0)}"
                              for i in range(len(self.class_names)))
        except Exception:
            dist = "?"

        raw_sched = str(lr_scheduler or "none").strip().lower()
        if self._sched is None:
            sched_desc = ("none（恒定 lr）" if raw_sched in ("", "none", "off", "null")
                          else f"{raw_sched} ← 未识别/构建失败，实际按恒定 lr 跑")
        else:
            sched_desc = raw_sched + (f" + warmup {warmup_epochs}ep" if warmup_epochs else "")

        if self._amp_enabled:
            amp_desc = "开启"
        elif torch.cuda.is_available():
            amp_desc = "关闭"
        else:
            amp_desc = "关闭（无 CUDA，CPU 上不适用）"

        aug_desc = getattr(self.train_loader.dataset, "augment_desc", "?")
        wd = 0.0
        try:
            wd = float(self.optimizer.param_groups[0].get("weight_decay", 0.0))
        except Exception:
            pass

        log("──────── 本次训练自述（判『跑的是哪套』看这里）────────")
        log(f"[环境] {dev_desc} | torch {torch.__version__} | CPU 线程 {n_threads} | DataLoader workers {self._num_workers}")
        log(f"[数据] train {n_tr}（{dist}）| val {n_va} | 起始 epoch {start_epoch} | 本轮 {epochs} 轮")
        log(f"[输入] image_size={image_size} keep_aspect={keep_aspect} scale_factor={getattr(self, '_scale_factor', 1.0)}")
        log(f"[模型] {self.model_name} 预训练={pretrained} dropout={dropout} 类别={self.class_names}")
        if pretrained:
            cached, wfile = pretrained_weight_status(self.model_name)
            if cached is False:
                log(f"[权重] ⚠ 本地缓存没有 {self.model_name} 的预训练权重，"
                    f"训练会先从 download.pytorch.org 下载（现场网络受限时会卡住/失败）")
            elif cached:
                log(f"[权重] 预训练权重已缓存：{wfile}")
        log(f"[优化] {optimizer} lr={lr:g} weight_decay={wd:g} | loss={loss_func} "
            f"label_smoothing={label_smoothing} focal_gamma={focal_gamma}")
        log(f"[调度] {sched_desc} | AMP {amp_desc} | EMA {'开' if ema else '关'} | 类平衡 {'开' if class_balanced else '关'}")
        log(f"[批次] batch={batch_size} drop_last={self._drop_last}（每轮实际喂 {used}/{n_tr} 张）")
        log(f"[增强] {aug_desc} | 选最优模型指标 select_metric={self._select_metric}")
        log("─────────────────────────────────────────────────")

    def _build_loss(self, loss_name, label_smoothing=0.1, focal_gamma=2.0, class_balanced=False):
        weights = None
        if class_balanced:
            try:
                from .eval import compute_class_weights
                weights = compute_class_weights(str(self.project_dir)).to(self.device)
            except Exception:
                weights = None
        if loss_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        elif loss_name == "label_smoothing":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weights)
        elif loss_name == "focal":
            alpha = weights.tolist() if weights is not None else None
            self.criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=weights)

    def _prepare_data(self, batch_size, image_size, scale_factor=1.0, augment=None,
                      keep_aspect=False):
        # augment=None 表示「不增强」：UI 的增强总开关关掉时 _get_augment_flags() 返回 None。
        # 旧实现把 None 直接透给 Dataset，Dataset 在 augment_flags=None 时回落到内置增强
        # ⇒ UI 上写着「无增强/原生基线」，实际仍在翻转+旋转+色彩抖动（标签与行为不符）。
        flags = _resolve_cls_augment(augment)
        if flags is None:
            flags = {"enabled": False}
        train_ds = ClassificationDataset(
            str(self.project_dir), split="train", image_size=image_size,
            scale_factor=scale_factor, augment_flags=flags, keep_aspect=keep_aspect
        )
        val_ds = ClassificationDataset(
            str(self.project_dir), split="val", image_size=image_size,
            scale_factor=scale_factor, augment_flags=None, keep_aspect=keep_aspect
        )
        self._target_size = getattr(train_ds, "_target_size", None)
        self._scale_factor = scale_factor

        if len(train_ds) == 0:
            raise ValueError("Training set is empty!")
        if len(val_ds) == 0 and len(train_ds) >= 2:
            auto_split(str(self.project_dir), val_split=0.2)
            train_ds = ClassificationDataset(
                str(self.project_dir), split="train", image_size=image_size,
                scale_factor=scale_factor, augment_flags=flags, keep_aspect=keep_aspect
            )
            val_ds = ClassificationDataset(
                str(self.project_dir), split="val", image_size=image_size,
                scale_factor=scale_factor, augment_flags=None, keep_aspect=keep_aspect
            )
        if len(val_ds) == 0:
            raise ValueError("Validation set is empty! Not enough samples.")

        self.num_classes = train_ds.num_classes
        self.class_names = train_ds.class_names

        import multiprocessing
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # ── drop_last：小数据集上「丢尾巴」代价极高 ──────────────────
        # 旧实现 drop_last = len(train) > batch，157 张 / batch 32 时
        # 每轮丢掉最后 29 张（18%），等于白丢样本。
        # 只有尾巴小到会破坏 BN 统计（< batch/4）时才丢。
        rem = len(train_ds) % max(1, batch_size)
        drop = bool(rem != 0 and rem < max(2, batch_size // 4))
        self._drop_last = drop

        # ── worker 数：CPU 训练时 worker 会和主进程抢同一批物理核 ─────
        cpu_n = multiprocessing.cpu_count() or 4
        on_cuda = _device_is_cuda(self.device)
        if on_cuda:
            nw = min(4, cpu_n)
        else:
            # CPU 路径：小数据集 inline 取数更快（Windows spawn 每个 worker 要重新
            # import torch，一次 2~4s，比它省下的取数时间还贵）；大一点才开 2 个。
            nw = 0 if len(train_ds) <= 1024 else min(2, cpu_n)
        self._num_workers = nw

        collate = self._pad_collate if (image_size is None and getattr(train_ds, "_target_size", None) is None) else None

        def _loader(ds, bs, shuffle, drop_last=False):
            kw = dict(batch_size=bs, shuffle=shuffle, num_workers=nw,
                      collate_fn=collate, drop_last=drop_last,
                      # CPU 上 pin_memory 无意义（没有 H2D 拷贝），只是白占页锁定内存
                      pin_memory=on_cuda)
            if nw > 0:
                kw["persistent_workers"] = True   # 避免每个 epoch 重启 worker（Windows spawn 很贵）
                kw["prefetch_factor"] = 2
            return DataLoader(ds, **kw)

        self.train_loader = _loader(train_ds, batch_size, True, drop_last=drop)
        self.val_loader = _loader(val_ds, min(batch_size, len(val_ds)), False)

    def _train_epoch(self, batch_callback=None):
        self.model.train()
        total_loss = 0.0
        batches = 0
        pbar = tqdm(self.train_loader, desc="Training", ncols=80, file=sys.stderr if sys.stderr else None, disable=sys.stderr is None)

        for images, labels in pbar:
            if self._stop_flag or (self._stop_check_fn and self._stop_check_fn()):
                self._stop_flag = True
                return total_loss / max(batches, 1)

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self._amp_enabled:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self._ema is not None:
                    self._ema.update(self.model)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if self._ema is not None:
                    self._ema.update(self.model)

            total_loss += loss.item()
            batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_callback and batches % 10 == 0:
                batch_callback(batches, len(self.train_loader), total_loss / batches)

        return total_loss / max(batches, 1)

    @torch.no_grad()
    def _validate(self):
        """返回 (val_loss, val_acc, metrics)。metrics 含混淆矩阵/逐类 P-R-F1/漏检/过杀。"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc="Validating", ncols=80, file=sys.stderr if sys.stderr else None, disable=sys.stderr is None)
        for images, labels in pbar:
            if self._stop_flag:
                break

            images = images.to(self.device)
            labels_np = labels  # keep on CPU for counting
            labels = labels.to(self.device)

            if self._amp_enabled:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = outputs.max(1)
            correct += preds.cpu().eq(labels_np).sum().item()
            total += labels_np.size(0)
            # 留全量预测，用于混淆矩阵 / 逐类指标（验证集很小，内存可忽略）
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels_np.detach().cpu().numpy())

            pbar.set_postfix(acc=f"{correct/max(total,1):.4f}")

        avg_loss = total_loss / max(len(self.val_loader), 1)
        acc = correct / max(total, 1)

        metrics = {}
        if all_labels:
            try:
                metrics = confusion_metrics(
                    np.concatenate(all_labels), np.concatenate(all_preds),
                    self.class_names, self.num_classes)
            except Exception:
                metrics = {}
        return avg_loss, acc, metrics

    def _train_kfold(self, k, epochs, batch_size, image_size,
                     loss_func, optimizer, lr, pretrained,
                     weight_decay, momentum, label_smoothing, focal_gamma,
                     dropout, lr_scheduler, warmup_epochs, early_stop_patience,
                     progress_callback, log_callback, stop_check,
                     plot_callback, batch_callback,
                     ema=True, ema_decay=0.999, class_balanced=False,
                     keep_aspect=False, use_amp=True, select_metric="acc"):
        """K-fold cross-validation with stratified splits."""
        self._select_metric = str(select_metric or "acc").strip().lower()
        # Ensure split exists to get all samples
        split_file = self.project_dir / "train_test_split.json"
        if not split_file.exists():
            auto_split(str(self.project_dir), val_split=0.0)
            if log_callback:
                log_callback("Auto-generated labels for K-fold")

        # Load all samples
        all_ds = ClassificationDataset(
            str(self.project_dir), split="all", image_size=image_size,
            keep_aspect=keep_aspect
        )
        num_samples = len(all_ds)
        if num_samples < k:
            if log_callback:
                log_callback(f"Only {num_samples} samples, need at least {k}. Reducing k to {num_samples}.")
            k = num_samples
        if k < 2:
            if log_callback:
                log_callback("Need at least 2 samples for K-fold. Falling back to normal training.")
            return self.train(epochs=epochs, batch_size=batch_size, image_size=image_size,
                             loss_func=loss_func, optimizer=optimizer, lr=lr,
                             pretrained=pretrained, resume=False, k_folds=1,
                             weight_decay=weight_decay, momentum=momentum,
                             label_smoothing=label_smoothing, focal_gamma=focal_gamma,
                             dropout=dropout, lr_scheduler=lr_scheduler,
                             warmup_epochs=warmup_epochs, early_stop_patience=early_stop_patience,
                             ema=ema, ema_decay=ema_decay, class_balanced=class_balanced,
                             use_amp=use_amp, select_metric=self._select_metric,
                             progress_callback=progress_callback, log_callback=log_callback,
                             stop_check=stop_check, plot_callback=plot_callback,
                             batch_callback=batch_callback, keep_aspect=keep_aspect)

        # Stratified fold assignment
        all_labels = [pid for _, _, pid in all_ds._pairs]
        rng = np.random.RandomState(42)
        indices = np.arange(num_samples)
        rng.shuffle(indices)

        # Stratified: group by label, split into k folds
        from collections import defaultdict
        by_label = defaultdict(list)
        for idx in indices:
            by_label[all_labels[idx]].append(idx)

        folds = [[] for _ in range(k)]
        for label_grp in by_label.values():
            fold_size = max(1, len(label_grp) // k)
            for i, idx in enumerate(label_grp):
                fold_idx = min(i // fold_size, k - 1)
                folds[fold_idx].append(idx)

        fold_records = []
        for fold in range(k):
            if self._stop_flag or (stop_check and stop_check()):
                if log_callback:
                    log_callback("K-fold stopped by user")
                break

            val_indices = set(folds[fold])
            train_indices = [i for i in range(num_samples) if i not in val_indices]

            # Build fold-specific dataloaders
            train_subset = torch.utils.data.Subset(all_ds, train_indices)
            val_subset = torch.utils.data.Subset(all_ds, list(val_indices))

            drop = bool(len(train_subset) % max(1, batch_size) != 0
                        and (len(train_subset) % max(1, batch_size)) < max(2, batch_size // 4))
            self._drop_last = drop
            cpu_n = __import__("multiprocessing").cpu_count() or 4
            on_cuda = _device_is_cuda(self.device)
            nw = min(4, cpu_n) if on_cuda else (0 if len(train_subset) <= 1024 else min(2, cpu_n))
            self._num_workers = nw
            collate = self._pad_collate if (image_size is None and getattr(all_ds, "_target_size", None) is None) else None

            def _fold_loader(ds, bs, shuffle, dl):
                kw = dict(batch_size=bs, shuffle=shuffle, num_workers=nw,
                          collate_fn=collate, drop_last=dl, pin_memory=on_cuda)
                if nw > 0:
                    kw["persistent_workers"] = True
                    kw["prefetch_factor"] = 2
                return DataLoader(ds, **kw)

            self.train_loader = _fold_loader(train_subset, batch_size, True, drop)
            self.val_loader = _fold_loader(val_subset, min(batch_size, len(val_subset)), False, False)

            # Re-init model per fold
            self.model = _create_classification_model(
                self.model_name, self.num_classes, pretrained=pretrained, dropout=dropout
            ).to(self.device)
            self._build_optimizer(optimizer, lr, weight_decay, momentum)
            self._build_loss(loss_func, label_smoothing, focal_gamma, class_balanced=class_balanced)
            # AMP / 调度器：每个 fold 重新建（optimizer 是新的）
            self._amp_enabled = bool(use_amp) and torch.cuda.is_available()
            self.scaler = torch.amp.GradScaler("cuda") if self._amp_enabled else None
            self._build_scheduler(lr_scheduler, epochs, warmup_epochs, start_epoch=0)
            self._ema = ModelEMA(self.model, decay=ema_decay) if ema else None
            self._stop_flag = False

            if log_callback:
                log_callback(f"--- Fold {fold+1}/{k}: {len(train_indices)} train, {len(val_indices)} val "
                             f"(scheduler={lr_scheduler}, amp={'on' if self._amp_enabled else 'off'}) ---")

            best_acc = 0.0
            best_metric = 0.0
            for epoch in range(epochs):
                if self._stop_flag or (stop_check and stop_check()):
                    self._stop_flag = True
                    break
                train_loss = self._train_epoch(batch_callback)
                if self._ema is not None:
                    self._ema.apply_shadow(self.model)
                try:
                    val_loss, val_acc, vmet = self._validate()
                finally:
                    if self._ema is not None:
                        self._ema.restore(self.model)
                cur_metric = metric_for_selection(val_acc, vmet, self._select_metric)
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history.setdefault("val_select", []).append(cur_metric)
                self.history.setdefault("lr", []).append(self._current_lr())

                if val_acc > best_acc:
                    best_acc = val_acc
                if cur_metric > best_metric:
                    best_metric = cur_metric

                if self._sched is not None:
                    try:
                        if self._sched_step_with_metric:
                            self._sched.step(cur_metric)
                        else:
                            self._sched.step()
                    except Exception:
                        self._sched = None

                msg = (f"Fold {fold+1}/{k} Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                       f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | lr {self._current_lr():.2e}")
                if log_callback:
                    log_callback(msg)
                if progress_callback:
                    progress_callback(epoch + 1, epochs)
                if plot_callback:
                    plot_callback(self.history)

            fold_records.append(best_acc)
            if log_callback:
                log_callback(f"Fold {fold+1}/{k} best Acc: {best_acc:.4f}")

        if fold_records:
            avg_acc = np.mean(fold_records)
            std_acc = np.std(fold_records)
            if log_callback:
                log_callback(f"=== K-Fold Complete: avg Acc={avg_acc:.4f} ±{std_acc:.4f} ===")
            self._save_model("best_model.pth", ema=(self._ema is not None))
            self._save_model("last_model.pth")
            try:
                vdir = self.save_version(config={
                    "epochs": epochs, "batch_size": batch_size, "lr": lr,
                    "optimizer": optimizer, "loss": loss_func,
                    "image_size": image_size,
                    # 这里原来裸用 `scale_factor` —— 它不是本函数的参数/局部变量，
                    # k_folds>1 训练到"存折 checkpoint"这一步必 NameError。
                    # 用 self._scale_factor（train() 里已设，且本文件别处也这么取）。
                    "scale_factor": (list(self._scale_factor)
                                     if isinstance(self._scale_factor, (tuple, list))
                                     else self._scale_factor),
                    "k_folds": k, "resume": False,
                    "weight_decay": weight_decay, "momentum": momentum,
                    "label_smoothing": label_smoothing, "focal_gamma": focal_gamma,
                    "dropout": dropout, "lr_scheduler": lr_scheduler,
                    "warmup_epochs": warmup_epochs, "early_stop_patience": early_stop_patience,
                    "ema": ema, "ema_decay": ema_decay, "class_balanced": class_balanced,
                    "select_metric": str(getattr(self, "_select_metric", "acc")),
                    "fold_accuracies": [float(a) for a in fold_records],
                    "mean_acc": float(avg_acc),
                })
                if log_callback:
                    log_callback(f"模型版本 {vdir.name} 已归档")
            except Exception:
                pass
            return {"success": True, "mean_acc": float(avg_acc), "std_acc": float(std_acc), "fold_accuracies": fold_records}

    # ── Model persistence ───────────────────────────────────────────
    def _save_model(self, filename, ema=False):
        """Save model checkpoint (EMA smoothed weights when ema=True)."""
        models_dir = self.project_dir / "models" / "classification"
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / filename
        state = self.model.state_dict()
        if ema and self._ema is not None:
            state = {k: v.detach().clone() for k, v in self._ema.shadow.items()}
        torch.save({
            "model_state_dict": state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "model_name": self.model_name,
            "image_size": self._image_size,
            "scale_factor": getattr(self, "_scale_factor", 1.0),
            "target_size": getattr(self, "_target_size", None),
            "keep_aspect": getattr(self, "_keep_aspect", False),
            "history": self.history,
        }, path)

    def load_model(self, filename="best_model.pth"):
        """Load a trained model for inference."""
        # 已加载过同一个模型时直接复用，避免每张图重复 torch.load + load_state_dict
        # （批量推理对每张图调用 predict_single，会反复触发 load_model，是 0.5s/图 的主因之一）。
        if self.model is not None and getattr(self, "_loaded_model_file", None) == filename:
            return getattr(self, "_loaded_history", {})
        path = resolve_model_path(self.project_dir, filename)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.num_classes = ckpt["num_classes"]
        self.class_names = ckpt.get("class_names", [f"class_{i}" for i in range(self.num_classes)])
        self.model_name = ckpt.get("model_name", self.model_name)
        self._image_size = ckpt.get("image_size", 224)
        self._scale_factor = ckpt.get("scale_factor", 1.0)
        self._keep_aspect = ckpt.get("keep_aspect", False)
        self._target_size = ckpt.get("target_size", None)

        self.model = _create_classification_model(
            self.model_name, self.num_classes, pretrained=False
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._loaded_model_file = filename
        self._loaded_history = ckpt.get("history", {})
        return self._loaded_history

    def _resolve_inference_resize(self):
        """Return (h, w) resize tuple for inference, or None when no resize applies.

        Precedence: stored image_size (int=square 或 (W,H) tuple) -> stored target_size
        (from scale_factor training) -> project scale settings + train split max dims.
        """
        if self._image_size is not None:
            if isinstance(self._image_size, (tuple, list)):
                return (int(self._image_size[1]), int(self._image_size[0]))  # (H, W)
            return (self._image_size, self._image_size)
        if getattr(self, "_target_size", None):
            tw, th = self._target_size
            if tw and th:
                return (th, tw)
        # Fallback for checkpoints trained before target_size was persisted:
        # replicate dataset._compute_target_size() using train split + scale settings.
        try:
            from PIL import Image
            settings = {}
            sf_path = self.project_dir / ".cls_train_settings.json"
            if sf_path.exists():
                with open(sf_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            sw = float(settings.get("scale_w", 1.0))
            sh = float(settings.get("scale_h", 1.0))
            if sw == 1.0 and sh == 1.0:
                return None
            img_dir = self.project_dir / "images"
            if not img_dir.exists():
                return None
            split_map = {}
            sp = self.project_dir / "train_test_split.json"
            if sp.exists():
                with open(sp, "r", encoding="utf-8") as f:
                    split_map = json.load(f)
            candidates = [k for k, s in split_map.items() if s == "train"]
            if not candidates:
                candidates = [os.path.splitext(f)[0] for f in os.listdir(str(img_dir))
                              if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))]
            max_w = max_h = 0
            for base in candidates:
                for ext in (".bmp", ".png", ".jpg", ".jpeg"):
                    fp = img_dir / (base + ext)
                    if fp.exists():
                        try:
                            with Image.open(str(fp)) as im:
                                max_w = max(max_w, im.size[0])
                                max_h = max(max_h, im.size[1])
                        except Exception:
                            pass
                        break
            if max_w > 0 and max_h > 0:
                return (max(int(max_h * sh), 1), max(int(max_w * sw), 1))
        except Exception:
            pass
        return None

    def _export_for_inference(self, out_dir=None):
        """Export trained model to TorchScript and ONNX.

        out_dir: 导出目录；默认 models/classification（版本归档时传入版本目录）。
        """
        self.model.eval()
        models_dir = Path(out_dir) if out_dir else (self.project_dir / "models" / "classification")
        models_dir.mkdir(parents=True, exist_ok=True)

        rsz = self._resolve_inference_resize()
        img_h, img_w = rsz if rsz else (224, 224)
        dummy = torch.randn(1, 3, img_h, img_w).to(self.device)

        # TorchScript
        try:
            with torch.no_grad(), torch.amp.autocast("cuda"):
                traced = torch.jit.trace(self.model, dummy)
            ts_path = models_dir / "cls_model_scripted.pt"
            traced.save(str(ts_path))
        except Exception:
            pass

        # ONNX
        try:
            onnx_path = models_dir / "cls_model.onnx"
            torch.onnx.export(
                self.model, dummy, str(onnx_path),
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=14,
            )
        except Exception:
            pass

    def save_version(self, config=None):
        """把当前最优权重归档为新版本 versions/V{n}（含 meta/history/onnx）。

        config: dict，训练超参数等附加元数据。
        """
        n = next_version_number(self.project_dir)
        state = self.model.state_dict()
        if self._ema is not None:
            state = {k: v.detach().clone() for k, v in self._ema.shadow.items()}
        opt_state = {}
        try:
            if getattr(self, "optimizer", None) is not None:
                opt_state = self.optimizer.state_dict()
        except Exception:
            opt_state = {}
        meta = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "model_name": self.model_name,
            "image_size": getattr(self, "_image_size", None),
            "scale_factor": getattr(self, "_scale_factor", 1.0),
            "target_size": getattr(self, "_target_size", None),
            "keep_aspect": getattr(self, "_keep_aspect", False),
            "best_val_acc": max(self.history.get("val_acc", [0])) if self.history.get("val_acc") else 0.0,
        }
        if config:
            meta.update(config)
        vdir = write_version_archive(str(self.project_dir), n, state, opt_state, meta, self.history)
        try:
            self._export_for_inference(out_dir=vdir)
        except Exception:
            pass
        return vdir

    # ── Inference ───────────────────────────────────────────────────
    def predict_single(self, image_path, model_filename, top_k=5, tta=False):
        """Convenience method: load model and classify a single image.

        tta=True enables multi-crop + flip voting for a more stable prediction.
        Returns dict: {"predictions": [{"class": ..., "confidence": ...}, ...]}
        or None on failure.
        """
        self.load_model(model_filename)
        results = self.predict(image_path, top_k=top_k, tta=tta)
        if not results:
            return None
        return {
            "predictions": [
                {"class": cls_name, "confidence": conf}
                for cls_name, conf in results
            ]
        }

    def _predict_tta(self, img, transform):
        """Multi-crop + horizontal-flip TTA: average softmax over 10 views."""
        from PIL import Image
        w, h = img.size
        s = min(w, h)
        crop = max(int(s * 0.9) // 2 * 2 + 1, 17)
        crop = min(crop, s)
        if w - crop < 4 or h - crop < 4:
            crop = s
        positions = [
            (0, 0),
            (0, h - crop),
            (w - crop, 0),
            (w - crop, h - crop),
            ((w - crop) // 2, (h - crop) // 2),
        ]
        views = []
        for x, y in positions:
            v = img.crop((x, y, x + crop, y + crop))
            views.append(v)
            views.append(v.transpose(Image.FLIP_LEFT_RIGHT))
        probs = None
        with torch.amp.autocast("cuda"):
            for v in views:
                tv = transform(v).unsqueeze(0).to(self.device)
                p = self.model(tv).softmax(1)
                probs = p if probs is None else probs + p
        return (probs / len(views)).squeeze(0)

    @torch.inference_mode()
    def predict(self, images, top_k=5, tta=False):
        """Classify image(s). Returns list of [(class_name, confidence), ...] or list-of-lists.

        tta=True (single image only) averages softmax over multi-crop + flips.
        """
        from PIL import Image
        import torchvision.transforms as T

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        is_single = not isinstance(images, list)
        if is_single:
            images = [images]

        rsz = self._resolve_inference_resize()
        if rsz is not None and getattr(self, "_keep_aspect", False):
            from .dataset import KeepAspectResize
            resize_step = KeepAspectResize((rsz[1], rsz[0]))  # (W, H)
        else:
            resize_step = T.Resize(rsz) if rsz is not None else T.Lambda(lambda x: x)
        transform = T.Compose([
            resize_step,
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        results = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if tta and is_single:
                probs = self._predict_tta(img, transform)
            else:
                tensor = transform(img).unsqueeze(0).to(self.device)
                with torch.amp.autocast("cuda"):
                    output = self.model(tensor)
                probs = output.softmax(1).squeeze(0)

            topk_probs, topk_ids = probs.topk(min(top_k, self.num_classes))
            top_results = [
                (self.class_names[idx], float(conf))
                for conf, idx in zip(topk_probs.cpu().numpy(), topk_ids.cpu().numpy())
            ]
            results.append(top_results)

        return results[0] if is_single else results


def get_training_history(project_dir):
    """Load training history from best_model.pth."""
    path = Path(project_dir) / "models" / "classification" / "best_model.pth"
    if path.exists():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("history", {})
    return {}

# ======================== Dual-Input Classification Trainer ========================

class DualClassificationTrainer:
    """Trains a dual-input model using grayscale + height map pairs."""

    def __init__(self, project_dir, model_name="resnet18", device=None):
        self.project_dir = Path(project_dir)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.project_dir / "project.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # Determine num_classes
        from .dual_dataset import MixedClassificationDataset
        ds = MixedClassificationDataset(project_dir, split="all", image_size=224)
        self.num_classes = ds.num_classes
        self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        if hasattr(ds, "_class_names"):
            self.class_names = ds._class_names

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.optimizer = None
        self.scaler = None
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self._stop_requested = False
        self._stop_check_fn = None
        self._ema = None

    def _prepare_data(self, batch_size=32, image_size=224, augment=None):
        from .dual_dataset import MixedClassificationDataset

        flags = _resolve_cls_augment(augment)
        train_ds = MixedClassificationDataset(
            self.project_dir, split="train", image_size=image_size, augment=flags)
        val_ds = MixedClassificationDataset(
            self.project_dir, split="val", image_size=image_size, augment=False)

        if len(val_ds) == 0 and len(train_ds) >= 2:
            from .dual_dataset import auto_split
            auto_split(self.project_dir, val_ratio=0.2)
            train_ds = MixedClassificationDataset(
                self.project_dir, split="train", image_size=image_size, augment=flags)
            val_ds = MixedClassificationDataset(
                self.project_dir, split="val", image_size=image_size, augment=False)

        if len(train_ds) == 0:
            raise ValueError("Training set is empty")
        if len(val_ds) == 0:
            raise ValueError("Validation set is empty")

        self.num_classes = train_ds.num_classes
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                       num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=2, pin_memory=True)
        return len(train_ds), len(val_ds)

    def _make_criterion(self, class_balanced=False):
        weights = None
        if class_balanced:
            try:
                from .eval import compute_class_weights
                weights = compute_class_weights(str(self.project_dir)).to(self.device)
            except Exception:
                weights = None
        return nn.CrossEntropyLoss(weight=weights)

    def init_model(self, class_balanced=False):
        from .dual_model import create_dual_model
        self.model = create_dual_model(self.model_name, self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = self._make_criterion(class_balanced)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        batches = 0
        pbar = tqdm(self.train_loader, desc="Train", ncols=80, file=sys.stderr if sys.stderr else None, disable=sys.stderr is None)
        for (gray, height), labels in pbar:
            if self._stop_requested or (self._stop_check_fn and self._stop_check_fn()):
                self._stop_requested = True
                return total_loss / max(batches, 1)
            gray = gray.to(self.device)
            height = height.to(self.device)
            labels = labels.to(self.device)

            # Stack as 4-channel input
            inputs = torch.cat([gray, height], dim=1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if self._ema is not None:
                self._ema.update(self.model)

            total_loss += loss.item()
            batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(batches, 1)

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        batches = 0
        with torch.no_grad():
            for (gray, height), labels in self.val_loader:
                if self._stop_requested:
                    break
                gray = gray.to(self.device)
                height = height.to(self.device)
                labels = labels.to(self.device)

                inputs = torch.cat([gray, height], dim=1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                batches += 1

                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def train(self, epochs=200, batch_size=32, lr=0.001, image_size=224,
              progress_callback=None, log_callback=None, resume=False,
              stop_check=None, epoch_callback=None, augment=None,
              class_balanced=False, ema=True, ema_decay=0.999,
              optimizer=None, loss_func=None, scale_factor=1.0, use_amp=None,
              k_folds=1, weight_decay=None, momentum=None, label_smoothing=None,
              focal_gamma=None, dropout=None, lr_scheduler=None,
              warmup_epochs=None, early_stop_patience=None, plot_callback=None,
              keep_aspect=None):
        if resume:
            ckpt_path = self.project_dir / "models" / "classification" / "last_model.pth"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.num_classes = ckpt["num_classes"]
                self.model_name = ckpt.get("model_name", self.model_name)
                self.history = ckpt.get("history", {"train_loss": [], "val_loss": [], "val_acc": []})
                start_epoch = len(self.history.get("train_loss", []))
                self._prepare_data(batch_size, image_size, scale_factor, augment=augment)
                from .dual_model import create_dual_model
                self.model = create_dual_model(self.model_name, self.num_classes).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                self.criterion = self._make_criterion(class_balanced)
                if "optimizer_state_dict" in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception:
                        if log_callback:
                            log_callback("Optimizer state mismatch, using fresh optimizer")
                if log_callback:
                    log_callback(f"Resuming from epoch {start_epoch} (Val Acc: {self.history.get('val_acc', [0])[-1] if self.history.get('val_acc') else 0:.4f})")
            else:
                resume = False
                if log_callback:
                    log_callback("No checkpoint found, training from scratch")

        if not resume:
            self._prepare_data(batch_size, image_size, augment=augment)
            self.init_model(class_balanced=class_balanced)
            start_epoch = 0
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        self._stop_requested = False
        self._stop_check_fn = stop_check
        self._ema = ModelEMA(self.model, decay=ema_decay) if ema else None

        best_acc = self.history.get("val_acc", [0])[-1] if self.history.get("val_acc") else 0
        no_improve = 0
        for epoch in range(start_epoch, epochs):
            train_loss = self.train_epoch()
            if self._ema is not None:
                self._ema.apply_shadow(self.model)
            try:
                val_loss, val_acc = self.validate()
            finally:
                if self._ema is not None:
                    self._ema.restore(self.model)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if stop_check and stop_check():
                self._stop_requested = True
                if log_callback:
                    log_callback("Training stopped by user")
                break

            msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            if log_callback:
                log_callback(msg)

            # Save checkpoint every epoch
            os.makedirs(self.project_dir / "models" / "classification", exist_ok=True)
            ckpt = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_classes": self.num_classes,
                "model_name": self.model_name,
                "history": self.history,
            }
            torch.save(ckpt, self.project_dir / "models" / "classification" / "last_model.pth")

            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                if self._ema is not None:
                    ckpt = dict(ckpt)
                    ckpt["model_state_dict"] = {k: v.detach().clone() for k, v in self._ema.shadow.items()}
                torch.save(ckpt, self.project_dir / "models" / "classification" / "best_model.pth")
                if log_callback:
                    log_callback(f"  Best saved (Acc: {best_acc:.4f})")
            elif early_stop_patience and early_stop_patience > 0:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    if log_callback:
                        log_callback(f"Early stop at epoch {epoch+1}: no val improvement for {early_stop_patience} epochs (best {best_acc:.4f})")
                    break

            if progress_callback:
                progress_callback(epoch + 1, epochs)
            if epoch_callback:
                epoch_callback(epoch + 1, epochs, train_loss, val_acc)

        if log_callback:
            log_callback(f"Training complete. Best accuracy: {best_acc:.4f}")

        # Roll back to the best-validated checkpoint before archiving
        try:
            self.load_model("best_model.pth")
            self._ema = None
            if log_callback:
                log_callback(f"Rolled back to best_model.pth (Best Acc: {best_acc:.4f})")
        except Exception as e:
            if log_callback:
                log_callback(f"Warning: best-model rollback failed: {e}")

        # 模型版本归档
        try:
            vdir = self.save_version(config={
                "epochs": epochs, "batch_size": batch_size, "lr": lr,
                "image_size": image_size,
                "scale_factor": list(scale_factor) if isinstance(scale_factor, (tuple, list)) else scale_factor,
                "k_folds": k_folds, "resume": resume,
                "ema": ema, "ema_decay": ema_decay, "class_balanced": class_balanced,
                "select_metric": str(getattr(self, "_select_metric", "acc")),
                "augment": dict(augment) if isinstance(augment, dict) else (augment if augment else None),
            })
            if log_callback:
                log_callback(f"模型版本 {vdir.name} 已归档")
        except Exception as e:
            if log_callback:
                log_callback(f"模型版本归档失败: {e}")

    def load_model(self, filename="best_model.pth"):
        """加载双输入分类模型（支持版本名 V{n}）。"""
        from .dual_model import create_dual_model
        path = resolve_model_path(self.project_dir, filename)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.num_classes = ckpt["num_classes"]
        self.model_name = ckpt.get("model_name", self.model_name)
        if getattr(self, "_class_names", None):
            self.class_names = self._class_names
        else:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        self.model = create_dual_model(self.model_name, self.num_classes).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()
        return ckpt.get("history", {})

    def save_version(self, config=None):
        """把双输入模型最优权重归档为新版本 versions/V{n}。"""
        n = next_version_number(self.project_dir)
        state = self.model.state_dict()
        if self._ema is not None:
            state = {k: v.detach().clone() for k, v in self._ema.shadow.items()}
        meta = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "model_name": self.model_name,
            "image_size": getattr(self, "_image_size", None),
            "best_val_acc": max(self.history.get("val_acc", [0])) if self.history.get("val_acc") else 0.0,
            "mixed_input": True,
        }
        if config:
            meta.update(config)
        vdir = write_version_archive(str(self.project_dir), n, state, {}, meta, self.history)
        return vdir

    def predict(self, gray_path, height_path, image_size=224):
        """Inference on a single pair."""
        from PIL import Image as PILImage
        import torchvision.transforms as T

        gray = PILImage.open(gray_path).convert("RGB")
        height = PILImage.open(height_path).convert("L")

        gray = gray.resize((image_size, image_size), PILImage.BILINEAR)
        height = height.resize((image_size, image_size), PILImage.BILINEAR)

        gray_t = T.ToTensor()(gray)
        for c in range(3):
            gray_t[c] = (gray_t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        height_t = T.ToTensor()(height)
        height_t = (height_t - 0.5) / 0.25

        inputs = torch.cat([gray_t, height_t], dim=0).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        probs = outputs.softmax(1).squeeze(0)
        top_prob, top_idx = probs.max(0)
        return {
            "class": self.class_names[top_idx.item()],
            "class_id": top_idx.item(),
            "confidence": top_prob.item(),
        }

