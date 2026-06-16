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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
}


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


# ── Model factory ───────────────────────────────────────────────────────
def _create_classification_model(model_name, num_classes, pretrained=True):
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
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif family == "mobilenet":
        import torchvision.models as models
        model_fn = getattr(models, "mobilenet_v2", None)
        if model_fn is None:
            model_fn = models.mobilenet_v2
        model = model_fn(weights="DEFAULT" if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif family == "mobilenet3":
        import torchvision.models as models
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            model_fn = models.mobilenet_v3_large
        model = model_fn(weights="DEFAULT" if pretrained else None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif family == "vit":
        import torchvision.models as models
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            model_fn = models.vit_b_16
        model = model_fn(weights="DEFAULT" if pretrained else None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model family: {family}")

    return model


# ── Trainer ─────────────────────────────────────────────────────────────
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
        self.scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
        self._stop_flag = False
        self._stop_check_fn = None
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self._image_size = 224

    # ── Public API ──────────────────────────────────────────────────
    def train(self, epochs=50, batch_size=32, image_size=224,
              loss_func="cross_entropy", optimizer="adam", lr=1e-4,
              pretrained=True, resume=False, k_folds=1,
              augment="none", use_amp=True,
              progress_callback=None, log_callback=None,
              stop_check=None, plot_callback=None, batch_callback=None):
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
            augment: Augmentation level (none/light/medium/heavy) — accepted for UI compat
            use_amp: Use AMP (accepted for UI compat)
            progress_callback: (epoch, total) → None
            log_callback: (msg) → None
            stop_check: () → bool
            plot_callback: (history) → None
            batch_callback: (batch, total, avg_loss) → None
        """
        self._image_size = image_size
        self._stop_flag = False
        self._stop_check_fn = stop_check

        # K-fold branch
        if k_folds > 1 and not resume:
            return self._train_kfold(
                k_folds, epochs, batch_size, image_size,
                loss_func, optimizer, lr, pretrained,
                progress_callback, log_callback, stop_check,
                plot_callback, batch_callback
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
                self.history = ckpt.get("history", {"train_loss": [], "val_loss": [], "val_acc": []})
                start_epoch = len(self.history.get("train_loss", []))

                self.model = _create_classification_model(
                    self.model_name, self.num_classes, pretrained=False
                ).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.train()
                self._build_optimizer(optimizer, lr)
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
                self.model_name, self.num_classes, pretrained=pretrained
            ).to(self.device)
            self._build_optimizer(optimizer, lr)
            start_epoch = 0

        # Build datasets and dataloaders
        self._prepare_data(batch_size, image_size)

        # Build loss function
        self._build_loss(loss_func)

        # Run epochs
        epochs_total = start_epoch + epochs
        best_acc = max(self.history.get("val_acc", [0])) if self.history.get("val_acc") else 0.0

        for epoch in range(start_epoch, epochs_total):
            if self._stop_flag or (stop_check and stop_check()):
                self._stop_flag = True
                if log_callback:
                    log_callback("Training stopped by user")
                break

            train_loss = self._train_epoch(batch_callback)
            val_loss, val_acc = self._validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            msg = f"Epoch {epoch+1}/{epochs_total} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            if log_callback:
                log_callback(msg)

            if val_acc > best_acc:
                best_acc = val_acc
                self._save_model("best_model.pth")
                if log_callback:
                    log_callback(f"  → Best saved (Acc: {best_acc:.4f})")

            self._save_model("last_model.pth")

            if progress_callback:
                progress_callback(epoch + 1 - start_epoch, epochs_total - start_epoch)
            if plot_callback:
                plot_callback(self.history)

        if log_callback:
            log_callback(f"Training complete! Best Acc: {best_acc:.4f}")

        # Export for inference
        try:
            self._export_for_inference()
            if log_callback:
                log_callback("Exported TorchScript/ONNX models")
        except Exception:
            pass

        return {"success": True, "best_acc": float(best_acc), "history": self.history}

    def stop(self):
        self._stop_flag = True

    # ── Internal methods ────────────────────────────────────────────
    def _build_optimizer(self, optimizer_name, lr):
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _build_loss(self, loss_name):
        if loss_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == "label_smoothing":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        elif loss_name == "focal":
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _prepare_data(self, batch_size, image_size):
        train_ds = ClassificationDataset(
            str(self.project_dir), split="train", image_size=image_size
        )
        val_ds = ClassificationDataset(
            str(self.project_dir), split="val", image_size=image_size
        )

        if len(train_ds) == 0:
            raise ValueError("Training set is empty!")
        if len(val_ds) == 0:
            raise ValueError("Validation set is empty! Use auto_split() first.")

        self.num_classes = train_ds.num_classes
        self.class_names = train_ds.class_names

        import multiprocessing
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        drop = len(train_ds) > batch_size
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=min(4, multiprocessing.cpu_count() or 4),
            pin_memory=True, prefetch_factor=2, drop_last=drop
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False,
            num_workers=2, pin_memory=True, prefetch_factor=2
        )

    def _train_epoch(self, batch_callback=None):
        self.model.train()
        total_loss = 0.0
        batches = 0
        pbar = tqdm(self.train_loader, desc="Training", ncols=80)

        for images, labels in pbar:
            if self._stop_flag or (self._stop_check_fn and self._stop_check_fn()):
                self._stop_flag = True
                return total_loss / max(batches, 1)

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_callback and batches % 10 == 0:
                batch_callback(batches, len(self.train_loader), total_loss / batches)

        return total_loss / max(batches, 1)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc="Validating", ncols=80)
        for images, labels in pbar:
            if self._stop_flag:
                break

            images = images.to(self.device)
            labels_np = labels  # keep on CPU for counting
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = outputs.max(1)
            correct += preds.cpu().eq(labels_np).sum().item()
            total += labels_np.size(0)

            pbar.set_postfix(acc=f"{correct/max(total,1):.4f}")

        avg_loss = total_loss / max(len(self.val_loader), 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    def _train_kfold(self, k, epochs, batch_size, image_size,
                     loss_func, optimizer, lr, pretrained,
                     progress_callback, log_callback, stop_check,
                     plot_callback, batch_callback):
        """K-fold cross-validation with stratified splits."""
        # Ensure split exists to get all samples
        split_file = self.project_dir / "train_test_split.json"
        if not split_file.exists():
            auto_split(str(self.project_dir), val_split=0.0)
            if log_callback:
                log_callback("Auto-generated labels for K-fold")

        # Load all samples
        all_ds = ClassificationDataset(
            str(self.project_dir), split="all", image_size=image_size
        )
        num_samples = len(all_ds)
        if num_samples < k:
            if log_callback:
                log_callback(f"Only {num_samples} samples, need at least {k}. Reducing k to {num_samples}.")
            k = num_samples
        if k < 2:
            if log_callback:
                log_callback("Need at least 2 samples for K-fold. Falling back to normal training.")
            return self.train(epochs, batch_size, image_size, loss_func, optimizer, lr,
                             pretrained, resume=False, k_folds=1,
                             progress_callback=progress_callback, log_callback=log_callback,
                             stop_check=stop_check, plot_callback=plot_callback,
                             batch_callback=batch_callback)

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

            drop = len(train_subset) > batch_size
            self.train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=drop
            )
            self.val_loader = DataLoader(
                val_subset, batch_size=min(batch_size, len(val_subset)),
                shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2
            )

            # Re-init model per fold
            self.model = _create_classification_model(
                self.model_name, self.num_classes, pretrained=pretrained
            ).to(self.device)
            self._build_optimizer(optimizer, lr)
            self._build_loss(loss_func)
            self._stop_flag = False

            if log_callback:
                log_callback(f"--- Fold {fold+1}/{k}: {len(train_indices)} train, {len(val_indices)} val ---")

            best_acc = 0.0
            for epoch in range(epochs):
                if self._stop_flag or (stop_check and stop_check()):
                    self._stop_flag = True
                    break
                train_loss = self._train_epoch(batch_callback)
                val_loss, val_acc = self._validate()
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if val_acc > best_acc:
                    best_acc = val_acc

                msg = f"Fold {fold+1}/{k} Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
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
            self._save_model("best_model.pth")
            self._save_model("last_model.pth")
            return {"success": True, "mean_acc": float(avg_acc), "std_acc": float(std_acc), "fold_accuracies": fold_records}

    # ── Model persistence ───────────────────────────────────────────
    def _save_model(self, filename):
        models_dir = self.project_dir / "models" / "classification"
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "model_name": self.model_name,
            "image_size": self._image_size,
            "history": self.history,
        }, path)

    def load_model(self, filename="best_model.pth"):
        """Load a trained model for inference."""
        path = self.project_dir / "models" / "classification" / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.num_classes = ckpt["num_classes"]
        self.class_names = ckpt.get("class_names", [f"class_{i}" for i in range(self.num_classes)])
        self.model_name = ckpt.get("model_name", self.model_name)
        self._image_size = ckpt.get("image_size", 224)

        self.model = _create_classification_model(
            self.model_name, self.num_classes, pretrained=False
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        return ckpt.get("history", {})

    def _export_for_inference(self):
        """Export trained model to TorchScript and ONNX."""
        self.model.eval()
        models_dir = self.project_dir / "models" / "classification"
        models_dir.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(1, 3, self._image_size, self._image_size).to(self.device)

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

    # ── Inference ───────────────────────────────────────────────────
    def predict_single(self, image_path, model_filename, top_k=5):
        """Convenience method: load model and classify a single image.

        Returns dict: {"predictions": [{"class": ..., "confidence": ...}, ...]}
        or None on failure.
        """
        self.load_model(model_filename)
        results = self.predict(image_path, top_k=top_k)
        if not results:
            return None
        return {
            "predictions": [
                {"class": cls_name, "confidence": conf}
                for cls_name, conf in results
            ]
        }

    @torch.inference_mode()
    def predict(self, images, top_k=5):
        """Classify image(s). Returns list of [(class_name, confidence), ...] or list-of-lists."""
        from PIL import Image
        import torchvision.transforms as T

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        is_single = not isinstance(images, list)
        if is_single:
            images = [images]

        transform = T.Compose([
            T.Resize((self._image_size, self._image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        results = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
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