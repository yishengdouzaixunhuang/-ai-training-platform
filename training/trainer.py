"""训练器"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from training.dataset import SegmentationDataset
from training.models import create_model
from training.losses_ext import build_loss, LOSS_NAMES
from training.augment_ext import AUGMENT_PRESETS

class Trainer:
    def __init__(self, project_dir, model_name="deeplabv3", device=None):
        self.project_dir = project_dir
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 读取工程信息
        with open(os.path.join(project_dir, "project.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        raw = self.meta.get("classes", ["background"])
        self.classes = [c for c in raw if c == "background" or ("ignore" not in c.lower() and "gnore" not in c.lower())]
        if "background" not in self.classes:
            self.classes.insert(0, "background")
        self.num_classes = len(self.classes)
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        self._stop_requested = False
        self.history = {"train_loss": [], "val_loss": [], "val_miou": []}
    
    def _train_kfold(self, k, epochs, batch_size, image_size,
                     progress_callback, log_callback, stop_check,
                     plot_callback, batch_callback, loss_name="cross_entropy", augment="none"):
        """K-fold cross-validation training."""
        # Load all annotated images
        full_ds = SegmentationDataset(self.project_dir, split="all", image_size=image_size, augment_fn=AUGMENT_PRESETS.get(augment))
        num_images = len(full_ds._image_samples)
        if num_images < k:
            if log_callback:
                log_callback(f"Only {num_images} images, need at least {k} for {k}-fold. Using {num_images}-fold.")
            k = num_images
        if k < 2:
            if log_callback:
                log_callback("Need at least 2 images for K-fold. Falling back to normal training.")
            self.prepare_data(batch_size, image_size=image_size, loss_name=loss_name, augment=augment)
            self.init_model()
            return self._run_epochs(epochs, batch_callback, stop_check, log_callback,
                                    progress_callback, plot_callback)

        # Shuffle image indices
        rng = np.random.RandomState(42)
        indices = list(range(num_images))
        rng.shuffle(indices)
        fold_size = num_images // k

        if log_callback:
            log_callback(f"=== {k}-Fold Cross-Validation ({num_images} images, ~{fold_size}/fold) ===")

        fold_mious = []
        for fold in range(k):
            if stop_check and stop_check():
                if log_callback:
                    log_callback("K-fold stopped by user")
                break

            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else num_images
            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]

            # Create fold datasets
            train_ds = SegmentationDataset.from_image_subset(full_ds, train_indices)
            val_ds = SegmentationDataset.from_image_subset(full_ds, val_indices)

            drop = len(train_ds) > batch_size
            self._image_size = image_size
            self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=drop)
            self.val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False, num_workers=0)

            # Compute class weights from cached masks (fast, no DataLoader pass)
            class_counts = np.zeros(self.num_classes, dtype=np.float64)
            for idx in range(len(train_ds._image_samples)):
                mask = train_ds._get_mask(idx)
                for c in range(self.num_classes):
                    class_counts[c] += (mask == c).sum()
            class_counts = np.maximum(class_counts, 1)
            weights = class_counts.sum() / (self.num_classes * class_counts)
            weights[0] = 1.0
            self.criterion = build_loss(loss_name, self.num_classes, self.device, weights)

            # Re-init model for each fold
            self.init_model()
            self._stop_requested = False
            self._stop_check_fn = stop_check

            if log_callback:
                log_callback(f"--- Fold {fold+1}/{k}: {len(train_indices)} train, {len(val_indices)} val ---")

            best_miou = 0
            for epoch in range(epochs):
                if self._stop_requested or (stop_check and stop_check()):
                    self._stop_requested = True
                    break
                train_loss = self.train_epoch(batch_callback=batch_callback)
                val_loss, miou, per_class = self.validate()
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_miou"].append(miou)
                if miou > best_miou:
                    best_miou = miou
                sorted_cls = sorted(per_class.items(), key=lambda x: x[1])
                worst = ", ".join("%s:%.3f" % (self.classes[c] if c < len(self.classes) else str(c), v) for c, v in sorted_cls[:3])
                msg = f"Fold {fold+1}/{k} Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f} | Worst: {worst}"
                if log_callback:
                    log_callback(msg)
                if progress_callback:
                    progress_callback(epoch + 1, epochs)
                if plot_callback:
                    plot_callback(self.history)

            fold_mious.append(best_miou)
            if log_callback:
                log_callback(f"Fold {fold+1}/{k} best mIoU: {best_miou:.4f}")

        if fold_mious:
            avg_miou = np.mean(fold_mious)
            std_miou = np.std(fold_mious)
            if log_callback:
                log_callback(f"=== K-Fold Complete: avg mIoU={avg_miou:.4f} ?{std_miou:.4f} ===")
            # Save last fold's model as best
            self.save_model("best_model.pth")
            self.save_model("last_model.pth")

    def prepare_data(self, batch_size=4, val_split=0.2, image_size=(512, 512), loss_name="cross_entropy", augment="none"):
        # Use user-assigned train/val splits from train_test_split.json
        train_ds = SegmentationDataset(self.project_dir, split="train", image_size=image_size, augment_fn=AUGMENT_PRESETS.get(augment))
        val_ds = SegmentationDataset(self.project_dir, split="val", image_size=image_size)

        # num_classes already set in __init__ via project.json
        train_size = len(train_ds)
        val_size = len(val_ds)

        # Fallback: if no val split assigned, split at IMAGE level (not tile level)
        if val_size == 0 and train_size >= 2:
            num_images = len(train_ds._image_samples)
            if num_images >= 2:
                val_img_count = max(1, int(num_images * val_split))
                train_img_count = num_images - val_img_count
                rng = np.random.RandomState(42)
                img_indices = list(range(num_images))
                rng.shuffle(img_indices)
                orig_ds = train_ds  # keep reference before overwriting
                train_ds = SegmentationDataset.from_image_subset(orig_ds, img_indices[:train_img_count])
                val_ds = SegmentationDataset.from_image_subset(orig_ds, img_indices[train_img_count:])
                train_size = len(train_ds)
                val_size = len(val_ds)
                self.num_classes = train_ds.num_classes

        if train_size == 0:
            raise ValueError("Train set is empty! Please assign train images first.")

        if val_size == 0:
            raise ValueError("Val set is empty! Please assign val/test images or use Auto Split.")

        import multiprocessing
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        drop = train_size > batch_size
        self._image_size = image_size
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=drop)
        self.val_loader = DataLoader(val_ds, batch_size=min(batch_size, val_size), shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=False)

        # Compute class weights from cached masks (no DataLoader pass)
        class_counts = np.zeros(self.num_classes, dtype=np.float64)
        for idx in range(len(train_ds._image_samples)):
            mask = train_ds._get_mask(idx)
            for c in range(self.num_classes):
                class_counts[c] += (mask == c).sum()
        class_counts = np.maximum(class_counts, 1)
        weights = class_counts.sum() / (self.num_classes * class_counts)
        weights[0] = 1.0  # background stays at 1.0
        if loss_name in ("cross_entropy", "dice", "focal"):
            self.criterion = build_loss(loss_name, self.num_classes, self.device, weights)
        else:
            self.criterion = build_loss(loss_name, self.num_classes, self.device, weights)
        return train_size, val_size
    


    def init_model(self):
        self.model = create_model(self.num_classes, self.model_name).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_epoch(self, batch_callback=None):
        self.model.train()
        total_loss = 0
        batches = 0
        pbar = tqdm(self.train_loader, desc="??", ncols=80)
        for images, masks in pbar:
            if self._stop_requested or (self._stop_check_fn and self._stop_check_fn()):
                self._stop_requested = True
                return total_loss / max(batches, 1)
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)["out"]
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)["out"]
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
            batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if batch_callback and batches % 10 == 0:
                batch_callback(batches, len(self.train_loader), total_loss / batches)
        return total_loss / max(batches, 1)

    def validate(self):
        self.model.eval()
        total_loss = 0
        batches = 0
        intersections = torch.zeros(self.num_classes).to(self.device)
        unions = torch.zeros(self.num_classes).to(self.device)

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="??", ncols=80):
                if self._stop_requested:
                    break
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)["out"]
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                batches += 1

                preds = outputs.argmax(1)
                for c in range(self.num_classes):
                    pred_c = (preds == c)
                    mask_c = (masks == c)
                    intersections[c] += (pred_c & mask_c).sum().float()
                    unions[c] += (pred_c | mask_c).sum().float()

        avg_loss = total_loss / max(batches, 1)
        ious = intersections / (unions + 1e-8)
        miou = ious.mean().item()
        per_class = {c: ious[c].item() for c in range(self.num_classes)}
        return avg_loss, miou, per_class

    def train(self, epochs=50, batch_size=4, image_size=(512, 512), loss_name="cross_entropy", augment="none",
              progress_callback=None, log_callback=None, resume=False,
              stop_check=None, plot_callback=None, batch_callback=None,
              k_folds=1):
        # K-fold cross-validation mode
        if k_folds > 1 and not resume:
            self._train_kfold(k_folds, epochs, batch_size, image_size,
                              progress_callback, log_callback, stop_check,
                              plot_callback, batch_callback)
            return

        if resume:
            ckpt_path = os.path.join(self.project_dir, "models", "last_model.pth")
            if not os.path.exists(ckpt_path):
                if log_callback:
                    log_callback("No checkpoint found, training from scratch")
                resume = False
            else:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.num_classes = ckpt["num_classes"]
                self.classes = ckpt["classes"]
                self.model_name = ckpt.get("model_name", self.model_name)
                self.history = ckpt.get("history", {})
                start_epoch = len(self.history.get("train_loss", []))

                self.model = create_model(self.num_classes, self.model_name).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.train()

                self.prepare_data(batch_size, image_size=image_size, loss_name=loss_name, augment=augment)

                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
                if "optimizer_state_dict" in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception:
                        if log_callback:
                            log_callback("Optimizer state mismatch, using fresh optimizer")

                if log_callback:
                    log_callback("Resuming from epoch %d (mIoU: %.4f)" % (
                        start_epoch + 1,
                        self.history.get("val_miou", [0])[-1] if self.history.get("val_miou") else 0))
        else:
            start_epoch = 0

        if not resume:
            self.prepare_data(batch_size, image_size=image_size, loss_name=loss_name, augment=augment)
            self.init_model()
            start_epoch = 0

        self._stop_requested = False
        self._stop_check_fn = stop_check
        epochs = start_epoch + epochs
        self._run_epochs(epochs, batch_callback, stop_check, log_callback,
                         progress_callback, plot_callback, start_epoch)

    def _run_epochs(self, epochs, batch_callback, stop_check, log_callback,
                    progress_callback, plot_callback, start_epoch=0):
        best_miou = self.history.get("val_miou", [0])[-1] if self.history.get("val_miou") else 0
        for epoch in range(start_epoch, epochs):
            train_loss = self.train_epoch(batch_callback=batch_callback)
            val_loss, miou, per_class = self.validate()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_miou"].append(miou)
            if stop_check and stop_check():
                self._stop_requested = True
                if log_callback:
                    log_callback("Training stopped by user")
                break
            sorted_cls = sorted(per_class.items(), key=lambda x: x[1])
            worst = ", ".join("%s:%.3f" % (self.classes[c] if c < len(self.classes) else str(c), v) for c, v in sorted_cls[:3])
            msg = "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | mIoU: %.4f | Worst: %s" % (
                epoch + 1, epochs, train_loss, val_loss, miou, worst)
            if log_callback:
                log_callback(msg)
            if miou > best_miou:
                best_miou = miou
                self.save_model("best_model.pth")
                if log_callback:
                    pclog = ", ".join("%s:%.3f" % (self.classes[c] if c < len(self.classes) else str(c), per_class[c]) for c in range(self.num_classes))
                    log_callback("  -> Best saved (mIoU: %.4f) [%s]" % (miou, pclog))
            self.save_model("last_model.pth")
            if progress_callback:
                progress_callback(epoch + 1 - start_epoch, epochs - start_epoch)
            if plot_callback:
                plot_callback(self.history)
        if log_callback:
            log_callback("Training complete!")
        # Export optimized models
        try:
            self.export_for_inference()
            if log_callback:
                log_callback("Exported TorchScript/ONNX models")
        except Exception:
            pass

    def save_model(self, filename):
        os.makedirs(os.path.join(self.project_dir, "models"), exist_ok=True)
        path = os.path.join(self.project_dir, "models", filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": self.num_classes,
            "classes": self.classes,
            "model_name": self.model_name,
            "history": self.history,
        }, path)

    def export_for_inference(self):
        """Export model to TorchScript and ONNX for faster inference."""
        self.model.eval()
        models_dir = os.path.join(self.project_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        dummy = torch.randn(1, 3, self.image_size[0], self.image_size[0]).to(self.device)

        # TorchScript
        try:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                traced = torch.jit.trace(self.model, dummy)
            ts_path = os.path.join(models_dir, "model_scripted.pt")
            traced.save(ts_path)
        except Exception:
            pass  # TorchScript export optional

        # ONNX
        try:
            onnx_path = os.path.join(models_dir, "model.onnx")
            torch.onnx.export(
                self.model, dummy, onnx_path,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=14)
        except Exception:
            pass  # ONNX export optional

    @property
    def image_size(self):
        return getattr(self, '_image_size', (512, 512))

    def load_model(self, filename="best_model.pth"):
        path = os.path.join(self.project_dir, "models", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.num_classes = ckpt["num_classes"]
        self.classes = ckpt["classes"]
        self.model_name = ckpt.get("model_name", "deeplabv3")
        self.model = create_model(self.num_classes, self.model_name).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        return ckpt.get("history", {})
