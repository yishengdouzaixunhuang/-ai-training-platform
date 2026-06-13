# -*- coding: utf-8 -*-
"""Detection Trainer — YOLO training wrapper.

Wraps ultralytics YOLO for object detection training with progress callbacks,
checkpoint resume, and K-fold cross-validation support.
"""
import os, json, time, threading, urllib.request, shutil
from pathlib import Path
import numpy as np

# Environment setup for ultralytics on restricted systems
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.environ.get("TEMP", "/tmp"), "ultralytics_cfg"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

_YOLO_IMPORTED = False
_YOLO_MODEL = None

# Mirror URLs for downloading YOLO weights (GitHub is often blocked in some regions)
_WEIGHTS_MIRRORS = [
    "https://modelscope.cn/models/ultralytics/YOLOv8/resolve/main/{model_name}.pt",
    "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/{model_name}.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}.pt",
]
_WEIGHTS_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

def _download_weights(model_name):
    """Try to download YOLO weights from mirrors, return local path or None."""
    _WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _WEIGHTS_CACHE_DIR / f"{model_name}.pt"
    if local_path.exists() and local_path.stat().st_size > 1024 * 1024:
        return str(local_path)
    
    for mirror in _WEIGHTS_MIRRORS:
        url = mirror.format(model_name=model_name)
        try:
            print(f"[Weights] Trying {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(local_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
            if local_path.stat().st_size > 1024 * 1024:
                print(f"[Weights] Downloaded {model_name}.pt ({local_path.stat().st_size / 1e6:.1f} MB)")
                return str(local_path)
        except Exception as e:
            print(f"[Weights] Mirror failed: {url} — {e}")
            continue
    
    # Clean up partial download
    if local_path.exists():
        local_path.unlink()
    return None


def _ensure_yolo():
    """Lazy import YOLO with env setup."""
    global _YOLO_IMPORTED, _YOLO_MODEL
    if not _YOLO_IMPORTED:
        from ultralytics import YOLO as _Y
        _YOLO_MODEL = _Y
        _YOLO_IMPORTED = True
    return _YOLO_MODEL


# Available YOLO models for detection
DET_MODELS = {
    "yolov8n":  {"name": "YOLOv8 Nano",    "size": "n", "params": "3.2M"},
    "yolov8s":  {"name": "YOLOv8 Small",   "size": "s", "params": "11.2M"},
    "yolov8m":  {"name": "YOLOv8 Medium",  "size": "m", "params": "25.9M"},
    "yolov8l":  {"name": "YOLOv8 Large",   "size": "l", "params": "43.7M"},
    "yolov8x":  {"name": "YOLOv8 XLarge",  "size": "x", "params": "68.2M"},
    "yolov8n-obb": {"name": "YOLOv8 Nano OBB",  "size": "n-obb", "params": "3.2M"},
    "yolov8s-obb": {"name": "YOLOv8 Small OBB", "size": "s-obb", "params": "11.2M"},
    "yolov8m-obb": {"name": "YOLOv8 Medium OBB","size": "m-obb", "params": "25.9M"},
}


class DetectionTrainer:
    """Wraps YOLO training with progress hooks for Qt UI integration."""
    
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self._stop_flag = False
        self._model = None
        self._results = None
        self._current_epoch = 0
        self._total_epochs = 0
        self._best_map = 0.0
        self._callback = None  # (epoch, total, train_loss, val_map) 
    
    def stop(self):
        self._stop_flag = True
    
    def train(self, data_yaml, model_name="yolov8s", epochs=200, batch_size=16,
              image_size=640, resume=False, k_folds=1,
              progress_callback=None, epoch_callback=None):
        """Train YOLO detection model.
        
        Args:
            data_yaml: Path to data.yaml
            model_name: YOLO variant key from DET_MODELS
            epochs: Number of epochs
            batch_size: Batch size
            image_size: Training image size (pixels)
            resume: Resume from last checkpoint
            k_folds: Number of cross-validation folds (1 = no CV)
            progress_callback: Called with (epoch, total, metrics_dict)
            epoch_callback: Called after each epoch with (epoch, total, metrics)
        
        Returns:
            dict with best_map, best_map50, final_weights
        """
        YOLO = _ensure_yolo()
        self._stop_flag = False
        self._callback = epoch_callback
        
        model_info = DET_MODELS.get(model_name, DET_MODELS["yolov8s"])
        model_variant = model_name  # Use full key (e.g. "yolov8n") not shortened "size"
        
        # Determine OBB mode
        is_obb = "obb" in model_name
        
        # Build model path or resume
        if resume:
            weights_dir = self.project_dir / "models" / "detection"
            weights_dir.mkdir(parents=True, exist_ok=True)
            last_pt = weights_dir / "last.pt"
            if last_pt.exists():
                self._model = YOLO(str(last_pt))
            else:
                self._model = self._load_model_with_fallback(YOLO, model_variant)
        else:
            self._model = self._load_model_with_fallback(YOLO, model_variant)
        
        # K-fold training
        if k_folds > 1:
            return self._train_kfold(data_yaml, model_name, epochs, batch_size,
                                     image_size, k_folds, progress_callback)
        
        # Single training run
        results_dir = self.project_dir / "models" / "detection"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        self._total_epochs = epochs
        self._current_epoch = 0
        self._best_map = 0.0
        
        # Monkey-patch progress callback
        class ProgressTracker:
            def __init__(self, trainer_ref):
                self.trainer = trainer_ref
            def __call__(self, metrics):
                if not self.trainer._stop_flag:
                    self.trainer._current_epoch += 1
                    if self.trainer._callback:
                        self.trainer._callback(
                            self.trainer._current_epoch,
                            self.trainer._total_epochs,
                            metrics
                        )
                return not self.trainer._stop_flag
        
        try:
            self._results = self._model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                project=str(results_dir),
                name="train",
                exist_ok=True,
                resume=resume,
                verbose=False,
                plots=False,
            )
            
            # Find best weights
            best_pt = results_dir / "train" / "weights" / "best.pt"
            last_pt = results_dir / "train" / "weights" / "last.pt"
            
            # Copy best/last to detection models dir
            if best_pt.exists():
                import shutil
                shutil.copy2(best_pt, results_dir / "best.pt")
            if last_pt.exists():
                import shutil
                shutil.copy2(last_pt, results_dir / "last.pt")
            
            # Extract best mAP
            best_map50 = 0.0
            best_map = 0.0
            if self._results and hasattr(self._results, 'results_dict'):
                rd = self._results.results_dict
                best_map50 = rd.get("metrics/mAP50(B)", 0.0)
                best_map = rd.get("metrics/mAP50-95(B)", 0.0)
            
            return {
                "best_map": float(best_map),
                "best_map50": float(best_map50),
                "weights_dir": str(results_dir),
                "best_weights": str(results_dir / "best.pt"),
            }
            
        except Exception as e:
            if self._stop_flag:
                return {"error": "stopped"}
            raise
    
    def _train_kfold(self, data_yaml, model_name, epochs, batch_size, image_size,
                     k_folds, progress_callback):
        """K-fold cross-validation training."""
        from detection.dataset import build_yolo_dataset
        
        # Read original data.yaml
        import yaml
        with open(data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        
        # We already have a split from build_yolo_dataset.
        # Re-build with k-fold by re-shuffling.
        all_results = []
        for fold in range(k_folds):
            if self._stop_flag:
                break
            if progress_callback:
                progress_callback(fold + 1, k_folds, {"fold": fold + 1})
            # Re-build dataset with different seed
            ds_info = build_yolo_dataset(
                self.project_dir,
                output_dir=self.project_dir / f"yolo_dataset_fold{fold}",
                val_split=1.0 / k_folds,
                seed=42 + fold
            )
            result = self.train(
                ds_info["data_yaml"], model_name, epochs, batch_size,
                image_size, resume=False, k_folds=1
            )
            result["fold"] = fold + 1
            all_results.append(result)
        
        # Average results
        maps = [r.get("best_map", 0) for r in all_results if "best_map" in r]
        map50s = [r.get("best_map50", 0) for r in all_results if "best_map50" in r]
        return {
            "kfold_results": all_results,
            "mean_map": float(np.mean(maps)) if maps else 0.0,
            "mean_map50": float(np.mean(map50s)) if map50s else 0.0,
            "best_map": float(max(maps)) if maps else 0.0,
            "best_map50": float(max(map50s)) if map50s else 0.0,
        }
    
    def _load_model_with_fallback(self, YOLO, model_name):
        """Load YOLO model, trying mirrors/download then falling back to from-scratch.
        
        Priority:
        1) Cached weights in ~/.cache/torch/hub/checkpoints/
        2) Download from mirror sources (ModelScope → HuggingFace → GitHub)
        3) If all downloads fail, create model from scratch with nc from dataset
        """
        local_path = _download_weights(model_name)
        if local_path:
            print(f"[Weights] Using cached: {local_path}")
            return YOLO(local_path)
        
        # All mirrors failed — train from scratch
        print(f"[Weights] All download mirrors failed. Training {model_name} from scratch.")
        # Create the model architecture from config (no pre-trained weights)
        return YOLO(f"{model_name}.yaml")
    
    def predict(self, image_path, weights=None, conf=0.25, iou=0.45):
        """Run inference on a single image.
        
        Returns list of dicts: [{bbox, category, confidence}, ...]
        """
        YOLO = _ensure_yolo()
        if self._model is None:
            if weights:
                self._model = YOLO(weights)
            else:
                best_pt = self.project_dir / "models" / "detection" / "best.pt"
                if best_pt.exists():
                    self._model = YOLO(str(best_pt))
                else:
                    raise FileNotFoundError("No detection model found")
        
        results = self._model.predict(
            image_path, conf=conf, iou=iou, verbose=False
        )
        
        boxes = []
        if results and len(results) > 0:
            r = results[0]
            if hasattr(r, 'obb') and r.obb is not None:
                # OBB results
                obb_data = r.obb.data.cpu().numpy() if hasattr(r.obb.data, 'cpu') else r.obb.data
                for d in obb_data:
                    boxes.append({
                        "bbox": d[:5].tolist(),  # [cx, cy, w, h, angle]
                        "confidence": float(d[5]),
                        "category_id": int(d[6]),
                        "category": self._model.names.get(int(d[6]), f"Class{int(d[6])}"),
                        "bbox_type": "obb",
                    })
            elif r.boxes is not None:
                box_data = r.boxes.data.cpu().numpy() if hasattr(r.boxes.data, 'cpu') else r.boxes.data
                for d in box_data:
                    x1, y1, x2, y2 = d[:4]
                    boxes.append({
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "confidence": float(d[4]),
                        "category_id": int(d[5]),
                        "category": self._model.names.get(int(d[5]), f"Class{int(d[5])}"),
                        "bbox_type": "hbb",
                    })
        
        return boxes
    
    def validate(self, data_yaml, weights=None):
        """Run validation/mAP evaluation.
        
        Returns dict with map50, map, precision, recall.
        """
        YOLO = _ensure_yolo()
        if weights:
            model = YOLO(weights)
        elif self._model is not None:
            model = self._model
        else:
            best_pt = self.project_dir / "models" / "detection" / "best.pt"
            if best_pt.exists():
                model = YOLO(str(best_pt))
            else:
                raise FileNotFoundError("No detection model found")
        
        results = model.val(data=data_yaml, verbose=False)
        
        metrics = {
            "map50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            "map": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            "precision": float(results.box.p[0]) if hasattr(results.box, 'p') and len(results.box.p) > 0 else 0.0,
            "recall": float(results.box.r[0]) if hasattr(results.box, 'r') and len(results.box.r) > 0 else 0.0,
        }
        return metrics
