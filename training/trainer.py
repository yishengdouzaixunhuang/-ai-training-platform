"""训练器"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ⚠️ 为什么 num_workers=0（单进程加载）：
#   Windows 的 DataLoader 用 spawn 起子进程，需要把 augment_fn **pickle** 过去；
#   而 build_augment_fn() 返回的是个**闭包**（本地函数），pickle 不了 →
#   `AttributeError: Can't get local object 'build_augment_fn.<locals>.augment'`
#   直接训练就崩。改成 0 就完全绕开。
#   （注意 workers=0 时不能再传 prefetch_factor，否则 torch 报
#    "prefetch_factor option could only be specified in multiprocessing"）
#   本项目 200×200 小图 + 几十张，单进程加载一点不慢。
#   —— 以后要恢复多进程，得先把 augment 改成可 pickle 的**类**（实现 __call__），
#      再把这里改回 4 / prefetch_factor=2。
from tqdm import tqdm
import numpy as np
from training.dataset import SegmentationDataset
from training.models import create_model
from training.losses_ext import build_loss, LOSS_NAMES
from training.augment_ext import AUGMENT_PRESETS
from training.augment_config import build_augment_fn

def _resolve_augment_fn(augment):
    """augment 兼容旧预设字符串与新增强开关 dict（增强参数面板）。"""
    if isinstance(augment, dict):
        return build_augment_fn(augment)
    return AUGMENT_PRESETS.get(augment)


class Trainer:
    def __init__(self, project_dir, model_name="deeplabv3", device=None,
                 tile_size=0, tile_stride_pct=75, tile_mode="slide"):
        self.project_dir = project_dir
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ★★ 2026-10-29：单视图也接上「分块」这组参数。
        #   原来界面上的「分块（滑窗）/ 滑窗大小 / 步长%」只传给了 MultiViewTrainer，
        #   单视图走 `Trainer(project_dir, model_name=...)`，三个参数**直接被丢掉** ——
        #   用户填了 768/75% 也不会生效，实际永远按 Image Size 无重叠切块。
        #   tile_size>0 ⇒ 块尺寸 = 网络输入尺寸（原分辨率裁，1:1），步长按百分比算重叠。
        self.tile_size = int(tile_size or 0)
        self.tile_stride_pct = float(tile_stride_pct or 75)
        self.tile_mode = str(tile_mode or "slide")
        # 写进 checkpoint，推理端据此自动选滑窗/整图（见 predictor._resolve_train_mode）。
        # ⚠️ 恒为 "tiled"：`SegmentationDataset._build_tiles` 目前**总是**切块
        #   （图比 image_size 大就滑窗），连"不勾分块"也是切块、只是无重叠。
        #   等将来真做了"整图 + 采样密度"训练，这里再按实际模式写 "full"。
        self.train_mode = "tiled"
        
        # 读取工程信息
        with open(os.path.join(project_dir, "project.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        raw = self.meta.get("classes", ["background"])
        self.classes = [c for c in raw if c == "background" or ("ignore" not in c.lower() and "gnore" not in c.lower())]
        if "background" not in self.classes:
            self.classes.insert(0, "background")
        self.num_classes = len(self.classes)

        # ★ 多视图（三图）工程：project.json 里 views=[...] ⇒ 输入是 3N 通道而不是 3。
        #   不识别的话：模型是 3 通道、数据是 3 通道 ⇒ **能训但只用了第一个视图**，
        #   看起来"训练成功"其实三图信息全丢（静默降级，比报错更坏）。
        _views = self.meta.get("views") or []
        self.views = tuple(v for v in _views if isinstance(v, str) and v)
        self.in_channels = 3 * (len(self.views) if self.views else 1)
        if self.views:
            _declared = self.meta.get("in_channels")
            if _declared and int(_declared) != self.in_channels:
                # 以 views 为准（views 是真数据来源），但把不一致记下来，别闷着
                print("[Trainer] 警告：project.json 的 in_channels=%s 与 views=%s ×3=%d 不一致，"
                      "按 views 计算" % (_declared, list(self.views), self.in_channels), flush=True)

        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        self._stop_requested = False
        self.history = {"train_loss": [], "val_loss": [], "val_miou": []}

    def cleanup(self):
        """Release model, data and GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _train_kfold(self, k, epochs, batch_size, image_size,
                     progress_callback, log_callback, stop_check,
                     plot_callback, batch_callback, loss_name="cross_entropy", augment="none"):
        """K-fold cross-validation training."""
        # Load all annotated images
        full_ds = SegmentationDataset(self.project_dir, split="all", image_size=image_size, augment_fn=_resolve_augment_fn(augment), views=self.views or None)
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
            self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=drop)
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
                # ⚠️ nan 表示"该组验证图里根本没有这个类"，不是"这类得了 0 分"。
                #   直接排序会把 nan 排到最前面 ⇒ 日志看起来像"Zw 是最差的类"（实拍踩到）。
                #   这里把 nan 剔出排名，另附一句"未出现"，避免误读。
                _valid = [(c, v) for c, v in per_class.items() if v == v]
                _absent = [self.classes[c] if c < len(self.classes) else str(c)
                           for c, v in per_class.items() if v != v]
                sorted_cls = sorted(_valid, key=lambda x: x[1])
                worst = ", ".join("%s:%.3f" % (self.classes[c] if c < len(self.classes) else str(c), v)
                                  for c, v in sorted_cls[:3])
                if _absent:
                    worst += "  (本验证图未出现: %s)" % "/".join(_absent)
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
        train_ds = SegmentationDataset(self.project_dir, split="train", image_size=image_size,
                                       augment_fn=_resolve_augment_fn(augment), views=self.views or None,
                                       tile_size=self.tile_size, tile_stride_pct=self.tile_stride_pct)
        val_ds = SegmentationDataset(self.project_dir, split="val", image_size=image_size,
                                     views=self.views or None,
                                     tile_size=self.tile_size, tile_stride_pct=self.tile_stride_pct)

        # num_classes already set in __init__ via project.json
        train_size = len(train_ds)
        val_size = len(val_ds)

        # Fallback: if no val split assigned, split at IMAGE level (not tile level)
        _auto_split = False
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
                _auto_split = True
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
        # ★ 2026-10-23：把划分实情记下来（训练开始时打日志）。
        #   用户界面上「验证集」列是 0，而这里偷偷切了 1 张 —— 不说清楚，
        #   日志里的"(未出现: Zw/Ps)"会被读成"训练集没有这两类"（用户实拍就是这个误会）。
        try:
            _vn = [os.path.splitext(os.path.basename(x[0]))[0]
                   for x in getattr(val_ds, "_image_samples", [])]
            _tn = [os.path.splitext(os.path.basename(x[0]))[0]
                   for x in getattr(train_ds, "_image_samples", [])]
            self._split_info = {"auto_split": _auto_split,
                                "train_images": len(_tn), "val_images": len(_vn),
                                "train_names": _tn[:10], "val_names": _vn[:10]}
        except Exception:
            self._split_info = None
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=drop)
        self.val_loader = DataLoader(val_ds, batch_size=min(batch_size, val_size), shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

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
        self.model = create_model(self.num_classes, self.model_name,
                                 in_channels=self.in_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_epoch(self, batch_callback=None):
        self.model.train()
        total_loss = 0
        batches = 0
        pbar = tqdm(self.train_loader, desc="训练", ncols=80)
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
        # 各类在**真值**里出现过的像素数：mIoU 只在这些类上平均
        gt_pixels = torch.zeros(self.num_classes).to(self.device)

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="验证", ncols=80):
                if self._stop_requested:
                    break
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)["out"]
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                batches += 1

                preds = outputs.argmax(1)
                # ★★ 2026-10-22 修：**ignore 像素必须整体剔除**。
                #   原来 `unions[c] += (pred_c | mask_c)` 里 pred_c 包含了"落在 ignore 区的预测"，
                #   而 mask_c 在那些位置恒为 0 ⇒ 分子不进、分母照加，
                #   IoU 被系统性压低（本工程 ignore 占 3%~14% 面积，能把 mIoU 拉下去一大截）。
                #   用户标 IGNORE 的语义就是"这块我不标，别拿它算账"——
                #   训练损失里已经剔了（CE ignore_index / Lovasz 剔除），**指标也必须剔**。
                valid = masks != 255
                for c in range(self.num_classes):
                    pred_c = (preds == c) & valid
                    mask_c = (masks == c) & valid
                    intersections[c] += (pred_c & mask_c).sum().float()
                    unions[c] += (pred_c | mask_c).sum().float()
                    gt_pixels[c] += (masks == c).sum().float()

        avg_loss = total_loss / max(batches, 1)
        ious = intersections / (unions + 1e-8)
        # ★★ 2026-10-22 第二个指标 bug：**图上根本没出现的类别不能按 0 计入平均**。
        #   这张验证图（上大面_OK_011）真值只有 background + Ps，Zw/Hh 一个像素都没有，
        #   旧代码仍把它们的 IoU(0) 算进 mean ⇒ mIoU 天花板被人为压到 0.5，
        #   实测 (0.974 + 0 + 0 + 0.554)/4 = 0.382，而真实表现是 0.764。
        #   标准做法（mmseg 等）：只对**真值里出现过**的类别求平均（union > 0）。
        #   ⚠️ 注意这只影响"报告出来的指标"，不影响训练与选模逻辑本身。
        # ⚠️ 判据必须是"**真值里**出现过"，不能是 `unions > 0`：
        #   模型在 GT 没有的类上乱预测时 union 也会 > 0，那类就会被算进来（实测踩过）。
        present = gt_pixels > 0
        miou = ious[present].mean().item() if bool(present.any()) else 0.0
        per_class = {c: (ious[c].item() if bool(present[c]) else float("nan"))
                     for c in range(self.num_classes)}
        return avg_loss, miou, per_class

    def train(self, epochs=50, batch_size=4, image_size=(512, 512), loss_name="cross_entropy", augment="none",
              progress_callback=None, log_callback=None, resume=False,
              stop_check=None, plot_callback=None, batch_callback=None,
              k_folds=1, early_stop_patience=0):
        """early_stop_patience: 连续多少轮 val_loss 与 mIoU **都没改善**就停（0 = 关闭）。

        为什么要有：8 张图那种小数据集，训到 2000+ 轮时 train loss 一路降、val loss 反而
        翻 7 倍（实测 0.20 → 1.57），后面全是在把模型训坏 —— 纯烧时间。"""
        # K-fold cross-validation mode
        if k_folds > 1 and not resume:
            self._train_kfold(k_folds, epochs, batch_size, image_size,
                              progress_callback, log_callback, stop_check,
                              plot_callback, batch_callback)
            return

        if resume:
            # ★★ 2026-10-23：resume **自动挑更好的起点**。
            #   原来固定读 last_model.pth —— 但 last 是"最后一轮"的权重，早停/过拟合之后
            #   它往往比 best 差（实测：best mIoU 0.8393@46，last 0.8324@74）。
            #   从 last 继承等于从退化的权重继续，白跑。
            #   现在比较两个文件**最后记录的 val_miou**，取高的那个；不复制/不修改任何文件。
            ckpt_path, _pick_info = self._pick_resume_ckpt()
            if not ckpt_path:
                if log_callback:
                    log_callback("No checkpoint found, training from scratch")
                resume = False
            else:
                if log_callback:
                    log_callback(_pick_info)
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                ckpt_num_classes = ckpt["num_classes"]
                ckpt_classes = ckpt["classes"]
                self.model_name = ckpt.get("model_name", self.model_name)
                self.history = ckpt.get("history", {})
                start_epoch = len(self.history.get("train_loss", []))

                # Check if classes changed since last training
                if ckpt_num_classes != self.num_classes:
                    if log_callback:
                        log_callback(f"Class count changed ({ckpt_num_classes} -> {self.num_classes}), reinitializing model head")
                    self.model = create_model(self.num_classes, self.model_name).to(self.device)
                    # Try to load backbone weights from checkpoint (ignore classifier head mismatch)
                    try:
                        model_state = self.model.state_dict()
                        pretrained_state = {k: v for k, v in ckpt["model_state_dict"].items()
                                           if k in model_state and model_state[k].shape == v.shape}
                        model_state.update(pretrained_state)
                        self.model.load_state_dict(model_state, strict=False)
                        if log_callback:
                            loaded = len(pretrained_state)
                            total = len(model_state)
                            log_callback(f"Loaded {loaded}/{total} layers from checkpoint (head reinitialized)")
                    except Exception as e:
                        if log_callback:
                            log_callback(f"Could not transfer backbone weights: {e}")
                    self.classes = self.classes  # keep current project classes
                else:
                    self.num_classes = ckpt_num_classes
                    self.classes = ckpt_classes
                    self.model = create_model(self.num_classes, self.model_name).to(self.device)
                    self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.train()

                self.prepare_data(batch_size, image_size=image_size, loss_name=loss_name, augment=augment)

                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
                # When classes changed, skip optimizer state (head params don't match)
                if ckpt_num_classes == self.num_classes and "optimizer_state_dict" in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception:
                        if log_callback:
                            log_callback("Optimizer state mismatch, using fresh optimizer")
                else:
                    if log_callback:
                        log_callback("Class count changed, using fresh optimizer")

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
        self._early_stop_patience = int(early_stop_patience or 0)
        epochs = start_epoch + epochs
        self._run_epochs(epochs, batch_callback, stop_check, log_callback,
                         progress_callback, plot_callback, start_epoch)

    def _pick_resume_ckpt(self):
        """挑 resume 的起点：best_model.pth 与 last_model.pth 里"最后记录 val_miou"更高的那个。

        返回 (路径 or None, 给人看的说明)。
        不复制文件 —— 只决定从哪份权重继续，避免像以前那样手动 cp 覆盖。
        """
        mdir = os.path.join(self.project_dir, "models")
        cands = []
        for fn in ("best_model.pth", "last_model.pth"):
            p = os.path.join(mdir, fn)
            if not os.path.exists(p):
                continue
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                mi = (ck.get("history") or {}).get("val_miou") or []
                last_val = float(mi[-1]) if mi else -1.0
                cands.append((last_val, len(mi), fn, p))
            except Exception:
                continue
        if not cands:
            return None, ""
        # 按"最后记录的 mIoU"降序；并列时取轮数多的（多训过总归信息更多）
        cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best = cands[0]
        skip = cands[1] if len(cands) > 1 else None
        msg = "Resume 起点：%s（记录 %d 轮，最后一轮 val mIoU %.4f）" % (best[2], best[1], best[0])
        if skip:
            msg += "；已跳过 %s（%d 轮，%.4f）" % (skip[2], skip[1], skip[0])
        return best[3], msg

    def _run_epochs(self, epochs, batch_callback, stop_check, log_callback,
                    progress_callback, plot_callback, start_epoch=0):
        best_miou = self.history.get("val_miou", [0])[-1] if self.history.get("val_miou") else 0
        # ── 早停 + 最佳轮次记录 ──────────────────────────────────────────
        # 判据用"val_loss 或 mIoU 任一创新高"：只看 val_loss 会在 mIoU 还在涨的时候
        # 提前停掉（实测本工程 val_loss 最低在第 1983 轮，而 mIoU 峰在第 2269 轮）。
        patience = int(getattr(self, "_early_stop_patience", 0) or 0)
        best_vloss = float("inf")
        best_vloss_ep = 0
        best_miou_ep = int(len(self.history.get("val_miou", [])) and
                           __import__("numpy").argmax(self.history["val_miou"]) + 1 or 0)
        no_improve = 0
        _si = getattr(self, "_split_info", None)
        if _si and log_callback:
            _vn = "、".join(_si.get("val_names") or []) or "-"
            if _si.get("auto_split"):
                log_callback(
                    "⚠️ 数据集没有显式验证集：已从训练集按 seed=42 随机切 %d 张做验证 — %s"
                    "（实际参与训练 %d 张）。日志里「未出现: X」指的是**这张验证图**没有 X，"
                    "不是训练集没有 X。" % (_si.get("val_images", 0), _vn, _si.get("train_images", 0)))
            else:
                log_callback("验证集 %d 张（来自数据集划分）：%s｜训练 %d 张"
                             % (_si.get("val_images", 0), _vn, _si.get("train_images", 0)))
        self.best_summary = {"best_miou": best_miou, "best_miou_epoch": best_miou_ep,
                             "best_val_loss": None, "best_val_loss_epoch": None,
                             "early_stopped": False, "stopped_epoch": None,
                             "patience": patience}
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
            # 同上：nan = 该验证图没有这个类，不参与"最差类"排名
            _valid = [(c, v) for c, v in per_class.items() if v == v]
            _absent = [self.classes[c] if c < len(self.classes) else str(c)
                       for c, v in per_class.items() if v != v]
            sorted_cls = sorted(_valid, key=lambda x: x[1])
            worst = ", ".join("%s:%.3f" % (self.classes[c] if c < len(self.classes) else str(c), v)
                              for c, v in sorted_cls[:3])
            if _absent:
                worst += "  (本验证图未出现: %s)" % "/".join(_absent)
            msg = "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | mIoU: %.4f | Worst: %s" % (
                epoch + 1, epochs, train_loss, val_loss, miou, worst)
            if log_callback:
                log_callback(msg)
            # 早停计数：val_loss 创新低 或 mIoU 创新高，都算"还在改善"
            _improved = False
            if val_loss < best_vloss - 1e-4:
                best_vloss, best_vloss_ep, _improved = val_loss, epoch + 1, True
            _miou_better = miou > best_miou
            if _miou_better:
                best_miou, best_miou_ep, _improved = miou, epoch + 1, True
            no_improve = 0 if _improved else no_improve + 1
            # ⚠️ 必须**每轮**更新：best_model.pth 是在最佳那一轮就落盘的，
            #   如果只在早停时更新，存的会是初始值（实测踩到：best_summary 全是 0/None）。
            try:
                self.best_summary.update({
                    "best_miou": best_miou, "best_miou_epoch": best_miou_ep,
                    "best_val_loss": (best_vloss if best_vloss != float("inf") else None),
                    "best_val_loss_epoch": best_vloss_ep,
                    "no_improve": no_improve,
                })
            except Exception:
                pass
            if _miou_better:
                self.save_model("best_model.pth")
                if log_callback:
                    pclog = ", ".join("%s:%.3f" % (self.classes[c] if c < len(self.classes) else str(c), per_class[c]) for c in range(self.num_classes))
                    log_callback("  -> Best saved (mIoU: %.4f) [%s]  epoch %d" % (miou, pclog, epoch + 1))
            self.save_model("last_model.pth")
            if progress_callback:
                progress_callback(epoch + 1 - start_epoch, epochs - start_epoch)
            if plot_callback:
                plot_callback(self.history)
            # ── 早停 ──
            if patience and no_improve >= patience:
                self.best_summary.update({"best_miou": best_miou, "best_miou_epoch": best_miou_ep,
                                          "best_val_loss": best_vloss,
                                          "best_val_loss_epoch": best_vloss_ep,
                                          "early_stopped": True, "stopped_epoch": epoch + 1})
                # ⚠️ last_model 是在早停判定**之前**存的，所以这里要再存一次，
                #   否则 checkpoint 里 early_stopped 永远是 False（实测踩到）。
                try:
                    self.save_model("last_model.pth")
                except Exception:
                    pass
                if log_callback:
                    log_callback(
                        "早停：连续 %d 轮 val_loss 与 mIoU 都没改善（停在第 %d 轮）｜"
                        "最佳 mIoU %.4f @第%d轮，最佳 val loss %.4f @第%d轮"
                        % (patience, epoch + 1, best_miou, best_miou_ep,
                           best_vloss if best_vloss != float("inf") else -1.0, best_vloss_ep))
                break
        if log_callback:
            log_callback("Training complete!")
        # Export optimized models
        try:
            self.export_for_inference()
            if log_callback:
                log_callback("Exported TorchScript/ONNX models")
        except Exception:
            pass

    def _本次版本目录(self):
        """每次训练一个独立版本目录（对标友商：SG_Tiny_V1/V2/V3…）。

        为什么：原来只有 best_model.pth / last_model.pth 两个文件，
        第二次训练就把上一次覆盖了 —— 历史模型全丢，没法回头比。
        现在每次训练建 `models/V<n>_<模型名>/`，里面存 best/last + training_info.json。
        版本号按已有目录扫描取 max+1，不依赖时间戳（跨机器也稳定）。
        """
        if getattr(self, "_版本目录缓存", None):
            return self._版本目录缓存
        import re as _re
        根 = os.path.join(self.project_dir, "models")
        os.makedirs(根, exist_ok=True)
        序号 = []
        try:
            for d in os.listdir(根):
                if os.path.isdir(os.path.join(根, d)):
                    m = _re.match(r"V(\d+)_", d)
                    if m:
                        序号.append(int(m.group(1)))
        except Exception:
            pass
        n = (max(序号) if 序号 else 0) + 1
        路径 = os.path.join(根, "V%d_%s" % (n, self.model_name))
        os.makedirs(路径, exist_ok=True)
        self._版本目录缓存 = 路径
        return 路径

    def save_model(self, filename):
        os.makedirs(os.path.join(self.project_dir, "models"), exist_ok=True)
        path = os.path.join(self.project_dir, "models", filename)
        _版本留档 = None
        try:
            _版本留档 = self._本次版本目录()
        except Exception as _e:
            print('[版本留档] 建版本目录失败（只存 best/last）：%s' % _e, flush=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": self.num_classes,
            "classes": self.classes,
            "model_name": self.model_name,
            # ★ 推理端 `_build_model` 会读这两个字段来复现同样的输入通道
            #   （predictor.py: `ck.get("in_channels")` 与工程 views 交叉校验）
            "in_channels": self.in_channels,
            "views": list(self.views),
            "history": self.history,
            # ★ 2026-10-22：把训练裁剪尺寸写进 checkpoint。
            #   推理端要按**同一个块尺寸**滑窗，否则同一块缺陷在块里大小对不上，
            #   "训练 mIoU 0.95、一推理全漏" 这种诡异现象就是这么来的
            #   （老模型没这个字段，推理端会退回 200 或用 .train_settings.json 兜底）。
            "image_size": (list(self.image_size) if isinstance(self.image_size, (tuple, list))
                           else self.image_size),
            # ★★ 2026-10-29：把**训练模式**也写进去，推理端据此自动选滑窗还是整图。
            #   用户要求"界面上不要 Tiled 勾选、软件自己判断"，那就必须让模型自己
            #   说清楚当初是怎么训的 —— 否则推理端只能猜，猜错就是
            #   "训练 mIoU 0.92、一推理啥也检不到"（尺度错位，差 原图宽/tile 倍）。
            #   目前训练一律走滑窗切块（dataset._build_tiles），所以写 "tiled"；
            #   将来支持"整图 + 采样密度"训练时，这里改成按实际模式写。
            "train_mode": getattr(self, "train_mode", "tiled"),
            "tile_stride": getattr(self, "tile_stride", None),
            # ★ 2026-10-23：早停/最佳轮次摘要（best_model 里能看到"它是最佳的第几轮"）
            #   saved_as 标明这份文件是"按 mIoU 存的最佳快照"还是"最后一轮"，
            #   否则同一轮 run 里 best/last 两个文件会给出不同的 best_val_loss，看着像矛盾。
            "best_summary": (dict(getattr(self, "best_summary", None) or {},
                                  saved_as=filename)),
        }, path)

        # ★ 版本留档（对标友商 SG_Tiny_V1/V2/V3…）：往 models/V<n>_<模型名>/ 也放一份，
        #   并写 training_info.json 记下这次训练用的模型/类别/超参/指标 ——
        #   以后想回头找"上次那个 mIoU 更好的版本"就不用翻日志了。
        if _版本留档:
            try:
                import json as _json
                import shutil as _sh
                import time as _time
                _sh.copy2(path, os.path.join(_版本留档, filename))
                _info = {
                    "版本目录": os.path.basename(_版本留档),
                    "模型": self.model_name,
                    "类别": list(self.classes or []),
                    "输入尺寸": (list(self.image_size) if isinstance(self.image_size, (tuple, list))
                                 else getattr(self, "image_size", None)),
                    "保存时间": _time.strftime("%Y-%m-%d %H:%M:%S"),
                    "最近指标": (self.history[-1] if getattr(self, "history", None) else None),
                    "历史条数": len(getattr(self, "history", []) or []),
                }
                with open(os.path.join(_版本留档, "training_info.json"), "w",
                          encoding="utf-8") as _f:
                    _json.dump(_info, _f, ensure_ascii=False, indent=2)
            except Exception as _e:
                print('[版本留档] 留档失败（best/last 已存）：%s' % _e, flush=True)

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
