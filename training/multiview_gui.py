# -*- coding: utf-8 -*-
"""三视图（9 通道）分割训练的 GUI 适配器

作用：让 GUI 的「Model Mgmt → Start Seg Training」在检测到**三视图工程**时，
自动改用 training.multiview 的训练循环，而**不需要改动 GUI 的线程/进度/日志/停止逻辑**。

接口与 training.trainer.Trainer 完全一致：
    __init__(project_dir, model_name=..., device=...)
    train(epochs, batch_size, image_size, loss_name, augment, k_folds, resume,
          stop_check, log_callback, progress_callback, plot_callback, batch_callback)
    cleanup()

几个必须做的"翻译"：
  * GUI 的 loss 名（cross_entropy / ce+dice / dice）→ 本训练器的（ce / ce_dice / dice）
  * GUI 的 augment 是 **dict 或 None**（不是字符串）→ 转成 bool
  * GUI 的 history 图表要 {"train_loss":[], "val_loss":[], "val_miou":[]}（字典套列表），
    而本训练器内部是 list[dict] → 需要转换
  * GUI 保存 best_model.pth / last_model.pth，沿用同名以保持一致
  * K 折交叉验证本训练器暂不支持 → 明确告知并按单次训练执行（不静默忽略）
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

DEFAULT_VIEWS = ("te_0", "te_1", "te_2")
_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# GUI 的 loss 选项 → 三视图训练器的 loss 选项
GUI_LOSS_MAP = {
    "cross_entropy": "ce",
    "dice": "dice",
    "ce+dice": "ce_dice",
}
# 这些 GUI 选项三视图训练器没有实现，退到 ce_dice
GUI_LOSS_FALLBACK = ("lovasz", "focal", "ce+lovasz")


def project_views(project_dir):
    """读工程声明的视图列表。

    优先级：project.json 的 views > project.json 的 in_channels 推断 > 扫 images/ 数视图后缀。
    **兜底必须是"单张"，绝不能默认三视图** —— 否则任何单图工程都会被当成 9 通道直接崩。
    返回 (views_tuple, view_names_dict or None)；只有 ≥2 个视图才算多视图工程。
    """
    p = Path(project_dir)
    pj = p / "project.json"
    if pj.exists():
        try:
            d = json.loads(pj.read_text(encoding="utf-8"))
            v = d.get("views")
            if v:
                return tuple(str(x) for x in v), (d.get("view_names") or None)
            ich = int(d.get("in_channels") or 0)
            if ich and ich % 3 == 0 and ich > 3:
                n = ich // 3
                return tuple(f"te_{i}" for i in range(n)), None
        except Exception:
            pass
    # 退回扫文件名：数一数出现了多少个 te_N 后缀
    img = p / "images"
    if img.is_dir():
        try:
            fs = [f.name.lower() for f in img.iterdir() if f.is_file()]
        except OSError:
            fs = []
        found = set()
        for n in fs:
            stem = n.rsplit(".", 1)[0]
            if "_" in stem:
                found.add(stem.rsplit("_", 1)[1])
        vs = sorted(x for x in found if x.startswith("te_"))
        if vs:
            return tuple(vs), None
    return ("te_0",), None


def is_multiview_project(project_dir) -> bool:
    """是否多视图工程（≥2 视图 → 输入通道 = 3 × 视图数）。单视图走平台原有 3 通道流程。"""
    views, _ = project_views(project_dir)
    return len(views) >= 2


def multiview_stats(project_dir):
    """给 UI 显示用：(组数, 有掩膜的组数, 切片尺寸, 视图数, 通道数)"""
    p = Path(project_dir)
    img, ann = p / "images", p / "annotations"
    views, _ = project_views(project_dir)
    tile = 0
    try:
        tile = int(json.loads((p / "project.json").read_text(encoding="utf-8")).get("tile_size") or 0)
    except Exception:
        pass
    keys, masked = set(), set()
    if img.is_dir():
        try:
            files = [f.name for f in img.iterdir() if f.is_file()]
        except OSError:
            files = []
        for n in files:
            for v in views:
                for e in _EXTS:
                    if n.lower().endswith(f"_{v}{e}"):
                        k = n[: -(len(v) + 1 + len(e))]
                        keys.add(k)
                        if (ann / f"{k}_{v}_mask.png").exists() or (ann / f"{k}_mask.png").exists():
                            masked.add(k)
                        break
    return len(keys), len(masked), tile, len(views), 3 * len(views)


class MultiViewTrainer:
    """training.trainer.Trainer 的三视图替代品（drop-in）。"""

    def __init__(self, project_dir, model_name="unet", device=None,
                 unet_base=16, unet_depth=4, num_workers=None,
                 tile_size=0, tile_mode="slide", tile_stride_pct=75, neg_ratio=1.5):
        self._tile_size = int(tile_size or 0)
        self._tile_mode = str(tile_mode or "slide")
        self._tile_stride_pct = float(tile_stride_pct or 75)
        self._tile_neg_ratio = float(neg_ratio or 1.5)
        self.project_dir = str(project_dir)
        self.model_name = model_name or "unet"
        self.device = device
        self.unet_base = int(unet_base)
        self.unet_depth = int(unet_depth)
        # GUI 里训练跑在**子线程**，主模块有 __main__ 保护，所以多进程加载是安全的。
        self.num_workers = num_workers
        self.metrics = None
        self.model = None

    # ── 与平台 Trainer 同名的清理接口（GUI 的 _cleanup_trainer 会调）──
    def cleanup(self):
        self.model = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def train(self, epochs=50, batch_size=4, image_size=512, loss_name="cross_entropy",
              augment=None, k_folds=1, resume=False, stop_check=None, log_callback=None,
              progress_callback=None, plot_callback=None, batch_callback=None,
              tile_size=None, tile_stride_pct=75, tile_mode="slide", neg_ratio=1.5):
        import torch
        from training.multiview import train_multiview

        L = log_callback or (lambda m: print(m, flush=True))
        proj = Path(self.project_dir)

        # ---- 尺寸：切片工程必须按原分辨率训练，放大只会浪费算力并破坏像素细节 ----
        pj = {}
        try:
            pj = json.loads((proj / "project.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        tile = int(pj.get("tile_size") or 0)
        gui_size = image_size[0] if isinstance(image_size, (tuple, list)) else int(image_size)
        size = tile or gui_size
        if tile and gui_size and int(gui_size) != tile:
            L(f"提示: 本工程切片为 {tile}px，已按原分辨率训练（界面上选的 {gui_size} 不适用："
              f"放大只会浪费算力，缩小会把小缺陷抹掉）")

        # ---- loss 名称翻译 ----
        lname = (loss_name or "cross_entropy").strip().lower()
        loss = GUI_LOSS_MAP.get(lname)
        if loss is None:
            loss = "ce_dice"
            if lname in GUI_LOSS_FALLBACK:
                L(f"提示: 三视图训练器未实现 loss={loss_name}，已改用 ce_dice（小目标更稳）")
            else:
                L(f"提示: 未知 loss={loss_name}，已改用 ce_dice")

        # ---- augment 在 GUI 里是 dict/None，不是字符串 ----
        aug = bool(augment)
        if k_folds and int(k_folds) > 1:
            L(f"提示: 三视图训练暂不支持 K 折交叉验证（收到 k_folds={k_folds}），按单次训练执行")

        # ---- 续训 ----
        resume_path = None
        if resume:
            for cand in (proj / "models" / "last_model.pth", proj / "models" / "best_model.pth"):
                if cand.exists():
                    resume_path = str(cand)
                    L(f"继续训练: 载入 {cand.name}")
                    break
            if resume_path is None:
                L("未找到 last_model.pth / best_model.pth，改为从头训练")

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        nw = self.num_workers
        if nw is None:
            nw = 4 if dev == "cuda" else 0
        views, vnames = project_views(proj)
        n_keys, n_masked, _, _n_view, n_ch = multiview_stats(proj)
        vdesc = " + ".join((vnames or {}).get(v, v) for v in views)
        L("=" * 58)
        L(f"多视图语义分割训练（{vdesc} → {n_ch} 通道）")
        L(f"  模型      : {self.model_name}"
          + (f"  base={self.unet_base} depth={self.unet_depth}" if self.model_name == "unet" else ""))
        L(f"  数据      : {n_keys} 组（每组 {len(views)} 视图），{n_masked} 组有掩膜")
        L(f"  切片/批/轮: {size}px / {batch_size} / {epochs}")
        L(f"  loss/增强 : {loss} / {'开' if aug else '关'}")
        L(f"  设备      : {dev}" + (f"  {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))
        L("=" * 58)

        self.metrics = train_multiview(
            proj, model_name=self.model_name, epochs=int(epochs), batch_size=int(batch_size),
            # 验证集不切图，但**必须和训练同尺度**：开了分块就按滑窗大小验证，
            # 否则"训练看 256、验证看 512"两套尺度，mIoU 没法比。
            image_size=int(self._tile_size) if self._tile_size else int(size),
            num_workers=int(nw), device=dev, augment=aug,
            log_callback=L, save_name="best_model.pth", loss=loss, views=views,
            unet_base=self.unet_base, unet_depth=self.unet_depth, resume=resume_path,
            # ── 分块训练（对齐友商：滑窗大小 + 步长%）──
            #    大图**不缩**，按滑窗在原分辨率上裁块训练；界面勾了「分块」才生效。
            tile_size=int(tile_size if tile_size is not None else self._tile_size),
            tile_mode=str(tile_mode or self._tile_mode),
            tile_stride_pct=float(tile_stride_pct if tile_stride_pct is not None else self._tile_stride_pct),
            neg_ratio=float(neg_ratio if neg_ratio is not None else self._tile_neg_ratio),
            progress_callback=progress_callback, batch_callback=batch_callback,
            stop_check=stop_check,
        )

        # ---- 与平台一致的 last_model.pth（供界面续训与查找）----
        try:
            rolling = proj / "models" / "best_model_last.pth"
            if rolling.exists():
                shutil.copy2(rolling, proj / "models" / "last_model.pth")
                L("已更新 models/last_model.pth（可勾选 Resume 继续训练）")
        except Exception as e:
            L(f"提示: last_model.pth 更新失败: {type(e).__name__}: {e}")

        # ---- 转成 GUI 图表要的 history 格式（字典套列表）----
        h = {"train_loss": [], "val_loss": [], "val_miou": []}
        for r in (self.metrics or {}).get("history", []):
            h["train_loss"].append(r.get("loss"))
            h["val_loss"].append(r.get("val_loss"))
            h["val_miou"].append(r.get("mIoU_fg"))
        if plot_callback:
            try:
                plot_callback(h)
            except Exception:
                pass

        m = self.metrics or {}
        L(f"训练完成: best epoch={m.get('best_epoch')}  "
          f"前景 mIoU={m.get('best_mIoU_fg')}  模型: models/best_model.pth")
        L("注意: 三视图模型的推理入口当前在「三视图工具.bat → [4] 整图滑窗评估」，"
          "平台自带的 Run Inference 是单图 3 通道模型，不适用于本工程。")
        return self.metrics
