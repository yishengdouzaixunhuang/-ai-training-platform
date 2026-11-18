# -*- coding: utf-8 -*-
"""主窗口用到的独立自定义控件。

这些类原来都堆在 `ui/main_window.py` 顶部（那文件一度接近 1 万行），
它们不依赖 MainWindow 的任何状态，纯属「自定义控件」，所以单独搬出来。

- ElideLabel           按宽度中间省略的标签
- ShrinkStack          只跟随当前页报尺寸的 QStackedWidget
- LossCurveWindow      分类训练独立的 Loss/Acc 曲线窗
- TrainSettingsDialog  训练参数弹窗（主界面已改为直接读面板，保留以免回退）
"""
from __future__ import annotations
from core.i18n import tr

import json
import os

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# 项目根目录（ui/ 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ElideLabel(QLabel):
    """按可用宽度**中间省略**的标签：宽不够时自己变短，绝不把右边的兄弟控件顶出去。

    「图片信息栏」左边是文件名（可达 40+ 字符），右边是坐标/RGB/计数等短字段。
    文件名用普通 QLabel 会把整条栏撑宽 —— 在中心列只有 ~700px 的窗口里，
    右边的"耗时"就被挤到屏幕外（而"耗时"恰恰是没跑推理时最需要看到"是空的"那一项）。
    所以：文件名这一格可缩、缩到装不下就 ElideMiddle（头尾都留住：
    前缀能看到日期/机号/缺陷，后缀能看到 _te_0.jpg 是哪个视图），右边字段一律不动。
    """

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        # 宽度由布局说了算：不让 sizeHint 反过来跟布局讨价还价（会来回抖）
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(80)

    def setFullText(self, text):
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._apply_elide()

    def fullText(self):
        return self._full_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self):
        fm = QFontMetrics(self.font())
        w = max(0, self.width() - 2)
        if w <= 0:
            return
        self.setText(fm.elidedText(self._full_text, Qt.ElideMiddle, w))


class ShrinkStack(QStackedWidget):
    """只按**当前子页**报告尺寸的 QStackedWidget。

    原生 QStackedWidget 的 sizeHint/minimumSizeHint 取的是「所有子页里最大的那个」。
    训练栈里各子页高度差极大（语义分割 ~200px、分类 ~900px），套进 QScrollArea 后
    它会把**每一个**子页都撑到最大高度 —— 页内那个 addStretch() 于是吃掉几百像素，
    表现就是「Start Training 和 Epochs 之间夹着一大片空白」（用户截图反馈）。

    这里改成只跟随 currentWidget()：换页后由调用方 updateGeometry() 重新量高。
    """

    def _cur(self):
        try:
            return self.currentWidget()
        except Exception:
            return None

    def sizeHint(self):
        w = self._cur()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self):
        w = self._cur()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()


class LossCurveWindow(QWidget):
    """Independent Loss/Accuracy curve window for classification training.

    - Shown automatically when training starts
    - Can be minimized / closed / reopened anytime
    - Not affected by the collapsed sidebar panel
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("损失 / 准确率曲线"))
        self.resize(760, 480)
        self.setWindowFlag(Qt.Window)
        layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(7, 4.5), dpi=100))
        self.ax = self.canvas.figure.subplots()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss / Acc")
        self.ax.grid(True, alpha=0.3)
        self.canvas.figure.tight_layout(pad=0.5)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_curve(self, history):
        """Redraw loss/acc curves from training history dict."""
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        val_acc = history.get("val_acc", [])
        if not train_loss:
            return
        ax = self.ax
        ax.clear()
        epochs = list(range(1, len(train_loss) + 1))
        ax.plot(epochs, train_loss, "b-", label="Train Loss", alpha=0.6)
        if val_loss:
            ax.plot(epochs, val_loss, "r-", label="Val Loss", alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color="b")
        ax.tick_params(axis="y", labelcolor="b")
        ax.grid(True, alpha=0.3)
        if val_acc:
            ax2 = ax.twinx()
            ax2.plot(epochs, val_acc, "g-", label="Val Acc", alpha=0.8)
            # 缺陷召回曲线：比总准确率更贴近产线关心的「漏检」
            ng_rec = history.get("val_ng_recall", [])
            if ng_rec and len(ng_rec) == len(epochs):
                ax2.plot(epochs, ng_rec, "m--", label="NG Recall", alpha=0.8)
            ax2.set_ylabel("Accuracy", color="g")
            ax2.tick_params(axis="y", labelcolor="g")
            ax2.set_ylim(0, 1.05)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        else:
            ax.legend(loc="upper left")
        self.canvas.figure.tight_layout(pad=0.5)
        self.canvas.draw()

    def closeEvent(self, event):
        """Close button hides the window so it can be reopened later."""
        self.hide()
        event.ignore()


class TrainSettingsDialog(QDialog):
    """训练参数弹窗。

    ⚠️ 主界面目前已改为「直接读右侧面板」，不再弹这个窗
      （见 MainWindow._start_training 里的注释）。保留它是为了需要时能一行切回。

    历史 bug（已修）：
      原来 SETTINGS_FILE 用了三层 dirname，指到
      `<项目父目录>/AI_Training_Platform/.train_settings.json` ——
      那是**另一个不存在的位置**，于是这里存的东西主界面永远读不到、
      主界面存的东西这里也读不到。现在统一用项目根目录下的 .train_settings.json。
    """

    SETTINGS_FILE = os.path.join(PROJECT_ROOT, ".train_settings.json")

    def __init__(self, parent=None, project_dir=None):
        super().__init__(parent)
        self.setWindowTitle(tr("训练参数"))
        layout = QFormLayout(self)
        saved = self._load()
        # ── 三视图检测 ──
        # 三视图工程要用 9 通道训练器，它只支持 unet / lraspp / deeplabv3 / fcn；
        # 单图工程继续用平台原有列表（含 r101 / mobilenet 等变体）。
        # 两边模型名不通用，所以必须按工程类型切换列表，否则选错就崩。
        self._is_mv = False
        self._mv_tile = 0
        mv_info = ""
        if project_dir:
            try:
                from training.multiview_gui import is_multiview_project, multiview_stats
                if is_multiview_project(project_dir):
                    self._is_mv = True
                    k, m, t, nv, nch = multiview_stats(project_dir)
                    self._mv_tile = int(t or 0)
                    mv_info = (f"✓ 多视图工程（{nv} 视图 / {nch} 通道）：{k} 组 / {m} 组有掩膜"
                               + (f" / 切片 {t}px" if t else ""))
            except Exception as e:
                mv_info = f"（三视图检测失败：{type(e).__name__}: {e}）"
        if mv_info:
            lab = QLabel(f"<b style='color:#00E5FF'>{mv_info}</b>")
            lab.setWordWrap(True)
            layout.addRow(lab)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 2000); self.epochs_spin.setValue(saved.get("epochs", 200))
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 64); self.batch_spin.setValue(saved.get("batch_size", 8))
        self.model_combo = QComboBox()
        models = (["unet", "lraspp", "deeplabv3", "fcn"] if self._is_mv else
                  ["deeplabv3", "deeplabv3_r101", "deeplabv3_mobilenet", "fcn", "fcn_r101", "lraspp"])
        self.model_combo.addItems(models)
        dflt = "unet" if self._is_mv else "deeplabv3"
        idx = self.model_combo.findText(saved.get("model", dflt)); self.model_combo.setCurrentIndex(max(0, idx))
        if self._is_mv:
            self.model_combo.setToolTip("三视图推荐 unet：全分辨率跳连，救 <16px 小缺陷（实测 5 个尺寸档赢 4 个）")
        self.size_combo = QComboBox(); self.size_combo.addItems(["256", "512", "768", "1024"])
        if self._is_mv and self._mv_tile:
            # 切片工程的训练尺寸就是切片尺寸，直接锁定：
            # 放大只是浪费算力，缩小会把小缺陷一起抹掉（本项目核心问题就是 <16px 缺陷）
            self.size_combo.addItem(str(self._mv_tile))
            self.size_combo.setCurrentText(str(self._mv_tile))
            self.size_combo.setEnabled(False)
            self.size_combo.setToolTip(f"切片工程固定为 {self._mv_tile}px 原分辨率训练")
        else:
            idx2 = self.size_combo.findText(str(saved.get("image_size", 512))); self.size_combo.setCurrentIndex(max(0, idx2))
        layout.addRow(tr("轮数："), self.epochs_spin)
        layout.addRow(tr("批大小："), self.batch_spin)
        layout.addRow(tr("模型："), self.model_combo)
        layout.addRow(tr("图像尺寸："), self.size_combo)
        # ── U-Net 容量参数（仅三视图工程显示）──
        if self._is_mv:
            self.unet_base_spin = QSpinBox(); self.unet_base_spin.setRange(8, 64); self.unet_base_spin.setSingleStep(8)
            self.unet_base_spin.setValue(saved.get("unet_base", 16))
            self.unet_base_spin.setToolTip("首层通道数：16≈2.0M 参数（快）/ 32≈7.9M 参数（容量更大，小目标更稳）")
            self.unet_depth_spin = QSpinBox(); self.unet_depth_spin.setRange(2, 6)
            self.unet_depth_spin.setValue(saved.get("unet_depth", 4))
            layout.addRow(tr("U-Net 基础通道："), self.unet_base_spin)
            layout.addRow(tr("U-Net 深度："), self.unet_depth_spin)
        else:
            self.unet_base_spin = None
            self.unet_depth_spin = None
        # ── 分块训练（对齐友商平台：分块 / 滑窗大小 / 步长(%)）──
        #    大图（本项目 3860×1480）**以前被整图 resize 到 512 宽**，中位 234px 的 Atd
        #    缩完只剩 ~4px、127px 的 Zw 剩 ~2px —— 缺陷在数据加载阶段就被抹掉了。
        #    勾上「分块」后改成在**原分辨率**上按滑窗裁剪，缺陷像素一个不丢。
        self._is_mv = bool(getattr(self, "_is_mv", False))
        self.tile_check = QCheckBox(tr("分块（滑窗裁剪，不缩放整图）"))
        self.tile_check.setChecked(bool(saved.get("tile", self._is_mv)))
        self.tile_check.setToolTip(tr("大图必须分块：整图缩放会把小缺陷缩没"))
        layout.addRow(self.tile_check)
        # ⚠️ 勾上「分块」后 Image Size 就不再决定训练输入了（那是「滑窗大小」的事），
        #    它只剩"验证时整图缩放到多大"这一个作用。留着可编辑会让人以为改它有用 ——
        #    所以勾选时置灰 + 改提示，避免"两个参数含义重叠、改一个不生效"。
        def _sync_size_enabled(_checked=None):
            on = bool(self.tile_check.isChecked())
            self.size_combo.setEnabled(not on)
            self.size_combo.setToolTip(
                "已开分块：训练输入由「滑窗大小」决定，这里只影响验证/整图推理的缩放"
                if on else "训练输入尺寸（未开分块时 = 整图缩放到这个尺寸）")
        self.tile_check.toggled.connect(_sync_size_enabled)
        _sync_size_enabled()
        self.tile_spin = QSpinBox(); self.tile_spin.setRange(64, 1024); self.tile_spin.setSingleStep(64)
        self.tile_spin.setValue(int(saved.get("tile_size", 256)))
        layout.addRow(tr("滑窗大小:"), self.tile_spin)
        self.stride_spin = QSpinBox(); self.stride_spin.setRange(10, 100); self.stride_spin.setSingleStep(5)
        self.stride_spin.setValue(int(saved.get("tile_stride", 75)))
        self.stride_spin.setSuffix(" %")
        self.stride_spin.setToolTip(tr("步长(%)：75 = 25% 重叠。越小覆盖越全，块数按平方增长"))
        layout.addRow(tr("步长(%):"), self.stride_spin)
        self.tile_mode_combo = QComboBox()
        self.tile_mode_combo.addItems([tr("滑窗遍历（覆盖全）"), tr("缺陷居中+负样本（省算力）")])
        self.tile_mode_combo.setCurrentIndex(0 if saved.get("tile_mode", "slide") == "slide" else 1)
        self.tile_mode_combo.setToolTip("滑窗遍历：友商口径，覆盖无死角但 98% 是背景块\n"
                                        "缺陷居中+负样本：只抽缺陷居中块+定量背景块，省约 10 倍算力")
        layout.addRow(tr("取样方式:"), self.tile_mode_combo)

        # ★ 2026-11-13：这四个控件的值来自文件，但 `_save_and_accept()` **不写回它们**
        #   （分块设置的权威在主面板上；写回去会和面板互相覆盖，见那个方法的注释）。
        #   以前它们看着可以改、改了点「确定」却什么都没发生 —— **UI 在撒谎**：
        #   用户把 1024 改成 512，下次打开还是 1024，只会来问"我改的分块怎么没了"。
        #   所以置灰 + 明写去哪儿改（和上面 Augment 那行同一个处理方式）。
        self.tile_hint = QLabel("分块 / 滑窗大小 / 步长 / 取样方式：请到主界面「分块」设置，"
                                "这里只显示当前值")
        self.tile_hint.setWordWrap(True)
        self.tile_hint.setStyleSheet("color: #8A93A6;")
        layout.addRow(tr("分块："), self.tile_hint)
        _tile_ro_tip = "以主界面「分块」设置为准；这里只显示当前值，改了不会保存"
        for _w_ro in (self.tile_check, self.tile_spin,
                      self.stride_spin, self.tile_mode_combo):
            _w_ro.setEnabled(False)
            _w_ro.setToolTip(_tile_ro_tip)
        self.kfold_spin = QSpinBox(); self.kfold_spin.setRange(1, 10); self.kfold_spin.setValue(saved.get("k_folds", 1))
        self.kfold_spin.setToolTip(tr("1=不交叉验证，5=5 折交叉验证"))
        layout.addRow(tr("K 折交叉验证："), self.kfold_spin)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["cross_entropy", "lovasz", "dice", "focal", "ce+lovasz", "ce+dice"])
        idx3 = self.loss_combo.findText(saved.get("loss", "cross_entropy"))
        self.loss_combo.setCurrentIndex(max(0, idx3))
        self.loss_combo.setToolTip(tr("损失函数：小缺陷建议 ce+lovasz"))
        layout.addRow(tr("损失函数："), self.loss_combo)
        self.augment_hint = QLabel(tr("数据增强：请在主界面右侧「增强」面板设置（全局生效）"))
        self.augment_hint.setWordWrap(True)
        self.augment_hint.setStyleSheet("color: #8A93A6;")
        layout.addRow(tr("数据增强："), self.augment_hint)
        self.resume_check = QCheckBox(tr("从上次检查点继续"))
        self.resume_check.setChecked(saved.get("resume", False))
        layout.addRow(self.resume_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save_and_accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _load(self):
        try:
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def get_settings(self):
        return {
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "model": self.model_combo.currentText(),
            "image_size": int(self.size_combo.currentText()),
            "k_folds": self.kfold_spin.value(),
            "loss": self.loss_combo.currentText(),
            "resume": self.resume_check.isChecked(),
            "is_multiview": bool(getattr(self, "_is_mv", False)),
            # 分块训练（滑窗大小 / 步长% / 取样方式）
            "tile": bool(self.tile_check.isChecked()),
            "tile_size": int(self.tile_spin.value()),
            "tile_stride": int(self.stride_spin.value()),
            "tile_mode": "slide" if self.tile_mode_combo.currentIndex() == 0 else "defect",
            "unet_base": self.unet_base_spin.value() if self.unet_base_spin else 16,
            "unet_depth": self.unet_depth_spin.value() if self.unet_depth_spin else 4,
        }

    def _save_and_accept(self):
        """把对话框上的值写回 `.train_settings.json`（**合并写 + 原子替换**）。

        ★ 2026-11-13：改前是 `open(self.SETTINGS_FILE, "w")` **全量覆盖**，
          而且只写 9 个键。文件里实际有 15 个键 ⇒ 点一次「确定」就抹掉 6 个：

              augment_flags / early_stop_patience / infer_mode /
              tile / tile_size / tile_stride

          用户在对话框里只改一个 `batch_size`，分块训练设置、早停耐心值、
          推理档位、增强参数全部被删掉（下次启动按默认值来），日志里一个字都没有。
          现在走 `core.settings.save("train", ...)`：只覆盖本对话框负责的键，
          其余原样保留，写入是原子的。

        ⚠️ 这个对话框**当前没有被打开**：`main_window` 里构造它的两行被注释掉了
           （注释原文"不再弹 Training Settings 窗（用户要求）"，改成直接读面板），
           类本身留着是为了"回退成弹窗模式"。
           正因如此这条路径**没人跑过**，烂在里面更不容易被发现 —— 所以照样修。
        """
        from core import settings as _settings

        data = {
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "model": self.model_combo.currentText(),
            "image_size": int(self.size_combo.currentText()),
            "k_folds": self.kfold_spin.value(),
            "loss": self.loss_combo.currentText(),
            "resume": self.resume_check.isChecked(),
            "unet_base": self.unet_base_spin.value() if self.unet_base_spin else 16,
            "unet_depth": self.unet_depth_spin.value() if self.unet_depth_spin else 4,
        }
        try:
            os.makedirs(os.path.dirname(self.SETTINGS_FILE), exist_ok=True)
        except OSError:
            pass                    # 目录建不出来时让 save() 去报，别在这里静默掉

        res = _settings.save("train", data, self.SETTINGS_FILE)
        if not res.ok:
            # 对话框没有日志栏，交给父窗口；再没有就弹一句 —— 总之不能静默
            msg = "[训练参数] %s" % res.code_text
            parent = self.parent()
            if parent is not None and hasattr(parent, "log"):
                parent.log(msg)
            else:
                QMessageBox.warning(self, tr("训练参数未保存"), msg)
        self.accept()
