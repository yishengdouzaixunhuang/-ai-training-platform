# -*- coding: utf-8 -*-
"""分类阈值调优面板 —— 扫阈值 + 漏检率/过杀率曲线 + 手动挑阈值 + 保存

Tools 菜单 "分类阈值调优 (过杀/漏检)..." 打开。
依赖 classification.threshold（collect_probs / binary_report / optimize_threshold）。
"""

import traceback
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QFileDialog, QLineEdit, QHeaderView, QSplitter, QMessageBox, QWidget,
    QGroupBox, QFormLayout,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class _SweepThread(QThread):
    """后台跑 collect_probs，避免卡住界面。"""
    done = pyqtSignal(object)

    def __init__(self, project_dir, split, tta, image_size):
        super().__init__()
        self.project_dir = project_dir
        self.split = split
        self.tta = tta
        self.image_size = image_size

    def run(self):
        try:
            from classification.threshold import collect_probs
            probs, labels, names = collect_probs(
                self.project_dir, split=self.split, image_size=self.image_size,
                tta=self.tta, log_callback=None,
            )
            self.done.emit({"probs": probs, "labels": labels, "names": names})
        except Exception as e:
            self.done.emit({"error": f"{e}\n{traceback.format_exc()}"})


class ThresholdPanelDialog(QDialog):
    def __init__(self, parent=None, project_dir=None):
        super().__init__(parent)
        self.setWindowTitle("分类阈值调优 —— 过杀率 / 漏检率")
        self.resize(1280, 800)
        self.project_dir = str(project_dir or "")
        self.rows = []
        self.pos_name = "NG"
        self._thread = None

        root = QVBoxLayout(self)

        # ── 顶部：工程与参数 ────────────────────────────────
        top = QGroupBox("参数")
        form = QFormLayout(top)
        row1 = QHBoxLayout()
        self.ed_dir = QLineEdit(self.project_dir)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse)
        row1.addWidget(self.ed_dir, 1)
        row1.addWidget(btn_browse)
        w1 = QWidget(); w1.setLayout(row1)
        form.addRow("工程目录", w1)

        row2 = QHBoxLayout()
        self.cb_split = QComboBox(); self.cb_split.addItems(["val", "train", "all"])
        self.chk_tta = QCheckBox("TTA(多视图平均)"); self.chk_tta.setChecked(True)
        self.sp_size = QSpinBox(); self.sp_size.setRange(64, 1024); self.sp_size.setValue(224); self.sp_size.setSingleStep(32)
        self.sp_miss = QDoubleSpinBox(); self.sp_miss.setDecimals(4); self.sp_miss.setRange(0.0, 0.5)
        self.sp_miss.setSingleStep(0.001); self.sp_miss.setValue(0.002)
        self.cb_obj = QComboBox(); self.cb_obj.addItems(["min_overkill", "youden", "f1"])
        self.btn_run = QPushButton("开始扫描"); self.btn_run.clicked.connect(self._run)
        row2.addWidget(QLabel("数据集")); row2.addWidget(self.cb_split)
        row2.addWidget(QLabel("图像尺寸")); row2.addWidget(self.sp_size)
        row2.addWidget(self.chk_tta)
        row2.addWidget(QLabel("最大漏检率")); row2.addWidget(self.sp_miss)
        row2.addWidget(QLabel("目标")); row2.addWidget(self.cb_obj)
        row2.addStretch(1); row2.addWidget(self.btn_run)
        w2 = QWidget(); w2.setLayout(row2)
        form.addRow("选项", w2)

        self.lbl_status = QLabel("就绪。点【开始扫描】在验证集上扫阈值。")
        self.lbl_status.setStyleSheet("color:#00E5FF;")
        form.addRow("状态", self.lbl_status)
        root.addWidget(top)

        # ── 中部：曲线 + 表格 ──────────────────────────────
        split = QSplitter(Qt.Horizontal)
        self.fig = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        split.addWidget(self.canvas)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            ["阈值", "漏检率%", "过杀率%", "精确率", "召回", "F1", "TP", "FN", "FP", "TN"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        split.addWidget(self.table)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        # ── 底部：挑阈值 / 保存 ────────────────────────────
        bottom = QHBoxLayout()
        self.sp_thr = QDoubleSpinBox(); self.sp_thr.setDecimals(3)
        self.sp_thr.setRange(0.001, 0.999); self.sp_thr.setSingleStep(0.005); self.sp_thr.setValue(0.5)
        self.sp_thr.valueChanged.connect(self._update_vline)
        btn_best = QPushButton("用最优阈值"); btn_best.clicked.connect(self._use_best)
        btn_save = QPushButton("保存该阈值到模型"); btn_save.clicked.connect(self._save)
        self.lbl_at = QLabel("")
        self.lbl_at.setStyleSheet("color:#E6F0FF;")
        bottom.addWidget(QLabel("当前阈值")); bottom.addWidget(self.sp_thr)
        bottom.addWidget(btn_best); bottom.addWidget(btn_save)
        bottom.addWidget(self.lbl_at, 1)
        root.addLayout(bottom)

    # ── 交互 ────────────────────────────────────────────
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择分类工程目录", self.ed_dir.text() or "")
        if d:
            self.ed_dir.setText(d)

    def _run(self):
        pd = self.ed_dir.text().strip()
        if not pd or not Path(pd).exists():
            QMessageBox.warning(self, "提示", "请先选择有效的工程目录")
            return
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("正在推理验证集…")
        self._thread = _SweepThread(pd, self.cb_split.currentText(),
                                    self.chk_tta.isChecked(), self.sp_size.value())
        self._thread.done.connect(self._on_sweep)
        self._thread.start()

    def _on_sweep(self, res):
        self.btn_run.setEnabled(True)
        if "error" in res:
            self.lbl_status.setText("失败：" + res["error"].splitlines()[0])
            QMessageBox.critical(self, "扫描失败", res["error"][:2000])
            return
        from classification.threshold import binary_report, optimize_threshold, pick_pos_idx
        probs, labels, names = res["probs"], res["labels"], res["names"]
        pos = pick_pos_idx(names, None)
        self.pos_name = names[pos]
        self.rows = binary_report(probs, labels, pos)

        # 填表
        self.table.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            vals = [f"{r['threshold']:.2f}", f"{r['miss_rate']*100:.3f}", f"{r['overkill_rate']*100:.3f}",
                    f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                    str(r["tp"]), str(r["fn"]), str(r["fp"]), str(r["tn"])]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, it)

        # 画曲线
        xs = [r["threshold"] for r in self.rows]
        miss = [r["miss_rate"] * 100 for r in self.rows]
        over = [r["overkill_rate"] * 100 for r in self.rows]
        f1s = [r["f1"] * 100 for r in self.rows]
        self.ax.clear()
        self.ax.plot(xs, miss, color="#FF4D4D", label="漏检率 %")
        self.ax.plot(xs, over, color="#00E5FF", label="过杀率 %")
        self.ax.plot(xs, f1s, color="#9AE66E", alpha=0.6, linestyle="--", label="F1 ×100")
        self.ax.set_xlabel("判定阈值 (NG概率 ≥ 阈值 → 判NG)")
        self.ax.set_ylabel("%")
        self.ax.set_title(f"阈值扫描（{len(labels)} 张/{self.cb_split.currentText()}，正类={self.pos_name}）")
        self.ax.grid(alpha=0.25)
        self.ax.legend(loc="upper center", fontsize=8)
        self._vline = self.ax.axvline(self.sp_thr.value(), color="#FFD400", linestyle=":", linewidth=1.5)
        self.canvas.draw()

        # 最优阈值
        best = optimize_threshold(probs, labels, pos,
                                  max_miss_rate=self.sp_miss.value(),
                                  objective=self.cb_obj.currentText())
        self._best = best
        self.sp_thr.blockSignals(True)
        self.sp_thr.setValue(float(best["threshold"]))
        self.sp_thr.blockSignals(False)
        self._update_vline()
        self.lbl_status.setText(
            f"完成：验证 {len(labels)} 张，正类 {self.pos_name}。"
            f"最优阈值 {best['threshold']} → 漏检率 {best['miss_rate']*100:.3f}%、"
            f"过杀率 {best['overkill_rate']*100:.3f}%、F1 {best['f1']}")

    def _row_at(self, thr):
        if not self.rows:
            return None
        return min(self.rows, key=lambda r: abs(r["threshold"] - thr))

    def _update_vline(self):
        if hasattr(self, "_vline") and self._vline is not None:
            self._vline.set_xdata([self.sp_thr.value(), self.sp_thr.value()])
            self.canvas.draw_idle()
        r = self._row_at(self.sp_thr.value())
        if r:
            self.lbl_at.setText(
                f"阈值 {r['threshold']:.2f} → 漏检率 {r['miss_rate']*100:.3f}%、"
                f"过杀率 {r['overkill_rate']*100:.3f}%、F1 {r['f1']:.4f} "
                f"(TP{r['tp']} FN{r['fn']} FP{r['fp']} TN{r['tn']})")

    def _on_row_selected(self):
        r = self.table.currentRow()
        if 0 <= r < len(self.rows):
            self.sp_thr.setValue(float(self.rows[r]["threshold"]))

    def _use_best(self):
        if getattr(self, "_best", None):
            self.sp_thr.setValue(float(self._best["threshold"]))

    def _save(self):
        from classification.threshold import save_threshold_result
        if not self.rows:
            QMessageBox.warning(self, "提示", "请先扫描")
            return
        r = self._row_at(self.sp_thr.value())
        if not r:
            return
        try:
            p = save_threshold_result(self.ed_dir.text().strip(), r, self.pos_name, self.rows)
            self.lbl_status.setText(f"已保存阈值 {r['threshold']} → {p}")
            QMessageBox.information(self, "已保存",
                                    f"阈值 {r['threshold']} 已写入：\n{p}\n\n"
                                    f"推理端 Predictor.classify_with_threshold() 会自动读取该阈值判定 OK/NG。")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"{e}\n{traceback.format_exc()[:1200]}")
