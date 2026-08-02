# -*- coding: utf-8 -*-
"""分类训练参数智能推荐弹窗。

展示环境摘要 + 推荐配置表（参数/推荐值/理由），支持目标推理耗时输入，
提供「应用并开始训练 / 仅应用参数 / 取消」三个动作。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QHeaderView)

from classification.recommend import (PARAM_ORDER, collect_dataset_stats,
                                      collect_hardware, recommend)


class RecommendationDialog(QDialog):
    """训练参数推荐弹窗。apply_requested 信号返回 (params, start_training)。"""

    apply_requested = pyqtSignal(dict, bool)

    def __init__(self, project_dir: str = "", parent=None):
        super().__init__(parent)
        self._project_dir = project_dir or ""
        self._params: dict = {}
        self.setWindowTitle("智能推荐训练参数")
        self.resize(760, 640)
        self._build_ui()
        self._refresh()

    # ---- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)

        # 环境与数据集摘要
        self.summary_box = QGroupBox("环境与数据集")
        self.summary_form = QFormLayout(self.summary_box)
        self.summary_labels = {}
        for key, label in [("device", "设备"), ("dataset", "数据集"),
                           ("classes", "类别数"), ("imbalance", "类别不平衡"),
                           ("size", "图片尺寸(最大)")]:
            lb = QLabel("")
            lb.setWordWrap(True)
            self.summary_labels[key] = lb
            self.summary_form.addRow(label, lb)
        root.addWidget(self.summary_box)

        # 目标推理耗时
        lat = QHBoxLayout()
        lat.addWidget(QLabel("目标推理耗时(ms)（线上部署约束，0 = 不约束）:"))
        self.latency_spin = QSpinBox()
        self.latency_spin.setRange(0, 10000)
        self.latency_spin.setSingleStep(5)
        self.latency_spin.setValue(0)
        self.latency_spin.setSuffix(" ms")
        self.latency_spin.valueChanged.connect(self._refresh)
        lat.addWidget(self.latency_spin)
        lat.addStretch(1)
        root.addLayout(lat)

        # 推荐表
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["参数", "推荐值", "推荐理由"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        root.addWidget(self.table, 1)

        # 底部按钮
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_apply_start = QPushButton("应用并开始训练")
        self.btn_apply_only = QPushButton("仅应用参数")
        btn_cancel = QPushButton("取消")
        self.btn_apply_start.setStyleSheet("QPushButton { background:#2e7d32; color:white; font-weight:bold; padding:6px 14px; }")
        self.btn_apply_only.setStyleSheet("QPushButton { padding:6px 14px; }")
        btn_cancel.setStyleSheet("QPushButton { padding:6px 14px; }")
        self.btn_apply_start.clicked.connect(lambda: self._apply(True))
        self.btn_apply_only.clicked.connect(lambda: self._apply(False))
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_apply_start)
        btns.addWidget(self.btn_apply_only)
        btns.addWidget(btn_cancel)
        root.addLayout(btns)

    # ---- 推荐与展示 ----
    def _refresh(self):
        try:
            dataset = collect_dataset_stats(self._project_dir) if self._project_dir else {}
            hw = collect_hardware()
            result = recommend(self._project_dir, dataset=dataset, hardware=hw,
                               target_latency_ms=self.latency_spin.value())
            self._params = result["params"]
            reasons = result["reasons"]
            s = result["summary"]
        except Exception as e:
            self.summary_labels["device"].setText(f"推荐失败: {e}")
            return

        ds, hw = s["dataset"], s["hardware"]
        self.summary_labels["device"].setText(
            f"{hw['device']} | CPU {hw['cpu_cores']} 核 | 内存 {hw['ram_gb']} GB")
        self.summary_labels["dataset"].setText(
            f"{ds['total']} 张样本（类别 {ds['num_classes']} 个，最少 {ds['min_per_class']} 张/类）")
        self.summary_labels["classes"].setText(str(ds["num_classes"]))
        self.summary_labels["imbalance"].setText(f"{ds['imbalance_ratio']}:1")
        self.summary_labels["size"].setText(f"最大 {ds['max_edge']}px（平均 {ds['avg_edge']}px）")

        self.table.setRowCount(0)
        for key, label in PARAM_ORDER:
            if key not in self._params:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.setItem(row, 1, QTableWidgetItem(str(self._params[key])))
            self.table.setItem(row, 2, QTableWidgetItem(reasons.get(key, "")))
        self.table.resizeRowsToContents()

    def _apply(self, start: bool):
        if self._params:
            self.apply_requested.emit(dict(self._params), start)
        self.accept()
