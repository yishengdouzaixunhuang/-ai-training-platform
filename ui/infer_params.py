# -*- coding: utf-8 -*-
"""推理参数面板（对标友商 友商平台 的「推理参数」）。

面板布局照友商的来：**类别下拉 + 一组 min~max 区间 + Apply All**，
所以同一时间只显示一个类别的参数，改完即存（防抖 600ms 写 inference_params.json）。

为什么不做成"一张大表、每类一行"：友商就是这么做的（一次看一个类别），
区间有 6 组共 12 个数字框，横着排一行会挤成看不清；竖着一类一块又太长。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox,
)

from inference.params import (
    RANGE_KEYS, SORT_CHOICES, default_class_params, load_params, save_params,
)

_NO_LIMIT_TIP = "上限填负数表示「不限」（与友商一致）\n角度上下限都为 -180 时表示不过滤角度"


class InferParamsPanel(QGroupBox):
    """推理参数：按类别过滤连通域（面积/宽度/高度/角度/灰度/分数）+ 全局后处理。"""

    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("推理参数", parent)
        self._project_dir = None
        self._params = None
        self._loading = False
        self._current = ""
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(self._flush_save)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 4, 4)
        root.setSpacing(3)

        # ── 类别选择 + Apply All ──
        top = QHBoxLayout(); top.setSpacing(4)
        top.addWidget(QLabel("类别:"))
        self.cls_combo = QComboBox()
        self.cls_combo.setMinimumWidth(90)
        self.cls_combo.currentTextChanged.connect(self._on_class_changed)
        top.addWidget(self.cls_combo, 1)
        self.apply_all_btn = QPushButton("Apply All")
        self.apply_all_btn.setToolTip("把当前类别的这组区间应用到**所有类别**")
        self.apply_all_btn.clicked.connect(self._apply_all)
        top.addWidget(self.apply_all_btn)
        root.addLayout(top)

        # ── 6 组区间 ──
        grid = QGridLayout(); grid.setHorizontalSpacing(6); grid.setVerticalSpacing(2)
        self._spins = {}
        for row, (key, label, _default) in enumerate(RANGE_KEYS):
            grid.addWidget(QLabel(label + ":"), row, 0)
            lo = QDoubleSpinBox(); hi = QDoubleSpinBox()
            for sp in (lo, hi):
                sp.setDecimals(4 if key == "score" else 1)
                sp.setRange(-1e9, 1e9)
                sp.setSingleStep(1.0)
                sp.setKeyboardTracking(False)
                sp.valueChanged.connect(self._on_value_changed)
                sp.setToolTip(_NO_LIMIT_TIP)
            grid.addWidget(lo, row, 1)
            grid.addWidget(QLabel("~"), row, 2)
            grid.addWidget(hi, row, 3)
            if key == "angle":
                tip = ("角度归一到 -45~45：0 = 横平竖直，±45 = 45° 斜；\n"
                       "上下限都填 -180 表示不看角度")
                lo.setToolTip(tip); hi.setToolTip(tip)
            elif key == "width":
                tip = "宽度/高度按**外接矩形**算（和结果表里的宽度/高度同一口径）"
                lo.setToolTip(tip); hi.setToolTip(tip)
            elif key == "score":
                tip = "分数 = 连通域内的平均置信度；\n当前推理后端没有概率图时会自动跳过这一项（日志里会写明）"
                lo.setToolTip(tip); hi.setToolTip(tip)
            self._spins[key] = (lo, hi)
        root.addLayout(grid)

        # ── 全局后处理 ──
        g = QGridLayout(); g.setHorizontalSpacing(6); g.setVerticalSpacing(2)
        g.addWidget(QLabel("孔洞填充:"), 0, 0)
        self.hole_spin = QSpinBox(); self.hole_spin.setRange(0, 1000000)
        self.hole_spin.setToolTip("填掉面积 ≤ N 的内部空洞（0=关）")
        self.hole_spin.setKeyboardTracking(False)
        self.hole_spin.valueChanged.connect(self._on_value_changed)
        g.addWidget(self.hole_spin, 0, 1)
        g.addWidget(QLabel("中心半径:"), 0, 2)
        self.radius_spin = QSpinBox(); self.radius_spin.setRange(0, 100000)
        self.radius_spin.setToolTip("等效半径 sqrt(面积/π) 的下限（0=关）")
        self.radius_spin.setKeyboardTracking(False)
        self.radius_spin.valueChanged.connect(self._on_value_changed)
        g.addWidget(self.radius_spin, 0, 3)
        g.addWidget(QLabel("排序:"), 1, 0)
        self.sort_combo = QComboBox(); self.sort_combo.addItems(SORT_CHOICES)
        self.sort_combo.currentTextChanged.connect(self._on_value_changed)
        g.addWidget(self.sort_combo, 1, 1)
        g.addWidget(QLabel("最大检测个数:"), 1, 2)
        self.maxdet_spin = QSpinBox(); self.maxdet_spin.setRange(1, 99999)
        self.maxdet_spin.setKeyboardTracking(False)
        self.maxdet_spin.setToolTip("每类最多保留多少个连通域")
        self.maxdet_spin.valueChanged.connect(self._on_value_changed)
        g.addWidget(self.maxdet_spin, 1, 3)
        root.addLayout(g)

        bottom = QHBoxLayout(); bottom.setSpacing(4)
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.clicked.connect(self._reset_current)
        bottom.addWidget(self.reset_btn)
        self.hint = QLabel("上限填负数=不限")
        self.hint.setStyleSheet("color:#8A93A6; font-size:10px;")
        bottom.addWidget(self.hint, 1)
        root.addLayout(bottom)

        self.setEnabled(False)

    # ────────── 对外 ──────────
    def set_project(self, project_dir, classes=None):
        """打开工程时调用：载入该工程的推理参数。"""
        self._project_dir = project_dir
        self._classes = [c for c in (classes or []) if c != "background"]
        self._params = load_params(project_dir, classes) if project_dir else None
        self._loading = True
        try:
            self.cls_combo.clear()
            self.cls_combo.addItems(self._classes)
            if self._classes:
                self.cls_combo.setCurrentIndex(0)
                self._current = self._classes[0]
                self._load_class_into_ui(self._current)
        finally:
            self._loading = False
        self.setEnabled(bool(self._classes))
        if not self._classes:
            self.hint.setText("该工程还没有类别（先导入标注 JSON）")
        else:
            self.hint.setText("上限填负数=不限")

    def current_params(self):
        return self._params

    # ────────── 内部 ──────────
    def _cls_params(self, name):
        if self._params is None:
            self._params = load_params(self._project_dir or "", self._classes)
        return self._params["classes"].setdefault(name, default_class_params())

    def _load_class_into_ui(self, name):
        rng = self._cls_params(name)
        self._loading = True
        try:
            for key, _label, _d in RANGE_KEYS:
                lo, hi = self._spins[key]
                v = rng.get(key) or list(default_class_params()[key])
                lo.setValue(float(v[0])); hi.setValue(float(v[1]))
            g = self._params.get("global") or {}
            self.hole_spin.setValue(int(g.get("hole_fill", 0) or 0))
            self.radius_spin.setValue(int(g.get("center_radius", 0) or 0))
            idx = self.sort_combo.findText(str(g.get("sort", "面积降序")))
            self.sort_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.maxdet_spin.setValue(int(g.get("max_det", 99999) or 99999))
        finally:
            self._loading = False

    def _on_class_changed(self, name):
        if self._loading or not name:
            return
        self._current = name
        self._load_class_into_ui(name)

    def _on_value_changed(self, *_a):
        if self._loading or self._params is None or not self._current:
            return
        rng = self._cls_params(self._current)
        for key, _label, _d in RANGE_KEYS:
            lo, hi = self._spins[key]
            rng[key] = [lo.value(), hi.value()]
        g = self._params.setdefault("global", {})
        g["hole_fill"] = self.hole_spin.value()
        g["center_radius"] = self.radius_spin.value()
        g["sort"] = self.sort_combo.currentText()
        g["max_det"] = self.maxdet_spin.value()
        self._save_timer.start()
        self.params_changed.emit()

    def _apply_all(self):
        if self._params is None or not self._current:
            return
        rng = dict(self._cls_params(self._current))
        for c in self._classes:
            self._params["classes"][c] = {k: list(v) for k, v in rng.items()}
        self._save_timer.start()
        self.params_changed.emit()

    def _reset_current(self):
        if self._params is None or not self._current:
            return
        self._params["classes"][self._current] = default_class_params()
        self._load_class_into_ui(self._current)
        self._save_timer.start()
        self.params_changed.emit()

    def _flush_save(self):
        if self._project_dir and self._params is not None:
            save_params(self._project_dir, self._params)
