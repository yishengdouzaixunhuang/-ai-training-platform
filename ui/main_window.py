"""Main Window - Optimized for large datasets"""
import os, json, re, sys, threading, shutil, cv2
import torch
from datetime import datetime
import time
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QButtonGroup,
    QMainWindow, QMenu, QAction, QSplitter, QTreeWidget, QTreeWidgetItem,
    QTextEdit, QFileDialog, QMessageBox, QInputDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QComboBox, QSpinBox, QProgressBar, QGroupBox,
    QListWidget, QListWidgetItem, QSlider, QDialog, QFormLayout, QDialogButtonBox,
    QLineEdit, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView
,
    QStackedWidget
)
from PyQt5.QtGui import QFont, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from core.config import APP_NAME, APP_VERSION, load_config, save_config
from core.project_manager import ProjectManager
from annotation.label_manager import LabelManager
from annotation.mask_editor import AnnotationCanvas
from annotation.labelme_io import save_mask_to_json, save_labelme_json, mask_to_shapes
from annotation.version_manager import list_versions, restore_version, get_version_diff
from training.trainer import Trainer
from training.dataset import get_train_test_split, save_train_test_split
from inference.predictor import Predictor
print("main_window imports OK")
class TrainSettingsDialog(QDialog):
    SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "AI_Training_Platform", ".train_settings.json")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Settings")
        layout = QFormLayout(self)
        saved = self._load()
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 2000); self.epochs_spin.setValue(saved.get("epochs", 200))
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 64); self.batch_spin.setValue(saved.get("batch_size", 8))
        self.model_combo = QComboBox(); self.model_combo.addItems(["deeplabv3", "deeplabv3_r101", "deeplabv3_mobilenet", "fcn", "fcn_r101", "lraspp"])
        idx = self.model_combo.findText(saved.get("model", "deeplabv3")); self.model_combo.setCurrentIndex(max(0, idx))
        self.size_combo = QComboBox(); self.size_combo.addItems(["256", "512", "768", "1024"])
        idx2 = self.size_combo.findText(str(saved.get("image_size", 512))); self.size_combo.setCurrentIndex(max(0, idx2))
        layout.addRow("Epochs:", self.epochs_spin)
        layout.addRow("Batch Size:", self.batch_spin)
        layout.addRow("Model:", self.model_combo)
        layout.addRow("Image Size:", self.size_combo)
        self.kfold_spin = QSpinBox(); self.kfold_spin.setRange(1, 10); self.kfold_spin.setValue(saved.get("k_folds", 1))
        self.kfold_spin.setToolTip("1=no cross-validation, 5=5-fold CV")
        layout.addRow("K-Fold CV:", self.kfold_spin)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["cross_entropy", "lovasz", "dice", "focal", "ce+lovasz", "ce+dice"])
        idx3 = self.loss_combo.findText(saved.get("loss", "cross_entropy"))
        self.loss_combo.setCurrentIndex(max(0, idx3))
        self.loss_combo.setToolTip("Loss function: ce+lovasz recommended for small defects")
        layout.addRow("Loss:", self.loss_combo)
        self.augment_combo = QComboBox()
        self.augment_combo.addItems(["none", "light", "medium", "strong"])
        idx4 = self.augment_combo.findText(saved.get("augment", "none"))
        self.augment_combo.setCurrentIndex(max(0, idx4))
        self.augment_combo.setToolTip("Data augmentation intensity")
        layout.addRow("Augment:", self.augment_combo)
        self.resume_check = QCheckBox("Resume from last checkpoint")
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
            "augment": self.augment_combo.currentText(),
            "resume": self.resume_check.isChecked(),
        }

    def _save_and_accept(self):
        data = {
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "model": self.model_combo.currentText(),
            "image_size": int(self.size_combo.currentText()),
            "k_folds": self.kfold_spin.value(),
            "loss": self.loss_combo.currentText(),
            "augment": self.augment_combo.currentText(),
            "resume": self.resume_check.isChecked(),
        }
        try:
            os.makedirs(os.path.dirname(self.SETTINGS_FILE), exist_ok=True)
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        self.accept()

class MainWindow(QMainWindow):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    image_list_signal = pyqtSignal(object)
    infer_progress_signal = pyqtSignal(int, int)
    stats_signal = pyqtSignal(object, object)
    chart_signal = pyqtSignal(dict)
    batch_signal = pyqtSignal(int, int, float)
    refresh_list_signal = pyqtSignal()
    image_load_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1400, 850)
        self.config = load_config()
        self.pm = ProjectManager(self.config["workspace"])
        self.current_project = None
        self._project_dir = None
        self._cached_classes = ["background"]
        self.trainer = None
        self._stop_flag = False
        self._project_gen = 0
        self._ignore_class_row = -1
        self._all_images = []
        self._saved_left = 200
        self._saved_right = 250
        self.log_signal.connect(self._append_log)
        self.progress_signal.connect(self._update_progress)
        self.image_list_signal.connect(self._populate_image_list)
        self.infer_progress_signal.connect(self._update_infer_progress)
        self.stats_signal.connect(self._update_stats)
        self.stats_signal.connect(lambda a, p: self._update_ann_stats())
        self._stats_timer = QTimer()
        self._stats_timer.setSingleShot(True)
        self._stats_timer.timeout.connect(self._do_update_stats)
        self._detection_mode = False
        self._box_manager = None
        self.chart_signal.connect(self._update_loss_chart)
        self.batch_signal.connect(self._update_batch_status)
        self.refresh_list_signal.connect(lambda: self._filter_images(self.search_input.text()))
        self._init_ui()
        self._refresh_projects()

    def _init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setContentsMargins(4, 4, 4, 4)
        self._create_menus()
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._left_panel = self._create_left_panel()
        self._center_panel = self._create_center_panel()
        self._right_panel = self._create_right_panel()
        self._main_splitter.addWidget(self._left_panel)
        self._main_splitter.addWidget(self._center_panel)
        self._main_splitter.addWidget(self._right_panel)
        self._main_splitter.setSizes([200, 850, 250])
        self._main_splitter.setStretchFactor(0, 0)  # left: no stretch
        self._main_splitter.setStretchFactor(1, 1)  # canvas: stretches
        self._main_splitter.setStretchFactor(2, 0)  # right: no stretch

        # Log panel in a vertical splitter so it's resizable
        self.log_widget = QTextEdit(); self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Consolas", 9))
        self.log_widget.setMinimumHeight(0)
        self._log_splitter = QSplitter(Qt.Vertical)
        self._log_splitter.addWidget(self._main_splitter)
        self._log_splitter.addWidget(self.log_widget)
        self._log_splitter.setSizes([700, 150])
        self._log_splitter.setStretchFactor(0, 1)  # main content stretches
        self._log_splitter.setStretchFactor(1, 0)  # log: fixed/minimum
        self._log_splitter.setCollapsible(1, True)
        self._saved_log = 150
        main_layout.addWidget(self._log_splitter)
        self.log(f"{APP_NAME} v{APP_VERSION} started")

    def _create_menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("File(&F)")
        fm.addAction("New Project(&N)", self._new_project)
        fm.addAction("Open Project(&O)", self._open_project)
        fm.addSeparator()
        fm.addAction("Close Project", self._close_project)
        fm.addSeparator()
        fm.addAction("Restart App", self._restart_app)
        fm.addAction("Exit(&Q)", self.close)
        tm = mb.addMenu("Tools(&T)")
        tm.addAction("Semantic Segmentation", lambda: self._set_task("segmentation"))
        tm.addAction("Object Detection", lambda: self._set_task("detection"))
        sm = mb.addMenu("Settings(&S)")
        sm.addAction("Workspace...", self._set_workspace)
        vm = mb.addMenu("View(&V)")
        vm.addAction("Toggle Log Panel	Ctrl+L", self._toggle_log_panel)
        vm.addAction("Toggle Left Panel	Ctrl+[", lambda: self._toggle_panel("left"))
        vm.addAction("Toggle Right Panel	Ctrl+]", lambda: self._toggle_panel("right"))
        vm.addSeparator()
        vm.addAction("Toggle Detection Mode	Ctrl+D", self._toggle_detection_mode)
        vm.addAction("Refresh Projects", self._refresh_projects)
        im = mb.addMenu("Image Mgmt")
        im.addAction("Import Images...", self._import_images)
        am = mb.addMenu("Annotation Mgmt")
        am.addAction("Manage Classes...", self._manage_classes)
        am.addAction("Import JSON...", self._import_json)
        am.addAction("Batch Import JSON...", self._batch_import_json)
        am.addAction("Remove Tiny Annotations (<10k px)...", self._clean_tiny_ann)
        am.addAction("Export JSON...", self._export_json)
        am.addSeparator()
        am.addAction("Export Annotations", self._export_annotations)
        mm = mb.addMenu("Model Mgmt")
        mm.addAction("Manage Train/Val/Test Split...", self._manage_split)
        mm.addAction("Start Seg Training...", self._start_training)
        mm.addAction("Start Det Training", self._start_det_training)
        mm.addAction("Run Inference", self._start_inference)
        mm.addAction("Batch Inference (All Images)...", self._batch_inference)
        hm = mb.addMenu("Help(&H)")
        hm.addAction("About(&A)", self._about)

    def _create_left_panel(self):
        p = QWidget(); outer_layout = QVBoxLayout(p); outer_layout.setContentsMargins(2, 2, 2, 2)
        hh = QHBoxLayout()
        hh.addWidget(QLabel("<b>Projects</b>"))
        hh.addStretch()
        btn = QPushButton("鈼€"); btn.setFixedSize(22, 22); btn.setToolTip("Collapse left panel")
        btn.clicked.connect(lambda: self._toggle_panel("left"))
        hh.addWidget(btn)
        outer_layout.addLayout(hh)

        # Resizable splitter for project tree and stat tables
        self._left_splitter = QSplitter(Qt.Vertical)
        self._left_splitter.setChildrenCollapsible(False)

        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderLabels(["Name", "Time"])
        self.project_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.project_tree.customContextMenuRequested.connect(self._on_project_context_menu)
        self.project_tree.itemDoubleClicked.connect(self._on_project_double_click)
        self.week_item = QTreeWidgetItem(["Within 1 week"])
        self.month_item = QTreeWidgetItem(["Within 1 month"])
        self.older_item = QTreeWidgetItem(["Older"])
        self.project_tree.addTopLevelItems([self.week_item, self.month_item, self.older_item])
        self._left_splitter.addWidget(self.project_tree)

        btn_new = QPushButton("+ New Project"); btn_new.clicked.connect(self._new_project)

        # Result blocks section
        res_widget = QWidget(); res_layout = QVBoxLayout(res_widget); res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.addWidget(QLabel("<b>Result Blocks</b>"))
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(8)
        self.stats_table.verticalHeader().setDefaultSectionSize(24)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setAlternatingRowColors(True)
        res_layout.addWidget(self.stats_table)
        self._left_splitter.addWidget(res_widget)

        # Annotation blocks section
        ann_widget = QWidget(); ann_layout = QVBoxLayout(ann_widget); ann_layout.setContentsMargins(0, 0, 0, 0)
        ann_layout.addWidget(QLabel("<b>Annotation Blocks</b>"))
        self.ann_table = QTableWidget()
        self.ann_table.setColumnCount(8)
        self.ann_table.verticalHeader().setDefaultSectionSize(24)
        self.ann_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ann_table.setAlternatingRowColors(True)
        ann_layout.addWidget(self.ann_table)
        self._left_splitter.addWidget(ann_widget)

        self._left_splitter.setSizes([300, 250, 200])
        outer_layout.addWidget(self._left_splitter)
        outer_layout.addWidget(btn_new)
        return p

    def _create_center_panel(self):
        outer = QSplitter(Qt.Vertical)
        outer.setCollapsible(0, False)  # canvas area stays visible
        outer.setCollapsible(1, True)   # image list can collapse
        # Canvas area
        cc = QWidget(); cl = QVBoxLayout(cc); cl.setContentsMargins(0, 0, 0, 0)
        tb = QHBoxLayout(); tb.setContentsMargins(0, 0, 0, 2)
        hider = QPushButton("_")
        hider.setFixedSize(20, 18)
        hider.setToolTip("Hide header")
        hider.clicked.connect(lambda: self._toggle_canvas_header())
        tb.addWidget(hider)
        tb.addWidget(QLabel("<b>Annot</b>")); tb.addStretch()
        self.image_count_label = QLabel(""); tb.addWidget(self.image_count_label)
        self._canvas_header = tb
        self._canvas_header_visible = True
        cl.addLayout(tb)
        self.canvas = AnnotationCanvas()
        self.canvas.status_message.connect(self.statusBar().showMessage)
        self._version_refresh_timer = QTimer()
        self._version_refresh_timer.setSingleShot(True)
        self._version_refresh_timer.setInterval(1500)
        self._version_refresh_timer.timeout.connect(self._refresh_versions)
        self.canvas.mask_changed.connect(lambda: self._version_refresh_timer.start())
        self.canvas.det_boxes_changed.connect(self._on_det_boxes_changed)
        cl.addWidget(self.canvas)
        outer.addWidget(cc)
        self._center_splitter = outer  # for toggling
        # Image list area (collapsible)
        lc = QWidget(); ll = QVBoxLayout(lc); ll.setContentsMargins(2, 2, 2, 2)
        hh = QHBoxLayout()
        btn_imglist = QPushButton("Hide List")
        btn_imglist.setFixedWidth(70)
        btn_imglist.clicked.connect(lambda: self._toggle_image_list())
        hh.addWidget(btn_imglist)
        hh.addWidget(QLabel("<b>Image List</b>"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search name or class...")
        self.search_input.textChanged.connect(self._filter_images)
        hh.addWidget(self.search_input)
        self.split_filter = QComboBox()
        self.split_filter.addItems(["All Sets", "Train Only", "Val Only", "Test Only"])
        self.split_filter.currentIndexChanged.connect(lambda: self._filter_images(self.search_input.text()))
        hh.addWidget(self.split_filter)
        self.annotated_check = QCheckBox("Annotated")
        self.annotated_check.setChecked(False)
        self.annotated_check.toggled.connect(lambda: self._filter_images(self.search_input.text()))
        hh.addWidget(self.annotated_check)
        self.recent_check = QCheckBox("Recent")
        self.recent_check.setToolTip("Sort by annotation date (newest first)")
        self.recent_check.toggled.connect(lambda: self._filter_images(self.search_input.text()))
        hh.addWidget(self.recent_check)
        # Prediction count filter
        hh2 = QHBoxLayout()
        hh2.addWidget(QLabel("Pred:"))
        self.pred_op_combo = QComboBox()
        self.pred_op_combo.addItems(["\u2265", "=", "\u2264"])
        self.pred_op_combo.setToolTip("Filter by prediction block count")
        self.pred_op_combo.currentIndexChanged.connect(lambda: self._filter_images(self.search_input.text()))
        hh2.addWidget(self.pred_op_combo)
        self.pred_count_spin = QSpinBox()
        self.pred_count_spin.setRange(0, 9999)
        self.pred_count_spin.setValue(0)
        self.pred_count_spin.setToolTip("0 = disabled. Filter images by number of prediction blocks.")
        self.pred_count_spin.valueChanged.connect(lambda: self._filter_images(self.search_input.text()))
        hh2.addWidget(self.pred_count_spin)
        ll.addLayout(hh2)
        ll.addLayout(hh)
        self.image_list_widget = QTableWidget()
        self.image_list_widget.setColumnCount(8)
        self.image_list_widget.setHorizontalHeaderLabels(["序号", "图片名", "大小", "数据集", "标注", "结果", "分数", "耗时"])
        self.image_list_widget.horizontalHeader().setStretchLastSection(True)
        self.image_list_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.image_list_widget.setSelectionMode(QTableWidget.ExtendedSelection)
        self.image_list_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.image_list_widget.setAlternatingRowColors(True)
        self.image_list_widget.verticalHeader().setVisible(False)
        self.image_list_widget.verticalHeader().setDefaultSectionSize(22)
        self.image_list_widget.setMaximumHeight(250)
        self.image_list_widget.currentCellChanged.connect(self._on_current_cell_changed)
        self.image_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list_widget.customContextMenuRequested.connect(self._on_image_context_menu)
        # Column widths
        self.image_list_widget.setColumnWidth(0, 40)
        self.image_list_widget.setColumnWidth(1, 200)
        self.image_list_widget.setColumnWidth(2, 100)
        self.image_list_widget.setColumnWidth(3, 60)
        self.image_list_widget.setColumnWidth(4, 50)
        self.image_list_widget.setColumnWidth(5, 50)
        self.image_list_widget.setColumnWidth(6, 60)
        ll.addWidget(self.image_list_widget)
        self.list_status = QLabel(""); ll.addWidget(self.list_status)
        outer.addWidget(lc)
        outer.setSizes([600, 250])
        return outer

    def _create_right_panel(self):
        p = QWidget()
        outer_layout = QVBoxLayout(p)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Top row: collapse button
        hh = QHBoxLayout()
        hh.setContentsMargins(2, 2, 2, 2)
        hh.addStretch()
        btn_collapse = QPushButton("\u25b6"); btn_collapse.setFixedSize(22, 22); btn_collapse.setToolTip("Collapse right panel")
        btn_collapse.clicked.connect(lambda: self._toggle_panel("right"))
        hh.addWidget(btn_collapse)
        outer_layout.addLayout(hh)

        # Horizontal layout: content (left) + tab bar (right)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(2, 0, 0, 0)
        content_layout.setSpacing(0)

        # ===== Stacked content panels =====
        self._right_stack = QStackedWidget()

        # Panel 0: Annotation Tools
        panel_annot = QWidget(); al = QVBoxLayout(panel_annot); al.setContentsMargins(4, 2, 2, 2)
        al.addWidget(QLabel("<b>Class:</b>"))
        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self._on_class_changed)
        al.addWidget(self.class_list)
        bl = QHBoxLayout(); bl.addWidget(QLabel("Brush:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(2, 100); self.brush_slider.setValue(15)
        self.brush_slider.valueChanged.connect(lambda v: setattr(self.canvas, "brush_size", v))
        bl.addWidget(self.brush_slider); al.addLayout(bl)
        al.addWidget(QLabel("<b>Mode:</b>"))
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self); self.mode_group.setExclusive(True)
        modes = [("Brush","brush"),("Polygon","polygon"),("Line","line"),("Eraser","eraser"),("Pan","pan")]
        self._mode_buttons = {}
        for label, mode_id in modes:
            btn = QPushButton(label); btn.setCheckable(True); btn.setMinimumWidth(56)
            def make_callback(m): return lambda: self._set_annotation_mode(m)
            btn.clicked.connect(make_callback(mode_id))
            if mode_id == "pan": btn.setChecked(True)
            self.mode_group.addButton(btn); self._mode_buttons[mode_id] = btn
            mode_layout.addWidget(btn)
        al.addLayout(mode_layout)
        # Detection mode toggle button
        self.det_btn = QPushButton("Detection Mode (Ctrl+D)")
        self.det_btn.setCheckable(True)
        self.det_btn.clicked.connect(self._toggle_detection_mode)
        self.det_btn.setStyleSheet("QPushButton { padding: 6px; } QPushButton:checked { background-color: #2d6a4f; color: white; }")
        al.addWidget(self.det_btn)
        al.addWidget(QPushButton("Reset View (Fit)", clicked=lambda: self.canvas.reset_view()))
        al.addWidget(QPushButton("Save Annotation", clicked=self._save_current_mask))
        al.addWidget(QPushButton("Clear Annotation", clicked=self._clear_current_mask))
        nl = QHBoxLayout()
        nl.addWidget(QPushButton("< Prev", clicked=self._prev_image))
        nl.addWidget(QPushButton("Next >", clicked=self._next_image))
        al.addLayout(nl)
        al.addStretch()
        self._right_stack.addWidget(panel_annot)  # 0

        # Panel 1: Version History
        panel_ver = QWidget(); vl = QVBoxLayout(panel_ver); vl.setContentsMargins(4, 2, 2, 2)
        vl.addWidget(QLabel("<b>Version History</b>"))
        self.version_list = QListWidget()
        self.version_list.setToolTip("Double-click to restore version")
        self.version_list.itemDoubleClicked.connect(self._restore_version)
        vl.addWidget(self.version_list)
        vbl = QHBoxLayout()
        vbl.addWidget(QPushButton("Diff", clicked=self._diff_version))
        vbl.addWidget(QPushButton("Restore", clicked=self._slot_restore_version))
        self.version_count_label = QLabel("0 versions"); vbl.addWidget(self.version_count_label)
        vl.addLayout(vbl)
        self._right_stack.addWidget(panel_ver)  # 1

        # Panel 2: Inference
        panel_infer = QWidget(); il = QVBoxLayout(panel_infer); il.setContentsMargins(4, 2, 2, 2)
        il.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox(); self.model_combo.addItems(["best_model.pth", "last_model.pth"])
        self.backend_combo = QComboBox(); self.backend_combo.addItems(["PyTorch", "Compiled", "TorchScript", "TensorRT (FP16)", "ONNX (CPU fallback)"])
        self.backend_combo.setToolTip("Inference backend")
        self.tiled_check = QCheckBox("Tiled")
        saved_tiled2 = self._load_ui_setting("tiled", True)
        self.tiled_check.setChecked(saved_tiled2)
        self.tiled_check.toggled.connect(lambda v: self._save_ui_setting("tiled", v))
        sl = QHBoxLayout(); sl.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Horizontal); self.scale_slider.setRange(25, 100)
        saved_scale = self._load_ui_setting("scale", 100)
        self.scale_slider.setValue(saved_scale)
        self.scale_slider.valueChanged.connect(lambda v: self._save_ui_setting("scale", v))
        self.scale_label = QLabel("1.00x")
        self.scale_slider.valueChanged.connect(lambda v: self.scale_label.setText(f"{v/100:.2f}x"))
        sl.addWidget(self.scale_slider); sl.addWidget(self.scale_label)
        il.addLayout(sl)
        il.addWidget(self.model_combo); il.addWidget(self.backend_combo); il.addWidget(self.tiled_check)
        il.addWidget(QPushButton("Run Inference", clicked=self._start_inference))
        self.infer_progress = QProgressBar(); il.addWidget(self.infer_progress)
        self.infer_status = QLabel(); il.addWidget(self.infer_status)
        hil = QHBoxLayout()
        self.btn_infer_stop = QPushButton("Stop"); self.btn_infer_stop.clicked.connect(self._stop_inference)
        self.btn_infer_stop.setEnabled(False); hil.addWidget(self.btn_infer_stop)
        self.infer_result_label = QLabel(""); hil.addWidget(self.infer_result_label)
        il.addLayout(hil); il.addStretch()
        self._right_stack.addWidget(panel_infer)  # 2

        # Panel 3: Training
        panel_train = QWidget(); tl = QVBoxLayout(panel_train); tl.setContentsMargins(4, 2, 2, 2)
        tl.addWidget(QLabel("<b>Training</b>"))
        self.train_progress = QProgressBar(); tl.addWidget(self.train_progress)
        self.train_status = QLabel("Ready"); tl.addWidget(self.train_status)
        self.btn_train_stop = QPushButton("Stop"); self.btn_train_stop.clicked.connect(self._stop_batch)
        self.btn_train_stop.setEnabled(False); tl.addWidget(self.btn_train_stop)
        tl.addStretch()
        fl = QFormLayout()
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 5000); self.epochs_spin.setValue(200)
        fl.addRow("Epochs:", self.epochs_spin)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 128); self.batch_spin.setValue(8)
        fl.addRow("Batch:", self.batch_spin)
        self.kfold_spin = QSpinBox(); self.kfold_spin.setRange(1, 10); self.kfold_spin.setValue(1)
        self.kfold_spin.setToolTip("1 = no cross-validation")
        fl.addRow("K-Fold CV:", self.kfold_spin)
        self.resume_check = QCheckBox("Resume from last")
        fl.addRow("", self.resume_check)
        tl.addLayout(fl)
        self._right_stack.addWidget(panel_train)  # 3

        # Panel 4: Loss Curve
        panel_loss = QWidget(); cl = QVBoxLayout(panel_loss); cl.setContentsMargins(4, 2, 2, 2)
        cl.addWidget(QLabel("<b>Loss Curve</b>"))
        self.loss_fig = Figure(figsize=(3, 2.5), dpi=100)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_title("Training Progress", fontsize=9)
        self.loss_ax.set_xlabel("Epoch", fontsize=7)
        self.loss_ax.set_ylabel("Loss", fontsize=7)
        self.loss_canvas = FigureCanvas(self.loss_fig)
        self.loss_canvas.setToolTip("Loss curve (click/drag to zoom, scroll to pan)")
        cl.addWidget(self.loss_canvas)
        self._right_stack.addWidget(panel_loss)  # 4

        # Panel 5: Detection Tools
        panel_det = QWidget(); dl = QVBoxLayout(panel_det); dl.setContentsMargins(4, 2, 2, 2)
        dl.addWidget(QLabel("<b>Detection Tools</b>"))
        dl.addWidget(QLabel("Class:"))
        self.det_class_list = QListWidget()
        self.det_class_list.currentRowChanged.connect(self._on_det_class_changed)
        dl.addWidget(self.det_class_list)
        btl = QHBoxLayout()
        self.det_hbb_btn = QPushButton("HBB"); self.det_hbb_btn.setCheckable(True); self.det_hbb_btn.setChecked(True)
        self.det_hbb_btn.clicked.connect(lambda: self._set_det_bbox_type("hbb"))
        self.det_obb_btn = QPushButton("OBB"); self.det_obb_btn.setCheckable(True)
        self.det_obb_btn.clicked.connect(lambda: self._set_det_bbox_type("obb"))
        btl.addWidget(QLabel("Box:")); btl.addWidget(self.det_hbb_btn); btl.addWidget(self.det_obb_btn)
        dl.addLayout(btl)
        dl.addWidget(QPushButton("Delete Selected Box (Del)", clicked=self._det_delete_box))
        dl.addWidget(QPushButton("Delete All Boxes", clicked=self._det_clear_all))
        dl.addStretch()
        self._right_stack.addWidget(panel_det)  # 5

        # Panel 6: Detection Training
        panel_det_train = QWidget(); dtl = QVBoxLayout(panel_det_train); dtl.setContentsMargins(4, 2, 2, 2)
        dtl.addWidget(QLabel("<b>Detection Training</b>"))
        self.det_train_progress = QProgressBar(); dtl.addWidget(self.det_train_progress)
        self.det_train_status = QLabel("Ready"); dtl.addWidget(self.det_train_status)
        self.btn_det_train_stop = QPushButton("Stop"); self.btn_det_train_stop.clicked.connect(self._stop_training)
        self.btn_det_train_stop.setEnabled(False); dtl.addWidget(self.btn_det_train_stop)
        dtl.addStretch()
        dfl = QFormLayout()
        self.det_model_combo = QComboBox()
        self.det_model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self.det_model_combo.setToolTip("YOLOv8 model variant (s recommended for balance)")
        dfl.addRow("Model:", self.det_model_combo)
        self.det_epochs_spin = QSpinBox(); self.det_epochs_spin.setRange(1, 2000); self.det_epochs_spin.setValue(200)
        dfl.addRow("Epochs:", self.det_epochs_spin)
        self.det_batch_spin = QSpinBox(); self.det_batch_spin.setRange(1, 128); self.det_batch_spin.setValue(16)
        dfl.addRow("Batch:", self.det_batch_spin)
        self.det_img_size_spin = QSpinBox(); self.det_img_size_spin.setRange(320, 1280); self.det_img_size_spin.setSingleStep(32)
        self.det_img_size_spin.setValue(640); dfl.addRow("Image Size:", self.det_img_size_spin)
        dtl.addLayout(dfl)
        dtl.addWidget(QPushButton("Start Training", clicked=self._start_det_training))
        dtl.addStretch()
        self._right_stack.addWidget(panel_det_train)  # 6

        content_layout.addWidget(self._right_stack)

        # ===== Vertical tab bar =====
        tab_bar = QWidget(); tab_bar.setFixedWidth(36)
        tab_layout = QVBoxLayout(tab_bar); tab_layout.setContentsMargins(1, 2, 1, 2); tab_layout.setSpacing(2)

        self._right_tabs = QButtonGroup(self); self._right_tabs.setExclusive(True)
        tab_defs = [
            ("An", "Annotation Tools", 0),
            ("Ve", "Version History", 1),
            ("In", "Inference", 2),
            ("Tr", "Training", 3),
            ("Lo", "Loss Curve", 4),
            ("De", "Detection Tools", 5),
            ("DT", "Det Training", 6),
        ]
        for icon_text, tooltip, idx in tab_defs:
            btn = QPushButton(icon_text)
            btn.setCheckable(True); btn.setFixedSize(32, 32)
            btn.setToolTip(tooltip)
            btn.setStyleSheet("QPushButton { font-size: 12px; font-weight: bold; padding: 2px; } QPushButton:checked { background-color: #2C5F8A; color: white; border-radius: 3px; }")
            btn.clicked.connect(lambda checked, i=idx: self._right_stack.setCurrentIndex(i))
            self._right_tabs.addButton(btn, idx)
            tab_layout.addWidget(btn)
        tab_layout.addStretch()
        content_layout.addWidget(tab_bar)

        outer_layout.addLayout(content_layout)

        # Default: show Annotation Tools
        self._right_stack.setCurrentIndex(0)
        self._right_tabs.button(0).setChecked(True)

        return p
    # ============ Logging & Progress ============
    def log(self, msg):
        self.log_signal.emit(msg)

    def _append_log(self, msg):
        if msg == "__DET_TRAINING_DONE__":
            self._on_det_training_done()
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_widget.append(f"[{ts}] {msg}")

    def _update_loss_chart(self, history):
        """Update loss curve from training thread."""
        try:
            self.loss_ax.clear()
            train_loss = history.get("train_loss", [])
            val_loss = history.get("val_loss", [])
            epochs = range(1, len(train_loss) + 1)
            if len(epochs) > 0:
                self.loss_ax.plot(epochs, train_loss, "b-", label="Train", linewidth=0.8, alpha=0.7)
                self.loss_ax.plot(epochs, val_loss, "r-", label="Val", linewidth=0.8, alpha=0.7)
                self.loss_ax.legend(fontsize=7)
                self.loss_ax.set_xlabel("Epoch", fontsize=7)
                self.loss_ax.set_ylabel("Loss", fontsize=7)
                self.loss_ax.tick_params(labelsize=6)
            self.loss_canvas.draw()
        except Exception:
            pass

    def _update_batch_status(self, batch, total, avg_loss):
        pct = batch * 100 // total
        self.train_status.setText(f"Training... {pct}% | batch {batch}/{total} | loss={avg_loss:.4f}")

    def _update_progress(self, cur, total):
        self.train_progress.setMaximum(total)
        self.train_progress.setValue(cur)
        self.train_status.setText(f"Epoch {cur}/{total}")

    def _update_infer_progress(self, cur, total):
        self.infer_progress.setMaximum(total)
        self.infer_progress.setValue(cur)
        self.infer_status.setText(f"Inferring... {cur}/{total}")

    # ============ Project Management ============
    def _refresh_projects(self):
        projects = self.pm.list_projects()
        now = datetime.now().timestamp()
        self.week_item.takeChildren()
        self.month_item.takeChildren()
        self.older_item.takeChildren()
        for p in projects:
            age = (now - p["modified"]) / 86400
            item = QTreeWidgetItem([p["name"], f"{int(age)}d ago"])
            item.setData(0, Qt.UserRole, p["name"])
            if age <= 7:
                self.week_item.addChild(item)
            elif age <= 30:
                self.month_item.addChild(item)
            else:
                self.older_item.addChild(item)
        self.project_tree.expandAll()

    def _on_project_context_menu(self, pos):
        item = self.project_tree.itemAt(pos)
        if item is None:
            return
        name = item.data(0, Qt.UserRole)
        if not name:
            return
        menu = QMenu(self)
        open_action = menu.addAction("Open in Explorer")
        open_action.triggered.connect(lambda: os.startfile(str(self.pm.get_project_dir(name))))
        menu.exec_(self.project_tree.viewport().mapToGlobal(pos))

    def _on_project_double_click(self, item):
        name = item.data(0, Qt.UserRole)
        if name:
            self._open_project_by_name(name)

    def _new_project(self):
        from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QLabel
        dlg = QDialog(self)
        dlg.setWindowTitle("New Project")
        form = QFormLayout(dlg)
        name_edit = QLineEdit()
        form.addRow("Project Name:", name_edit)
        classes_edit = QLineEdit()
        classes_edit.setPlaceholderText("e.g. defect, scratch, dent (comma separated)")
        form.addRow("Classes:", classes_edit)
        form.addRow(QLabel("Leave empty to auto-detect from imported JSONs"))
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec_() != QDialog.Accepted:
            return
        name = name_edit.text().strip()
        if not name:
            return QMessageBox.warning(self, "Error", "Project name is required")
        try:
            self.pm.create_project(name, "semantic_segmentation")
            # Register classes
            class_text = classes_edit.text().strip()
            if class_text:
                lm = LabelManager(str(self.pm.get_project_dir(name)))
                for c in [x.strip() for x in class_text.split(",") if x.strip()]:
                    lm.add_class(c)
            self._open_project_by_name(name)
            self._refresh_projects()
            self.log(f"Created project: {name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    def _open_project(self):
        self._refresh_projects()

    def _open_project_by_name(self, name):
        try:
            self.current_project = self.pm.open_project(name)
            self._project_dir = str(self.pm.get_project_dir(name))
            self.canvas.set_project(self.pm.get_project_dir(name))
            self._load_image_list_async()
            self._load_classes()
            self.log(f"Opened: {name}")
            self.setWindowTitle(f"{APP_NAME} - {name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _close_project(self):
        self._project_gen += 1  # Invalidate any pending scan threads
        self.current_project = None
        self._project_dir = None
        self._all_images = []
        self.canvas.image = None
        self.canvas.mask = None
        self.canvas.display_pixmap = None
        self.canvas.prediction_overlay = None
        self.canvas.current_image_path = None
        self.canvas._overlay_dirty = True
        self.canvas.update()
        self.image_list_widget.setRowCount(0)
        self.class_list.clear()
        self.image_count_label.setText("")
        self.list_status.setText("")
        self.log("Project closed")
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")

    def _set_task(self, task):
        """Switch between segmentation and detection task modes."""
        if task == "detection":
            self._detection_mode = True
            self.log("Task: Object Detection")
            
            from detection.box_manager import BoxManager
            from detection.box_overlay import DetectionOverlay
            
            classes = getattr(self, "_cached_classes", None) or ["background"]
            self._box_manager = BoxManager(categories=classes)
            self.canvas._det_overlay = DetectionOverlay(self.canvas, self._box_manager)
            
            if "pan" in getattr(self, "_mode_buttons", {}):
                self._mode_buttons["pan"].setChecked(True)
                self.canvas._mode = self.canvas.MODE_PAN
            
            self._load_det_annotations()
            self._populate_det_class_list()
            self._right_stack.setCurrentIndex(5)
            self._right_tabs.button(5).setChecked(True) if self._right_tabs.button(5) else None
        else:
            self._detection_mode = False
            self.log("Task: Semantic Segmentation")
            self.canvas._det_overlay = None
            self.canvas.update()
            self._right_stack.setCurrentIndex(0)
            self._right_tabs.button(0).setChecked(True) if self._right_tabs.button(0) else None

    def _set_workspace(self):
        path = QFileDialog.getExistingDirectory(self, "Select Workspace")
        if path:
            self.config["workspace"] = path
            save_config(self.config)
            self.pm = ProjectManager(path)
            self._refresh_projects()
            self.log(f"Workspace: {path}")

    # ============ Image Management ============
    def _import_images(self):
        if not self.current_project:
            return QMessageBox.information(self, "Hint", "Please open a project first")
        files, _ = QFileDialog.getOpenFileNames(
            self, "Import Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if files:
            self.log(f"Importing {len(files)} images...")
            img_dir = os.path.join(
                self.pm.get_project_dir(self.current_project["name"]), "images")
            def copy_files():
                for i, f in enumerate(files):
                    dst = os.path.join(img_dir, os.path.basename(f))
                    if not os.path.exists(dst):
                        shutil.copy2(f, dst)
                    if (i + 1) % 100 == 0:
                        self.log(f"Import progress: {i+1}/{len(files)}")
                self.log(f"Import complete: {len(files)} images")
                self._load_image_list_async()
            threading.Thread(target=copy_files, daemon=True).start()

    def _load_image_list_async(self):
        if not self.current_project:
            return
        self.log("Scanning image list...")
        gen = self._project_gen
        def scan():
            img_dir = os.path.join(
                self.pm.get_project_dir(self.current_project["name"]), "images")
            images = []
            if os.path.exists(img_dir):
                for f in sorted(os.listdir(img_dir)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        images.append(os.path.join(img_dir, f))
            self.image_list_signal.emit((gen, images))
        threading.Thread(target=scan, daemon=True).start()

    def _populate_image_list(self, data):
        gen, image_paths = data
        if gen != self._project_gen:
            return
        self._all_images = image_paths
        self._suppress_cell_change = True
        self._filter_images("")
        self._suppress_cell_change = False
        self._detection_mode = False  # False=segmentation, True=detection
        self._box_manager = None  # BoxManager for detection annotations
        # Async load output stats (pred counts + IoU scores)
        if self.current_project:
            pd = str(self.pm.get_project_dir(self.current_project["name"]))
            self._pred_count_cache_key = None  # force reload
            threading.Thread(target=self._load_output_stats_async, args=(pd,), daemon=True).start()
        self.image_count_label.setText(f"Total: {len(image_paths)} images")
        self.log(f"Loaded {len(image_paths)} images in list")
        if image_paths:
            self._suppress_cell_change = True
            self.image_list_widget.selectRow(0)
            self._suppress_cell_change = False
            self._load_image_by_index(0)

    def _load_output_stats_async(self, project_dir):
        """Load prediction counts and IoU scores from outputs dir in background thread."""
        out_dir = os.path.join(project_dir, "outputs")
        ann_dir = os.path.join(project_dir, "annotations")
        if not os.path.isdir(out_dir):
            return
        try:
            import numpy as _np
            from detection.coco_io import load_coco_json
            pred_count = {}
            img_scores = {}
            for f in os.listdir(out_dir):
                # Segmentation: _pred.png masks
                if f.endswith("_pred.png"):
                    base = f.replace("_pred.png", "")
                    try:
                        mask = _np.array(Image.open(os.path.join(out_dir, f)))
                        unique = _np.unique(mask)
                        pred_count[base] = len([u for u in unique if u not in (0, 255)])
                        mp = os.path.join(ann_dir, base + "_mask.png")
                        if os.path.exists(mp):
                            ann = _np.array(Image.open(mp)).astype(_np.uint8)
                            ious = []
                            for c in unique:
                                if c in (0, 255):
                                    continue
                                pred_c = (mask == c)
                                ann_c = (ann == c)
                                inter = int((pred_c & ann_c).sum())
                                union = int((pred_c | ann_c).sum())
                                if union > 0:
                                    ious.append(inter / union)
                            img_scores[base] = round(sum(ious) / len(ious), 4) if ious else 0.0
                    except Exception:
                        pred_count[base] = 0
                # Detection: _det.json boxes
                elif f.endswith("_det.json"):
                    base = f.replace("_det.json", "")
                    try:
                        det_boxes, _ = load_coco_json(os.path.join(out_dir, f))
                        if det_boxes:
                            pred_count[base] = len(det_boxes)
                    except Exception:
                        pass
            # Update on main thread
            self._pred_count = pred_count
            self._img_scores = img_scores
            self._pred_count_cache_key = project_dir
            self.refresh_list_signal.emit()
        except Exception:
            pass

    def _filter_images(self, text):
        self._suppress_cell_change = True
        self.image_list_widget.setRowCount(0)
        text_lower = text.lower().strip()
        sf_text = self.split_filter.currentText()
        annotated_only = self.annotated_check.isChecked()
        # Load prediction counts + IoU scores from outputs dir (triggered async by _populate_image_list)
        # Ensure fallback dicts exist
        if not hasattr(self, "_pred_count"):
            self._pred_count = {}
        if not hasattr(self, "_img_scores"):
            self._img_scores = {}
        # Sort by annotation date if "Recent" is checked
        if self.recent_check.isChecked():
            def _json_mtime(p):
                base = os.path.splitext(os.path.basename(p))[0]
                jp = os.path.join(os.path.dirname(p), base + ".json")
                return os.path.getmtime(jp) if os.path.exists(jp) else 0
            self._all_images.sort(key=_json_mtime, reverse=True)
        shown = 0
        total_train = 0
        total_val = 0
        total_test = 0
        total_annotated = 0
        total_shapes = 0
        ann_dir = os.path.join(os.path.dirname(self._all_images[0]) if self._all_images else "",
                               "..", "annotations")
        for path in self._all_images:
            fname = os.path.basename(path)
            base = os.path.splitext(fname)[0]
            mask_path = os.path.join(ann_dir, base + "_mask.png")
            json_path = os.path.join(os.path.dirname(path), base + ".json")
            det_json_path = os.path.join(os.path.dirname(path), base + "_det.json")
            has_ann = os.path.exists(mask_path) or os.path.exists(json_path) or os.path.exists(det_json_path)

            # Count shapes and collect labels for search
            shape_count = 0
            shape_labels = []
            if os.path.exists(det_json_path):
                try:
                    from detection.coco_io import load_coco_json
                    det_boxes, _ = load_coco_json(det_json_path)
                    if det_boxes:
                        shape_count += len(det_boxes)
                        for b in det_boxes:
                            cat = b.get("category", "")
                            if cat and cat not in shape_labels:
                                shape_labels.append(cat)
                except Exception:
                    pass
            if os.path.exists(json_path):
                try:
                    import json as _json
                    with open(json_path, "r", encoding="utf-8") as jf:
                        jdata = _json.load(jf)
                    shapes = jdata.get("shapes", [])
                    for s in shapes:
                        lbl = s.get("label", "")
                        if lbl and lbl != "background" and "ignore" not in lbl.lower() and "gnore" not in lbl.lower():
                            shape_count += 1
                            shape_labels.append(lbl)
                except Exception:
                    pass
            elif os.path.exists(mask_path):
                try:
                    mask = np.array(Image.open(mask_path))
                    unique = np.unique(mask)
                    shape_count = len([u for u in unique if u not in (0, 255)])
                except Exception:
                    pass

            # Search: match filename OR class label
            if text_lower and text_lower not in fname.lower():
                # Check if any shape label matches
                label_match = any(text_lower in lbl.lower() for lbl in shape_labels)
                if not label_match:
                    continue

            if annotated_only and not has_ann:
                continue

            # Prediction count filter
            pred_count = getattr(self, "_pred_count", {}).get(base, -1)
            if self.pred_count_spin.value() > 0 and pred_count >= 0:
                op = self.pred_op_combo.currentText()
                threshold = self.pred_count_spin.value()
                if op == "≥" and not (pred_count >= threshold):
                    continue
                elif op == "=" and not (pred_count == threshold):
                    continue
                elif op == "≤" and not (pred_count <= threshold):
                    continue

            # Get split status (unannotated images not in split file get no status)
            split_status = ""
            has_split_file = False
            try:
                if self.current_project:
                    pd = str(self.pm.get_project_dir(self.current_project["name"]))
                    sf = os.path.join(pd, "train_test_split.json")
                    if os.path.exists(sf):
                        has_split_file = True
                        with open(sf, "r", encoding="utf-8") as sfh:
                            sm = json.load(sfh)
                        v = sm.get(base)
                        if v == "val":
                            split_status = " [Val]"
                        elif v == "test":
                            split_status = " [Test]"
                        elif v == "train":
                            split_status = " [Train]"
                        elif has_ann:
                            split_status = " [Train]"
            except Exception:
                pass

            # Split filter
            if sf_text == "Train Only" and split_status != " [Train]":
                continue
            if sf_text == "Val Only" and split_status != " [Val]":
                continue
            if sf_text == "Test Only" and split_status != " [Test]":
                continue

            # Build display text
            # Image dimensions (cached)
            img_w, img_h = 0, 0
            cache_key = path
            if not hasattr(self, '_img_dims'):
                self._img_dims = {}
            if cache_key in self._img_dims:
                img_w, img_h = self._img_dims[cache_key]
            else:
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(path) as im:
                        img_w, img_h = im.size
                except Exception:
                    pass
                self._img_dims[cache_key] = (img_w, img_h)

            # Prediction count
            pred_cnt = getattr(self, "_pred_count", {}).get(base, 0)

            # Inference time
            infer_ms = getattr(self, "_infer_times", {}).get(base, "")

            # Score
            raw_score = getattr(self, "_img_scores", {}).get(base, None)
            score_str = f"{raw_score:.2f}" if raw_score is not None else ""

            row = self.image_list_widget.rowCount()
            self.image_list_widget.insertRow(row)
            items_data = [
                (str(shown + 1), True),
                (fname, False),
                (f"{img_w}x{img_h}" if img_w else "", True),
                (split_status.replace("[", "").replace("]", "").strip(), True),
                (str(shape_count) if shape_count > 0 else "", True),
                (str(pred_cnt) if pred_cnt > 0 else "", True),
                (score_str, True),
                (str(infer_ms) if infer_ms else "", True),
            ]
            for col, (val, center) in enumerate(items_data):
                item = QTableWidgetItem(val)
                if center:
                    item.setTextAlignment(Qt.AlignCenter)
                self.image_list_widget.setItem(row, col, item)
            # Store path in first column
            self.image_list_widget.item(row, 0).setData(Qt.UserRole, path)
            # Color: green=Train, orange=Val, red=Test, green=annotated
            if "Test" in split_status:
                clr = QColor(180, 0, 0)
            elif "Val" in split_status:
                clr = QColor(200, 150, 0)
            elif has_ann:
                clr = QColor(0, 120, 0)
            else:
                clr = QColor(0, 0, 0)
            for col in range(8):
                item = self.image_list_widget.item(row, col)
                if item:
                    item.setForeground(clr)
            shown += 1
            if split_status == " [Train]":
                total_train += 1
            elif split_status == " [Val]":
                total_val += 1
            elif split_status == " [Test]":
                total_test += 1
            if has_ann:
                total_annotated += 1
                total_shapes += shape_count

        parts = [f"Showing {shown}/{len(self._all_images)}"]
        if total_train:
            parts.append(f"Train: {total_train}")
        if total_val:
            parts.append(f"Val: {total_val}")
        if total_test:
            parts.append(f"Test: {total_test}")
        parts.append(f"Annotated: {total_annotated}")
        parts.append(f"Blocks: {total_shapes}")
        self.list_status.setText(" | ".join(parts))

        self._suppress_cell_change = False

    def _on_current_cell_changed(self, currentRow, currentColumn, previousRow, previousColumn):
        """Handle mouse click and keyboard navigation in image list."""
        if getattr(self, "_suppress_cell_change", False):
            return
        if currentRow >= 0 and currentRow != previousRow:
            item = self.image_list_widget.item(currentRow, 0)
            path = item.data(Qt.UserRole) if item else None
            if path and os.path.exists(path):
                self._load_image_by_index(currentRow)

    def _on_image_item_clicked(self, row, col):
        item = self.image_list_widget.item(row, 0)
        path = item.data(Qt.UserRole) if item else None
        if path and os.path.exists(path):
            self._load_image_by_index(row)

    def _do_update_stats(self):
        """Debounced stats update - called after user stops navigating."""
        self._update_stats()
        self._update_ann_stats()

    def _update_stats(self, ann_mask=None, pred_mask=None):
        """Update per-prediction-block statistics table (segmentation or detection)."""
        # --- Detection mode: show prediction boxes from BoxManager ---
        if self._detection_mode and self._box_manager is not None:
            bm = self._box_manager
            headers = ["序号", "类别", "置信度", "面积", "宽度", "高度", "x1", "y1"]
            self.stats_table.setColumnCount(len(headers))
            self.stats_table.setHorizontalHeaderLabels(headers)
            for i in range(len(headers)):
                self.stats_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
            self.stats_table.setRowCount(0)
            self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
            preds = bm.pred_boxes if bm.pred_boxes else []
            rows = []
            for idx, b in enumerate(preds):
                bbox = b.get("bbox", [0, 0, 0, 0])
                x, y, w, h = (bbox + [0, 0, 0, 0])[:4]
                score = b.get("score", b.get("confidence", 0.0))
                rows.append((
                    b.get("category", "unknown"),
                    "%.2f" % score if score else "",
                    "%.0f" % (w * h),
                    "%.0f" % w,
                    "%.0f" % h,
                    "%.0f" % x,
                    "%.0f" % y,
                ))
            self.stats_table.setRowCount(len(rows))
            for i, row_data in enumerate(rows):
                items = [str(i + 1)] + list(row_data)
                for j, val in enumerate(items):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.stats_table.setItem(i, j, item)
            return

        # --- Segmentation mode (original) ---
        headers = ["序号", "类别", "分数", "面积", "周长", "半径", "平均灰度", "交并比"]
        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setHorizontalHeaderLabels(headers)
        for i in range(len(headers)):
            self.stats_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        self.stats_table.setRowCount(0)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)

        arr = pred_mask if pred_mask is not None else getattr(self.canvas, "_prediction_mask", None)
        mask = ann_mask if ann_mask is not None else (self.canvas.mask if self.canvas.image is not None else None)
        if arr is None:
            return
        pred_classes = getattr(self.canvas, "_pred_classes", None)

        import numpy as _np
        img_gray = getattr(self.canvas, "_cached_gray", None)

        rows = []
        for cid in sorted(set(_np.unique(arr)) - {0, 255}):
            region = (arr == cid)
            area = int(region.sum())
            # Perimeter: approximate via contour arc length
            import cv2
            binary = region.astype(_np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = sum(int(cv2.arcLength(c, True)) for c in contours)
            radius = int(_np.sqrt(area / _np.pi))
            # Avg gray
            avg_gray = 0.0
            if img_gray is not None and img_gray.shape[:2] == region.shape[:2]:
                try:
                    avg_gray = float(img_gray[region].mean())
                except IndexError:
                    avg_gray = 0.0
            # IoU: best-matching annotation class
            iou_val = 0.0
            if mask is not None and mask.shape == arr.shape:
                best = 0.0
                for ac in set(_np.unique(mask)) - {0, 255}:
                    ann_region = (mask == ac)
                    inter = (region & ann_region).sum()
                    uni = (region | ann_region).sum()
                    if uni > 0:
                        ciou = inter / uni
                        if ciou > best:
                            best = ciou
                iou_val = best
            # Class name
            cls_name = (pred_classes[int(cid)] if pred_classes and int(cid) < len(pred_classes) else f"Class{int(cid)}")
            # Score: IoU*100 as approximate confidence
            score = "%.1f%%" % (iou_val * 100)
            rows.append((cls_name, score, str(area), str(perimeter), str(radius), "%.1f" % avg_gray, "%.3f" % iou_val))

        self.stats_table.setRowCount(len(rows))
        for i, (cls_name, score, area, perim, rad, gray, iou) in enumerate(rows):
            items = [str(i+1), cls_name, score, area, perim, rad, gray, iou]
            for j, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.stats_table.setItem(i, j, item)

    def _update_ann_stats(self):
        """Update per-annotation-block statistics table (segmentation or detection)."""
        # --- Detection mode: show annotation boxes from BoxManager ---
        if self._detection_mode and self._box_manager is not None:
            bm = self._box_manager
            headers = ["序号", "类别", "面积", "宽度", "高度", "x1", "y1"]
            self.ann_table.setColumnCount(len(headers))
            self.ann_table.setHorizontalHeaderLabels(headers)
            for i in range(len(headers)):
                self.ann_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
            self.ann_table.setRowCount(0)
            self.ann_table.setEditTriggers(QTableWidget.NoEditTriggers)
            boxes = bm.boxes if bm.boxes else []
            rows = []
            for idx, b in enumerate(boxes):
                bbox = b.get("bbox", [0, 0, 0, 0])
                x, y, w, h = (bbox + [0, 0, 0, 0])[:4]
                rows.append((
                    b.get("category", "unknown"),
                    "%.0f" % (w * h),
                    "%.0f" % w,
                    "%.0f" % h,
                    "%.0f" % x,
                    "%.0f" % y,
                ))
            self.ann_table.setRowCount(len(rows))
            for i, row_data in enumerate(rows):
                items = [str(i + 1)] + list(row_data)
                for j, val in enumerate(items):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.ann_table.setItem(i, j, item)
            return

        # --- Segmentation mode (original) ---
        headers = ["序号", "类别", "面积", "周长", "半径", "平均灰度"]
        self.ann_table.setColumnCount(len(headers))
        self.ann_table.setHorizontalHeaderLabels(headers)
        for i in range(len(headers)):
            self.ann_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        self.ann_table.setRowCount(0)
        self.ann_table.setEditTriggers(QTableWidget.NoEditTriggers)

        mask = self.canvas.mask if self.canvas.image is not None else None
        if mask is None:
            return
        import numpy as _np
        img_gray = None
        if self.canvas.image is not None:
            img_arr = _np.array(self.canvas.image)
            img_gray = _np.mean(img_arr, axis=2) if img_arr.ndim == 3 else img_arr.astype(_np.float64)

        # Get class names from project
        classes = (getattr(self, "_cached_classes", None) or ["background"])
        import cv2
        rows = []
        for cid in sorted(set(_np.unique(mask)) - {0, 255}):
            region = (mask == cid)
            area = int(region.sum())
            binary = region.astype(_np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = sum(int(cv2.arcLength(c, True)) for c in contours)
            radius = int(_np.sqrt(area / _np.pi))
            avg_gray = 0.0
            if img_gray is not None and img_gray.shape[:2] == region.shape[:2]:
                try:
                    avg_gray = float(img_gray[region].mean())
                except IndexError:
                    avg_gray = 0.0
            cls_name = classes[int(cid)] if int(cid) < len(classes) else f"Class{int(cid)}"
            rows.append((cls_name, str(area), str(perimeter), str(radius), "%.1f" % avg_gray))

        self.ann_table.setRowCount(len(rows))
        for i, (cls_name, area, perim, rad, gray) in enumerate(rows):
            items = [str(i+1), cls_name, area, perim, rad, gray]
            for j, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.ann_table.setItem(i, j, item)


    def _toggle_detection_mode(self):
        """Toggle between segmentation and detection annotation modes."""
        self._detection_mode = not self._detection_mode
        self.det_btn.blockSignals(True)
        if self._detection_mode:
            self.det_btn.setChecked(True)
            self.det_btn.setText("Detection: ON")
            self.det_btn.blockSignals(False)
            self.log("Switched to Detection mode")
            
            from detection.box_manager import BoxManager
            from detection.box_overlay import DetectionOverlay
            
            classes = getattr(self, "_cached_classes", None) or ["background"]
            self._box_manager = BoxManager(categories=classes)
            self.canvas._det_overlay = DetectionOverlay(self.canvas, self._box_manager)
            
            if "pan" in getattr(self, "_mode_buttons", {}):
                self._mode_buttons["pan"].setChecked(True)
                self.canvas._mode = self.canvas.MODE_PAN
            
            self._load_det_annotations()
        else:
            self.det_btn.setChecked(False)
            self.det_btn.setText("Detection: OFF")
            self.det_btn.blockSignals(False)
            self.log("Switched to Segmentation mode")
            
            self.canvas._det_overlay = None
            self.canvas.update()



    # ============ Detection support ============

    def _load_det_annotations(self):
        """Load detection annotations for current image if in detection mode."""
        if not self._detection_mode or self._box_manager is None:
            return
        path = getattr(self.canvas, "current_image_path", None)
        if path and os.path.exists(path) and self.canvas.image is not None:
            self._box_manager.set_image_size(*self.canvas.image.size)
            self._box_manager.load(path)
            # Auto-load batch inference predictions from outputs/ directory
            if self.current_project:
                outputs_dir = os.path.join(
                    str(self.pm.get_project_dir(self.current_project["name"])), "outputs")
                self._box_manager.load_predictions(outputs_dir)
            self.canvas.update()

    # ============ Image Loading ============

    def _load_image_by_index(self, idx):
        item = self.image_list_widget.item(idx, 0)
        if item:
            path = item.data(Qt.UserRole)
            if path and os.path.exists(path):
                self.canvas.load_image(path)
                self._refresh_versions()
                w, h = self.canvas.image.width, self.canvas.image.height
                self.statusBar().showMessage(
                    f"Image: {os.path.basename(path)}  |  {w}x{h}")
                self._stats_timer.start(150)
                self._load_det_annotations()

    # ============ Annotation Modes ============

    def _set_annotation_mode(self, mode):
        """Set the annotation tool mode (brush/pan/polygon/line/eraser)."""
        mode_map = {
            "pan": self.canvas.MODE_PAN,
            "brush": self.canvas.MODE_BRUSH,
            "polygon": self.canvas.MODE_POLYGON,
            "line": self.canvas.MODE_LINE,
            "eraser": self.canvas.MODE_ERASER,
        }
        if mode in mode_map:
            self.canvas._mode = mode_map[mode]
            self.canvas._update_mode_cursor()

    # ============ Right-click Context Menu ============

    def _on_image_context_menu(self, pos):
        idx = self.image_list_widget.indexAt(pos)
        row = idx.row() if idx.isValid() else -1
        selected_rows = set(it.row() for it in self.image_list_widget.selectedItems())
        if not selected_rows:
            return
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        from training.dataset import get_train_test_split, save_train_test_split
        split_map, _ = get_train_test_split(project_dir)

        menu = QMenu(self)
        for label, key in [("Set as Train", "train"), ("Set as Val", "val"), ("Set as Test", "test")]:
            action = menu.addAction(label)
            action.triggered.connect(lambda checked, k=key: self._set_selected_split(selected_rows, k, project_dir, split_map))
        menu.exec_(self.image_list_widget.viewport().mapToGlobal(pos))

    def _set_selected_split(self, selected_rows, split_key, project_dir, split_map):
        from training.dataset import save_train_test_split
        for r in selected_rows:
            item = self.image_list_widget.item(r, 0)
            p = item.data(Qt.UserRole) if item else None
            if not p:
                continue
            base = os.path.splitext(os.path.basename(p))[0]
            split_map[base] = split_key
        save_train_test_split(project_dir, split_map)
        self._filter_images(self.search_input.text())
        self.log(f"Set {len(selected_rows)} image(s) to {split_key}")

    # ============ Version History ============

    def _refresh_versions(self):
        """Refresh version history list for current image."""
        if not hasattr(self, "version_list"):
            return
        self.version_list.clear()
        path = getattr(self.canvas, "current_image_path", None)
        if not path or not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        try:
            from annotation.version_manager import list_versions
            versions = list_versions(pd, path)
            for vpath, ts_str, size_kb in versions:
                self.version_list.addItem(f"{ts_str} ({size_kb:.1f}KB)")
            if hasattr(self, "version_count_label"):
                self.version_count_label.setText(f"{len(versions)} versions")
        except Exception as e:
            self.log(f"[Version] Error: {e}")

    def _restore_version(self):
        """Restore selected version from history."""
        if not hasattr(self, "version_list") or self.version_list.currentRow() < 0:
            return
        path = getattr(self.canvas, "current_image_path", None)
        if not path or not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        idx = self.version_list.currentRow()
        try:
            from annotation.version_manager import list_versions, restore_version
            versions = list_versions(pd, path)
            if idx < len(versions):
                vpath = versions[idx][0]
                if restore_version(pd, path, vpath):
                    self.canvas.load_image(path)
                    self._refresh_versions()
                    self.log(f"Restored version {idx+1}")
        except Exception as e:
            self.log(f"[Version] Restore error: {e}")

    # ============ Class Management ============

    def _load_classes(self):
        """Load class list from project config."""
        if not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        config_path = os.path.join(pd, "project.json")
        classes = ["background"]
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                classes = meta.get("classes", ["background"])
            except Exception:
                pass
        self._cached_classes = classes
        if hasattr(self, "class_list"):
            self.class_list.clear()
            for c in classes:
                item = QListWidgetItem(c)
                self.class_list.addItem(item)

    def _on_class_changed(self, current):
        """Handle class selection change in class list."""
        if current < 0 or not hasattr(self.canvas, "label_manager"):
            return
        if current >= 0:
            self.canvas.label_manager.current_class = current

    # ============ UI Panel Toggles ============

    def _toggle_panel(self, side):
        """Toggle left or right panel visibility."""
        if side == "left":
            v = not self._left_panel.isVisible()
            self._left_panel.setVisible(v)
            self._saved_left = v
            self._save_ui_setting("left_panel_visible", v)
        elif side == "right":
            v = not self._right_panel.isVisible()
            self._right_panel.setVisible(v)
            self._saved_right = v
            self._save_ui_setting("right_panel_visible", v)

    def _toggle_canvas_header(self):
        self._canvas_header_visible = not getattr(self, "_canvas_header_visible", True)
        if hasattr(self, "_canvas_header"):
            self._canvas_header.setVisible(self._canvas_header_visible)
        self._save_ui_setting("canvas_header_visible", self._canvas_header_visible)

    def _toggle_image_list(self):
        v = not self.image_list_widget.isVisible()
        self.image_list_widget.setVisible(v)
        self._save_ui_setting("image_list_visible", v)

    def _toggle_log_panel(self):
        v = not self._log_splitter.isVisible() if hasattr(self, "_log_splitter") else False
        if hasattr(self, "_log_splitter"):
            self._log_splitter.setVisible(not v)
            self._saved_log = not v
            self._save_ui_setting("log_panel_visible", not v)

    def _save_ui_setting(self, key, value):
        try:
            if not hasattr(self, "_ui_settings"):
                self._ui_settings = {}
            self._ui_settings[key] = value
        except Exception:
            pass

    def _load_ui_setting(self, key, default=True):
        if hasattr(self, "_ui_settings") and key in self._ui_settings:
            return self._ui_settings[key]
        return default

    def _pred_count_cache_key(self, base):
        return base

    # ============ Stop Handlers ============

    def _stop_batch(self):
        self._stop_flag = True
        self.log("Stopping inference...")

    def _stop_inference(self):
        self._stop_flag = True
        self.log("Stopping...")

    def _stop_training(self):
        self._stop_flag = True
        if self.trainer is not None:
            self.trainer._stop_requested = True
        if hasattr(self, '_det_trainer') and self._det_trainer is not None:
            self._det_trainer.stop()
        self.log("Stopping...")
        self.btn_train_stop.setEnabled(False)
        if hasattr(self, 'btn_det_train_stop'):
            self.btn_det_train_stop.setEnabled(False)

    # ============ Training ============

    def _start_training(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        images_dir = os.path.join(project_dir, "images")
        if not os.path.exists(images_dir):
            return QMessageBox.warning(self, "Warning", "No images directory found")
        
        # Show training settings dialog
        dlg = TrainSettingsDialog(self)
        if dlg.exec_() != dlg.Accepted:
            return
        settings = dlg.get_settings()
        epochs = settings.get("epochs", 200)
        batch_size = settings.get("batch_size", 8)
        model_name = settings.get("model", "deeplabv3")
        image_size = settings.get("image_size", 512)
        loss_name = settings.get("loss", "cross_entropy")
        augment = settings.get("augment", "none")
        cv_folds = settings.get("k_folds", 1)

        self._stop_flag = False
        self.btn_train_stop.setEnabled(True)
        self.train_status.setText("Training...")
        self.train_progress.setVisible(True)
        self.train_progress.setValue(0)

        from training.trainer import Trainer
        self.trainer = Trainer(project_dir, model_name=model_name)

        self.log(f"Starting training: {model_name}, {epochs} epochs, batch={batch_size}"
                 + (f", {cv_folds}-fold CV" if cv_folds > 1 else "")
                 + f", loss={loss_name}, aug={augment}")
        
        def run():
            try:
                self.trainer.train(
                    epochs=epochs, batch_size=batch_size,
                    image_size=image_size, loss_name=loss_name,
                    augment=augment, k_folds=cv_folds,
                    stop_check=lambda: self._stop_flag,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                    progress_callback=lambda c, t: self.progress_signal.emit(c, t),
                    plot_callback=lambda h: self.chart_signal.emit(h),
                    batch_callback=lambda b, t, l: (self.batch_signal.emit(b, t, l), self._log_batch_progress(b, t, l)))
                self.log_signal.emit("Training complete!")
                self.train_status.setText("Complete")
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Training error: {e}")
                self.log_signal.emit(traceback.format_exc())
                self.train_status.setText("Failed")
            finally:
                self.btn_train_stop.setEnabled(False)
                self.train_progress.setVisible(False)
                self._cleanup_trainer()

        threading.Thread(target=run, daemon=True).start()

    def _start_det_training(self):
        """Start detection model training."""
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        
        model_name = self.det_model_combo.currentText() if hasattr(self, 'det_model_combo') else "yolov8s"
        epochs = self.det_epochs_spin.value() if hasattr(self, 'det_epochs_spin') else 200
        batch_size = self.det_batch_spin.value() if hasattr(self, 'det_batch_spin') else 16
        image_size = self.det_img_size_spin.value() if hasattr(self, 'det_img_size_spin') else 640
        
        self._stop_flag = False
        self.btn_det_train_stop.setEnabled(True)
        self.det_train_status.setText("Preparing...")
        self.det_train_progress.setVisible(True)
        self.det_train_progress.setValue(0)
        
        def run():
            try:
                # Step 1: Build dataset
                self.log_signal.emit("Building YOLO dataset...")
                from detection.dataset import build_yolo_dataset, get_annotated_count
                ann_count = get_annotated_count(project_dir)
                if ann_count < 1:
                    self.log_signal.emit("ERROR: No detection annotations found")
                    return
                ds_info = build_yolo_dataset(project_dir, val_split=0.2)
                self.log_signal.emit(
                    f"Dataset: {ds_info['train_images']} train + {ds_info['val_images']} val, "
                    f"{ds_info['num_classes']} classes: {', '.join(ds_info['class_names'])}")
                
                # Step 2: Train
                self.log_signal.emit(
                    f"Training: {model_name}, {epochs} epochs, batch={batch_size}, imgsz={image_size}")
                from detection.trainer import DetectionTrainer
                det_trainer = DetectionTrainer(project_dir)
                
                def epoch_cb(ep, tot, m):
                    self.log_signal.emit(
                        f"Epoch {ep}/{tot} | mAP50: {m.get('metrics/mAP50(B)', 0):.4f} | "
                        f"mAP: {m.get('metrics/mAP50-95(B)', 0):.4f}")
                    self.progress_signal.emit(ep, tot)
                
                result = det_trainer.train(
                    data_yaml=ds_info["data_yaml"],
                    model_name=model_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    image_size=image_size,
                    epoch_callback=epoch_cb,
                )
                
                if "error" not in result:
                    self.log_signal.emit(
                        f"Training complete! mAP50: {result.get('best_map50', 0):.4f}, "
                        f"mAP: {result.get('best_map', 0):.4f}")
                else:
                    self.log_signal.emit("Training stopped by user")
                    
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Detection training error: {e}")
                self.log_signal.emit(traceback.format_exc())
            finally:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.log_signal.emit("GPU memory released")
                # Use invokeMethod to safely update UI from thread
                self.log_signal.emit("__DET_TRAINING_DONE__")
        
        threading.Thread(target=run, daemon=True).start()
    
    def _on_det_training_done(self):
        """Called from main thread via signal to update UI after training."""
        self.btn_det_train_stop.setEnabled(False)
        self.det_train_progress.setVisible(False)
        self.det_train_status.setText("Ready")

    def _cleanup_trainer(self):
        if self.trainer is not None:
            self.trainer.cleanup()
        self.trainer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.log("GPU memory released")

    def _log_batch_progress(self, batch, total, loss):
        pct = int(batch / total * 100) if total > 0 else 0
        self.log(f"  [{pct}%] batch {batch}/{total} | loss={loss:.4f}")

    # ============ Inference ============

    def _backend_key(self):
        text = self.backend_combo.currentText().lower()
        if "tensorrt" in text:
            return "tensorrt"
        if "onnx" in text:
            return "onnx"
        if "compiled" in text:
            return "compiled"
        if "torchscript" in text:
            return "torchscript"
        return "pytorch"

    def _start_inference(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        if self.canvas.image is None:
            return QMessageBox.warning(self, "Warning", "Please load an image first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        
        if self._detection_mode:
            self._run_det_inference_single(project_dir)
        else:
            self._run_seg_inference_single(project_dir)
    
    def _run_seg_inference_single(self, project_dir):
        model_file = self.model_combo.currentText()
        model_path = os.path.join(project_dir, "models", model_file)
        if not os.path.exists(model_path):
            return QMessageBox.warning(self, "Error",
                f"Model not found: {model_path}\nPlease train a model first.")
        self.log(f"Running inference: {model_file}")
        try:
            from inference.predictor import Predictor
            predictor = Predictor(project_dir, model_file, backend=self._backend_key())
            pred, overlay = predictor.predict(self.canvas.image, return_overlay=True,
                tiled=self.tiled_check.isChecked(), scale=self.scale_slider.value() / 100.0)
            h, w, ch = overlay.shape
            qimg = QImage(overlay.data, w, h, 3 * w, QImage.Format_RGB888)
            self.canvas._prediction_qimage = qimg.copy()
            self.canvas._prediction_mask = pred.copy()
            self.canvas._pred_stats_cache = {}
            self.canvas._pred_classes = predictor.classes
            self.canvas.update()
            self.log("Inference complete - Space to view")
            self._update_stats()
        except Exception as e:
            import traceback
            self.log(traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Inference failed: {e}")
    
    def _run_det_inference_single(self, project_dir):
        """Run detection inference on current image and display boxes."""
        model_path = os.path.join(project_dir, "models", "detection", "best.pt")
        if not os.path.exists(model_path):
            return QMessageBox.warning(self, "Error",
                f"Detection model not found: {model_path}\nPlease train a detection model first.")
        self.log(f"Running detection inference: {os.path.basename(model_path)}")
        try:
            from detection.trainer import DetectionTrainer
            dt = DetectionTrainer(project_dir)
            boxes = dt.predict(self.canvas.current_image_path)
            if boxes:
                self._box_manager.delete_all()
                for b in boxes:
                    self._box_manager.add_box(
                        bbox=b["bbox"],
                        category=b.get("category", "object"),
                        score=b.get("confidence"),
                    )
                self.canvas.update()
            self.log(f"Detection complete: {len(boxes)} objects found")
        except Exception as e:
            import traceback
            self.log(traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Detection inference failed: {e}")

    def _batch_inference(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        
        if self._detection_mode:
            self._run_det_batch_inference(project_dir)
        else:
            self._run_seg_batch_inference(project_dir)
    
    def _run_seg_batch_inference(self, project_dir):
        model_file = self.model_combo.currentText()
        model_path = os.path.join(project_dir, "models", model_file)
        if not os.path.exists(model_path):
            return QMessageBox.warning(self, "Error",
                "Model not found: %s\nPlease train a model first." % model_path)
        img_dir = os.path.join(project_dir, "images")
        out_dir = os.path.join(project_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                  if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))]
        if not images:
            return QMessageBox.warning(self, "Warning", "No images found")
        total = len(images)
        self._stop_flag = False
        self.btn_infer_stop.setEnabled(True)
        self.infer_progress.setVisible(True)
        self.infer_progress.setMaximum(total)
        self.infer_progress.setValue(0)
        self.infer_status.setText("Running...")
        self.log(f"Batch inference on {total} images...")
        from inference.predictor import Predictor
        import time

        def run():
            try:
                predictor = Predictor(project_dir, model_file, backend=self._backend_key())
                for i, img_path in enumerate(images):
                    t0 = time.time()
                    if self._stop_flag:
                        self.log_signal.emit("Batch inference stopped by user at %d/%d" % (i, total))
                        break
                    try:
                        image = Image.open(img_path).convert("RGB")
                    except Exception:
                        self.log_signal.emit("[%d/%d] SKIP: cannot read %s" % (i + 1, total, os.path.basename(img_path)))
                        continue
                    pred, overlay = predictor.predict(image, return_overlay=True,
                        tiled=self.tiled_check.isChecked(), scale=self.scale_slider.value() / 100.0)
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    mask_out = os.path.join(out_dir, base + "_pred.png")
                    Image.fromarray(pred.astype(np.uint8)).save(mask_out)
                    overlay_out = os.path.join(out_dir, base + "_overlay.jpg")
                    Image.fromarray(overlay).save(overlay_out)
                    elapsed = time.time() - t0
                    self.log_signal.emit("[%d/%d] %s (%.1fs)" % (i + 1, total, base, elapsed))
                    self.infer_progress_signal.emit(i + 1, total)
                self.log_signal.emit("Batch inference complete! Results: %s" % out_dir)
                self.infer_status.setText("Complete")
                self._pred_count_cache_key = None  # force re-scan outputs
                threading.Thread(target=self._load_output_stats_async, args=(project_dir,), daemon=True).start()
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Batch inference error: {e}")
                self.log_signal.emit(traceback.format_exc())
                self.infer_status.setText("Failed")
            finally:
                self.btn_infer_stop.setEnabled(False)
                self.infer_progress.setVisible(False)

        threading.Thread(target=run, daemon=True).start()
    
    def _run_det_batch_inference(self, project_dir):
        """Batch detection inference on all images."""
        model_path = os.path.join(project_dir, "models", "detection", "best.pt")
        if not os.path.exists(model_path):
            return QMessageBox.warning(self, "Error",
                "Detection model not found: %s\nPlease train a detection model first." % model_path)
        img_dir = os.path.join(project_dir, "images")
        out_dir = os.path.join(project_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                  if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))]
        if not images:
            return QMessageBox.warning(self, "Warning", "No images found")
        total = len(images)
        self._stop_flag = False
        self.btn_infer_stop.setEnabled(True)
        self.infer_progress.setVisible(True)
        self.infer_progress.setMaximum(total)
        self.infer_progress.setValue(0)
        self.infer_status.setText("Running...")
        self.log(f"Batch detection inference on {total} images...")
        import time
        
        def run():
            try:
                from detection.trainer import DetectionTrainer
                dt = DetectionTrainer(project_dir)
                for i, img_path in enumerate(images):
                    t0 = time.time()
                    if self._stop_flag:
                        self.log_signal.emit("Batch detection stopped by user at %d/%d" % (i, total))
                        break
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    try:
                        boxes = dt.predict(img_path)
                        # Remap category_id to match project categories (YOLO uses its own internal indexing)
                        classes = self._cached_classes if hasattr(self, "_cached_classes") else ["background"]
                        for b in boxes:
                            cat_name = b.get("category", "unknown")
                            b["category_id"] = classes.index(cat_name) if cat_name in classes else 0
                        # Save detection results as COCO JSON
                        from detection.coco_io import save_coco_json
                        try:
                            img_w, img_h = Image.open(img_path).size
                        except Exception:
                            img_w, img_h = 0, 0
                        json_out = os.path.join(out_dir, base + "_det.json")
                        save_coco_json(
                            json_out, img_path, img_w, img_h,
                            boxes, classes,
                            bbox_type="hbb"
                        )
                        elapsed = time.time() - t0
                        self.log_signal.emit("[%d/%d] %s: %d boxes (%.1fs)" % (i + 1, total, base, len(boxes), elapsed))
                    except Exception as ex:
                        self.log_signal.emit("[%d/%d] %s: FAILED — %s" % (i + 1, total, base, ex))
                    self.infer_progress_signal.emit(i + 1, total)
                self.log_signal.emit("Batch detection complete! Results: %s" % out_dir)
                self.infer_status.setText("Complete")
                self._pred_count_cache_key = None  # force re-scan outputs
                threading.Thread(target=self._load_output_stats_async, args=(project_dir,), daemon=True).start()
                # Reload predictions for current image
                if self._detection_mode and self._box_manager is not None:
                    self._box_manager.load_predictions(out_dir)
                    self.image_load_signal.emit(None)  # trigger canvas update
                self.stats_signal.emit(None, None)  # refresh Result & Annotation Blocks
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Batch detection error: {e}")
                self.log_signal.emit(traceback.format_exc())
                self.infer_status.setText("Failed")
            finally:
                self.btn_infer_stop.setEnabled(False)
                self.infer_progress.setVisible(False)
        
        threading.Thread(target=run, daemon=True).start()

    # ============ Import / Export ============

    def _import_json(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        json_file, _ = QFileDialog.getOpenFileName(self, "Import JSON Annotation", "",
            "JSON Files (*.json);;All Files (*)")
        if not json_file:
            return
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            img_name = data.get("imagePath", "")
            img_path = os.path.join(project_dir, "images", img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(os.path.dirname(json_file), img_name)
            if not os.path.exists(img_path):
                self.log(f"Image not found for {img_name}")
                return
            dest_json = os.path.join(os.path.dirname(img_path),
                                     os.path.splitext(os.path.basename(img_path))[0] + ".json")
            import shutil
            shutil.copy2(json_file, dest_json)
            self.log(f"Imported: {os.path.basename(json_file)}")
            if self.canvas.current_image_path == img_path:
                self.canvas.load_image(img_path)
        except Exception as e:
            self.log(f"Import failed: {e}")

    def _batch_import_json(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        folder = QFileDialog.getExistingDirectory(self, "Select JSON Folder")
        if not folder:
            return
        import shutil
        count = 0
        for f in os.listdir(folder):
            if f.lower().endswith(".json"):
                src = os.path.join(folder, f)
                try:
                    with open(src, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                    img_name = data.get("imagePath", os.path.splitext(f)[0] + ".bmp")
                    img_path = os.path.join(project_dir, "images", img_name)
                    if not os.path.exists(img_path):
                        continue
                    dest = os.path.join(os.path.dirname(img_path),
                                        os.path.splitext(os.path.basename(img_path))[0] + ".json")
                    shutil.copy2(src, dest)
                    count += 1
                except Exception:
                    pass
        self.log(f"Batch imported {count} JSON files")

    def _export_json(self):
        if self.canvas.image is None:
            return
        path = getattr(self.canvas, "current_image_path", "")
        if not path:
            return
        json_path = os.path.splitext(path)[0] + ".json"
        if not os.path.exists(json_path):
            self.log("No annotation JSON to export")
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Export JSON", os.path.basename(json_path),
            "JSON Files (*.json)")
        if dest:
            import shutil
            shutil.copy2(json_path, dest)
            self.log(f"Exported to {dest}")

    def _export_annotations(self):
        if not self.current_project:
            return
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        folder = QFileDialog.getExistingDirectory(self, "Export Annotations To")
        if not folder:
            return
        import shutil
        count = 0
        for f in os.listdir(os.path.join(project_dir, "images")):
            if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
                base = os.path.splitext(f)[0]
                json_path = os.path.join(project_dir, "images", base + ".json")
                if os.path.exists(json_path):
                    shutil.copy2(json_path, os.path.join(folder, base + ".json"))
                    count += 1
        self.log(f"Exported {count} annotations")

    def _clean_tiny_ann(self):
        if self.canvas.image is None:
            return
        mask = self.canvas.mask
        if mask is None:
            return
        threshold = 10000
        unique = np.unique(mask)
        for cid in unique:
            if cid == 0 or cid == 255:
                continue
            region = (mask == cid)
            area = region.sum()
            if area < threshold:
                mask[region] = 0
        self.canvas.mask = mask
        self.canvas._overlay_dirty = True
        self.canvas.update()
        self.canvas.mask_changed.emit()
        self.log(f"Removed tiny annotations (<{threshold}px)")

    # ============ Split Management ============

    def _manage_split(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        from ui.split_dialog import SplitDialog
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        dlg = SplitDialog(pd, self)
        if dlg.exec_():
            self._filter_images(self.search_input.text())

    def _manage_classes(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        from ui.class_dialog import ClassDialog
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        dlg = ClassDialog(pd, self)
        if dlg.exec_():
            self._load_classes()

    # ============ Settings ============



    def _on_det_class_changed(self, row):
        """Handle detection class selection change."""
        if self._box_manager is None or row < 0:
            return
        item = self.det_class_list.item(row)
        if item:
            self._box_manager._current_category = item.text()

    def _on_det_boxes_changed(self):
        """Called when detection boxes are added/deleted/modified."""
        # Refresh image list (annotation counts) and stats
        self._filter_images(self.search_input.text())
        self._update_ann_stats()

    def _set_det_bbox_type(self, bbox_type):
        """Switch between HBB and OBB box types."""
        if self._box_manager is None:
            return
        self._box_manager.bbox_type = bbox_type
        self.det_hbb_btn.setChecked(bbox_type == "hbb")
        self.det_obb_btn.setChecked(bbox_type == "obb")
        if self.canvas._det_overlay:
            self.canvas._det_overlay.switch_bbox_type(bbox_type)
        self.log(f"Box type: {bbox_type.upper()}")

    def _det_delete_box(self):
        """Delete the currently selected detection box."""
        if self.canvas._det_overlay:
            self.canvas._det_overlay.delete_active_box()
            self.canvas.update()
            self.canvas.det_boxes_changed.emit()

    def _det_clear_all(self):
        """Delete all detection boxes."""
        if self._box_manager is None:
            return
        reply = QMessageBox.question(self, "Clear All Boxes",
            "Delete all bounding boxes?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._box_manager.delete_all()
            self.canvas.update()
            self.canvas.det_boxes_changed.emit()
            self.log("All boxes deleted")

    def _populate_det_class_list(self):
        """Populate detection class list from project categories."""
        if not hasattr(self, "det_class_list"):
            return
        self.det_class_list.clear()
        classes = getattr(self, "_cached_classes", ["background"])
        for c in classes:
            if c.lower() == "background":
                continue
            self.det_class_list.addItem(c)

    def _about(self):
        """Show about dialog."""
        QMessageBox.about(self, "AI Training Platform",
            "AI Training Platform v0.1.0\n\n"
            "Semantic Segmentation + Object Detection\n"
            "Built with PyTorch + PyQt5")

    def _save_current_mask(self):
        """Save current annotation mask to JSON."""
        if self.canvas.image is None:
            return
        path = getattr(self.canvas, "current_image_path", "")
        if not path:
            return
        try:
            from annotation.labelme_io import save_mask_to_json
            save_mask_to_json(path, self.canvas.mask, self.canvas.label_manager.classes, self.current_project)
            self.log("Annotation saved")
        except Exception as e:
            self.log(f"Save failed: {e}")

    def _clear_current_mask(self):
        """Clear current annotation mask."""
        if self.canvas.image is None:
            return
        reply = QMessageBox.question(self, "Clear Annotation",
            "Clear all annotations for this image?", 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            h, w = self.canvas.mask.shape
            self.canvas.mask = np.zeros((h, w), dtype=np.uint8)
            self.canvas._overlay_dirty = True
            self.canvas.update()
            self.canvas.mask_changed.emit()
            self.log("Annotation cleared")

    def _next_image(self):
        """Navigate to next image in the list."""
        row = self.image_list_widget.currentRow()
        if row < self.image_list_widget.rowCount() - 1:
            self.image_list_widget.setCurrentCell(row + 1, 0)

    def _prev_image(self):
        """Navigate to previous image in the list."""
        row = self.image_list_widget.currentRow()
        if row > 0:
            self.image_list_widget.setCurrentCell(row - 1, 0)

    def _diff_version(self):
        """Compare two versions from version history."""
        self.log("Version diff not yet implemented")

    def _slot_restore_version(self):
        """Slot wrapper for version restore."""
        self._restore_version()
    def _restart_app(self):
        """Restart the application."""
        self.log("Restarting...")
        import subprocess, sys
        from PyQt5.QtWidgets import QApplication
        QApplication.quit()
        subprocess.Popen([sys.executable, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "main.py")])
