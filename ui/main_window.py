"""Main Window - Optimized for large datasets"""
import os, json, re, sys, threading, shutil, cv2
import torch
from datetime import datetime
import time
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QFrame,
    QButtonGroup,
    QMainWindow, QMenu, QAction, QSplitter, QTreeWidget, QTreeWidgetItem,
    QTextEdit, QFileDialog, QMessageBox, QInputDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QGroupBox,
    QListWidget, QSlider, QDialog, QFormLayout, QDialogButtonBox, QToolButton,
    QLineEdit, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout,
    QStackedWidget, QApplication, QScrollArea, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QFont, QColor, QImage, QPixmap, QPainter, QLinearGradient, QKeySequence, QCursor, QFontMetrics
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from core.config import APP_NAME, APP_VERSION, load_config, save_config, CLASS_COLORS
from core.imgio import resize_mask_u, pil_from_array, ensure_uint8
from ui.qt_thread import ui_call, run_bg, ui_call_sync
from ui.widgets import ElideLabel, ShrinkStack, LossCurveWindow, TrainSettingsDialog
from ui.ocr_ocv_mixin import OcrOcvMixin
from core.project_manager import ProjectManager
from core import marks as marks_mod
from core.filter_expr import FilterExpr
from ui.mark_panel import MarkToolbar, MarkStatsPanel
from ui.filter_builder import FilterBuilderDialog
from annotation.label_manager import LabelManager
from annotation.mask_editor import AnnotationCanvas
from ui.mixed_viewer import MixedViewer
from ui.class_table import ClassTableView
from ui.crop_tool import CropToolDialog
from ui.resize_tool import ResizeToolDialog
from ui.heatmap_analyzer import HeatmapAnalyzerDialog
from ui.mixed_add_dialog import MixedAddImagesDialog
from annotation.labelme_io import save_mask_to_json, save_labelme_json, mask_to_shapes
from annotation.version_manager import list_versions, restore_version
from training.trainer import Trainer
from training.dataset import get_train_test_split, save_train_test_split
from inference.predictor import Predictor
from ui.system_monitor import SystemMonitor
from ui.screen_ocr import screen_ocr_and_show
from ui.annotation_toolbar import AnnotationToolbar
from ui.det_toolbar import DetToolbar
from ui.global_hotkey import register_global_hotkey, unregister_global_hotkey, WM_HOTKEY
import ctypes
from ctypes import wintypes
print("main_window imports OK")

# ────────── 底部「图片信息栏」右端的交互件（2026-10-14 顶栏并进底栏时加的）──────────
#
# 缩放用**相对"适应窗口"的百分比**（与画布 `_fit_zoom` 同口径，见 MultiViewPreview.zoom_percent）。
# 不用绝对倍率：绝对倍率在不同尺寸的图上读数完全不一样，"100% = 铺满窗口"
# 才是用户心里的那个基准（友商底栏右下角那个 100% 就是这个意思）。
#
# 刻度用**对数**，不用线性：
#   · 线性的 5..800 会让 100%（= 什么都没动的默认态）落在轨道**最左 12%** 处 ——
#     手柄贴着左端、右边空一大截，看着像坏了；而且"缩小"那 20 倍range 挤在 12%
#     的行程里，往左轻轻一拖就掉到 5%，往右（放大，复核凹点最常干的事）反倒占 88%。
#   · 对数刻度把 100% 放在**轨道中段**（pos≈525/1000），左右两边"倍率行程"相等：
#         pct = MIN * (MAX/MIN) ** (pos / POS_MAX)
#     顺带一个好处：± 每格固定走 40 个位置 ⇒ 各倍率下"点一下"的幅度都是 ×1.19，
#     手感一致（线性档位在 20% 处点一下是 +50%、在 700% 处点一下几乎不动）。
ZOOM_PCT_MIN = 10
ZOOM_PCT_MAX = 800
ZOOM_POS_MAX = 1000           # 滑块**内部位置** 0..1000（不是百分比，别当 pct 用）
ZOOM_STEP_POS = 40            # −/+ 一格 = 位置 40 ⇒ 固定 ×1.19（各倍率下点一下幅度一致）


def zoom_pos_to_pct(pos):
    """滑块位置 -> 缩放百分比（**对数刻度**）。"""
    p = max(0, min(ZOOM_POS_MAX, int(pos)))
    ratio = float(ZOOM_PCT_MAX) / float(ZOOM_PCT_MIN)
    return int(round(ZOOM_PCT_MIN * ratio ** (p / float(ZOOM_POS_MAX))))


def zoom_pct_to_pos(pct):
    """缩放百分比 -> 滑块位置（`zoom_pos_to_pct` 的反函数）。"""
    import math
    p = max(ZOOM_PCT_MIN, min(ZOOM_PCT_MAX, int(pct)))
    span = math.log(float(ZOOM_PCT_MAX) / float(ZOOM_PCT_MIN))
    return int(round(ZOOM_POS_MAX * math.log(float(p) / float(ZOOM_PCT_MIN)) / span))


# ────────── 竖排页签的样式：右栏 / 左栏**共用这一份** ──────────
#
# 左栏重构（2026-10-14 晚）时把右栏那份内联样式原样搬了过来。与其两边各写一份
# 迟早漂移，不如只留一处：形状/字号/选中态都在这。
#
# ⚠️ 这里**只写主题皮肤色**（`#D6E4FF` / `#2A3A5C` / `#0E3A52` / `#00E5FF`，
#    全在 `ui.themes.THEME_COLORS` 白名单里）—— 切浅色时 `sweep_inline_styles`
#    会把这些颜色剥掉、度量留住，配色交回主题表。**别在这一段里塞语义色**（红/绿），
#    一旦混进一个，整段就不会被清扫，浅色主题下会留下深色皮肤（见 dlstudio-ui-verify）。
_V_TAB_QSS = (
    "QPushButton { font-size: 12px; font-weight: bold; color: #D6E4FF;"
    " padding: 2px 0; border: 1px solid transparent; border-radius: 3px; }"
    "QPushButton:hover { border-color: #2A3A5C; }"
    "QPushButton:checked { background-color: #0E3A52; color: #00E5FF;"
    " border: 1px solid #00E5FF; }"
)


# 底栏那两个 ± 小按钮的样式：**只带度量，颜色交给主题表**。
# ⚠️ 这里原来还写着 `color: #C8D2E0` / `border: 1px solid #2A3A5C` —— 全是
#    `ui.themes.THEME_COLORS` 白名单里的色。窗口建好后 `sweep_inline_styles`
#    会把这类内联样式里的颜色声明**整段剥掉**（只留度量），配色交回主题表；
#    也就是说：写在这里的颜色**从来没有生效过**，只是让这份源码在骗人
#    （`QToolButton:hover` 被剥成空规则 ⇒ 悬停其实一点反馈都没有）。
#    2026-10-14 的教训比"没生效"严重得多：底栏滑块的内联 QSS 三个颜色同样全在
#    白名单里 ⇒ 槽和手柄被剥成"没有背景色" ⇒ **占着 84px 宽一个像素都不画**，
#    而主题表里当时**没有** QSlider 规则可兜底。功能断言（value/geometry/信号）
#    全绿，肉眼却是一段空槽。
#    规则：**颜色一律写进 themes.py 的主题表，内联样式只写度量**。
_BAR_BTN_QSS = """
QToolButton {
    border-width: 1px; border-style: solid; border-radius: 3px;
    padding: 0px; margin: 0px;
    font-size: 12px;
    background: transparent;
}
"""


def _mask_stats_of(mask_path):
    """读一张掩膜 PNG，返回 (类别数, 是否有内容)。

    - 类别数：排除 0（背景）和 255（ignore）之后的不同取值个数
      （列表「标注」列显示的就是这个数，语义沿用原实现，不改业务口径）
    - 是否有内容：存在非 0 像素

    为什么不用 np.unique：
      这类掩膜 99% 是 0，np.unique 仍要对整个 1500² 数组排序，实测 7.5ms；
      而 PIL 的 getcolors() 在 C 层直接统计颜色直方图，只要 4.0ms（快 46%）。
      1690 张算下来 12.6s → 6.8s。省下的时间在打开工程时是能感觉到的。
    """
    with Image.open(mask_path) as im:
        cols = im.getcolors(256)          # uint8 掩膜最多 256 色，够用
        if cols is not None:
            vals = {v for _, v in cols if not isinstance(v, tuple)}
            classes = len([v for v in vals if v not in (0, 255)])
            return classes, bool(vals - {0})
        # 颜色数超 256（多通道误存之类）→ 退回 numpy 兜底
        arr = np.array(im.convert("L"))
        u = np.unique(arr)
        return len([x for x in u if x not in (0, 255)]), bool(arr.any())


class MainWindow(OcrOcvMixin, QMainWindow):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    image_list_signal = pyqtSignal(object)
    infer_progress_signal = pyqtSignal(int, int)
    stats_signal = pyqtSignal(object, object)
    chart_signal = pyqtSignal(dict)
    batch_signal = pyqtSignal(int, int, float)
    refresh_list_signal = pyqtSignal()
    image_load_signal = pyqtSignal(object)
    # 后台预解码完掩膜统计后通知主线程刷新列表（参数=工程代次，用于丢弃过期结果）
    mask_stats_signal = pyqtSignal(int)

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
        # 掩膜统计缓存 + "哪个代次已经预热完毕"标记。
        # 未预热完时主线程**只读缓存、绝不自己解码**（否则会和后台预热线程
        # 抢同一批 PNG，实测反而更慢：主线程 6 秒 + 预热 2 秒白烧）。
        self._mask_stat_cache = {}
        self._mask_stats_ready_gen = -1
        self._ignore_class_row = -1
        self._all_images = []
        self._saved_left = 200
        self._saved_right = 250
        self.log_signal.connect(self._append_log)
        self.progress_signal.connect(self._update_progress)
        self.image_list_signal.connect(self._populate_image_list)
        self.mask_stats_signal.connect(self._on_mask_stats_ready)
        self.infer_progress_signal.connect(self._update_infer_progress)
        self.stats_signal.connect(self._update_stats)
        self.stats_signal.connect(lambda a, p: self._update_ann_stats())
        self._stats_timer = QTimer()
        self._stats_timer.setSingleShot(True)
        self._stats_timer.timeout.connect(self._do_update_stats)
        self._detection_mode = False
        self._cls_mode = False
        self._mixed_cls_mode = False
        self._ocr_mode = False
        self._ocv_mode = False
        self._box_manager = None
        self._page_size = 50
        self._current_page = 0
        self._filtered_data = []
        # ── 标记（对标 iCogtiveFlow 的四态：正常/过杀/漏杀/存疑）──
        # 与 label 物理分离：不进训练集，只做统计与筛选，是难例闭环的入口。
        self._marks = {}            # base -> {mark,note,by,at}
        self._marks_project = None  # 标记已加载的工程目录（避免每次刷新都读盘）
        self._mark_filter = ""      # 工具栏/统计面板点出来的快速筛选（kind）
        self._defect_filter = ""    # 快速筛选的缺陷类型
        self._filter_expr = FilterExpr()   # 条目式表达式筛选（高/低级筛选可叠加）

        # Install event filter for global Space key handling
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)
        self.chart_signal.connect(self._update_loss_chart)
        self.batch_signal.connect(self._update_batch_status)
        self.refresh_list_signal.connect(lambda: self._filter_images(self.search_input.text()))
        self._init_ui()
        self._init_augment_panel()
        self._refresh_projects()

        # System monitor in status bar (bottom-right)
        self._sys_monitor = SystemMonitor()
        self.statusBar().addPermanentWidget(self._sys_monitor)

        # 全局热键：Shift+X 截图 OCR（窗口未聚焦也能触发）
        self._screen_ocr_busy = False
        self._register_screen_ocr_hotkey()


    def closeEvent(self, event):
        """Clean up system monitor on close."""
        if hasattr(self, "_sys_monitor"):
            self._sys_monitor.stop()
        # 保存分类训练面板参数，避免重启后回滚
        if getattr(self, "_project_dir", None):
            try:
                self._save_cls_train_settings()
            except Exception:
                pass
        if getattr(self, "_screen_ocr_global", False):
            try:
                unregister_global_hotkey(int(self.winId()), self._screen_ocr_hotkey_id)
            except Exception:
                pass
        super().closeEvent(event)
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
        # 左栏 200 -> 240：左栏重构后左边多了一条 40px 的竖排页签条，
        # 补回这 40px，**内容区宽度与重构前一致**（不然图片列表平白窄一截）。
        self._main_splitter.setSizes([240, 800, 300])
        self._main_splitter.setStretchFactor(0, 0)  # left: no stretch
        self._main_splitter.setStretchFactor(1, 1)  # canvas: stretches
        self._main_splitter.setStretchFactor(2, 0)  # right: no stretch

        # Log panel in a vertical splitter so it's resizable
        self.log_widget = QTextEdit(); self.log_widget.setReadOnly(True)
        # ★ 2026-10-23：日志控件原来**没有上限**，批量推理 489 张能堆几千行、只增不减。
        #   2 万行足够回溯一次完整训练，再多也只是拖慢滚动。
        try:
            # ⚠️ QTextEdit 本身没有 setMaximumBlockCount（那是 QPlainTextEdit 的），
            #   要挂到它的 document 上，否则这行会静默失败、日志依旧无限增长。
            self.log_widget.document().setMaximumBlockCount(20000)
        except Exception:
            pass
        self.log_widget.setFont(QFont("Consolas", 9))
        self.log_widget.setMinimumHeight(0)
        self._log_splitter = QSplitter(Qt.Vertical)
        self._log_splitter.addWidget(self._main_splitter)
        self._log_splitter.addWidget(self.log_widget)
        self._log_splitter.setSizes([700, 150])
        self._log_splitter.setStretchFactor(0, 1)  # main content stretches
        self._log_splitter.setStretchFactor(1, 0)  # log: fixed/minimum
        self._log_splitter.setCollapsible(1, True)
        # 拖日志分隔条会改左栏高度（左栏挂在 main_splitter 里），但**不会**触发
        # MainWindow.resizeEvent ⇒ 表格那个"最多能长多高"得在这里补一次重量，
        # 否则拖过之后表格就一直停在旧限制上（收下去后展不开）。
        self._log_splitter.splitterMoved.connect(lambda *_: self._schedule_fit_list_height())
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
        tm.addAction("Image Classification", lambda: self._set_task("classification"))
        tm.addAction("Mixed Classification", lambda: self._set_task("mixed_classification"))
        tm.addSeparator()
        tm.addAction("OCR Text Recognition", lambda: self._set_task("ocr"))
        self._act_screen_ocr = tm.addAction("Screen OCR (截图识别)...", self._open_screen_ocr)
        # 快捷键统一在 _register_screen_ocr_hotkey() 里设置：
        # 优先注册系统级全局热键（窗口未聚焦也能触发），失败则回退为窗口内快捷键
        tm.addAction("OCV Quality Inspection", lambda: self._set_task("ocv"))
        tm.addSeparator()
        tm.addAction("Image Crop Tool...", self._open_crop_tool)
        tm.addAction("Image Resize Tool...", self._open_resize_tool)
        tm.addAction("Heatmap Region Analyzer...", self._open_heatmap_analyzer)
        tm.addAction("分类阈值调优 (过杀/漏检)...", self._open_threshold_panel)
        # 「管理数据集划分…」的**零门槛兜底入口**：图片列表右键里也有一条，但那条要
        # 先选中至少一行才出菜单（`_on_image_context_menu` 的早退），而"划分"是工程级动作、
        # 跟选没选中图无关 —— 没这个兜底，面板按钮删掉后就会出现"想划分却点不到"。
        tm.addAction("数据集划分管理...", self._manage_split)
        tm.addSeparator()
        tm.addAction("Flow Editor...", self._open_flow_editor)
        sm = mb.addMenu("Settings(&S)")
        sm.addAction("Workspace...", self._set_workspace)
        vm = mb.addMenu("View(&V)")
        vm.addAction("Toggle Log Panel	Ctrl+L", self._toggle_log_panel)
        vm.addAction("Toggle Left Panel	Ctrl+[", lambda: self._toggle_panel("left"))
        vm.addAction("Toggle Right Panel	Ctrl+]", lambda: self._toggle_panel("right"))
        vm.addSeparator()
        vm.addAction("Toggle Detection Mode	Ctrl+D", self._toggle_detection_mode)
        vm.addAction("Refresh Projects", self._refresh_projects)
        # 主题（深色/浅色）：勾选框指示器在深色下容易看不见，给个可切换的主题
        tm = vm.addMenu("主题 / Theme")
        for _label, _key in (("深色（默认）", "dark"), ("浅色", "light")):
            act = tm.addAction(_label)
            act.setCheckable(True)
            act.setChecked(self._load_ui_setting("theme", "dark") == _key)
            act.triggered.connect(lambda _c, k=_key: self._set_theme(k))
        # ⚠️ Image Mgmt / Annotation Mgmt / Model Mgmt 三个英文菜单**已删除**：
        #    它们和右侧「图像/标注/模型管理」模块完全重叠，而且同名不同内容
        #    （菜单里的 Model Mgmt 管的是训练/推理，右侧模块管的是模型文件），
        #    同一动作两个入口迟早出现"改了一处、另一处行为不同"。
        #    其中原先**只有菜单能点**的四项已补进面板：
        #      导入图片…            -> 左栏图片列表右键菜单（原在"图像管理/输出与工程"，
        #                              2026-10-14 该页删除时改挂到这里）
        #      导出 JSON / 导出标注 / 清理碎标注 -> 左栏图片列表右键菜单（原"标注管理/标注操作"）
        #      批量推理（全部图片）…  -> 模型管理 / 模型列表
        hm = mb.addMenu("Help(&H)")
        # 参照友商「帮助」：日志 / 使用说明 / 开发文档 / 示例工程 / 快捷键提示
        hm.addAction("日志面板\tCtrl+L", self._toggle_log_panel)
        hm.addAction("使用说明", self._open_user_guide)
        hm.addAction("开发文档", self._open_dev_docs)
        hx = hm.addMenu("示例工程")
        hx.addAction("生成三视图示例工程…", self._create_example_project)
        hx.addAction("打开文档目录", lambda: self._open_dir(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "docs")))
        hm.addAction("快捷键提示\tF1", self._show_shortcuts)
        hm.addSeparator()
        hm.addAction("About(&A)", self._about)

    # ────────── 帮助 ──────────

    def _open_doc(self, filename):
        """用系统默认程序打开 docs/ 下的文档（缺失时给可用替代并写日志）。"""
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
        path = os.path.normpath(os.path.join(base, filename))
        if not os.path.exists(path):
            alt = os.path.normpath(os.path.join(base, "架构文档_算子流重构路线.md"))
            self.log(f"[帮助] 找不到 {filename}，改开 {os.path.basename(alt)}")
            path = alt
        try:
            os.startfile(path)
            self.log(f"[帮助] 已打开 {os.path.basename(path)}")
        except Exception as e:
            self.log(f"[帮助] 打开文档失败: {e}")

    def _open_user_guide(self):
        self._open_doc("使用说明.md")

    def _open_dev_docs(self):
        self._open_doc("开发文档.md")

    def _show_shortcuts(self):
        """快捷键提示（只列代码里真实存在的，不写没实现的）。"""
        rows = [
            ("预览 / 标注", "Ctrl+Space", "显示或隐藏**缺陷类别**叠加层"),
            ("", "Ctrl+Alt（或 Ctrl+Alt+I）", "显示或隐藏**忽略区**（__ignore__）"),
            ("", "Space", "显示或隐藏**预测结果**"),
            ("", "Ctrl+Z / Ctrl+Y", "撤销 / 重做"),
            ("", "Ctrl+S", "保存当前标注（编辑时也会自动保存）"),
            ("", "B / L / R / C / O / P", "画笔 / 直线 / 矩形 / 圆 / 椭圆 / 多边形"),
            ("", "E / I / A / H", "橡皮 / 忽略区 / SAM / 平移"),
            ("", "1 ~ 9", "切换当前类别"),
            ("", "Enter / Esc / Backspace", "多边形：闭合 / 取消 / 退一个点"),
            ("界面", "Ctrl+L", "日志面板显隐"),
            ("", "Ctrl+[ / Ctrl+]", "左栏 / 右栏显隐"),
            ("", "Ctrl+D", "切换检测模式"),
            ("", "Ctrl+A", "图片列表全选（列表右键里也能选）"),
            ("", "F1", "本提示"),
        ]
        html = ("<table cellspacing='0' cellpadding='4'>"
                "<tr><th align='left'>范围</th><th align='left'>快捷键</th>"
                "<th align='left'>说明</th></tr>")
        for scope, keys, desc in rows:
            html += (f"<tr><td style='color:#8A93A6'>{scope}</td>"
                     f"<td><b>{keys}</b></td><td>{desc.replace('**','')}</td></tr>")
        html += "</table>"
        box = QMessageBox(self)
        box.setWindowTitle("快捷键提示")
        box.setTextFormat(Qt.RichText)
        box.setText(html)
        box.exec_()

    def _create_example_project(self):
        """生成一个小的三视图示例工程（合成图+JSON+掩膜），用来试功能/给新人练手。

        为什么要"生成"而不是"放一个现成目录"：示例工程里是合成图，
        可以随代码一起演进；也不用往用户工程目录里塞几十兆真实图片。
        """
        try:
            name = "示例工程_三视图分割"
            pd = os.path.join(str(self.pm.workspace), name)
            os.makedirs(os.path.join(pd, "images"), exist_ok=True)
            os.makedirs(os.path.join(pd, "annotations"), exist_ok=True)
            os.makedirs(os.path.join(pd, "models"), exist_ok=True)
            os.makedirs(os.path.join(pd, "outputs"), exist_ok=True)
            classes = ["background", "Zw", "Ys", "Zz", "Hh", "Atd"]
            with open(os.path.join(pd, "project.json"), "w", encoding="utf-8") as f:
                json.dump({"name": name, "task_type": "语义分割", "classes": classes,
                           "views": ["te_0", "te_1", "te_2"], "view_count": 3,
                           "in_channels": 9,
                           "view_names": {"te_0": "漫反射", "te_1": "形状图1", "te_2": "形状图2"},
                           "class_cn": {"background": "背景", "Zw": "脏污", "Ys": "压伤",
                                        "Zz": "褶皱", "Hh": "划痕", "Atd": "凹凸点"}},
                          f, indent=2, ensure_ascii=False)
            import numpy as _np
            from PIL import Image as _Img
            H, W = 480, 640
            rng = _np.random.default_rng(7)
            for gi in range(3):
                key = "示例电芯%02d" % (gi + 1)
                # 三视图：同一块区域、不同"成像"
                base = _np.full((H, W), 45, _np.uint8)
                base = _np.clip(base + rng.normal(0, 4, (H, W)), 0, 255).astype(_np.uint8)
                shapes, mask = [], _np.zeros((H, W), _np.uint8)
                # 一个压伤(Ys=2) + 一个脏污(Zw=1)
                for cid, (cx, cy, w, h) in ((2, (180 + 40 * gi, 160, 26, 14)),
                                            (1, (420, 300 + 20 * gi, 34, 22))):
                    shapes.append({"label": classes[cid], "shape_type": "polygon",
                                   "points": [[cx - w // 2, cy - h // 2], [cx + w // 2, cy - h // 2],
                                              [cx + w // 2, cy + h // 2], [cx - w // 2, cy + h // 2]],
                                   "flags": {}, "group_id": None})
                    mask[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2] = cid
                # 忽略区
                shapes.append({"label": "__ignore__", "shape_type": "polygon",
                               "points": [[0, H - 60], [W, H - 60], [W, H], [0, H]],
                               "flags": {}, "group_id": None})
                mask[H - 60:, :] = 255
                for vi, v in enumerate(("te_0", "te_1", "te_2")):
                    img = base.copy()
                    if vi == 1:
                        img = _np.clip(255 - img.astype(_np.int16), 0, 255).astype(_np.uint8)
                    elif vi == 2:
                        img = (img // 2 + 20).astype(_np.uint8)
                    img = _np.stack([img] * 3, axis=2)
                    img[mask == 2] = (img[mask == 2] * 0.6 + _np.array([90, 90, 0]) * 0.4)
                    f = os.path.join(pd, "images", f"{key}_{v}.jpg")
                    _Img.fromarray(img).save(f, quality=92)
                    with open(os.path.join(pd, "images", f"{key}_{v}.json"), "w", encoding="utf-8") as jf:
                        json.dump({"version": "5.3.1", "flags": {}, "imageData": None,
                                   "imagePath": f"{key}_{v}.jpg", "imageWidth": W, "imageHeight": H,
                                   # ★ 空白标注不预设任何数据集：以前写 "Train"，
                                  #   于是导入 500 张图就有 500 张"未标注的训练集"，全都不该训。
                                  "imageSet": "", "shapes": shapes}, jf, ensure_ascii=False)
                _Img.fromarray(mask, mode="L").save(
                    os.path.join(pd, "annotations", f"{key}_te_0_mask.png"))
            self._refresh_projects()
            self.log(f"[示例工程] 已生成 {pd}")
            QMessageBox.information(self, "示例工程",
                                    f"示例工程已生成：\n{pd}\n\n"
                                    "含 3 组三视图 + 标注 JSON + 掩膜（Zw/Ys + 忽略区）。\n"
                                    "左栏工程列表里双击即可打开。")
        except Exception as e:
            self.log(f"[示例工程] 生成失败: {type(e).__name__}: {e}")
            QMessageBox.warning(self, "示例工程", f"生成失败: {e}")

    def _create_left_panel(self):
        # ★ 2026-10-15 改成「**常驻把手条 + 可折叠内容**」：
        #   原来折叠按钮在面板**里面**，一按就跟着面板一起消失 ⇒ 想再展开只能去 View 菜单，
        #   而鼠标明明就停在刚才那个位置（维护者 反馈"这很不方便"）。
        #   现在把手条（只放那一枚箭头）**永远可见**，收起时只藏内容：
        #     展开态 = 把手 + 内容；折叠态 = 只剩把手、箭头反向 ⇒ 点它就展开。
        #   `self._left_panel` 现在是这个**容器**（splitter 里放的仍是它，引用不用改）。
        container = QWidget()
        _cl = QHBoxLayout(container)
        _cl.setContentsMargins(0, 0, 0, 0)
        _cl.setSpacing(0)

        handle = QWidget()
        _hl = QVBoxLayout(handle)
        _hl.setContentsMargins(2, 2, 2, 2)
        _hl.setSpacing(0)
        btn_left = QPushButton("\u25c0")
        btn_left.setFixedSize(20, 20)
        btn_left.setToolTip("收起左栏（再点一下展开；Ctrl+[）")
        btn_left.clicked.connect(lambda: self._toggle_panel("left"))
        _hl.addWidget(btn_left)
        _hl.addStretch()
        _cl.addWidget(handle)
        self._left_handle = handle
        self._left_toggle_btn = btn_left

        p = QWidget(); outer_layout = QVBoxLayout(p); outer_layout.setContentsMargins(2, 2, 2, 2)
        outer_layout.setSpacing(2)
        _cl.addWidget(p, 1)
        self._left_content = p          # 折叠时被 setVisible(False) 的就是它

        # ==== 主体：竖排页签（最左） + 页栈（右）====
        # 参照右栏那套（`_create_right_panel`）：40px 宽的竖排按钮条 + QStackedWidget，
        # 只是把页签条摆到**最左端** —— 整窗左右各一条竖栏，视觉上对称。
        #
        # 为什么必须从"上下叠放"改成"分页"：
        #   原来工程树 `setMaximumHeight(150)` 钉死，图片列表吃"剩下的那点高度"，
        #   两者**互相抢空间**（用户 2026-10-14 反馈左栏"挺丑的"就是这个）。
        #   分页之后每一页都拿到**整栏高度**，工程多 / 图片多都不再互相压。
        body = QHBoxLayout(); body.setContentsMargins(0, 0, 0, 0); body.setSpacing(0)

        tab_bar = QWidget(); tab_bar.setFixedWidth(40)
        tab_layout = QVBoxLayout(tab_bar)
        tab_layout.setContentsMargins(1, 2, 1, 2); tab_layout.setSpacing(2)

        # ⚠️ 这里**不能**用 setExclusive(True)：独占组里"再点一次已选中的那个"点不下去，
        #    而本轮的诉求恰恰是"再点一下 = 收起"（照友商：点图像管理即显示/隐藏图像列表）。
        #    所以改成非独占、互斥自己管 —— 见 `_on_left_tab_clicked`。
        self._left_tabs = QButtonGroup(self); self._left_tabs.setExclusive(False)
        #: 左栏模块页下标（与右栏 `_right_tab_index` 同一个套路：按名字取，不写死数字）
        self._left_tab_index = {"proj": 0, "img": 1, "task": 2}
        tab_defs = [
            ("工程管理", "工程列表：打开 / 新建 / 删除工程", "proj"),
            ("图像管理", "图片列表与标记统计", "img"),
            # 任务队列（2026-10-16，对标友商同名模块）：训练/推理统一登记，
            # 显示进度与剩余时间，每行可取消。串行执行（单卡并发只会 OOM）。
            ("任务队列", "训练 / 推理任务队列：进度、剩余时间、取消", "task"),
        ]
        self._left_tab_btns = {}
        for icon_text, tooltip, key in tab_defs:
            # 竖排：一个字一行（与右栏完全一致；横排 4 个字塞进 40px 会溢出到按钮外面）
            btn = QPushButton("\n".join(icon_text))
            btn.setCheckable(True)
            btn.setToolTip(tooltip + f"（{icon_text}：点一下显示，再点一下收起）")
            btn.setMinimumHeight(56)
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            btn.setStyleSheet(_V_TAB_QSS)          # 与右栏共用同一份样式
            # ⚠️ 用 clicked 而不是 toggled：clicked 只在**用户点击**后发，
            #    所以 `setChecked()` 回写状态时不会绕回来（无需 blockSignals 那套）。
            btn.clicked.connect(lambda checked, k=key: self._on_left_tab_clicked(k, checked))
            self._left_tabs.addButton(btn)
            self._left_tab_btns[key] = btn
            tab_layout.addWidget(btn, 1)          # ← stretch=1：两块均分整条竖栏
        body.addWidget(tab_bar)
        # ⚠️ 页签条必须**钉在最左端**，不能跟着内容收起而居中：
        #    `_left_stack` 一隐藏，body 里就只剩 tab_bar 一个控件，而它被
        #    setFixedWidth(40) 钉死、吃不下多余宽度 ⇒ QHBoxLayout 会把剩下的
        #    237px **平均分到两边**，竖条就飘到栏中间去了（实测踩到）。
        #    微测试（tmp/micro_align2.py）对比过四种修法，只有 AlignLeft 既能贴左
        #    又不影响展开态（常驻 addStretch 会让页栈只拿到一半宽）。
        body.setAlignment(tab_bar, Qt.AlignLeft)

        self._left_stack = QStackedWidget()

        # ── 页 0：工程管理（工程列表）──
        page_proj = QWidget()
        pl = QVBoxLayout(page_proj); pl.setContentsMargins(0, 2, 0, 0); pl.setSpacing(2)
        pl.addWidget(self._make_block_header("工程列表"))

        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderLabels(["Name", "Time"])
        # Name 列宽度根据工程名自适应，避免工程名被截断、用户手动拉伸
        self.project_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.project_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.project_tree.customContextMenuRequested.connect(self._on_project_context_menu)
        self.project_tree.itemDoubleClicked.connect(self._on_project_double_click)
        self.week_item = QTreeWidgetItem(["Within 1 week"])
        self.month_item = QTreeWidgetItem(["Within 1 month"])
        self.older_item = QTreeWidgetItem(["Older"])
        self.project_tree.addTopLevelItems([self.week_item, self.month_item, self.older_item])
        # ⚠️ 这里原来钉死 `setMaximumHeight(150)`，理由是"别把下面的图片列表挤没了"。
        #    现在两者是**两页**、不会再互相挤，所以把这个限制撤掉：
        #    工程多的时候让它吃满整页（自带滚动条），不必再手动拖分隔条。
        pl.addWidget(self.project_tree, 1)
        self._left_stack.addWidget(page_proj)      # 0

        # ── 页 1：图像管理（图片列表 + 标记统计）──
        # 原来这条分隔线（sep）是用来在"上下叠放"里分开工程树与图片列表的，
        # 现在两页互斥显示，线就没有意义了。
        page_img = QWidget()
        pg = QVBoxLayout(page_img); pg.setContentsMargins(0, 0, 0, 0); pg.setSpacing(1)

        # ⚠️ 2026-10-15（第二轮）用户判定这整行都冗余，已删：
        #   · 「图片列表 (n)」标题 -> 底部已有「共 N 组（M 张）」+ 左侧竖排页签「图像管理」，
        #     同一个信息说了三遍；计数唯一来源交给 _update_accordion_counts 写底部。
        #   · 「标记统计」按钮 -> 标记工具条里的「统计」按钮是同一个动作（_switch_accordion(1)），
        #     入口只留一处；从统计页返回走 MarkStatsPanel 顶部的「← 返回列表」（back_requested）。
        #    这行删掉后，表格直接顶到筛选行下面，省 26px。

        # Accordion content: QStackedWidget (fills all remaining space)
        self._accordion_stack = QStackedWidget()
        pg.addWidget(self._accordion_stack, 1)  # stretch=1

        # -- Page 2: Image List --
        imglist_page = QWidget(); il = QVBoxLayout(imglist_page); il.setContentsMargins(0, 2, 0, 0); il.setSpacing(1)

        # ── 筛选区：两行，按"用得多的在前、用得少的在后"排 ──
        # 原来 5 个控件挤一行（Search / All Sets / Annotated / All Labels / Recent），
        # 在 ~480px 的左栏里最后一个「Recent」直接被截掉看不见（用户 2026-10-14 截图里
        # 就是截断的样子）。现在：行1 = 搜索 + 两个"选什么"的下拉；
        # 行2 = Pred 门限 + 两个"要不要"的勾选（原来 Pred 独占一行、右边一大片空白）。
        ft = QHBoxLayout(); ft.setContentsMargins(0, 0, 0, 0); ft.setSpacing(2)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.setMinimumWidth(110)
        self.search_input.textChanged.connect(self._filter_images)
        ft.addWidget(self.search_input, 1)
        self.split_filter = QComboBox()
        # 对标友商 友商平台 的筛选下拉：除数据集外，还要能按"是否标注 / 是否有结果"筛。
        # 后半段这几项在 _filter_images 里实现。
        self.split_filter.addItems(["All Sets", "Train Only", "Val Only", "Test Only",
                                    "Annotated Only", "Unannotated Only",
                                    "Has Result", "No Result"])
        self.split_filter.setToolTip("筛选：数据集 / 已标注 / 未标注 / 有结果 / 无结果")
        self.split_filter.currentIndexChanged.connect(lambda: self._filter_images(self.search_input.text()))
        ft.addWidget(self.split_filter)
        # 标注名字筛选（对标友商下拉里的「标注名字」）：选一个类别，只看含该类的图。
        # 选项在 _filter_images 里按工程类别懒同步，不必到处挂钩子。
        self.label_filter = QComboBox()
        self.label_filter.addItem("All Labels")
        self.label_filter.setToolTip("只看含指定标注类别的图片")
        self.label_filter.currentIndexChanged.connect(
            lambda: self._filter_images(self.search_input.text()))
        ft.addWidget(self.label_filter)
        il.addLayout(ft)
        # Pred filter row
        pf = QHBoxLayout(); pf.setContentsMargins(0, 0, 0, 0); pf.setSpacing(2)
        pf.addWidget(QLabel("Pred:"))
        self.pred_op_combo = QComboBox()
        self.pred_op_combo.addItems(["\u2265", "=", "\u2264"])
        self.pred_op_combo.currentIndexChanged.connect(lambda: self._filter_images(self.search_input.text()))
        pf.addWidget(self.pred_op_combo)
        self.pred_count_spin = QSpinBox(); self.pred_count_spin.setRange(0, 9999); self.pred_count_spin.setValue(0)
        self.pred_count_spin.setToolTip("0=disabled")
        self.pred_count_spin.valueChanged.connect(lambda: self._filter_images(self.search_input.text()))
        pf.addWidget(self.pred_count_spin)
        pf.addSpacing(10)
        self.annotated_check = QCheckBox("Annotated")
        self.annotated_check.toggled.connect(lambda: self._filter_images(self.search_input.text()))
        pf.addWidget(self.annotated_check)
        self.recent_check = QCheckBox("Recent")
        self.recent_check.setToolTip("Sort by annotation date")
        self.recent_check.toggled.connect(lambda: self._filter_images(self.search_input.text()))
        pf.addWidget(self.recent_check)
        pf.addStretch()
        il.addLayout(pf)

        # ── 标记工具条 + 表达式筛选（对标 iCogtiveFlow 的标记栏） ──
        # 作用域 = 列表里选中的行（支持多选批量）；统计按钮切到右侧「标记统计」页。
        self.mark_toolbar = MarkToolbar()
        self.mark_toolbar.mark_requested.connect(self._on_mark_requested)
        self.mark_toolbar.note_requested.connect(self._on_mark_note)
        self.mark_toolbar.stats_requested.connect(lambda: self._switch_accordion(1))
        il.addWidget(self.mark_toolbar)

        # ── 筛选操作行 ──
        # ⚠️ 2026-10-15 用户反馈左栏列表上方控件堆了太多行、按钮又挤又小：
        #   · 「筛选…」（条目式高级筛选）低频 → 收进图片列表右键菜单（见 _on_image_context_menu）
        #   · 「结果转标注…」 → 右键菜单「标记」子菜单里已有同名入口（_run_convert_by_mark），
        #     面板上删掉，不再重复挂第三个地方
        #   · 「清除筛选」是高频 + mark_filter_label 的提示文案指向它 → 保留在面板上
        mf = QHBoxLayout(); mf.setContentsMargins(0, 0, 0, 0); mf.setSpacing(2)
        self.btn_filter_clear = QPushButton("清除筛选")
        self.btn_filter_clear.setToolTip("清掉条目式筛选/标记筛选/缺陷筛选，恢复显示全部图片")
        self.btn_filter_clear.clicked.connect(self._clear_mark_filters)
        mf.addWidget(self.btn_filter_clear)
        mf.addStretch(1)
        il.addLayout(mf)
        self.mark_filter_label = QLabel("")
        self.mark_filter_label.setWordWrap(True)
        self.mark_filter_label.setVisible(False)
        il.addWidget(self.mark_filter_label)

        # Image table
        self.image_list_widget = QTableWidget()
        self.image_list_widget.setColumnCount(9)
        self.image_list_widget.setHorizontalHeaderLabels(
            ["序号", "图片名", "大小", "数据集", "标注", "结果", "分数", "耗时ms", "标记"])
        # 「分数」= 该图所有检出块的**平均模型置信度**（与 Result 表同口径），
        # 没有置信度图（<base>_prob.png）时留空 —— 别让人误以为是 IoU。
        try:
            self.image_list_widget.horizontalHeaderItem(6).setToolTip(
                "分数：本图所有检出块的平均模型置信度（0-100%）\n"
                "没有置信度图时留空（它不再表示与标注的 IoU）")
        except Exception:
            pass
        # ⚠️ 不要让「标记」列当拉伸列：stretchLastSection 会把它在宽面板下拉成一大条，
        #    而它承载的是"一个色块 + 两个字"，宽度固定 54px 才是它的正确形态。
        #    多余宽度交给「图片名」（它本来就是最长的字段）。
        self.image_list_widget.horizontalHeader().setStretchLastSection(False)
        self.image_list_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.image_list_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.image_list_widget.setSelectionMode(QTableWidget.ExtendedSelection)
        # ⚠️ 选中态配色**不能写死在这里**：这段内联样式优先级高于全局主题表，
        #    浅色主题下会变成"深蓝底 + 青色字"，而「标注/结果」两列的文字颜色是
        #    代码里按类别色单独 setForeground 过的（内联前景色又压过 QSS 的 color）
        #    -> 深色类别字压在深蓝选中底上 = **整行看不见**（用户实测反馈）。
        #    现在改成 objectName 交给主题表管：深色给深底青字，浅色给浅底深字。
        self.image_list_widget.setObjectName("imageList")
        self.image_list_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.image_list_widget.setAlternatingRowColors(True)
        self.image_list_widget.verticalHeader().setVisible(False)
        self.image_list_widget.verticalHeader().setDefaultSectionSize(22)
        self.image_list_widget.currentCellChanged.connect(self._on_current_cell_changed)
        self.image_list_widget.itemSelectionChanged.connect(self._update_selection_count)
        self.image_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list_widget.customContextMenuRequested.connect(self._on_image_context_menu)
        self.image_list_widget.setColumnWidth(0, 40)
        self.image_list_widget.setColumnWidth(1, 200)
        self.image_list_widget.setColumnWidth(2, 100)
        self.image_list_widget.setColumnWidth(3, 60)
        self.image_list_widget.setColumnWidth(4, 50)
        self.image_list_widget.setColumnWidth(5, 50)
        self.image_list_widget.setColumnWidth(6, 60)
        # 第 7 列（耗时ms）以前**没设宽度**、用的是默认值 100px：列头还只写「耗时」，
        # 既白占宽度又看不出单位，而值本身以前存的还是秒。宽度给够列头、值本身不超过 4 位。
        self.image_list_widget.setColumnWidth(7, 62)
        self.image_list_widget.setColumnWidth(8, 54)
        il.addWidget(self.image_list_widget, 1)

        # Pagination
        pag_bar = QHBoxLayout(); pag_bar.setContentsMargins(0, 0, 0, 0); pag_bar.setSpacing(2)
        self.stats_summary = QLabel(""); pag_bar.addWidget(self.stats_summary)
        pag_bar.addStretch()
        pag_bar.addWidget(QLabel("Per page:"))
        self.page_size_combo = QComboBox()
        # "全部" = 不分页，一次列出所有图片（滑块会随数量变短），方便直接全选。
        # 对标友商 友商平台：列表不分页，一眼能看全并 Ctrl+A。
        self.page_size_combo.addItems(["50", "100", "200", "500", "全部"])
        self.page_size_combo.setCurrentText("全部")
        self.page_size_combo.setToolTip("每页显示数量；选「全部」不分页，便于全选（图片很多时会略慢）")
        self.page_size_combo.currentTextChanged.connect(self._on_page_size_changed)
        # 注意：上面是先 setCurrentText 再 connect，所以这里必须手动同步初值，
        # 否则下拉显示"全部"而 _page_size 还是 50，行为与界面不一致。
        self._page_size = 10 ** 9
        pag_bar.addWidget(self.page_size_combo)
        self.page_prev_btn = QPushButton("\u25c0"); self.page_prev_btn.setFixedWidth(30)
        self.page_prev_btn.clicked.connect(lambda: self._go_to_page(self._current_page - 1))
        pag_bar.addWidget(self.page_prev_btn)
        self.page_info_label = QLabel("Page 0/0"); pag_bar.addWidget(self.page_info_label)
        self.page_next_btn = QPushButton("\u25b6"); self.page_next_btn.setFixedWidth(30)
        self.page_next_btn.clicked.connect(lambda: self._go_to_page(self._current_page + 1))
        pag_bar.addWidget(self.page_next_btn); pag_bar.addStretch()
        il.addLayout(pag_bar)
        self.list_status = QLabel(""); il.addWidget(self.list_status)

        # ⚠️ 这行位置**曾短暂放过**两个小按钮（「管理划分…」「输出目录」）：右栏「图像管理」
        #    页整页删除时，把这两个"别处没有入口"的动作搬到了这里。
        #    维护者 当晚看到实机后判定**这一行本身冗余**（面板上多一行视觉噪音），已删。
        #    现在两个动作都只在**右键菜单 / 菜单栏**里，面板上不占位：
        #      · 打开输出目录     -> 图片列表右键 + 工程树右键（`_open_output_dir`）
        #      · 管理数据集划分…  -> 图片列表右键 + 顶部 Tools（`_manage_split`）
        #    ⚠️ 别再把它们摆回面板：`输出目录` 光右键就有两处，再挂面板就是第三处入口。
        # 表格收成内容高度后，空出来的那截必须有个"接盘侠"，否则 Qt 会把它**平摊给其它
        # 控件**：实测（--dump-layout）标记工具栏被撑成 69px、状态行被撑成 68px，
        # 按钮和文字各自飘在一大块空白中间，比原来"表格下面一片空白"更难看。
        # 所以尾部放一个可切换的弹簧：表格被限高时它 Expanding 吸走余量（余量落在最下面），
        # 表格撑满时它退回 Fixed 让位给表格（否则弹簧会和表格五五分）。
        self._list_bottom_spacer = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Fixed)
        il.addSpacerItem(self._list_bottom_spacer)
        self._accordion_stack.addWidget(imglist_page)  # 0
        # 1：标记统计页（四态计数与占比 + 缺陷类型 × 标记 交叉表）
        self.mark_stats = MarkStatsPanel()
        self.mark_stats.cell_filter_requested.connect(self._on_mark_cell_filter)
        self.mark_stats.kind_filter_requested.connect(self._on_mark_kind_filter)
        # 页头行删除后（2026-10-15），从统计页不带筛选回列表的唯一入口
        self.mark_stats.back_requested.connect(lambda: self._switch_accordion(0))
        self._accordion_stack.addWidget(self.mark_stats)

        # Default: show Image List
        self._switch_accordion(0)

        # 收口：图片页进栈 → 栈进主体 → 主体进左栏
        self._left_stack.addWidget(page_img)       # 1

        # ── 页 2：任务队列 ──
        from ui.task_queue import TaskQueue
        from ui.task_queue_panel import TaskQueuePanel
        if getattr(self, "_task_queue", None) is None:
            self._task_queue = TaskQueue()
        page_task = QWidget()
        pt = QVBoxLayout(page_task); pt.setContentsMargins(0, 0, 0, 0); pt.setSpacing(0)
        self.task_queue_panel = TaskQueuePanel(self._task_queue, page_task)
        self.task_queue_panel.cancel_requested.connect(self._on_task_cancel)
        self.task_queue_panel.clear_requested.connect(self._on_task_clear)
        pt.addWidget(self.task_queue_panel, 1)
        self._left_stack.addWidget(page_task)      # 2
        body.addWidget(self._left_stack, 1)
        outer_layout.addLayout(body, 1)            # stretch=1：页栈吃掉整栏余高

        # ==== BOTTOM: New Project (fixed) ====
        # 位置**不动**（用户明确说过别挪它）：仍在左栏最底下、不随页签收起而消失。
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setFrameShadow(QFrame.Sunken)
        outer_layout.addWidget(sep2)
        btn_new = QPushButton("+ New Project"); btn_new.clicked.connect(self._new_project)
        outer_layout.addWidget(btn_new)

        # ★ 导入图片 —— 放在 New Project 下面（它的位置按注释要求不动）。
        # 样式**故意不写 setStyleSheet**：主题扫描器会处理内联 qss，上次加 :disabled
        # 选择器直接把程序搞崩了（40 秒内必退，无 traceback）。跟隔壁按钮同款最稳。
        btn_imp = QPushButton("＋ 导入图片")
        btn_imp.clicked.connect(self._import_images)
        self._btn_import_images = btn_imp
        outer_layout.addWidget(btn_imp)

        # 供「表格高度自适应内容」用：需要页面高度与"除表格外的其它行高度"来算可用高度
        # （见 _fit_list_height 里为什么不能直接量 table.height()）。
        self._list_page = imglist_page
        self._list_layout = il

        # Default: 落在「工程管理」（维护者 2026-10-15 拍板：重启后先看工程列表）。
        # 为什么这样更合理：没打开工程时图片列表**本来就是空的**，落在一个空列表页
        # 等于"右侧画布是灰的、左边也是空的"，用户第一眼无事可做；
        # 落在工程列表则一眼看到"有哪些工程、上次做到哪儿"。
        # 配套：打开工程成功后会自动切到「图像管理」（见 _open_project_by_name 末尾），
        # 否则从工程列表打开工程后还停在一页空图片列表上。
        self.show_left_tab("proj")
        return container        # 把手条 + 内容；splitter 里放的是它

    def _fit_list_height(self):
        """图片列表高度跟着内容走：行少时收成内容高度，行多时照旧撑满。

        用户 2026-10-14 反馈："表格下方一大片空白"。场景是**筛完之后**列表只剩几行，
        而表格仍按拉伸占满剩余高度，看着像没加载完。

        ⚠️ 可用高度**不能**拿 `table.height()` 量：表格一旦被限过高度，再量到的就是
        被限后的那个值 ⇒ 限制永远解不开（自锁，窗口拉大表格也不跟着长）。
        所以这里按"页面高度 - 其它行高度 - 行间距"算，与表格当前被限没被限无关。
        """
        t = getattr(self, "image_list_widget", None)
        lay = getattr(self, "_list_layout", None)
        page = getattr(self, "_list_page", None)
        if t is None or lay is None or page is None:
            return
        try:
            # 切到"标记统计"页时列表页是隐藏的（高度 0）：此时不量、也不动限制
            if page.height() <= 0 or not page.isVisible():
                return
            other = 0
            for i in range(lay.count()):
                it = lay.itemAt(i)
                w = it.widget()
                if w is t:
                    continue
                if w is not None:
                    # ⚠️ 隐藏的控件必须跳过：它不占位，但 height() 给的是**上次的旧高度**。
                    #    实测（2026-10-14，--dump-layout 量出来的）mark_filter_label 隐藏时
                    #    仍报 480px ⇒ 可用高度算成 -249 ⇒ 自适应永远不触发，
                    #    现象就是"改了但看起来没改"。
                    if not w.isVisible():
                        continue
                    other += w.height()
                else:
                    sub = it.layout()
                    if sub is not None:
                        other += sub.geometry().height() or sub.sizeHint().height()
            avail = page.height() - other - lay.spacing() * max(0, lay.count() - 1)
            hh = 0 if t.horizontalHeader().isHidden() else t.horizontalHeader().height()
            body = sum(t.rowHeight(r) for r in range(t.rowCount()))
            need = hh + body + 2 * t.frameWidth() + 6
            # need < avail 才收：行多的时候让 maximumHeight 放开，交给布局去拉伸。
            # 同时切换尾部弹簧（见 _create_left_panel 里的注释）：收起来时余量落到最下面。
            spacer = getattr(self, "_list_bottom_spacer", None)
            if need < avail:
                t.setMaximumHeight(need)
                if spacer is not None:
                    spacer.changeSize(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
            else:
                t.setMaximumHeight(16777215)
                if spacer is not None:
                    spacer.changeSize(0, 0, QSizePolicy.Minimum, QSizePolicy.Fixed)
        except Exception:
            pass

    def _schedule_fit_list_height(self):
        """下一轮事件循环再量高：布局几何要等这一轮结束才落到控件上。"""
        try:
            QTimer.singleShot(0, self._fit_list_height)
        except Exception:
            pass

    # ────────── 工具行折叠 / 展开（2026-10-15 照友商"点标注显示隐藏工具栏"）──────────
    def _on_toolbar_collapsed(self, _on):
        """工具行藏起来 = 画布长高了约 30px ⇒ 把下游"按尺寸排版"的东西都叫醒。

        ⚠️ 必须延后一轮：`toggled` 是在**点击处理中**发的，此刻布局几何还没重算，
           立刻去量画布宽高拿到的是**旧值** ⇒ 重排完还是错的，而且不报错（最难查的那类）。
           （同 `_schedule_fit_list_height` 的道理。）
        """
        def _relayout():
            # 画布/viewer 的重画**不用管**：Qt 在几何变化时会发 resizeEvent，
            # MultiViewPreview 的 resizeEvent 自己把 `_state` 置空并标脏（见该文件 465行）。
            # 这里只补"不靠 resizeEvent 的那一个"：图片信息栏按可用宽度排版。
            # ⚠️ 不要写 `if hasattr(mv, "refresh_state"): ...` 这种兜底调用 ——
            #    方法不存在时它静默什么都不做，看着"排过了"，其实没排。
            try:
                self._schedule_fit_img_info_bar()
            except Exception:
                pass
        try:
            QTimer.singleShot(0, _relayout)
        except Exception:
            pass

    def resizeEvent(self, event):
        """窗口尺寸变了，表格能长多少也跟着变 —— 不重量的话"收起来"就再也展不开了。

        底栏右端同理：宽了就放得下 −/滑块/+，窄了就得收（见 `_fit_img_info_bar`）。
        """
        super().resizeEvent(event)
        self._layout_edge_buttons()      # 边缘恢复按钮是浮层，窗口一变大小就得重贴
        self._schedule_fit_list_height()
        self._schedule_fit_img_info_bar()

    def _switch_accordion(self, idx):
        """切左侧页：0=图片列表，1=标记统计。"""
        # 2026-10-15 第二轮减负后，页头那一整行（标题 + 「标记统计」按钮）已删：
        # 去统计页走标记工具条的「统计」按钮，回列表走统计页里的「← 返回列表」。
        idx = max(0, min(idx, 1))
        self._accordion_stack.setCurrentIndex(idx)
        # 切回图片列表页时页面高度才确定，表格高度自适应要重量一次
        self._schedule_fit_list_height()

    # ────────── 左栏模块页签（参照右栏，2026-10-14 重构）──────────

    def show_left_tab(self, key, notify=True):
        """按**名字**切左栏模块页（"proj" = 工程管理 / "img" = 图像管理）。

        与右栏 `show_right_tab` 同一个套路：按名字取下标，不写死数字 ——
        页序一改，写死的 `setCurrentIndex(1)` 就变成"点了跳到别的页"，且不报错。

        `notify=False` 给构造期用（那时窗口还没显示，算高度没意义）。
        """
        idx = getattr(self, "_left_tab_index", {}).get(key)
        if idx is None:
            return
        self._left_stack.setCurrentIndex(idx)
        # ⚠️ 勾选状态必须**由下标唯一决定**：这里原来只勾目标按钮、从不去取消别的。
        #    默认页改成「工程管理」+ 打开工程再自动切「图像管理」之后，
        #    两个按钮会**同时勾着**（探针实测 checked=['proj','img']）——
        #    页签是互斥的一组，两颗都亮等于"我也不知道现在在哪一页"。
        #    setChecked 依然安全：按钮连的是 `clicked`（只有用户点击才发），
        #    不是 `toggled` —— 否则这一下就会绕回 _on_left_tab_clicked 来回打架。
        _idx_map = getattr(self, "_left_tab_index", {}) or {}
        for _k, _b in (getattr(self, "_left_tab_btns", {}) or {}).items():
            _b.setChecked(_idx_map.get(_k) == idx)
        self._left_stack.setVisible(True)
        if notify:
            # 图片页刚可见时才量得到真实高度（_fit_list_height 依赖 page.height()），
            # 且必须延后一拍：布局几何要等这一轮事件循环结束才落到控件上。
            self._schedule_fit_list_height()

    def _on_left_tab_clicked(self, key, checked):
        """左栏页签点击：**点一下显示该列表，再点一下收起整块内容**。

        这是照友商的行为（"只要点击图像管理就会显示或者隐藏图像列表"）做的，
        所以页签按钮**不能**进独占组（独占组里"再点已选中的那个"点不下去）。
        两个页签都弹起来 = 整块内容收起。
        """
        if checked:
            # 互斥：把另一个放开（它连的是 clicked，setChecked 不会绕回来）
            for k, b in self._left_tab_btns.items():
                if k != key and b.isChecked():
                    b.setChecked(False)
            self.show_left_tab(key)
            return
        if not any(b.isChecked() for b in self._left_tab_btns.values()):
            # ⚠️ 只收**内容**，竖条本身绝不能隐藏 —— 否则就再也点不回来了。
            self._left_stack.setVisible(False)
            self._schedule_fit_list_height()

    def left_tab_state(self):
        """诊断用：当前左栏状态（哪个页签按着、内容可不可见）。"""
        return {
            "checked": [k for k, b in getattr(self, "_left_tab_btns", {}).items() if b.isChecked()],
            "content_visible": bool(self._left_stack.isVisible()) if hasattr(self, "_left_stack") else None,
            "page": self._left_stack.currentIndex() if hasattr(self, "_left_stack") else None,
        }

    def _update_accordion_counts(self, result_count=None, ann_count=None, img_count=None):
        """更新右侧「Result (n) / Annotation (n)」小标题的计数。

        2026-10-15 起不再管左侧：「图片列表 (n)」页头标题已删（同一份计数底部
        stats_summary 已写「共 N 组（M 张）」，见 _filter_images 里那几行）。
        img_count 形参保留只为兼容旧调用点，不再消费。
        """
        if result_count is not None or ann_count is not None:
            self._update_block_counts(result_count, ann_count)

    def _update_block_counts(self, result_count=None, ann_count=None):
        """右侧「标注」页里 Class / Annotation / Result 三个小标题的计数。

        用 dict 按键取控件，不用列表下标 —— 分段顺序变过两次，
        下标写死最容易悄悄把两个数字对调（这类错不会报错，只会显示错）。
        """
        lb = getattr(self, "_block_labels", None)
        if not lb:
            return
        if ann_count is not None and "ann" in lb:
            lb["ann"].setText(f"Annotation ({ann_count})")
        if result_count is not None and "res" in lb:
            lb["res"].setText(f"Result ({result_count})")

    def _update_class_count_label(self, n=None):
        """「Class (n)」小标题的计数（n 为空时按 _cached_classes 算）。"""
        lb = getattr(self, "_block_labels", None)
        if not lb or "cls" not in lb:
            return
        if n is None:
            n = len(getattr(self, "_cached_classes", None) or [])
        lb["cls"].setText(f"Class ({n})")

    def _create_center_panel(self):
        outer = QSplitter(Qt.Vertical)
        # Canvas area (no header - canvas fills entire space)
        cc = QWidget(); cl = QVBoxLayout(cc); cl.setContentsMargins(0, 0, 0, 0)
        self.image_count_label = QLabel(""); self.image_count_label.setVisible(False)

        # Stacked widget: index 0 = MultiViewPreview(内含 AnnotationCanvas), index 1 = MixedViewer
        self._viewer_stack = QStackedWidget()
        self.canvas = AnnotationCanvas()
        self.canvas.status_message.connect(self.statusBar().showMessage)
        self.mixed_viewer = MixedViewer()
        # index 0 换成"多视图预览容器"：它内部第 0 格就是上面的 canvas，
        # 单图模式下只显示 canvas，外观与改动前完全一致（不引入回归）。
        from ui.multiview_preview import MultiViewPreview
        self.mv_preview = MultiViewPreview(self.canvas)

        self._viewer_stack.addWidget(self.mv_preview)    # index 0
        self._viewer_stack.addWidget(self.mixed_viewer)  # index 1

        self._version_refresh_timer = QTimer()
        self._version_refresh_timer.setSingleShot(True)
        self._version_refresh_timer.setInterval(1500)
        self._version_refresh_timer.timeout.connect(self._refresh_versions)
        self.canvas.mask_changed.connect(lambda: self._version_refresh_timer.start())
        self.canvas.det_boxes_changed.connect(self._on_det_boxes_changed)

        # 标注变更后防抖刷新列表「标注」列与 Class 表统计
        self._ann_refresh_timer = QTimer()
        self._ann_refresh_timer.setSingleShot(True)
        self._ann_refresh_timer.setInterval(600)
        self._ann_refresh_timer.timeout.connect(self._refresh_annotation_views)
        self.canvas.mask_changed.connect(lambda: self._ann_refresh_timer.start())

        # ★★ 2026-10-22：**标注改动后同步重生成 annotations/<base>_mask.png**。
        #   原来只有 `_auto_save_json()` 在每次编辑后写 JSON，`_mask.png` 只在
        #   「导入标注」或手动 `save_mask()` 时生成 ⇒ 掩膜会长期滞后于 JSON。
        #   实测：用户给 008 新画了 Hh×2 + Ps×1，JSON 已经是 3 个缺陷，
        #   而 _mask.png 里只剩 background + ignore（缺陷全缺）——
        #   训练走 JSON 侥幸没事，但评估工具/任何读 mask.png 的地方都会拿错真值。
        #   防抖 500ms、每串操作只写一次盘（630×400 约 3ms，20MP 约 40ms）。
        self._mask_png_timer = QTimer()
        self._mask_png_timer.setSingleShot(True)
        self._mask_png_timer.setInterval(500)
        self._mask_png_timer.timeout.connect(self._sync_mask_png_for_current)
        self.canvas.mask_changed.connect(lambda: self._mask_png_timer.start())

        # 横向标注工具栏（画布上方）
        self.annotation_toolbar = AnnotationToolbar()
        self.annotation_toolbar.mode_selected.connect(self._set_annotation_mode)
        self.annotation_toolbar.width_changed.connect(self._on_canvas_width_changed)
        self.annotation_toolbar.undo_requested.connect(self.canvas.undo)
        self.annotation_toolbar.redo_requested.connect(self.canvas.redo)
        self.annotation_toolbar.clear_requested.connect(self._clear_current_mask)
        self.annotation_toolbar.cancel_requested.connect(self.canvas.cancel_current)
        self.annotation_toolbar.reload_requested.connect(self._on_toolbar_reload)
        self.annotation_toolbar.save_requested.connect(self._save_current_mask)
        self.annotation_toolbar.prev_requested.connect(self._prev_image)
        self.annotation_toolbar.next_requested.connect(self._next_image)
        self.canvas.mode_changed.connect(self.annotation_toolbar.set_mode)
        self.canvas.width_changed.connect(self.annotation_toolbar.set_width)
        self.canvas.class_select_requested.connect(self._on_canvas_class_select)
        self.canvas.save_requested.connect(self._save_current_mask)
        # 工具栏右侧状态区（2026-10-14 顶栏并进底栏时加的）：
        # 三个图层开关与当前类别原先只以**纯文本**挂在多视图顶栏上，顶栏删掉后
        # 它们在全屏没有第二个显示处（只能靠快捷键盲切），所以搬过来并做成可点按钮。
        # 类别那一格直接复用数字键那条路（`_on_canvas_class_select`）——
        # 同一个动作只留一个实现，两个入口才不会各自跑偏。
        self.annotation_toolbar.layer_toggled.connect(self._on_layer_toggled)
        # 工具行折叠/展开（2026-10-15 照友商"点标注显示隐藏工具栏"）：
        # 工具行是这个布局里的兄弟控件，藏起来画布会**真的变大** ——
        # 所以必须通知下游"视口尺寸变了"：图片信息栏按可用宽排版、底栏滑块重算适配，
        # 否则会出现"画布长高了但信息栏还按旧宽度摆"的错位。
        self.annotation_toolbar.tools_toggled.connect(self._on_toolbar_collapsed)
        cl.addWidget(self.annotation_toolbar)
        # 轻量级目标检测标注工具栏（仅 detection 项目显示，与分割工具栏同位置互斥）
        self.det_toolbar = DetToolbar()
        self.det_toolbar.draw_toggled.connect(self._set_det_draw_mode)
        self.det_toolbar.clear_requested.connect(self._det_clear_all)
        self.det_toolbar.delete_requested.connect(self._det_delete_box)
        self.det_toolbar.cancel_requested.connect(self._det_cancel)
        self.det_toolbar.undo_requested.connect(self._det_undo)
        self.det_toolbar.redo_requested.connect(self._det_redo)
        self.det_toolbar.tools_toggled.connect(self._on_toolbar_collapsed)
        self.canvas.det_selection_changed.connect(self._sync_det_toolbar)
        cl.addWidget(self.det_toolbar)
        cl.addWidget(self._viewer_stack)
        # 图像区正下方的「图片信息栏」（文件/宽高/坐标/RGB/已标注/已检出/耗时/平均耗时）
        # 注意要把 cc 作为 parent 传进去，不能只靠 addWidget 事后认父（见该函数文档里的坑）
        cl.addWidget(self._build_img_info_bar(cc))

        outer.addWidget(cc)

        # ── 折叠后的「边缘恢复按钮」（2026-10-15 改版）──
        # 上一版：折叠 = 面板收窄成 26px 把手条 ⇒ 画布**没有占满整宽**，两侧各留一条空白
        #   （维护者："折叠后，预览窗口直接占满左右栏就好"）。
        # 这一版：折叠 = **整个面板藏起来**（宽度归零，画布吃满），
        #   恢复按钮改成**贴在画布左右边缘的浮层小按钮**，用 move() 定位、不进布局 ⇒
        #   不占任何布局宽度，同时又满足"随时能点回来"（上一版的需求）。
        # 为什么浮层不用布局：`cc` 的布局是 QVBoxLayout，加进去就等于给它宽度/高度。
        self._edge_left_btn = QToolButton(cc)
        self._edge_right_btn = QToolButton(cc)
        for _b, _tip in ((self._edge_left_btn, "展开左栏（Ctrl+[）"),
                         (self._edge_right_btn, "展开右栏（Ctrl+]）")):
            _b.setText("▶" if _tip.startswith("展开左") else "◀")
            _b.setToolTip(_tip)
            _b.setFixedSize(16, 46)
            _b.setCursor(Qt.PointingHandCursor)
            _b.setFocusPolicy(Qt.NoFocus)
            _b.hide()                      # 默认两侧都是展开态 ⇒ 不显示
        self._edge_left_btn.clicked.connect(lambda: self._toggle_panel("left"))
        self._edge_right_btn.clicked.connect(lambda: self._toggle_panel("right"))

        return outer

    def _layout_edge_buttons(self):
        """把两枚边缘恢复按钮贴到图像区的左右边缘（浮层，不占布局宽度）。"""
        for side, btn in (("left", getattr(self, "_edge_left_btn", None)),
                          ("right", getattr(self, "_edge_right_btn", None))):
            if btn is None or btn.isHidden():
                continue
            host = btn.parentWidget()
            if host is None:
                continue
            y = max(0, (host.height() - btn.height()) // 2)
            if side == "left":
                btn.move(0, y)
            else:
                btn.move(max(0, host.width() - btn.width()), y)
            btn.raise_()                  # 压在图像之上，不被布局压到底层

    # ────────── 底部「图片信息栏」（对标成熟视觉软件的图片信息栏字段表） ──────────
    def _build_img_info_bar(self, parent=None):
        """文件 / 宽高 / 坐标 / RGB / 已标注 / 已检出 / 耗时 / 平均耗时。

        两条规矩：
        1. **数据只从图片列表那一行的同一条记录取**（`self._filtered_data` 里按 base 找），
           不另算一份 —— 否则"列表说 89 块、底栏说 90 块"这种互相打脸迟早出现；
        2. 宽度不够时**只让文件名变短**（中间省略），右边的坐标/RGB/计数/耗时一律不截 ——
           "耗时是不是空的"恰恰是没跑推理时最需要看见的一项。

        ⚠️ 容器控件（box/sep/bar）必须**在构造时就把 parent 给死**（调用点传 cc 进来）。
        踩过的坑（2026-10-14，真窗口探针抓的，静默崩）：中间那个 `bar` QFrame 既没有父对象、
        又没被 self 引用 ⇒ PyQt 认为它归 Python 所有 ⇒ 本函数一返回它就**被 GC 删掉**，
        孩子（这些 QLabel）跟着一起被删。之后再碰就是
        `RuntimeError: wrapped C/C++ object of type ElideLabel has been deleted`，
        而这个异常发生在 Qt 槽里 ⇒ PyQt 走 qFatal/abort **直接杀进程**，
        stderr 还没 flush ⇒ 现场表现是"打开工程就静默退出、没有 traceback、崩溃日志也没写"。
        """
        box = QWidget(parent)
        vb = QVBoxLayout(box)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(0)
        sep = QFrame(box)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        vb.addWidget(sep)
        bar = QFrame(box)
        lay = QHBoxLayout(bar)
        # 间距/边距都收紧一点：数值项的"最小宽度"是钉死的（防抖动），紧出来的宽度
        # 全部让给左边那个可缩的文件名 —— 中心列只有 ~700px 时不至于把名字砍得太狠。
        lay.setContentsMargins(6, 1, 6, 1)
        lay.setSpacing(7)
        self._info_file = ElideLabel("文件 --")
        lay.addWidget(self._info_file, 1)
        self._info_dims = QLabel("宽高 --")
        self._info_xy = QLabel("坐标 --")
        self._info_rgb = QLabel("RGB --")
        self._info_ann = QLabel("已标注 --")
        self._info_det = QLabel("已检出 --")
        self._info_ms = QLabel("耗时 --")
        self._info_avg = QLabel("平均 --")
        self._img_info_bar = box          # 供探针/后续代码定位这一条（断言它挂在图像区下方）
        fm = QFontMetrics(self.font())
        # 数字位数一变（1129,1962 -> 999,1962）整条栏就会左右抖，看着像在跳。
        # 用"最宽的样本串"给每一项钉一个最小宽度：位置固定、只有数字本身在变。
        for lb, sample in ((self._info_dims, "宽高 0000×0000"), (self._info_xy, "坐标 0000,0000"),
                           (self._info_rgb, "RGB 255,255,255"), (self._info_ann, "已标注 000"),
                           (self._info_det, "已检出 000"), (self._info_ms, "耗时 0000ms"),
                           (self._info_avg, "平均 0000ms")):
            lb.setMinimumWidth(fm.horizontalAdvance(sample) + 2)
            lay.addWidget(lb)
        self._info_bar = bar            # 供 `_fit_img_info_bar` 量可用宽（见那里的自锁说明）
        # 右端：视图布局 + 缩放控件（照友商把缩放摆在同一栏的最右端）
        lay.addWidget(self._build_zoom_controls(bar))
        # ⚠️ 这一行不能漏：给了 parent 只解决"会不会被回收"，**不等于进了布局** ——
        #    漏掉它就是"有父对象但没人管它"的野孩子，实测整条栏只剩 3px 高（分隔线的高度），
        #    标签一个都看不见，而 isVisible() 仍然是 True（探针第一版就是这么被骗过去的，
        #    后来改成量 box.height() 才露出来）。
        vb.addWidget(bar)
        # 光标坐标 / RGB 从**三格里的任意一格**上报，不只主画布。
        # 用户 2026-10-14 报的正是这个："光标停留在形状1和形状2上时，这些值都是空的" ——
        # 参考窗那时是纯 QLabel、连鼠标跟踪都没开，压根不产生事件，更没人接。
        # 挂 MultiViewPreview 那一路（它已经把主画布与各参考窗汇成一个出口）；
        # 老版本没有这个信号时退回只挂画布（至少主视图那格还是好的，不崩）。
        _mv = getattr(self, "mv_preview", None)
        if _mv is not None and hasattr(_mv, "cursor_moved"):
            _mv.cursor_moved.connect(self._on_canvas_cursor_moved)
        else:
            self.canvas.cursor_moved.connect(self._on_canvas_cursor_moved)
        # 顶栏拆掉后的**状态出口**（三个图层开关 / 当前类别 / 缩放百分比）。
        # 同样是"老版本没有就跳过"：缺这个信号时工具栏那几个控件只是不同步，
        # 不该因此让整个窗口起不来。
        if _mv is not None and hasattr(_mv, "state_changed"):
            _mv.state_changed.connect(self._on_mv_state)
        self._refresh_img_info()
        return box

    def _build_zoom_controls(self, parent=None):
        """底栏右端那组控件：`[视图布局] − [滑块] + 140%`。

        为什么缩放读数挪到这里（原话是"顶栏和底栏可以合二为一，参考友商"）：
        原先"缩放 140%（三窗同步）"在多视图顶栏、而调缩放只能滚轮 ——
        读数与操作分在两处；友商把读数和 −/滑块/+ 摆在同一栏的最右端，照做。

        ⚠️ **一切缩放只走 `set_zoom_percent`**（滑块、±、以后可能的键盘）。
        三个入口各自算一遍 zoom/pan 的话，迟早出现"滑块说 140%、画面是 200%"。
        """
        box = QWidget(parent)
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)

        # 「视图布局」只是个按钮（QToolButton + 菜单）。这个下拉原先在多视图顶栏，
        # 顶栏删掉后无处可去：塞标注工具栏实测放不下（工具栏 sizeHint 已 773px，
        # 中央列左栏展开时只有 757px），所以跟缩放一起放在底栏右端，只有在多视图模式下才出现。
        self._view_layout_btn = QToolButton(box)
        self._view_layout_btn.setText("视图布局")
        self._view_layout_btn.setToolTip("多视图排列方式（1*1 / 2*1 / 3*1 / 2*2 / 4*1）")
        self._view_layout_btn.setCursor(Qt.PointingHandCursor)
        self._view_layout_btn.setPopupMode(QToolButton.InstantPopup)
        self._view_layout_btn.setStyleSheet(_BAR_BTN_QSS)
        self._layout_menu = QMenu(self._view_layout_btn)
        self._view_layout_btn.setMenu(self._layout_menu)
        self._view_layout_btn.setVisible(False)     # 单图模式没有布局可设
        h.addWidget(self._view_layout_btn)

        self._zoom_out_btn = QToolButton(box)
        self._zoom_out_btn.setText("\u2212")        # U+2212：连字符"-"在居中时偏上，视觉不对称
        self._zoom_out_btn.setToolTip("缩小（三窗同步，长按连续）")
        self._zoom_out_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_out_btn.setAutoRepeat(True)
        self._zoom_out_btn.setAutoRepeatDelay(420)
        self._zoom_out_btn.setAutoRepeatInterval(110)
        self._zoom_out_btn.setStyleSheet(_BAR_BTN_QSS)
        self._zoom_out_btn.setFixedSize(18, 17)
        h.addWidget(self._zoom_out_btn)

        self._zoom_slider = QSlider(Qt.Horizontal, box)
        self._zoom_slider.setRange(0, ZOOM_POS_MAX)      # 内部位置，不是百分比（见 zoom_pos_to_pct）
        self._zoom_slider.setValue(zoom_pct_to_pos(100))
        self._zoom_slider.setSingleStep(ZOOM_STEP_POS)
        self._zoom_slider.setPageStep(ZOOM_STEP_POS * 3)
        self._zoom_slider.setFixedWidth(84)
        self._zoom_slider.setFixedHeight(17)
        self._zoom_slider.setToolTip("缩放（相对「适应窗口」，三窗同步）")
        # ⚠️ **不要**在这里 setStyleSheet 写颜色：颜色写在内联样式里会被
        #    `sweep_inline_styles` 剥掉（白名单色）⇒ 槽和手柄没有背景色 ⇒ 空槽。
        #    配色在 ui/themes.py 的 `_DARK_SLIDER` / `_LIGHT_SLIDER`。
        h.addWidget(self._zoom_slider)

        self._zoom_in_btn = QToolButton(box)
        self._zoom_in_btn.setText("+")
        self._zoom_in_btn.setToolTip("放大（三窗同步，长按连续）")
        self._zoom_in_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_in_btn.setAutoRepeat(True)
        self._zoom_in_btn.setAutoRepeatDelay(420)
        self._zoom_in_btn.setAutoRepeatInterval(110)
        self._zoom_in_btn.setStyleSheet(_BAR_BTN_QSS)
        self._zoom_in_btn.setFixedSize(18, 17)
        h.addWidget(self._zoom_in_btn)

        self._zoom_pct = QLabel("100%", box)
        # 按"最宽样本"钉死：从 100% 跳到 800% 时整条底栏不能跟着左右抖
        self._zoom_pct.setMinimumWidth(QFontMetrics(self.font()).horizontalAdvance("1000%") + 2)
        h.addWidget(self._zoom_pct)

        self._zoom_out_btn.clicked.connect(lambda: self._nudge_zoom(-1))
        self._zoom_in_btn.clicked.connect(lambda: self._nudge_zoom(+1))
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)
        return box

    def _nudge_zoom(self, sign):
        """−/+ 按钮：一格 = 滑块位置走 `ZOOM_STEP_POS`。

        对数刻度下这就是**固定倍率**（×1.19）：在 20% 处和在 700% 处点一下，
        画面变化的幅度一样 —— 线性档位在两端的手感会差出一个数量级。
        改的是**滑块值**、不直接调 `set_zoom_percent`：保持"缩放入口只有一条路"
        （`valueChanged` -> `_on_zoom_slider` -> `set_zoom_percent`）。
        """
        pos = int(self._zoom_slider.value()) + int(sign) * ZOOM_STEP_POS
        pos = max(0, min(ZOOM_POS_MAX, pos))
        if pos != self._zoom_slider.value():
            self._zoom_slider.setValue(pos)

    def _on_zoom_slider(self, val):
        """滑块（或 ± 按钮）改缩放。拖动时高频触发，所以这里只干一件事：交给 MultiViewPreview。"""
        if getattr(self, "_zoom_sync_lock", False):
            return
        pct = zoom_pos_to_pct(val)          # 位置 -> 百分比只在这里换算
        mv = getattr(self, "mv_preview", None)
        if mv is not None and hasattr(mv, "set_zoom_percent"):
            mv.set_zoom_percent(pct)
        self._zoom_pct.setText("%d%%" % pct)

    def _apply_zoom_ui(self, pct):
        """把外面的缩放值回写到滑块与读数（滚轮缩放 / 打开工程 / 程序性改视图时走这里）。

        `_zoom_sync_lock` 不能省：`setValue` 会触发 `valueChanged` → `_on_zoom_slider`
        → 又去 set_zoom_percent → 又回来 → 无限回环，而且每圈都重画三格。

        ⚠️ 回写前要先问一句"滑块自己的位置**已经是**这个百分比了吗"：
        画布上报的是**取整**过的百分比（`zoom_percent()` 返回 int），而位置->百分比
        是连着的一条曲线，取整会来回蹭。实测（2026-10-14）：滑块在 356（=48%），
        画布报 48%、反算位置得 358 ⇒ 无条件 setValue 会把滑块从 356 挪到 358，
        于是"± 一格固定走 40"变成 41/42，而且每点一下滑块都被无意义地拖动一次。
        判据用**滑块自己的位置**算回来的百分比，不是拿位置跟位置比。
        """
        self._zoom_sync_lock = True
        try:
            pct = max(ZOOM_PCT_MIN, min(ZOOM_PCT_MAX, int(pct)))
            if zoom_pos_to_pct(self._zoom_slider.value()) != pct:
                self._zoom_slider.setValue(zoom_pct_to_pos(pct))
            self._zoom_pct.setText("%d%%" % pct)
        finally:
            self._zoom_sync_lock = False

    def _rebuild_layout_menu(self):
        """重建「视图布局」菜单（模式列表从 MultiViewPreview 问，不在这里抄一份常量）。"""
        mv = getattr(self, "mv_preview", None)
        btn = getattr(self, "_view_layout_btn", None)
        if mv is None or btn is None or not hasattr(mv, "layout_modes"):
            return
        m = self._layout_menu
        m.clear()
        cur = mv.layout_mode()
        for text in mv.layout_modes():
            act = m.addAction(text)
            act.setCheckable(True)
            act.setChecked(text == cur)
            act.triggered.connect(lambda *a, t=text: self._on_layout_pick(t))

    def _on_layout_pick(self, text):
        mv = getattr(self, "mv_preview", None)
        if mv is not None and hasattr(mv, "set_layout_mode"):
            mv.set_layout_mode(text)

    def _on_layer_toggled(self, key, on):
        """工具栏右侧三个图层开关被点。

        ⚠️ 一律调画布自己的 `toggle_*`，**不直接改字段**：画布那边还要负责
        `_overlay_dirty`、`status_message`、`view_changed`；绕过它就是
        "标志位改了但叠加层没重画"这类只在某些图上才看得出来的 bug。
        按钮的亮灭与画布真实状态由 `state_changed` 兜底校正（最多差一个 40ms 轮询），
        所以万一模式对不上（比如检测模式下"标注"切的是框），按钮会自己弹回去 —— 这是对的。
        """
        c = getattr(self, "canvas", None)
        if c is None:
            return
        try:
            if key == "annot":
                c.toggle_annotation_visibility()
            elif key == "ignore":
                c.toggle_ignore_visibility()
            elif key == "pred":
                c.toggle_prediction_visibility()
        except Exception as e:
            self.log(f"[图层开关] 切换 {key} 失败：{type(e).__name__}: {e}")

    def _on_mv_state(self, st):
        """接 MultiViewPreview 的状态汇总（顶栏拆掉后这些值的**唯一出口**）。

        三件事：
        ① 工具栏右侧三个图层开关 + 类别格 —— **只在多视图模式显示**：
           单图模式下画布自己叠着一行状态文本（`set_status_overlay(True)`），
           这里再显示一处就是"同一个状态同屏两遍"（正是这次要消掉的毛病）；
        ② 底栏的缩放读数与滑块位置；
        ③ 底栏「视图布局」按钮的可见性（同理，单图模式没有布局可设）。
        """
        if not isinstance(st, dict):
            return
        multi = int(st.get("n_views") or 0) > 1
        tb = getattr(self, "annotation_toolbar", None)
        if tb is not None:
            try:
                tb.set_layers_visible(multi)
                tb.set_layer_states(st.get("annot", True), st.get("ignore", True),
                                    st.get("pred", False))
            except Exception:
                pass
        btn = getattr(self, "_view_layout_btn", None)
        if btn is not None and bool(btn.isVisible()) != multi:
            btn.setVisible(multi)
            if multi:
                self._rebuild_layout_menu()
        self._apply_zoom_ui(st.get("zoom_pct", 100))
        self._schedule_fit_img_info_bar()

    def _fit_img_info_bar(self):
        """底栏放不下时，按优先级收起右端的交互件（**读数一项都不收**）。

        收起顺序：视图布局按钮 → **滑块** → −/+ → 「100%」读数（这个永不收）。

        为什么这个顺序：
          · 「视图布局」排第一 —— 多视图顶栏那个下拉删掉后，它是**唯一**的布局入口，
            收掉就等于"这个功能没了"（别处再也没有）。
          · 滑块排第二、−/+ 排最后 —— 两者都是缩放入口，但滑块是照友商做的主控件，
            而且拖动比连点精确；更关键的是 ± 与"指针在图上滚轮"完全重复，
            窄窗口下先牺牲它，用户照样能缩放（滚轮一直在）。
            反过来的顺序（先收滑块、留两个小 ± ）会把友商那个特征控件弄丢，
            屏幕上只剩下两个看不出干什么的方块。
          · 「100%」读数永不收：它不占位、又是唯一能一眼看出"当前是不是 100%"的地方。

        为什么读数一项都不收：它们全是"扫一眼就知道"的东西，收起来就得悬停，
        等于没有。而右端那几个是**可交互**的，短时间收掉不影响看数（滚轮缩放一直在）。
        实测（tmp/measure_bars2.py）：7 个数字项钉死宽合计 530px + 间隙 + 边距 ≈ 601px，
        再加文件名最小 80px ⇒ 中央列窄到 757px 时右端只剩 ~76px，必须收。

        ⚠️ **只算一遍就定完，不做"收起来再量"。** 收起会让可用宽变大，
        于是又"够放了" ⇒ 显示 ⇒ 又不够 —— 来回抖（表格高度自适应踩过同一个自锁）。
        所以这里量的是固定项的最小需求，与当前显示状态无关。
        """
        bar = getattr(self, "_info_bar", None)
        if bar is None or bar.width() <= 8:
            return
        fixed = 0
        for lb in (getattr(self, "_info_dims", None), getattr(self, "_info_xy", None),
                   getattr(self, "_info_rgb", None), getattr(self, "_info_ann", None),
                   getattr(self, "_info_det", None), getattr(self, "_info_ms", None),
                   getattr(self, "_info_avg", None)):
            if lb is not None:
                fixed += lb.minimumWidth()
        lay = bar.layout()
        gap = lay.spacing() if lay is not None else 7
        fixed += gap * 7
        room = bar.width() - 12 - fixed - 80       # 12=边距, 80=文件名最小宽
        try:
            multi = bool(getattr(self.mv_preview, "is_multiview", lambda: False)())
        except Exception:
            multi = False
        for w, need_multi in ((getattr(self, "_view_layout_btn", None), True),
                              (getattr(self, "_zoom_slider", None), False),
                              (getattr(self, "_zoom_out_btn", None), False),
                              (getattr(self, "_zoom_in_btn", None), False)):
            if w is None:
                continue
            cost = w.sizeHint().width() + 3
            show = (not need_multi or multi)
            if show and room >= cost:
                room -= cost
            else:
                show = False
            if bool(w.isVisible()) != show:
                w.setVisible(show)
        # 按钮刚变可见时补一次菜单重建 —— 菜单原先只在 `_on_mv_state` 里建，
        # 而那条路要靠 `state_changed` 推；这里兜底，免得出现"按钮能点、点开是空的"
        # （空菜单在界面上完全看不出是坏的，只会觉得"这个按钮没用"）。
        btn = getattr(self, "_view_layout_btn", None)
        if btn is not None and btn.isVisible() and not self._layout_menu.actions():
            self._rebuild_layout_menu()

    def _schedule_fit_img_info_bar(self):
        """延到事件循环下一拍再量：布局还没走完时量到的宽度是旧的（"改了但看起来没改"）。"""
        try:
            QTimer.singleShot(0, self._fit_img_info_bar)
        except Exception:
            pass

    def _img_info_record(self, base):
        """当前图对应的**列表记录**（多视图工程里 base = 主视图 stem，列表一行 = 一颗电芯）。

        对不上就返回 None（版本预览、外部目录里打开的图）：此时只显示画布能自报的文件与尺寸，
        计数项老实显示 "--"，不假装是 0。
        """
        if not base:
            return None
        cached = getattr(self, "_img_info_rec_cache", None)
        if cached and cached[0] == base:
            return cached[1]
        rec = None
        for r in (getattr(self, "_filtered_data", None) or []):
            if r.get("base") == base:
                rec = r
                break
        self._img_info_rec_cache = (base, rec)
        return rec

    def _refresh_img_info(self):
        """刷新信息栏（低频路径：换图 / 列表重建 / 标注变化时调用）。"""
        if getattr(self, "_info_file", None) is None:
            return
        path = getattr(self.canvas, "current_image_path", "") or ""
        base = os.path.splitext(os.path.basename(path))[0] if path else ""
        self._img_info_rec_cache = None
        rec = self._img_info_record(base)
        self._info_file.setFullText(("文件 " + os.path.basename(path)) if path else "文件 --")
        w = int(rec.get("img_w") or 0) if rec else 0
        h = int(rec.get("img_h") or 0) if rec else 0
        if not w or not h:
            img = getattr(self.canvas, "image", None)
            if img is not None:
                w, h = img.size
        self._info_dims.setText("宽高 %s" % ("%d×%d" % (w, h) if (w and h) else "--"))
        self._info_ann.setText("已标注 %s" % (rec.get("shape_count", 0) if rec else "--"))
        self._info_det.setText("已检出 %s" % (rec.get("pred_cnt", 0) if rec else "--"))
        ms = rec.get("infer_ms") if rec else None
        self._info_ms.setText("耗时 %s" % ("%dms" % int(round(float(ms))) if ms not in (None, "") else "--"))
        # 平均耗时：本项目已跑过推理的图取平均（与列表「耗时」列同一个字典，同一种单位）
        times = [float(v) for v in (getattr(self, "_infer_times", None) or {}).values()
                 if isinstance(v, (int, float))]
        self._info_avg.setText("平均 %s" % ("%dms" % int(round(sum(times) / len(times))) if times else "--"))
        if not path:
            self._on_canvas_cursor_moved(-1, -1)

    def _on_canvas_cursor_moved(self, x, y, source=None):
        """画布/参考窗上报光标像素坐标（高频路径：只碰这两个标签，别的什么都不管）。

        `source` = 光标所在的那一格（主画布，或某个参考窗面板）：
        RGB 读的是**光标底下那一格显示的图** —— 漫反射 / 形状图1 / 形状图2 各自为政，
        显示什么就读什么。否则"对着形状图读漫反射灰度"这种读数比没有读数更误导。
        坐标则三格共用主视图的图像坐标系（三窗同步就是按它对齐的），
        所以同一个数字在漫反射和形状图上指的是同一处 —— 这正是它有用的原因。

        值形态：三格**统一 RGB**（用户 2026-10-14 的原话："形状1和形状2都要显示RGB"）。
        漫反射那格本来就已经是 RGB —— `AnnotationCanvas.image` 加载时` .convert("RGB")`，
        而形状图两格早先按文件 mode 走（L 灰度就写「灰度 127」），两种形态并存。
        现在参考窗一律给三元组，底栏那栏三格都写 `RGB r,g,b`。
        """
        if getattr(self, "_info_xy", None) is None:
            return
        if x < 0 or y < 0:
            self._info_xy.setText("坐标 --")
            self._info_rgb.setText("RGB --")
            return
        self._info_xy.setText("坐标 %d,%d" % (x, y))
        v = self._pixel_at(source if source is not None else self.canvas, x, y)
        if isinstance(v, (tuple, list)):
            if len(v) >= 3:
                self._info_rgb.setText("RGB %d,%d,%d" % (int(v[0]), int(v[1]), int(v[2])))
            else:
                self._info_rgb.setText("灰度 %d" % int(v[0]))
        elif v is None:
            self._info_rgb.setText("RGB --")
        else:
            self._info_rgb.setText("灰度 %d" % int(v))

    @staticmethod
    def _pixel_at(src, x, y):
        """读某一格 (x, y) 处的像素。

        形态（维护者 2026-10-14 定的口径：三格统一 RGB）：
        - 主画布（漫反射）：走 PIL `getpixel`。`AnnotationCanvas.image` 加载时已经
          `.convert("RGB")` ⇒ 给 (r, g, b)；
        - 参考窗（形状图1 / 形状图2）：面板自己的 `pixel_at()` **一律给 (r, g, b)**，
          哪怕源文件是 L 灰度、三值相等。

        ⇒ 底栏那条 `RGB r,g,b` 现在对三格都成立。`灰度 N` 那个分支留着：万一哪天
        画布上真的放了 L 模式图（`mask_editor` 里切灰度视图那条路），它仍然是对的。

        格式化逻辑只有一份（见 `_on_canvas_cursor_moved`）—— 谁给什么形态由数据源决定，
        底栏只管照着渲染。
        """
        fn = getattr(src, "pixel_at", None)
        if callable(fn):
            try:
                return fn(x, y)
            except Exception:
                return None
        img = getattr(src, "image", None)
        if img is None:
            return None
        try:
            return img.getpixel((x, y))
        except Exception:
            return None

    def _create_right_panel(self):
        # ★ 与左栏同款（2026-10-15）：把手条**常驻**，折叠只藏内容 ——
        #   原来箭头在面板里面，一按就跟着消失，想展开只能去 View 菜单。
        container = QWidget()
        _cr = QHBoxLayout(container)
        _cr.setContentsMargins(0, 0, 0, 0)
        _cr.setSpacing(0)

        p = QWidget()
        outer_layout = QVBoxLayout(p)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        _cr.addWidget(p, 1)
        self._right_content = p         # 折叠时被 setVisible(False) 的就是它

        # 把手条贴在**最右端**（右栏的箭头在右边）
        handle = QWidget()
        _hr = QVBoxLayout(handle)
        _hr.setContentsMargins(2, 2, 2, 2)
        _hr.setSpacing(0)
        btn_collapse = QPushButton("\u25b6")
        btn_collapse.setFixedSize(20, 20)
        btn_collapse.setToolTip("收起右栏（再点一下展开；Ctrl+]）")
        btn_collapse.clicked.connect(lambda: self._toggle_panel("right"))
        _hr.addWidget(btn_collapse)
        _hr.addStretch()
        _cr.addWidget(handle)
        self._right_handle = handle
        self._right_toggle_btn = btn_collapse

        # Horizontal layout: content (left) + tab bar (right)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(2, 0, 0, 0)
        content_layout.setSpacing(0)

        # ===== Stacked content panels =====
        self._right_stack = QStackedWidget()

        # Panel 0: Annotation Tools
        panel_annot = QWidget(); al = QVBoxLayout(panel_annot); al.setContentsMargins(4, 2, 2, 2)
        # ── 三张表：Class（类别）/ Annotation（标注块）/ Result（结果块） ──
        #    原来 Class 表独立在 stack 里、两张块表在另一个 splitter 里 —— 三段之间
        #    只有一段有分隔条。现在**合成一个三段 QSplitter**：每两段之间都有可拖的
        #    分隔条，三张表的高度都能自己调，也彻底没有"Class 表下面一大片空白"。
        #    表格对象本身没换（还是 self.class_list / self.stats_table / self.ann_table），
        #    所有刷新逻辑一行都不用改。
        al.addSpacing(6)
        self._panel_split = QSplitter(Qt.Vertical)
        self._panel_split.setChildrenCollapsible(False)   # 不许拖到 0 高度（拉回来很烦）
        self._panel_split.setHandleWidth(4)
        self._block_labels = {}

        # 第 1 段：Class（类别表）
        cls_box = QWidget(); cl_layout = QVBoxLayout(cls_box)
        cl_layout.setContentsMargins(0, 0, 0, 0); cl_layout.setSpacing(1)
        self._block_labels["cls"] = self._make_block_header("Class (0)")
        cl_layout.addWidget(self._block_labels["cls"])
        self.class_list = ClassTableView()
        self.class_list.currentRowChanged.connect(self._on_class_changed)
        self.class_list.add_requested.connect(self._add_class)
        self.class_list.delete_requested.connect(self._delete_class)
        self.class_list.refresh_requested.connect(self._refresh_class_stats)
        self.class_list.cell_filter_requested.connect(self._on_class_cell_filter)
        self.class_list.move_up_requested.connect(lambda: self._move_class(-1))
        self.class_list.move_down_requested.connect(lambda: self._move_class(1))
        self.class_list.rename_requested.connect(self._rename_class)
        cl_layout.addWidget(self.class_list, 1)
        self._panel_split.addWidget(cls_box)

        # 第 2 段：Annotation（标注块）—— 用户要求：标注表在上
        ann_box = QWidget(); ann_layout = QVBoxLayout(ann_box)
        ann_layout.setContentsMargins(0, 0, 0, 0); ann_layout.setSpacing(1)
        self._block_labels["ann"] = self._make_block_header("Annotation (0)")
        ann_layout.addWidget(self._block_labels["ann"])
        self.ann_table = QTableWidget()
        self.ann_table.setColumnCount(8)
        self.ann_table.verticalHeader().setDefaultSectionSize(24)
        self.ann_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ann_table.setAlternatingRowColors(True)
        ann_layout.addWidget(self.ann_table, 1)
        self._panel_split.addWidget(ann_box)

        # 第 3 段：Result（预测/结果块）—— 结果表在下
        res_box = QWidget(); res_layout = QVBoxLayout(res_box)
        res_layout.setContentsMargins(0, 0, 0, 0); res_layout.setSpacing(1)
        self._block_labels["res"] = self._make_block_header("Result (0)")
        res_layout.addWidget(self._block_labels["res"])
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(8)
        self.stats_table.verticalHeader().setDefaultSectionSize(24)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setAlternatingRowColors(True)
        res_layout.addWidget(self.stats_table, 1)
        self._panel_split.addWidget(res_box)

        # 多出来的高度按 0 : 1 : 2 分给 Result（类别表只有几行，结果表可能几百行）
        self._panel_split.setStretchFactor(0, 0)
        self._panel_split.setStretchFactor(1, 1)
        self._panel_split.setStretchFactor(2, 2)
        self._panel_split.setSizes([230, 160, 380])
        al.addWidget(self._panel_split, 1)         # 占满「标注管理」里这一块
        # 注意：**不要再往 _right_stack 里加这个 panel** ——
        # 右侧页面统一在函数末尾按"3 个模块"组装（参照友商右栏）。
        # 这里若残留一行 addWidget，模块页的下标就会整体后移、页签指错页，
        # 而且错得很安静（点了没反应/跳到别的页），不会报错。

        # Panel 1: Version History
        panel_ver = QWidget(); vl = QVBoxLayout(panel_ver); vl.setContentsMargins(4, 2, 2, 2)
        vl.addWidget(QLabel("<b>Version History</b>"))
        self.version_list = QListWidget()
        self.version_list.setToolTip("Double-click to restore version")
        self.version_list.itemDoubleClicked.connect(self._restore_version)
        vl.addWidget(self.version_list)
        vbl = QHBoxLayout()
        vbl.addWidget(QPushButton("Restore", clicked=self._slot_restore_version))
        self.version_count_label = QLabel("0 versions"); vbl.addWidget(self.version_count_label)
        vl.addLayout(vbl)
        # 同上：panel_ver 由末尾的模块组装挂进「模型管理」

        # Panel 2: Inference
        panel_infer = QWidget(); il = QVBoxLayout(panel_infer); il.setContentsMargins(4, 2, 2, 2)
        # ══ ① 模型 ══
        # ★ 顺序原来是：Model 标签 → **Scale 滑块** → model 下拉 → backend → Tiled
        #   （下拉框被自己的标签和 Scale 隔开，看着莫名其妙）→ 现已按"标签+控件配对"重排。
        il.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox(); self.model_combo.addItems(["best_model.pth", "last_model.pth"])
        il.addWidget(self.model_combo)
        self.backend_combo = QComboBox(); self.backend_combo.addItems(["PyTorch", "Compiled", "TorchScript", "TensorRT (FP16)", "ONNX (CPU fallback)"])
        self.backend_combo.setToolTip("Inference backend")
        il.addWidget(self.backend_combo)
        sl = QHBoxLayout(); sl.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Horizontal); self.scale_slider.setRange(25, 100)
        saved_scale = self._load_ui_setting("scale", 100)
        self.scale_slider.setValue(saved_scale)
        self.scale_slider.valueChanged.connect(lambda v: self._save_ui_setting("scale", v))
        self.scale_label = QLabel("1.00x")
        self.scale_slider.valueChanged.connect(lambda v: self.scale_label.setText(f"{v/100:.2f}x"))
        sl.addWidget(self.scale_slider); sl.addWidget(self.scale_label)
        il.addLayout(sl)
        self.tiled_check = QCheckBox("Tiled")
        # ★★ 2026-10-29（用户要求）：**这个勾选框不再显示给用户**。
        #   推理该不该分块，取决于**训练时是怎么训的**，不是问用户 ——
        #   训练在原分辨率上按 tile 切块，推理却把整图缩成方形喂进去，同一块缺陷在
        #   输入里的尺寸会差 `原图宽/tile` 倍（1500×1500 图、tile 256 ⇒ 5.86 倍），
        #   现场表现就是"训练 mIoU 0.92、一推理啥也检不到"。
        #   现在由 `Predictor._resolve_train_mode()` 自己判断（读 checkpoint 的
        #   train_mode，老模型用"块尺寸 vs 图尺寸"启发式），调用处传 tiled=None。
        #   控件本身保留（还有代码在引用它，且想强制指定时可临时 setChecked），
        #   只是不放进界面。
        self.tiled_check.setToolTip("已改为自动：推理方式跟随训练模式\n"
                                    "（训练分块→滑窗推理；训练整图→整图推理）")
        self.tiled_check.setChecked(True)
        saved_tiled2 = self._load_ui_setting("tiled", False)
        self.tiled_check.setChecked(saved_tiled2)
        self.tiled_check.toggled.connect(lambda v: self._save_ui_setting("tiled", v))
        # ← 不再 il.addWidget(self.tiled_check)：界面上不显示

        # ── 分隔线 ──
        _sep1 = QFrame(); _sep1.setFrameShape(QFrame.HLine); _sep1.setFrameShadow(QFrame.Sunken)
        il.addWidget(_sep1)

        # ══ ② 运行 ══
        il.addWidget(QPushButton("Run Inference", clicked=self._start_inference))
        # 批量推理：走的是模型管理里那个同一个入口（_batch_inference），
        # 按任务类型自动分发（分割/多视图/检测/分类），结果写到 <工程>/outputs/。
        # 这里加一个更顺手的入口 —— 用户原来在「模型管理」面板最底部，得滚很远才找得到。
        _btn_batch_all = QPushButton("批量推理（全部图片）…")
        _btn_batch_all.setToolTip("对工程内所有图片跑一遍推理，结果写到 <工程>/outputs/")
        _btn_batch_all.clicked.connect(self._batch_inference)
        il.addWidget(_btn_batch_all)
        self.infer_progress = QProgressBar(); il.addWidget(self.infer_progress)
        self.infer_status = QLabel(); il.addWidget(self.infer_status)
        hil = QHBoxLayout()
        self.btn_infer_stop = QPushButton("Stop"); self.btn_infer_stop.clicked.connect(self._stop_inference)
        self.btn_infer_stop.setEnabled(False); hil.addWidget(self.btn_infer_stop)
        self.infer_result_label = QLabel(""); hil.addWidget(self.infer_result_label)
        il.addLayout(hil)

        # ── 分隔线 ──
        _sep2 = QFrame(); _sep2.setFrameShape(QFrame.HLine); _sep2.setFrameShadow(QFrame.Sunken)
        il.addWidget(_sep2)

        # ══ ③ 结果过滤 ══

        # 推理参数（对标友商 友商平台）：按类别过滤连通域 + 全局后处理。
        # 放在推理控件下面、同一个滚动区里，改完即存到 <工程>/inference_params.json。
        from ui.infer_params import InferParamsPanel
        self.infer_params = InferParamsPanel()
        il.addWidget(self.infer_params)
        il.addStretch()

        # Panel 3: Training
        panel_train = QWidget(); tl = QVBoxLayout(panel_train); tl.setContentsMargins(4, 2, 2, 2)
        tl.addWidget(QLabel("<b>Training</b>"))
        self.train_progress = QProgressBar(); tl.addWidget(self.train_progress)
        self.train_status = QLabel("Ready"); tl.addWidget(self.train_status)
        btns = QHBoxLayout()
        self.btn_train_start = QPushButton("Start Training")
        self.btn_train_start.clicked.connect(self._start_training)
        self.btn_train_start.setStyleSheet("QPushButton { background-color: #00FF9C; color: #061018; font-weight: bold; padding: 4px 12px; }")
        btns.addWidget(self.btn_train_start)
        self.btn_train_stop = QPushButton("Stop")
        self.btn_train_stop.clicked.connect(self._stop_batch)
        self.btn_train_stop.setEnabled(False)
        btns.addWidget(self.btn_train_stop)
        tl.addLayout(btns)
        # 参数表紧跟按钮，stretch 挪到最底部 —— 否则 Start Training 与 Epochs 之间
        # 会夹一大片空白（见 ShrinkStack 的说明）
        fl = QFormLayout()
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 5000); self.epochs_spin.setValue(200)
        fl.addRow("Epochs:", self.epochs_spin)
        # ★ 2026-10-23：早停。8 张图那种小数据集训到 2000+ 轮时 val loss 会翻几倍（实测 0.20→1.57），
        #   后面全在把模型训坏。0 = 关闭（默认，保持老行为）。
        self.early_stop_spin = QSpinBox()
        self.early_stop_spin.setRange(0, 2000); self.early_stop_spin.setValue(0)
        self.early_stop_spin.setToolTip(
            "连续 N 轮「val loss 与 mIoU 都没改善」就自动停止训练。\n"
            "0 = 不早停。小数据集建议 100~200。")
        fl.addRow("早停耐心:", self.early_stop_spin)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 128); self.batch_spin.setValue(8)
        fl.addRow("Batch:", self.batch_spin)
        self.kfold_spin = QSpinBox(); self.kfold_spin.setRange(1, 10); self.kfold_spin.setValue(1)
        self.kfold_spin.setToolTip("1 = no cross-validation")
        fl.addRow("K-Fold CV:", self.kfold_spin)
        self.resume_check = QCheckBox("Resume from last")
        fl.addRow("", self.resume_check)

        # ══ ★ 训练参数全部开放到面板 ══
        # 用户要求：点了 Start Training 再弹一个「Training Settings」窗太冗余，
        # 参数应该直接摆在面板上，点按钮就开训。
        # 这些和 TrainSettingsDialog 里的**同名同义**，只是搬到面板上常驻；
        # 对话框类还留着（不再调用），要回退只要把 _start_training 里那两行换回去。
        self.tr_model_combo = QComboBox()
        # ⚠️ 只能放 create_model 真正支持的：单图工程 = deeplabv3 / fcn（training/models.py:39,49）
        #    unet / lraspp 是**多视图工程**专用（training/multiview.py），单图选了会
        #    ValueError: Unknown model —— 打开多视图工程时下面 _start_training 会自动换成对的列表。
        # unet 也放进来：单图工程现在也支持（小模型，200×200 小图推理 2-3ms）
        self.tr_model_combo.addItems(["deeplabv3", "fcn", "unet"])
        self.tr_model_combo.setCurrentText("deeplabv3")
        self.tr_model_combo.setToolTip("deeplabv3：默认，小目标分割稳；fcn：更轻更快")
        fl.addRow("Model:", self.tr_model_combo)

        self.tr_size_combo = QComboBox()
        self.tr_size_combo.addItems(["256", "512", "768", "1024"])
        self.tr_size_combo.setCurrentText("256")
        self.tr_size_combo.setToolTip("训练输入尺寸（未开分块时 = 整图缩放到这个尺寸）。"
                                   "200×200 的图选 256；不能再小，U-Net 4 层下采样要求能被 16 整除")
        fl.addRow("Image Size:", self.tr_size_combo)

        self.tr_mode_combo = QComboBox()
        self.tr_mode_combo.addItems(["精细（缩到1024 + 滑窗200）", "快速（缩到1024 整图）"])
        self.tr_mode_combo.setToolTip(
            "精细：大图缩到长边1024后按200滑窗拼回，保留细节，慢\n"
            "快速：大图缩到长边1024整图一次前向，对标友商，快")
        fl.addRow("推理模式:", self.tr_mode_combo)

        self.tr_loss_combo = QComboBox()
        self.tr_loss_combo.addItems(["ce+lovasz", "cross_entropy", "lovasz", "dice"])
        self.tr_loss_combo.setCurrentText("ce+lovasz")
        fl.addRow("Loss:", self.tr_loss_combo)

        self.tr_tile_check = QCheckBox("分块（滑窗）")
        self.tr_tile_check.setToolTip("只对大图有意义；200×200 的图**不要勾**（整图训练就够）")
        self.tr_tile_check.setChecked(False)
        fl.addRow("", self.tr_tile_check)

        self.tr_tile_spin = QSpinBox(); self.tr_tile_spin.setRange(64, 1024); self.tr_tile_spin.setSingleStep(64)
        self.tr_tile_spin.setValue(256)
        fl.addRow("滑窗大小:", self.tr_tile_spin)

        self.tr_tile_stride_spin = QSpinBox(); self.tr_tile_stride_spin.setRange(10, 100); self.tr_tile_stride_spin.setValue(75)
        fl.addRow("步长%:", self.tr_tile_stride_spin)

        def _sync_tile_enabled():
            on = self.tr_tile_check.isChecked()
            self.tr_tile_spin.setEnabled(on)
            self.tr_tile_stride_spin.setEnabled(on)
            self.tr_size_combo.setEnabled(not on)
        self.tr_tile_check.toggled.connect(_sync_tile_enabled)
        _sync_tile_enabled()

        tl.addLayout(fl)
        # 2026-10-22：面板参数持久化（用户反馈重启后参数不保留）
        #   原来只有 Training Settings 对话框读写 .train_settings.json，
        #   而且它的 SETTINGS_FILE 多算了一层目录（指到不存在的 AI_Training_Platform），
        #   面板上的控件更是从没跟设置文件挂钩。
        self._load_panel_train_settings()
        tl.addStretch()

        # Panel 4: Detection Tools
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

        # Panel 5: Detection Training
        panel_det_train = QWidget(); dtl = QVBoxLayout(panel_det_train); dtl.setContentsMargins(4, 2, 2, 2)
        dtl.addWidget(QLabel("<b>Detection Training</b>"))
        self.det_train_progress = QProgressBar(); dtl.addWidget(self.det_train_progress)
        self.det_train_status = QLabel("Ready"); dtl.addWidget(self.det_train_status)
        self.btn_det_train_stop = QPushButton("Stop"); self.btn_det_train_stop.clicked.connect(self._stop_training)
        self.btn_det_train_stop.setEnabled(False); dtl.addWidget(self.btn_det_train_stop)
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

        # Panel 6: Classification Training
        panel_cls_train = QWidget(); ctl = QVBoxLayout(panel_cls_train); ctl.setContentsMargins(4, 2, 2, 2)
        ctl.addWidget(QLabel("<b>Classification Training</b>"))
        self.cls_mixed_check = QCheckBox("Mixed (Gray + Height Map)")
        self.cls_mixed_check.setToolTip("Use paired grayscale (.bmp) + height (.tif) images for training")
        self.cls_mixed_check.toggled.connect(self._on_cls_mixed_toggled)
        ctl.addWidget(self.cls_mixed_check)
        self.cls_train_progress = QProgressBar(); ctl.addWidget(self.cls_train_progress)
        self.cls_train_status = QLabel("Ready"); ctl.addWidget(self.cls_train_status)
        btn_cls_stop = QPushButton("Stop"); btn_cls_stop.clicked.connect(self._stop_training)
        btn_cls_stop.setEnabled(False); ctl.addWidget(btn_cls_stop)
        self._cls_stop_btn = btn_cls_stop
        cfl = QFormLayout()
        self.cls_model_combo = QComboBox()
        self.cls_model_combo.addItems(["resnet18", "resnet34", "resnet50", "resnet101", "efficientnet_b0",
                                        "efficientnet_b3", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
                                        "vit_b_16", "vit_b_32",
                                        "convnext_tiny", "convnext_base", "efficientnet_v2_s", "efficientnet_v2_m",
                                        "efficientnet_v2_l", "swin_t", "swin_s", "swin_b",
                                        "densenet121", "densenet161", "densenet201"])
        cfl.addRow("Model:", self.cls_model_combo)
        self.cls_epochs_spin = QSpinBox(); self.cls_epochs_spin.setRange(1, 2000); self.cls_epochs_spin.setValue(200)
        cfl.addRow("Epochs:", self.cls_epochs_spin)
        self.cls_batch_spin = QSpinBox(); self.cls_batch_spin.setRange(1, 256); self.cls_batch_spin.setValue(32)
        cfl.addRow("Batch:", self.cls_batch_spin)
        self.cls_optim_combo = QComboBox()
        self.cls_optim_combo.addItems(["AdamW", "Adam", "SGD"])
        cfl.addRow("Optimizer:", self.cls_optim_combo)
        self.cls_lr_spin = QDoubleSpinBox(); self.cls_lr_spin.setRange(0.00001, 0.1); self.cls_lr_spin.setSingleStep(0.0001)
        self.cls_lr_spin.setDecimals(5); self.cls_lr_spin.setValue(0.001)
        cfl.addRow("LR:", self.cls_lr_spin)
        self.cls_loss_combo = QComboBox()
        self.cls_loss_combo.addItems(["cross_entropy", "label_smoothing", "focal"])
        cfl.addRow("Loss:", self.cls_loss_combo)
        cls_aug_hint = QLabel("数据增强：右侧「增强」面板（全局生效）")
        cls_aug_hint.setWordWrap(True)
        cls_aug_hint.setStyleSheet("color: #8A93A6;")
        cfl.addRow("Augment:", cls_aug_hint)
        hl_scale = QHBoxLayout()
        self.cls_scale_w_spin = QDoubleSpinBox(); self.cls_scale_w_spin.setRange(0.1, 1.0); self.cls_scale_w_spin.setSingleStep(0.05)
        self.cls_scale_w_spin.setDecimals(2); self.cls_scale_w_spin.setValue(1.0)
        self.cls_scale_w_spin.setToolTip("Width scale factor (0.1~1.0)")
        hl_scale.addWidget(QLabel("W:"))
        hl_scale.addWidget(self.cls_scale_w_spin)
        self.cls_scale_h_spin = QDoubleSpinBox(); self.cls_scale_h_spin.setRange(0.1, 1.0); self.cls_scale_h_spin.setSingleStep(0.05)
        self.cls_scale_h_spin.setDecimals(2); self.cls_scale_h_spin.setValue(1.0)
        self.cls_scale_h_spin.setToolTip("Height scale factor (0.1~1.0)")
        hl_scale.addWidget(QLabel("H:"))
        hl_scale.addWidget(self.cls_scale_h_spin)
        self.cls_scale_link_btn = QPushButton("🔗"); self.cls_scale_link_btn.setFixedWidth(30)
        self.cls_scale_link_btn.setCheckable(True); self.cls_scale_link_btn.setChecked(True)
        self.cls_scale_link_btn.setToolTip("Lock W/H ratio")
        self.cls_scale_link_btn.toggled.connect(self._on_cls_scale_link_toggled)
        hl_scale.addWidget(self.cls_scale_link_btn)
        self.cls_scale_w_spin.valueChanged.connect(lambda: self._sync_cls_scale("w"))
        self.cls_scale_h_spin.valueChanged.connect(lambda: self._sync_cls_scale("h"))
        cfl.addRow("Pre-Scale:", hl_scale)
        hl_input = QHBoxLayout()
        self.cls_input_w_spin = QSpinBox(); self.cls_input_w_spin.setRange(32, 2048); self.cls_input_w_spin.setSingleStep(32)
        self.cls_input_w_spin.setValue(448)
        self.cls_input_w_spin.setToolTip("输入尺寸宽（letterbox 目标画布宽）")
        hl_input.addWidget(QLabel("W:")); hl_input.addWidget(self.cls_input_w_spin)
        self.cls_input_h_spin = QSpinBox(); self.cls_input_h_spin.setRange(32, 2048); self.cls_input_h_spin.setSingleStep(32)
        self.cls_input_h_spin.setValue(192)
        self.cls_input_h_spin.setToolTip("输入尺寸高（letterbox 目标画布高）")
        hl_input.addWidget(QLabel("H:")); hl_input.addWidget(self.cls_input_h_spin)
        cfl.addRow("Input Size:", hl_input)
        self.cls_keep_aspect_check = QCheckBox("保持宽高比 Letterbox（不拉伸变形）")
        self.cls_keep_aspect_check.setChecked(True)
        self.cls_keep_aspect_check.setToolTip("等比缩放后居中黑边填充到输入尺寸；关闭=直接拉伸到 W×H")
        cfl.addRow("", self.cls_keep_aspect_check)
        self.cls_amp_check = QCheckBox("AMP (mixed precision)"); self.cls_amp_check.setChecked(True)
        cfl.addRow("", self.cls_amp_check)
        self.cls_kfold_spin = QSpinBox(); self.cls_kfold_spin.setRange(1, 10); self.cls_kfold_spin.setValue(1)
        self.cls_kfold_spin.setToolTip("1 = no cross-validation")
        cfl.addRow("K-Fold:", self.cls_kfold_spin)
        self.cls_resume_check = QCheckBox("Resume from last")
        cfl.addRow("", self.cls_resume_check)
        self.cls_balanced_check = QCheckBox("类别平衡权重（缓解样本不平衡）")
        self.cls_balanced_check.setToolTip("按各类训练样本数反比加权损失，类别不平衡时通常能明显提升准确率")
        cfl.addRow("", self.cls_balanced_check)
        self.cls_ema_check = QCheckBox("EMA 权重平均（原生基线建议关闭）")
        self.cls_ema_check.setToolTip("开启后保存 EMA 平滑权重（训练更稳但偏离原始权重）；关闭=直接保存训练出的原始权重")
        cfl.addRow("", self.cls_ema_check)

        # --- Hyperparameters ---
        self.cls_wd_spin = QDoubleSpinBox(); self.cls_wd_spin.setRange(0, 0.1); self.cls_wd_spin.setSingleStep(0.0001)
        self.cls_wd_spin.setDecimals(5); self.cls_wd_spin.setValue(0.0001)
        self.cls_wd_spin.setToolTip("L2 regularization strength (0 = off)")
        cfl.addRow("Weight Decay:", self.cls_wd_spin)

        self.cls_momentum_spin = QDoubleSpinBox(); self.cls_momentum_spin.setRange(0, 0.999); self.cls_momentum_spin.setSingleStep(0.1)
        self.cls_momentum_spin.setDecimals(3); self.cls_momentum_spin.setValue(0.9)
        self.cls_momentum_spin.setToolTip("SGD momentum (ignored for Adam/AdamW)")
        cfl.addRow("Momentum:", self.cls_momentum_spin)

        self.cls_ls_spin = QDoubleSpinBox(); self.cls_ls_spin.setRange(0, 0.5); self.cls_ls_spin.setSingleStep(0.05)
        self.cls_ls_spin.setDecimals(2); self.cls_ls_spin.setValue(0.1)
        self.cls_ls_spin.setToolTip("Label smoothing epsilon (0 = hard labels)")
        cfl.addRow("Label Smooth:", self.cls_ls_spin)

        self.cls_focal_gamma_spin = QDoubleSpinBox(); self.cls_focal_gamma_spin.setRange(0.5, 5.0); self.cls_focal_gamma_spin.setSingleStep(0.5)
        self.cls_focal_gamma_spin.setDecimals(1); self.cls_focal_gamma_spin.setValue(2.0)
        self.cls_focal_gamma_spin.setToolTip("Focal loss gamma (higher = more focus on hard samples)")
        cfl.addRow("Focal Gamma:", self.cls_focal_gamma_spin)

        self.cls_dropout_spin = QDoubleSpinBox(); self.cls_dropout_spin.setRange(0, 0.8); self.cls_dropout_spin.setSingleStep(0.1)
        self.cls_dropout_spin.setDecimals(2); self.cls_dropout_spin.setValue(0.0)
        self.cls_dropout_spin.setToolTip("Dropout rate in classifier head (0 = off)")
        cfl.addRow("Dropout:", self.cls_dropout_spin)

        self.cls_scheduler_combo = QComboBox()
        self.cls_scheduler_combo.addItems(["none", "step", "cosine", "plateau"])
        self.cls_scheduler_combo.setToolTip("LR scheduler type")
        cfl.addRow("LR Scheduler:", self.cls_scheduler_combo)

        self.cls_warmup_spin = QSpinBox(); self.cls_warmup_spin.setRange(0, 50); self.cls_warmup_spin.setValue(0)
        self.cls_warmup_spin.setToolTip("Linear LR warmup epochs (0 = off)")
        cfl.addRow("Warmup Epochs:", self.cls_warmup_spin)

        self.cls_early_stop_spin = QSpinBox(); self.cls_early_stop_spin.setRange(0, 200); self.cls_early_stop_spin.setValue(0)
        self.cls_early_stop_spin.setToolTip("Stop if val loss not improved for N epochs (0 = off)")
        cfl.addRow("Early Stop:", self.cls_early_stop_spin)

        self.cls_select_metric_combo = QComboBox()
        self.cls_select_metric_combo.addItems(["acc", "balanced_acc", "macro_f1", "ng_recall"])
        self.cls_select_metric_combo.setToolTip(
            "用哪个指标挑『最优模型』：\n"
            "acc = 总体准确率（默认，口径不变）\n"
            "balanced_acc = 类别召回的平均（类别不均衡时更稳）\n"
            "macro_f1 = 逐类 F1 平均\n"
            "ng_recall = 只看缺陷召回（最贴近『少漏检』）\n"
            "验证集很小时建议用 balanced_acc 或 macro_f1")
        cfl.addRow("Best Metric:", self.cls_select_metric_combo)

        ctl.addLayout(cfl)
        self.cls_recommend_btn = QPushButton("🧠 智能推荐训练参数", clicked=self._open_cls_recommend)
        self.cls_recommend_btn.setStyleSheet(
            "QPushButton { background:#7B2FBE; color:white; font-weight:bold; padding:6px; border:none; }"
            "QPushButton:hover { background:#B967FF; }")
        ctl.addWidget(self.cls_recommend_btn)
        self.cls_native_btn = QPushButton("⚡ 原生基线参数（最简，无增强/EMA/类平衡）", clicked=self._apply_cls_native_params)
        self.cls_native_btn.setStyleSheet(
            "QPushButton { background:#00A86B; color:#061018; font-weight:bold; padding:6px; border:none; }"
            "QPushButton:hover { background:#00FF9C; }")
        ctl.addWidget(self.cls_native_btn)
        self._cls_start_btn = QPushButton("Start Training", clicked=self._start_cls_training)
        ctl.addWidget(self._cls_start_btn)
        # Loss/Acc curve now lives in an independent window
        self._cls_loss_window = LossCurveWindow(self)
        self._cls_loss_open_btn = QPushButton("打开 Loss/Acc Curve Window", clicked=self._show_cls_loss_window)
        ctl.addWidget(self._cls_loss_open_btn)
        ctl.addStretch()

        # Panel 7: Classification Inference
        panel_cls_infer = QWidget(); cil = QVBoxLayout(panel_cls_infer); cil.setContentsMargins(4, 2, 2, 2)
        cil.addWidget(QLabel("<b>Classification Inference</b>"))
        self.cls_infer_model_combo = QComboBox()
        self.cls_infer_model_combo.setToolTip("Select model checkpoint")
        cil.addWidget(self.cls_infer_model_combo)
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Top-K:"))
        self.cls_topk_spin = QSpinBox(); self.cls_topk_spin.setRange(1, 20); self.cls_topk_spin.setValue(5)
        hl2.addWidget(self.cls_topk_spin); cil.addLayout(hl2)
        self.cls_infer_progress = QProgressBar(); cil.addWidget(self.cls_infer_progress)
        self.cls_infer_status = QLabel("Ready"); cil.addWidget(self.cls_infer_status)
        self.cls_result_label = QLabel(""); self.cls_result_label.setWordWrap(True)
        self.cls_result_label.setStyleSheet("QLabel { font-family: Consolas; font-size: 11px; padding: 4px; }")
        cil.addWidget(self.cls_result_label)
        ih = QHBoxLayout()
        ih.addWidget(QPushButton("Single Inference", clicked=self._start_cls_inference))
        ih.addWidget(QPushButton("Batch Inference", clicked=self._start_cls_batch_inference))
        cil.addLayout(ih)
        hl3 = QHBoxLayout()
        hl3.addWidget(QPushButton("Heatmap", clicked=self._generate_cls_heatmap))
        self.cls_auto_heatmap = QCheckBox("Auto-heatmap after inference")
        self.cls_auto_heatmap.setChecked(True)
        hl3.addWidget(self.cls_auto_heatmap)
        cil.addLayout(hl3)
        self.cls_tta_check = QCheckBox("TTA 多裁剪投票（单张推理更稳）")
        saved_tta = self._load_ui_setting("cls_tta", False)
        self.cls_tta_check.setChecked(saved_tta)
        self.cls_tta_check.toggled.connect(lambda v: self._save_ui_setting("cls_tta", v))
        cil.addWidget(self.cls_tta_check)

        # Confusion Matrix section
        cil.addWidget(QLabel("<b>Confusion Matrix</b>"))
        eh = QHBoxLayout()
        eh.addWidget(QPushButton("Evaluate", clicked=self._eval_cls_model))
        self.cls_eval_status = QLabel("")
        eh.addWidget(self.cls_eval_status)
        cil.addLayout(eh)
        self.cls_cm_table = QTableWidget()
        self.cls_cm_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cls_cm_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.cls_cm_table.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
        cil.addWidget(self.cls_cm_table)
        # 模型版本管理
        vgb = QGroupBox("模型版本")
        vv = QVBoxLayout(vgb)
        self.cls_version_table = QTableWidget()
        self.cls_version_table.setColumnCount(6)
        self.cls_version_table.setHorizontalHeaderLabels(["版本", "时间", "验证Acc", "批推Acc", "平均耗时", "模型"])
        self.cls_version_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cls_version_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cls_version_table.setMaximumHeight(150)
        self.cls_version_table.doubleClicked.connect(self._cls_version_activate)
        vv.addWidget(self.cls_version_table)
        vhb = QHBoxLayout()
        vhb.addWidget(QPushButton("刷新版本", clicked=self._refresh_cls_versions))
        vhb.addWidget(QPushButton("设为当前", clicked=self._cls_version_activate))
        vhb.addWidget(QPushButton("导出ZIP", clicked=self._cls_version_export))
        vv.addLayout(vhb)
        cil.addWidget(vgb)
        cil.addStretch()

        # Panel 8: OCR Inference
        panel_ocr = QWidget(); oil = QVBoxLayout(panel_ocr); oil.setContentsMargins(4, 2, 2, 2)
        oil.addWidget(QLabel("<b>OCR Inference</b>"))
        self.ocr_status = QLabel("Ready"); oil.addWidget(self.ocr_status)
        self.ocr_progress = QProgressBar(); oil.addWidget(self.ocr_progress)
        oh = QHBoxLayout()
        oh.addWidget(QPushButton("Run OCR (Single)", clicked=self._run_ocr_single))
        oh.addWidget(QPushButton("Batch OCR", clicked=self._run_ocr_batch))
        oil.addLayout(oh)
        self.ocr_show_check = QCheckBox("Show OCR overlay on canvas")
        self.ocr_show_check.setChecked(True)
        self.ocr_show_check.toggled.connect(lambda v: self._update_ocr_overlay())
        oil.addWidget(self.ocr_show_check)
        self.ocr_result_table = QTableWidget()
        self.ocr_result_table.setColumnCount(5)
        self.ocr_result_table.setHorizontalHeaderLabels(["#", "Text", "Conf", "Pos", "Size"])
        # 允许双击/再次单击选中文本（仅 Text 列可编辑，其他列仍只读），配合右键复制
        self.ocr_result_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked)
        self.ocr_result_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ocr_result_table.customContextMenuRequested.connect(self._on_ocr_result_menu)
        oil.addWidget(self.ocr_result_table)
        oil.addWidget(QPushButton("Export CSV", clicked=self._export_ocr_csv))
        oil.addStretch()

        # Panel 9: OCV Inspection
        panel_ocv = QWidget(); ovl = QVBoxLayout(panel_ocv); ovl.setContentsMargins(4, 2, 2, 2)
        ovl.addWidget(QLabel("<b>OCV Quality Inspection</b>"))
        self.ocv_status = QLabel("Ready"); ovl.addWidget(self.ocv_status)
        ovl.addWidget(QLabel("NG Threshold:"))
        self.ocv_threshold = QDoubleSpinBox()
        self.ocv_threshold.setRange(0.1, 50.0); self.ocv_threshold.setValue(3.0); self.ocv_threshold.setSingleStep(0.5)
        ovl.addWidget(self.ocv_threshold)
        ovh = QHBoxLayout()
        ovh.addWidget(QPushButton("Build OCV Model", clicked=self._build_ocv_model))
        ovh.addWidget(QPushButton("Load OCV Model", clicked=self._load_ocv_model))
        ovl.addLayout(ovh)
        ovh2 = QHBoxLayout()
        ovh2.addWidget(QPushButton("Inspect Current", clicked=self._ocv_inspect_current))
        ovh2.addWidget(QPushButton("Batch Inspect", clicked=self._ocv_batch_inspect))
        ovl.addLayout(ovh2)
        self.ocv_result_label = QLabel(""); self.ocv_result_label.setWordWrap(True)
        ovl.addWidget(self.ocv_result_label)
        ovl.addStretch()

        # Panel 10: Height Map Controls
        panel_hm = QWidget(); hml = QVBoxLayout(panel_hm); hml.setContentsMargins(4, 2, 2, 2)
        hml.addWidget(QLabel("<b>Height Map Colormap</b>"))
        self.hm_status = QLabel("No height map loaded")
        self.hm_status.setStyleSheet("color: #8A93A6; font-size: 11px;")
        hml.addWidget(self.hm_status)
        self.hm_data_range = QLabel("")
        self.hm_data_range.setStyleSheet("color: #8A93A6; font-size: 10px;")
        hml.addWidget(self.hm_data_range)
        hml.addWidget(QLabel("Mapping Range (mm):"))
        rng = QHBoxLayout()
        self.hm_vmin = QDoubleSpinBox()
        self.hm_vmin.setRange(-999, 999); self.hm_vmin.setDecimals(4)
        self.hm_vmin.setSingleStep(0.01)
        self.hm_vmin.setToolTip("Min height (maps to blue/cold color)")
        rng.addWidget(self.hm_vmin)
        rng.addWidget(QLabel("~"))
        self.hm_vmax = QDoubleSpinBox()
        self.hm_vmax.setRange(-999, 999); self.hm_vmax.setDecimals(4)
        self.hm_vmax.setSingleStep(0.01)
        self.hm_vmax.setToolTip("Max height (maps to red/warm color)")
        rng.addWidget(self.hm_vmax)
        hml.addLayout(rng)
        self.hm_colorbar = QLabel()
        self.hm_colorbar.setFixedHeight(20)
        self.hm_colorbar.setMinimumWidth(100)
        hml.addWidget(self.hm_colorbar)
        hml.addWidget(QLabel("Color Scheme:"))
        self.hm_cmap_combo = QComboBox()
        self.hm_cmap_combo.addItems(["JET", "TURBO", "HOT", "RAINBOW", "VIRIDIS", "PLASMA", "INFERNO", "MAGMA", "HSV"])
        self.hm_cmap_combo.setCurrentText("JET")
        hml.addWidget(self.hm_cmap_combo)
        bth = QHBoxLayout()
        auto_btn = QPushButton("Auto Range")
        auto_btn.clicked.connect(self._hm_auto_range)
        bth.addWidget(auto_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._hm_apply)
        apply_btn.setStyleSheet("QPushButton { background-color: #0E3A52; color: #00E5FF; border: 1px solid #00E5FF; font-weight: bold; }")
        bth.addWidget(apply_btn)
        hml.addLayout(bth)
        hml.addStretch()

        # Panel 11: 数据增强（全局，所有训练任务共用）
        panel_aug = QWidget(); augl = QVBoxLayout(panel_aug); augl.setContentsMargins(4, 2, 2, 2)
        augl.addWidget(QLabel("<b>数据增强（全局）</b>"))
        aug_tip = QLabel("对所有训练任务生效：分割 / 分类 / 检测\n（随机应用已勾选的增强方式）")
        aug_tip.setStyleSheet("color: #8A93A6; font-size: 11px;")
        aug_tip.setWordWrap(True)
        augl.addWidget(aug_tip)
        self.aug_enabled_check = QCheckBox("启用数据增强")
        self.aug_enabled_check.toggled.connect(self._save_augment_ui)
        augl.addWidget(self.aug_enabled_check)
        augl.addWidget(QLabel("<b>增强方式：</b>"))
        aug_grid = QGridLayout(); aug_grid.setHorizontalSpacing(10); aug_grid.setVerticalSpacing(2)
        self._aug_checks = {}
        from training.augment_config import AUGMENT_FLAG_LABELS
        for i, (key, label) in enumerate(AUGMENT_FLAG_LABELS):
            cb = QCheckBox(label)
            cb.toggled.connect(self._save_augment_ui)
            self._aug_checks[key] = cb
            aug_grid.addWidget(cb, i // 2, i % 2)
        augl.addLayout(aug_grid)
        abtns = QHBoxLayout()
        b_all = QPushButton("全选"); b_all.clicked.connect(lambda: self._set_aug_all(True))
        b_none = QPushButton("全不选"); b_none.clicked.connect(lambda: self._set_aug_all(False))
        b_def = QPushButton("恢复默认"); b_def.clicked.connect(self._set_aug_default)
        abtns.addWidget(b_all); abtns.addWidget(b_none); abtns.addWidget(b_def)
        augl.addLayout(abtns)
        self.aug_status = QLabel(""); self.aug_status.setWordWrap(True)
        self.aug_status.setStyleSheet("color: #8A93A6; font-size: 11px;")
        augl.addWidget(self.aug_status)
        augl.addStretch()

        # ===== Inference group (auto-switch by project type) =====
        # ===== Inference 段（按项目类型自动切换子页）=====
        #    标题与内容分开：标题留在滚动区**外面**（滚动时不会跟着滚没），
        #    只有内容区滚动。两段拼装见下面「Training + Inference 合并面板」。
        self._infer_title = QLabel("推理 · 语义分割")
        self._infer_title.setStyleSheet("font-weight: bold; color: #00E5FF;")
        self._infer_stack = ShrinkStack()
        self._infer_stack.addWidget(panel_infer)      # 0: seg / det (shared)
        self._infer_stack.addWidget(panel_cls_infer)  # 1: classification
        self._infer_stack.addWidget(panel_ocr)        # 2: OCR
        self._infer_stack.addWidget(panel_ocv)        # 3: OCV

        # ===== Training 段（按项目类型自动切换子页）=====
        self._train_title = QLabel("训练 · 语义分割")
        self._train_title.setStyleSheet("font-weight: bold; color: #FF2A6D;")
        self._train_stack = ShrinkStack()
        self._train_stack.addWidget(panel_train)      # 0: segmentation
        self._train_stack.addWidget(panel_det_train)  # 1: detection
        self._train_stack.addWidget(panel_cls_train)  # 2: classification
        panel_train_na = QWidget(); nal = QVBoxLayout(panel_train_na)
        na_label = QLabel("该任务类型无独立训练流程\n请在「推理」面板中直接操作")
        na_label.setWordWrap(True); na_label.setStyleSheet("color: #8A93A6;")
        nal.addWidget(na_label); nal.addStretch()
        self._train_stack.addWidget(panel_train_na)   # 3: no training flow (OCR/OCV)

        # 换子页后要重新量高：ShrinkStack 只按当前页报告尺寸，
        # 不 updateGeometry 的话滚动区仍按旧页的高度留白。
        for _st in (self._train_stack, self._infer_stack):
            _st.currentChanged.connect(self._on_ti_stack_page_changed)

        # ===== Training + Inference 合并面板（用户要求：训练与推理整合成一个面板）=====
        #    为什么内容区都套一层 QScrollArea：QStackedWidget 的 minimumSizeHint 是
        #    "所有子页里最大的那个"（实测训练 991px、推理 546px）。直接塞进 splitter
        #    会把这个虚高的下界当成硬约束 —— 面板整体超高、底部控件被裁掉且**没有滚动条**。
        #    套一层滚动区后下界降到几十像素，两段都能自由压缩，够不着的内容滚动查看。
        #
        #    高度分配：**按内容实际高度自动分配**（`_fit_ti_split`），不再固定 65/35 ——
        #    固定比例会把"内容只有一百多像素的训练段"撑成一大片空白（用户实测截图反馈）。
        panel_ti = QWidget(); til = QVBoxLayout(panel_ti)
        til.setContentsMargins(0, 0, 0, 0); til.setSpacing(2)
        self._ti_split = QSplitter(Qt.Vertical)
        self._ti_split.setChildrenCollapsible(False)
        self._ti_split.setHandleWidth(4)
        self._ti_split.addWidget(self._make_ti_section(self._train_title, self._train_stack))
        self._ti_split.addWidget(self._make_ti_section(self._infer_title, self._infer_stack))
        self._ti_split.setSizes([150, 640])
        til.addWidget(self._ti_split)

        # ===== 右侧 3 个模块（参照友商 友商平台 右栏：参数管理 / 标注管理 / 模型管理）=====
        #    友商的右栏不是"一堆并列工具页"，而是**模块**，每个模块里竖着若干**区块**
        #    （区块 = 左侧一条彩色竖条 + 标题 + 内容），区块可折叠、各按内容占高。
        #    这样既没有"某一块撑出大片空白"，也不用靠比例分配高度 —— 不看的折叠起来就行。
        # 「高度图参数」只对**有 TIF/高度图**的工程有意义（混合分类、3D 高度图），
        # 普通三视图工程里它永远是 "No height map loaded"，白占一块地方（用户要求去掉）。
        self._hm_section = self._make_section_box("高度图参数", [panel_hm])
        page_params = self._make_module_page([
            ("训练参数", [self._train_title, self._train_stack]),
            ("增强参数", [panel_aug]),
            self._hm_section,          # 仅在"有高度图/TIF"或混合分类工程里显示

            ("推理参数", [self._infer_title, self._infer_stack]),
        ])
        # 「标注操作」区块已按用户要求去掉（导出/清理那几个动作挪到图片列表右键菜单，
        # 那里本来就有"批量导入 JSON"，放一起更顺）。_make_annotation_ops_section 保留备用。
        #
        # 「检测工具」是**目标检测工程专属**的（语义分割里那些画框工具全是死的），
        # 所以先建好、由 _set_task() 按任务类型显隐，而不是所有工程都摆着。
        self._det_section = self._make_section_box("检测工具", [panel_det])
        page_annot = self._make_module_page([
            self._panel_split,                     # 类别 / 标注统计 / 结果 三个区块（自带标题+计数）
            self._det_section,
        ])
        page_model = self._make_module_page([
            ("模型列表", [self._make_model_list_section()]),
            ("版本历史", [panel_ver]),
        ])
        # ⚠️ 原来的第 4 页「图像管理」（图像信息 / 数据集划分 / 输出与工程）**已整页删除**
        #    （2026-10-14 晚，维护者：左栏有了「图像管理」之后"右侧的图像管理已经冗余了"）。
        #    删前逐条核过"这项有没有第二处显示"，三块里两块半是**重复信息**，不是缺功能：
        #      · 图像信息（当前图名/尺寸/标注块数）-> 底栏状态区已有
        #        「文件: xxx.jpg 宽度 1160x3706 … 已标注1 已检出0」
        #      · 数据集划分数量（训练/测试/验证）  -> 左栏图片列表底行已有
        #        「Train: 28 | Val: 5 | Test: 3」（_filter_images 里算的那行）
        #      · 导入图片…                        -> 左栏图片列表右键菜单已有
        #      · 工程目录                          -> 工程树右键 Open in Explorer 已有
        #    真正**只有这一处**能点的两个动作已搬到左栏「图像管理」页（见 _create_left_panel）：
        #      「管理划分…」-> 底部计数行下方；「输出目录」-> 图片列表右键菜单 + 工程树右键菜单。
        for _page in (page_params, page_annot, page_model):
            self._right_stack.addWidget(_page)
        self._right_tab_index = {"params": 0, "annot": 1, "model": 2}
        content_layout.addWidget(self._right_stack)

        # ===== 右侧竖排页签（参照友商：文字**竖向排列**，4 块均分撑满整条竖栏）=====
        tab_bar = QWidget(); tab_bar.setFixedWidth(40)
        tab_layout = QVBoxLayout(tab_bar); tab_layout.setContentsMargins(1, 2, 1, 2); tab_layout.setSpacing(2)

        self._right_tabs = QButtonGroup(self); self._right_tabs.setExclusive(True)
        # 参照友商 友商平台 右栏，但只留 **3 个模块**（原来是 6 个并列工具页，
        # 2026-10-14 又砍掉冗余的「图像管理」——理由见上面 _right_stack 组装处）
        tab_defs = [
            ("参数管理", "训练参数 / 增强参数 / 高度图参数 / 推理参数", "params"),
            ("标注管理", "类别 / 标注统计 / 结果 / 检测工具", "annot"),
            ("模型管理", "模型列表 / 版本历史", "model"),
        ]
        self._right_tab_btns = {}
        for icon_text, tooltip, key in tab_defs:
            idx = self._right_tab_index[key]
            # 竖排：一个字一行。横排 4 个字塞进 40px 宽会溢出到按钮外面（用户截图就是那样）
            btn = QPushButton("\n".join(icon_text))
            btn.setCheckable(True)
            btn.setToolTip(tooltip + f"（{icon_text}）")
            btn.setMinimumHeight(56)
            # 高度策略 Expanding + 布局 stretch=1 -> 各块**均分并撑满**整条竖栏
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            # 样式与左栏共用同一份（模块级 `_V_TAB_QSS`）：两边各写一份迟早漂移
            btn.setStyleSheet(_V_TAB_QSS)
            btn.clicked.connect(lambda checked, k=key: self.show_right_tab(k))
            self._right_tabs.addButton(btn, idx)
            self._right_tab_btns[key] = btn
            tab_layout.addWidget(btn, 1)          # ← stretch=1：均分竖栏高度
        content_layout.addWidget(tab_bar)

        outer_layout.addLayout(content_layout)

        # Default: 打开工程后默认落在「标注管理」
        self.show_right_tab("annot")

        return container

    # ────────── 右侧模块/区块（参照友商右栏）──────────

    def _make_section_box(self, title, widgets):
        """一个**区块**：左侧彩色竖条 + 标题（点标题折叠）+ 内容。

        区块各按内容占高，所以不会出现"某一块撑出大片空白"；
        不看的直接折叠，比拖分隔条省事。
        """
        box = QWidget()
        lay = QVBoxLayout(box); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        head = QPushButton("▍" + title)
        head.setCheckable(True); head.setChecked(True)
        head.setCursor(Qt.PointingHandCursor)
        head.setToolTip("点击折叠 / 展开")
        head.setStyleSheet(
            "QPushButton { text-align:left; font-size:11px; font-weight:bold; color:#00E5FF;"
            " background:#101828; border:none; border-left:3px solid #00E5FF; padding:3px 6px; }"
            "QPushButton:checked { color:#00E5FF; }"
            "QPushButton:!checked { color:#8A93A6; border-left:3px solid #2A3A5C; }"
        )
        lay.addWidget(head)
        # ★★ 2026-10-22 修「双击 exe 后屏幕中间一闪一闪」的**真凶**：
        #    原来写的是 `body = QWidget()`（无父 ⇒ 它本身就是一个**顶层窗口**），
        #    后面又跟了一句 `body.setVisible(True)` —— 对顶层窗口来说那等于 show()，
        #    于是一个小窗在当前屏幕中心弹出来；紧接着 `lay.addWidget(body)` 把它收进布局
        #    （reparent ⇒ 自动 hide），小窗又消失。每个区块闪一次，右栏 7 个区块
        #    = 启动瞬间屏幕中间连闪 7 下（探针 _probe_startup_flash.py 实测 7 组 SHOW/HIDE）。
        #    两处都治：构造时就带上父对象（它天生不该是窗口），并删掉那句多余的 setVisible
        #    —— 子控件随父显示，本来就不需要单独 show。
        body = QWidget(box); bl = QVBoxLayout(body)
        bl.setContentsMargins(7, 4, 2, 6); bl.setSpacing(3)
        for w in (widgets or []):
            if w is not None:
                bl.addWidget(w)
        lay.addWidget(body)
        head.toggled.connect(body.setVisible)
        box._head = head
        box._body = body
        return box

    def _make_module_page(self, items):
        """一个**模块页** = 可滚动的竖排区块；items 里可以是区块元组 (标题, [控件])
        或已经拼好的控件（原样放进去，例如标注管理那三张自带标题的表）。"""
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); v = QVBoxLayout(inner)
        v.setContentsMargins(2, 2, 2, 2); v.setSpacing(3)
        for it in items:
            if isinstance(it, tuple):
                title, widgets = it
                v.addWidget(self._make_section_box(title, widgets))
            elif it is not None:
                v.addWidget(it)
        v.addStretch()
        sa.setWidget(inner)
        return sa

    # ────────── 模型管理 里的几个小区块 ──────────

    def _make_model_list_section(self):
        """模型列表：扫 models/*.pth（名称 / 类别数 / 大小 / 时间），双击即选为推理模型。"""
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(2)
        bar = QHBoxLayout(); bar.setSpacing(3)
        self.model_list_table = QTableWidget()
        self.model_list_table.setColumnCount(4)
        # 「来源」列：工程内 / 版本留档 / 链接（对应友商那句 GPU/CPU/iGPU 的信息位）
        self.model_list_table.setHorizontalHeaderLabels(["模型", "来源", "大小", "时间"])
        self.model_list_table.verticalHeader().setDefaultSectionSize(20)
        self.model_list_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.model_list_table.setAlternatingRowColors(True)
        self.model_list_table.setMaximumHeight(140)
        self.model_list_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.model_list_table.itemDoubleClicked.connect(self._on_model_row_activated)
        # ★ 2026-10-23：右键菜单（打开所在文件夹 / 复制路径 / 设为推理模型 / 移除链接）
        self.model_list_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_list_table.customContextMenuRequested.connect(self._on_model_list_context_menu)
        btn = QPushButton("刷新"); btn.setFixedWidth(46)
        btn.clicked.connect(self._refresh_model_list)
        # ★ 2026-10-23：对标友商「模型管理」——支持把**工程外**的 .pth 链接进来
        #   （不复制文件，只登记绝对路径；友商标成 GPU/CPU/iGPU，我们标来源）
        b_link = QPushButton("链接本地模型…")
        b_link.setToolTip("把工程目录之外的 .pth 登记到本列表（只记录路径，不复制文件）")
        b_link.clicked.connect(self._link_local_model)
        b_batch = QPushButton("批量推理（全部图片）…")
        b_batch.setToolTip("对工程内所有图片跑一遍推理，结果写到 outputs/")
        b_batch.clicked.connect(self._batch_inference)
        bar.addWidget(b_batch); bar.addWidget(b_link)
        bar.addWidget(QLabel("双击设为推理模型"))
        bar.addStretch(); bar.addWidget(btn)
        v.addWidget(self.model_list_table); v.addLayout(bar)
        return w

    # ── 链接模型登记（工程外 .pth，只存路径，不复制文件）──
    def _linked_models_file(self):
        pd = self._project_dir
        return os.path.join(pd, "models", ".linked_models.json") if pd else None

    def _load_linked_models(self):
        """{显示名: 绝对路径}。读失败就当空。"""
        p = self._linked_models_file()
        if not p or not os.path.exists(p):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {str(k): str(v) for k, v in (d or {}).items()}
        except Exception:
            return {}

    def _save_linked_models(self, d):
        p = self._linked_models_file()
        if not p:
            return
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"[模型] 保存链接失败: {type(e).__name__}: {e}")

    def _model_display_to_path(self):
        """显示名 -> 真实路径（工程内模型 + 链接模型）。

        推理/combo 只认"名字"，所以统一在这里翻译一次；
        链接模型的名字是别名，真实路径是工程外的绝对路径。
        """
        m = getattr(self, "_model_name_map", None)
        if m is None:
            m = {}
            self._model_name_map = m
        return m

    def _link_local_model(self):
        """把工程目录之外的 .pth 链接进模型列表（只登记路径）。"""
        if not self._project_dir:
            return QMessageBox.information(self, "链接模型", "先打开一个工程。")
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要链接的模型（可多选）", "",
            "模型文件 (*.pth *.pt);;所有文件 (*)")
        if not paths:
            return
        d = self._load_linked_models()
        added = 0
        for p in paths:
            p = os.path.abspath(p)
            name = "🔗 " + os.path.basename(p)
            # 同名不同路径：加序号，别互相覆盖
            base_name, k = name, 2
            while name in d and os.path.abspath(d[name]) != p:
                name = f"{base_name} ({k})"
                k += 1
            d[name] = p
            added += 1
        self._save_linked_models(d)
        self._model_list_sig = None          # 强制刷新
        self._refresh_model_list(force=True)
        self.log(f"[模型] 已链接 {added} 个本地模型")
        # ⚠️ 跨工程链模型最容易犯的错：别人的模型类别跟本工程不同（比如 2 类 vs 4 类），
        #   推理能跑但语义完全是错的。这里对一下 checkpoint 里的 classes，不一致就明说。
        try:
            proj_cls = list(getattr(self, "_cached_classes", []) or [])
            for p_ in paths:
                try:
                    ck = torch.load(os.path.abspath(p_), map_location="cpu", weights_only=False)
                    ck_cls = [str(c) for c in (ck.get("classes") or [])]
                    if proj_cls and ck_cls and ck_cls != proj_cls:
                        QMessageBox.warning(
                            self, "类别不一致",
                            "模型 %s 的类别与本工程不同：\n\n工程：%s\n模型：%s\n\n"
                            "仍会链接，但推理结果的类别语义会对不上，别拿它出正式结果。"
                            % (os.path.basename(p_), proj_cls, ck_cls))
                except Exception:
                    continue
        except Exception:
            pass

    def _unlink_selected_models(self):
        """把选中的链接模型从列表里移除（只删登记，不删磁盘文件）。"""
        tbl = getattr(self, "model_list_table", None)
        if tbl is None:
            return
        rows = sorted({i.row() for i in tbl.selectedIndexes()}, reverse=True)
        if not rows:
            return QMessageBox.information(self, "移除链接", "先在列表里选中链接模型。")
        d = self._load_linked_models()
        removed = 0
        for r in rows:
            name = tbl.item(r, 0).text() if tbl.item(r, 0) else ""
            if name in d:
                d.pop(name, None); removed += 1
        self._save_linked_models(d)
        self._model_list_sig = None
        self._refresh_model_list(force=True)
        self.log(f"[模型] 已移除 {removed} 个链接（磁盘文件未删除）")

    def _refresh_model_list(self, force=True):
        """重新扫描 <工程>/models（**含子目录的版本留档**）+ 链接模型。

        ★★ 2026-10-23：原来只 glob `models/*.pth` 这一层，`models/V1_unet/`、
           `models/V2_unet/` 这些**版本留档里的模型根本不在列表里** ——
           用户看到的现象就是"只显示最新的两个"（best/last）。
           现在递归扫 `models/**/*.pth`，版本模型显示成 `V1_unet/best_model.pth`；
           另外把「链接本地模型」登记进来的工程外模型也一起列出来（来源列标"链接"）。

        ⚡ 带指纹缓存：目录内容（名字+来源+大小+时间）没变就直接返回。
        """
        tbl = getattr(self, "model_list_table", None)
        if tbl is None:
            return
        import glob
        rows = []            # (显示名, 来源, 大小, 时间, 真实路径)
        pd = self._project_dir
        if pd:
            mdir = os.path.join(pd, "models")
            seen = set()
            for pat in ("*.pth", "*.pt", os.path.join("*", "*.pth"), os.path.join("*", "*.pt"),
                        os.path.join("multiview", "*.pth"),
                        os.path.join("detection", "*.pt"),
                        os.path.join("classification", "*.pth")):
                for p in glob.glob(os.path.join(mdir, pat)):
                    ap = os.path.abspath(p)
                    if ap in seen:
                        continue
                    seen.add(ap)
                    try:
                        st = os.stat(ap)
                        rel = os.path.relpath(ap, mdir).replace("\\", "/")
                        src = "版本留档" if "/" in rel else "工程"
                        rows.append((rel, src, "%.1f MB" % (st.st_size / 1024 / 1024),
                                     time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime)), ap))
                    except Exception:
                        continue
            # 链接模型
            name_map = self._model_display_to_path()
            for name, path in self._load_linked_models().items():
                try:
                    st = os.stat(path)
                    rows.append((name, "链接", "%.1f MB" % (st.st_size / 1024 / 1024),
                                 time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime)), path))
                    name_map[name] = path
                except Exception:
                    rows.append((name, "链接(失效)", "—", "—", path))
                    name_map[name] = path
        # 先按时间新→旧；再用**稳定排序**把失效链接沉底（Python sort 是稳定的）
        rows.sort(key=lambda r: r[3], reverse=True)
        rows.sort(key=lambda r: r[1].startswith("链接(失效)"))
        sig = tuple((r[0], r[1], r[2], r[3]) for r in rows)
        if not force and sig == getattr(self, "_model_list_sig", None):
            return
        self._model_list_sig = sig
        tbl.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j in range(4):
                it = QTableWidgetItem(r[j])
                it.setTextAlignment(Qt.AlignCenter)
                if j == 0:
                    it.setData(Qt.UserRole, r[4])      # 真实路径，双击时直接用
                    if r[1].startswith("链接"):
                        it.setToolTip(r[4])
                tbl.setItem(i, j, it)
        # ★ 同步「推理模型」下拉框：原来它初始化时写死 best/last 两项，
        #   版本留档和链接模型根本选不到。这里按列表内容重建，选中项保持不变。
        try:
            names = [r[0] for r in rows]
            cur = self.model_combo.currentText()
            items = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
            if names and names != items:
                self.model_combo.blockSignals(True)
                self.model_combo.clear()
                self.model_combo.addItems(names)
                idx = self.model_combo.findText(cur)
                if idx < 0:
                    idx = next((k for k, n in enumerate(names) if "best" in n.lower()), 0)
                self.model_combo.setCurrentIndex(max(0, idx))
                self.model_combo.blockSignals(False)
        except Exception:
            pass

    def _model_row_path(self, row):
        """取模型列表某行的**真实路径**（链接模型在工程外，靠 UserRole 带过来的绝对路径）。"""
        try:
            it = self.model_list_table.item(row, 0)
            return (it.data(Qt.UserRole) or "") if it is not None else ""
        except Exception:
            return ""

    def _reveal_in_explorer(self, path):
        """在资源管理器里打开该文件所在目录并选中它（跨平台给个兜底）。"""
        if not path or not os.path.exists(path):
            return QMessageBox.information(
                self, "打开文件夹",
                "文件不存在（可能被移动或删除）：\n%s" % (path or "—"))
        try:
            import subprocess
            path = os.path.normpath(path)
            if os.name == "nt":
                subprocess.Popen(["explorer", "/select,%s" % path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(path)])
        except Exception as e:
            QMessageBox.warning(self, "打开文件夹失败", "%s: %s" % (type(e).__name__, e))

    def _on_model_list_context_menu(self, pos):
        """模型列表右键菜单。"""
        from PyQt5.QtWidgets import QMenu
        tbl = self.model_list_table
        row = tbl.rowAt(pos.y())
        if row < 0:
            return
        tbl.selectRow(row)
        name = tbl.item(row, 0).text() if tbl.item(row, 0) is not None else ""
        path = self._model_row_path(row)
        is_link = name.startswith("🔗")

        m = QMenu(self)
        a_open = m.addAction("打开所在文件夹")
        a_copy = m.addAction("复制模型完整路径")
        m.addSeparator()
        a_use = m.addAction("设为推理模型")
        a_unlink = m.addAction("移除链接（不删磁盘文件）") if is_link else None
        act = m.exec_(tbl.viewport().mapToGlobal(pos))
        if act is None:
            return
        if act is a_open:
            self._reveal_in_explorer(path)
        elif act is a_copy:
            try:
                from PyQt5.QtWidgets import QApplication as _QA
                _QA.clipboard().setText(path)
                self.log(f"[模型] 已复制路径：{path}")
            except Exception:
                pass
        elif act is a_use:
            self._on_model_row_activated(tbl.item(row, 0))
        elif a_unlink is not None and act is a_unlink:
            self._unlink_selected_models()

    def _resolve_model_for_use(self, name):
        """把下拉/列表里的显示名翻成**真实路径**。

        链接模型的名字是别名（前缀 🔗），真实文件在工程外；工程内与版本留档
        直接用相对路径 join 就能找到，原样返回即可。
        """
        if not name:
            return name
        return self._model_display_to_path().get(name, name)

    def _on_model_row_activated(self, item):
        """双击模型列表某行 -> 把它设为推理用的模型文件。"""
        try:
            c0 = self.model_list_table.item(item.row(), 0)
            name = c0.text()
            real = c0.data(Qt.UserRole) or name
        except Exception:
            return
        self._model_display_to_path()[name] = real
        idx = self.model_combo.findText(name)
        if idx < 0:
            self.model_combo.addItem(name); idx = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(idx)
        self.log(f"[模型] 推理将使用 {name}")
        self.show_right_tab("params")

    # ⚠️ 这里原来还有三个小区块工厂：`_make_image_info_section`（图像信息）/
    #    `_make_split_section`（数据集划分）/ `_make_project_ops_section`（输出与工程）。
    #    它们只服务于右栏那页「图像管理」，2026-10-14 该页整页删除后**已无任何引用**，
    #    一并删掉 —— 留着就是"看着像能用、其实挂不上去"的死代码（`_make_annotation_ops_section`
    #    是当时明确说"保留备用"的，另算）。

    def _make_annotation_ops_section(self):
        """标注操作：导出/清理 这些原先**只有英文菜单能点**的动作，补进「标注管理」。

        补入口 → 验证能点通 → 才敢把菜单里那三条冗余菜单删掉。
        """
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(2)
        r1 = QHBoxLayout(); r1.setSpacing(3)
        b1 = QPushButton("导出 JSON…"); b1.clicked.connect(self._export_json)
        b2 = QPushButton("导出标注"); b2.clicked.connect(self._export_annotations)
        r1.addWidget(b1); r1.addWidget(b2)
        r2 = QHBoxLayout(); r2.setSpacing(3)
        b3 = QPushButton("清理碎标注(<10k px)…"); b3.setToolTip("把面积小于 1 万像素的标注块整块删掉")
        b3.clicked.connect(self._clean_tiny_ann)
        b4 = QPushButton("批量导入 JSON…"); b4.clicked.connect(self._batch_import_json)
        r2.addWidget(b3); r2.addWidget(b4)
        v.addLayout(r1); v.addLayout(r2)
        return w

    def _open_dir(self, path):
        if path and os.path.isdir(path):
            try:
                os.startfile(path)
            except Exception as e:
                self.log(f"打开目录失败: {e}")
        else:
            self.log("目录不存在（先打开工程）")

    def _open_output_dir(self, project_dir=None):
        """打开工程的 `outputs/`（推理结果 + 各次训练输出）。

        原先是右栏「图像管理」页里的一个按钮，该页 2026-10-14 删除后
        改挂到**左栏图片列表底部**与**图片列表/工程树右键菜单**（同一函数，三处入口）。
        传 `project_dir` 时用传入的那个（工程树右键 = 右键那个工程，不一定是当前打开的）。
        """
        pd = project_dir or self._project_dir
        if not pd:
            self.log("还没打开工程")
            return
        self._open_dir(os.path.join(pd, "outputs"))

    def _refresh_image_info(self):
        """（原右栏「图像管理」页的刷新钩子，2026-10-14 该页已整页删除）

        ⚠️ 现在**只剩一个职责**：顺带刷新「模型管理」的模型列表。
        函数名与 3 处调用点（切图 / 保存 / 打开工程后）**故意不动** —— 改名要连带改调用点，
        改动面越大越容易漏一处，漏了就是"模型列表不刷新"这种静默失效。
        """
        self._refresh_model_list(force=False)   # 指纹没变就直接返回（切图/保存后也会走到这里）

    def _make_ti_section(self, title, stack):
        """拼一段「训练」或「推理」：标题固定，只有内容滚动。"""
        box = QWidget(); lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 2, 2, 2); lay.setSpacing(2)
        lay.addWidget(title)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(stack)
        sa.setMinimumHeight(90)
        lay.addWidget(sa, 1)
        return box

    def _ti_section_need(self, i):
        """第 i 段内容实际需要多高（标题 + **当前子页**）。

        ⚠️ 不能用 `QStackedWidget.sizeHint()`：它是"所有子页里最大的那个"
        （实测训练栈 991px，而当前可见页只要 150px 左右），拿它算会得出完全离谱的高度。
        """
        sp = getattr(self, "_ti_split", None)
        if sp is None or i >= sp.count():
            return 90
        box = sp.widget(i)
        lay = box.layout()
        h = 0
        for j in range(lay.count()):
            w = lay.itemAt(j).widget()
            if w is None:
                continue
            if isinstance(w, QScrollArea):
                inner = w.widget()
                cur = inner.currentWidget() if isinstance(inner, QStackedWidget) else None
                h += (cur.sizeHint().height() if cur is not None else inner.sizeHint().height())
            else:
                h += w.sizeHint().height()
        m = lay.contentsMargins()
        h += m.top() + m.bottom() + max(0, lay.count() - 1) * lay.spacing()
        return max(90, int(h) + 8)

    def _fit_ti_split(self):
        """按两段内容的实际高度分配高度 —— 别让"内容很少的那段"占着一大片空白。

        规则：
          · 两段内容都装得下 -> 第一段（训练）只拿它需要的高度，**剩下的全给第二段（推理）**；
            多余空白落在整个面板最下面，看起来是"表单尾巴"，而不是夹在中间的大洞。
          · 装不下 -> 按内容比例分配，两段各自滚动。
        面板刚显示时高度还是 0，这里会自己重试几次（不是轮询，最多 5 次就停）。
        """
        sp = getattr(self, "_ti_split", None)
        if sp is None:
            return
        try:
            # ⚠️ `sp.count()` / `sp.height()` 必须**在 try 里面**：
            #    QSplitter 的 C++ 对象被销毁后，Python 这层 wrapper 还在
            #    （`sp is None` 为 False），一碰就抛
            #    `RuntimeError: wrapped C/C++ object of type QSplitter has been deleted`。
            #    这个函数是 QTimer.singleShot 的回调 —— PyQt5 里槽函数抛异常会直接
            #    终止整个进程，训练跑到一半也会被一起带走。
            #    crash_log.txt 2026-10-13 两次实测都崩在原先的 `if sp is None or sp.count() != 2`
            #    那一行（当时它在 try 外面），行号变了但结构没变，所以现在仍然会崩。
            if sp.count() != 2:
                return
            avail = sp.height()
            if avail < 80:
                tries = getattr(self, "_ti_fit_tries", 0) + 1
                self._ti_fit_tries = tries
                if tries <= 5:
                    QTimer.singleShot(80, self._fit_ti_split)
                return
            self._ti_fit_tries = 0
            need = [self._ti_section_need(0), self._ti_section_need(1)]
            if sum(need) <= avail:
                sizes = [need[0], avail - need[0]]
            else:
                k = avail / float(sum(need))
                sizes = [max(90, int(need[0] * k)), max(90, int(need[1] * k))]
            if list(sizes) != list(sp.sizes()):
                sp.setSizes(sizes)
        except Exception:
            pass

    def _on_ti_stack_page_changed(self, _idx=0):
        """训练/推理栈换子页后重新量高。

        ShrinkStack 只按 currentWidget() 报告尺寸，但 QScrollArea 的
        widgetResizable 只在它自己 resize 时重排内部 widget —— 光调
        updateGeometry() 不够：从"高页"切到"矮页"时会缩不回去，
        页内 stretch 又把空白吃掉，等于老问题复发（实测 991px 卡住不退）。
        """
        for st in (getattr(self, "_train_stack", None), getattr(self, "_infer_stack", None)):
            if st is not None:
                st.updateGeometry()
        QTimer.singleShot(0, self._refit_ti_viewports)

    @staticmethod
    def _stack_scroll(st):
        """找到栈所在的 QScrollArea。

        ⚠️ 不能直接取 `st.parentWidget()`：QScrollArea.setWidget() 把 widget 挂在
        **viewport** 下面，parentWidget() 拿到的是 viewport，不是 QScrollArea 本身
        （第一版就是这么写错的，结果 resize 分支永远不执行）。
        """
        w = st.parentWidget()
        while w is not None and not isinstance(w, QScrollArea):
            w = w.parentWidget()
        return w

    def _refit_ti_viewports(self):
        """按当前子页把栈重新撑到正确高度，并把两段的分割比例重新算一遍。"""
        self._fit_ti_split()
        # QSplitter 的几何要下一轮事件循环才落到控件上，所以量 viewport 得再延后一拍
        # （早一拍量到的是旧尺寸，会把这轮 resize 直接废掉 —— 实测"矮页"缩不回去）
        QTimer.singleShot(0, self._apply_ti_viewport_sizes)

    def _apply_ti_viewport_sizes(self):
        """按**当前子页**的实际需要，显式给两个栈定高、定宽。

        为什么要 setFixedHeight 这么霸道：QStackedLayout 的 minimumSize() 取的是
        "所有子页里最大的那个"（实测 649x991），Qt 会拿它当硬约束 —— 从高页切到矮页时
        直接把控件钉在 991 上缩不回去（实测切到占位页后卡在 1031px），页内空白照旧。
        我们自己按当前页的 sizeHint 定高，就绕开了这个缓存；超高时滚动条照常出现。
        """
        for st in (getattr(self, "_train_stack", None), getattr(self, "_infer_stack", None)):
            if st is None:
                continue
            sa = self._stack_scroll(st)
            if sa is None:
                continue
            vp = sa.viewport().size()
            if vp.width() <= 0:
                continue
            need = max(1, st.sizeHint().height())
            if st.minimumHeight() != need or st.maximumHeight() != need:
                st.setFixedHeight(need)
            if st.width() != vp.width():
                st.resize(vp.width(), need)
            st.updateGeometry()

    def show_right_tab(self, key):
        """按**名字**切右侧页签（"annot"/"ver"/"train"/"det"/"hm"/"aug"）。

        不用写死下标：页签合并过一次（训练+推理），下标一改就得满文件找 setCurrentIndex，
        漏一处就是"点了没反应"或"跳到别的页"，且不报错。
        """
        idx = getattr(self, "_right_tab_index", {}).get(key)
        if idx is None:
            return
        self._right_stack.setCurrentIndex(idx)
        btn = self._right_tabs.button(idx)
        if btn is not None:
            btn.setChecked(True)
        if key == "train":
            # 页签刚可见时才量得到真实高度，所以延后一拍再按内容分配高度
            QTimer.singleShot(0, self._refit_ti_viewports)

    def _make_block_header(self, text):
        """块表的小标题（只作标题、不可点 —— 免得看着像按钮却点不动）。

        ⚠️ 这个方法是**在 _create_right_panel 之后**定义的，不要为了"就近"把它挪进
        函数中间：那样会把函数后半段（Version/Inference/Training 等面板和 `return p`）
        变成它的不可达函数体，`_create_right_panel` 直接返回 None，
        外面 addWidget(None) 表现为 **0xC0000005 访问违例崩溃**，
        而 py_compile 完全查不出来（已踩过，靠 faulthandler 定位）。
        """
        lb = QLabel(text)
        lb.setStyleSheet("color:#00E5FF; font-size:10px; font-weight:bold; padding:0 2px;")
        return lb

    # ============ 全局数据增强（增强参数面板，所有训练任务共用） ============
    def _init_augment_panel(self):
        """启动时把全局增强配置加载到右侧「增强」面板。"""
        if not hasattr(self, "aug_enabled_check"):
            return
        from training.augment_config import load_augment_flags, FLAG_KEYS
        flags = load_augment_flags()
        self.aug_enabled_check.setChecked(bool(flags.get("enabled", True)))
        for key in FLAG_KEYS:
            cb = self._aug_checks.get(key)
            if cb is not None:
                cb.setChecked(bool(flags.get(key, False)))
        self._update_aug_status()

    def _collect_augment_flags(self):
        """读取面板控件 -> 增强开关 dict。"""
        from training.augment_config import FLAG_KEYS
        flags = {"enabled": bool(self.aug_enabled_check.isChecked())}
        for key in FLAG_KEYS:
            cb = self._aug_checks.get(key)
            flags[key] = bool(cb.isChecked()) if cb is not None else False
        return flags

    def _save_augment_ui(self):
        """面板任一开关变化时持久化到 .train_settings.json。"""
        if not hasattr(self, "aug_enabled_check"):
            return
        from training.augment_config import save_augment_flags
        save_augment_flags(self._collect_augment_flags())
        self._update_aug_status()

    def _update_aug_status(self):
        """刷新增强面板状态摘要。"""
        if not hasattr(self, "aug_enabled_check"):
            return
        from training.augment_config import flags_to_level, AUGMENT_FLAG_LABELS
        flags = self._collect_augment_flags()
        if not flags["enabled"]:
            self.aug_status.setText("当前：关闭（训练不使用数据增强）")
            return
        on = [label for key, label in AUGMENT_FLAG_LABELS if flags.get(key)]
        text = f"级别：{flags_to_level(flags)}"
        if on:
            text += "　已启用：" + "、".join(on)
        else:
            text += "　已启用：无"
        self.aug_status.setText(text)

    def _set_aug_all(self, checked):
        for cb in self._aug_checks.values():
            cb.setChecked(checked)
        self._save_augment_ui()

    def _set_aug_default(self):
        from training.augment_config import DEFAULT_AUGMENT_FLAGS
        flags = DEFAULT_AUGMENT_FLAGS
        self.aug_enabled_check.setChecked(bool(flags["enabled"]))
        for key, cb in self._aug_checks.items():
            cb.setChecked(bool(flags.get(key, False)))
        self._save_augment_ui()

    def _set_augment_level(self, level):
        """智能推荐等外部入口按预设级别（light/medium/strong/heavy/none）设置开关。"""
        from training.augment_config import preset_to_flags
        flags = preset_to_flags(level)
        self.aug_enabled_check.setChecked(bool(flags["enabled"]))
        for key, cb in self._aug_checks.items():
            cb.setChecked(bool(flags.get(key, False)))
        self._save_augment_ui()

    def _get_augment_flags(self):
        """返回当前增强开关 dict；未启用时返回 None（等效关闭）。"""
        if not hasattr(self, "aug_enabled_check"):
            return None
        flags = self._collect_augment_flags()
        if not flags.get("enabled"):
            return None
        return flags

    def _augment_level_text(self, flags):
        """把开关 dict 转成可读文本（日志用）。"""
        if not flags:
            return "off"
        from training.augment_config import flags_to_level
        return flags_to_level(flags)

    # ============ Height Map Controls ============
    def _update_height_controls(self):
        canvas = self.canvas
        if canvas._height_image is None:
            self.hm_status.setText("No height map loaded")
            self.hm_data_range.setText("")
            self._update_height_colorbar()
            return
        # Auto-switch to HM tab on first height map detection
        if not getattr(self, "_hm_tab_auto_shown", False):
            self._hm_tab_auto_shown = True
            self.show_right_tab("params")       # 高度图参数在「参数管理」模块里
        vmin, vmax = canvas._get_height_auto_range()
        self.hm_status.setText(f"Height map: {canvas._height_image.size[0]}x{canvas._height_image.size[1]}")
        self.hm_data_range.setText(f"Data range: [{vmin:.4f}, {vmax:.4f}] mm")
        if canvas._height_vmin is not None:
            self.hm_vmin.setValue(canvas._height_vmin)
        else:
            self.hm_vmin.setValue(vmin)
        if canvas._height_vmax is not None:
            self.hm_vmax.setValue(canvas._height_vmax)
        else:
            self.hm_vmax.setValue(vmax)
        self._update_height_colorbar()

    def _update_height_colorbar(self):
        w = max(self.hm_colorbar.width(), 200)
        pix = QPixmap(w, 20)
        pix.fill(QColor(40, 40, 40))
        painter = QPainter(pix)
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(0, 0, 180))
        grad.setColorAt(0.25, QColor(0, 200, 200))
        grad.setColorAt(0.5, QColor(0, 220, 0))
        grad.setColorAt(0.75, QColor(240, 240, 0))
        grad.setColorAt(1.0, QColor(220, 50, 0))
        painter.fillRect(0, 0, w, 20, grad)
        painter.end()
        self.hm_colorbar.setPixmap(pix)

    def _hm_auto_range(self):
        self.canvas.set_height_range(None, None, self._hm_get_colormap())
        self._update_height_controls()

    def _hm_apply(self):
        vmin = self.hm_vmin.value()
        vmax = self.hm_vmax.value()
        cmap = self._hm_get_colormap()
        self.canvas.set_height_range(vmin, vmax, cmap)
        self._update_height_controls()

    def _hm_get_colormap(self):
        import cv2
        cmap_map = {
            "JET": cv2.COLORMAP_JET, "TURBO": cv2.COLORMAP_TURBO,
            "HOT": cv2.COLORMAP_HOT, "RAINBOW": cv2.COLORMAP_RAINBOW,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS, "PLASMA": cv2.COLORMAP_PLASMA,
            "INFERNO": cv2.COLORMAP_INFERNO, "MAGMA": cv2.COLORMAP_MAGMA,
            "HSV": cv2.COLORMAP_HSV,
        }
        return cmap_map.get(self.hm_cmap_combo.currentText(), cv2.COLORMAP_JET)

    # ============ Logging & Progress ============
    def log(self, msg):
        self.log_signal.emit(msg)

    def _log_file_path(self):
        """UI 日志的落盘路径（按天一个文件）。

        位置：源码目录下 `logs/app_YYYY-MM-DD.log`（和 启动日志.txt 同级）。
        ⚠️ 与 `启动日志.txt` 不是一回事：那个只接 stdout/stderr（print 与崩溃栈），
           UI 日志栏的内容从来没有任何文件承接 —— 关掉软件就丢了。
        """
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            d = os.path.join(root, "logs")
            os.makedirs(d, exist_ok=True)
            return os.path.join(d, "app_%s.log" % datetime.now().strftime("%Y-%m-%d"))
        except Exception:
            return None

    def _append_log(self, msg):
        if msg == "__DET_TRAINING_DONE__":
            self._on_det_training_done()
            return
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_widget.append(line)
        # ★ 2026-10-23：**日志落盘**。原来只写控件，训练跑一晚、第二天想回看根本找不到。
        #   每行 flush：训练中断（点停止/关窗/崩溃）也不会丢最后几十行。
        try:
            fh = getattr(self, "_log_fh", None)
            if fh is None:
                p = self._log_file_path()
                fh = open(p, "a", encoding="utf-8") if p else False
                self._log_fh = fh
                if fh:
                    fh.write("\n===== 会话开始 %s =====\n"
                             % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if fh:
                fh.write(line + "\n")
                fh.flush()
        except Exception:
            pass

    def _ensure_loss_window(self):
        """获取/创建独立 Loss/Acc 曲线窗口（分割/分类共用，懒创建）。"""
        if not hasattr(self, "_cls_loss_window") or self._cls_loss_window is None:
            self._cls_loss_window = LossCurveWindow(self)
        return self._cls_loss_window

    def _update_loss_chart(self, history):
        """训练曲线统一在独立 Loss/Acc 弹窗中显示（分割/分类共用）。"""
        try:
            if not history or not history.get("train_loss"):
                return
            self._ensure_loss_window().update_curve(history)
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
        # 同一个数字也推给任务队列行（队列页与进度条必须一致，不能各算一套）
        ctx = getattr(self, "_active_infer_ctx", None)
        if ctx is not None:
            try:
                ctx.progress(cur, total)
            except Exception:
                pass
        self.infer_status.setText(f"Inferring... {cur}/{total}")

    # ============ Project Management ============
    def _set_theme(self, name):
        """切换主题（深色/浅色）并记住选择。"""
        from ui.themes import apply_theme
        from PyQt5.QtWidgets import QApplication
        apply_theme(QApplication.instance(), name)
        # 「指针 / 手掌」的选中态是**画进图标的**底色块（不靠 QSS，见 icons.make_nav_icon），
        # 深浅主题底色不同 ⇒ 切主题后必须重画一次，否则浅色主题下还是深色块。
        try:
            self.annotation_toolbar.refresh_nav_icons()
        except Exception:
            pass
        self._save_ui_setting("theme", name)
        self.log(f"[主题] 已切换到 {name}（下次启动保持）")

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
        # 空分组不显示："Within 1 week" 没子项时只会占一行标题，纯噪声
        # （而且会让人以为"还有内容没展开"）。
        for _grp in (self.week_item, self.month_item, self.older_item):
            _grp.setHidden(_grp.childCount() == 0)
        self.project_tree.expandAll()
        self.project_tree.resizeColumnToContents(0)

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
        # 「输出目录」：右栏「图像管理」页删除（2026-10-14）后归到这里 ——
        # 与"打开工程目录"并列，而且**不需要先在图片列表里选中一行**（图片列表右键
        # 在没选中任何行时是不出菜单的，所以工程树这条才是随时可用的那个入口）。
        act_out = menu.addAction("打开输出目录")
        act_out.triggered.connect(
            lambda _c=False, n=name: self._open_output_dir(str(self.pm.get_project_dir(n))))
        menu.addSeparator()
        del_action = menu.addAction("Delete Project")
        del_action.triggered.connect(lambda: self._delete_project(name, item))
        menu.exec_(self.project_tree.viewport().mapToGlobal(pos))

    def _delete_project(self, name, item):
        """Delete a project after confirmation."""
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete project '{name}' and all its data?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            # Close if currently open
            if self.current_project and self.current_project.get("name") == name:
                self._close_project()
            self.pm.delete_project(name)
            self._refresh_projects()
            self.log(f"Deleted project: {name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _on_project_double_click(self, item):
        name = item.data(0, Qt.UserRole)
        if name:
            self._open_project_by_name(name)

    def _new_project(self):
        from pathlib import Path
        from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QLabel, QComboBox, QHBoxLayout, QPushButton, QFileDialog
        dlg = QDialog(self)
        dlg.setWindowTitle("New Project")
        form = QFormLayout(dlg)
        name_edit = QLineEdit()
        form.addRow("Project Name:", name_edit)
        task_combo = QComboBox()
        task_combo.addItems(["语义分割", "目标检测", "图像分类", "混合分类", "OCR文字识别", "OCV字符质检"])
        form.addRow("Task Type:", task_combo)
        # ── 输入类型（视图数）──
        # 只有语义分割需要：单张 = 3 通道，三视图 = 9 通道（漫反射/形状图1/形状图2）。
        # 声明写进 project.json 后，训练端自动识别，不必靠扫文件名去猜。
        _VIEW_PRESETS = [
            (["te_0"], {"te_0": "图像"},
             "任意文件名。训练按单张 3 通道处理。"),
            (["te_0", "te_1"], {"te_0": "视图1", "te_1": "视图2"},
             "命名约定：xxx_te_0.jpg / xxx_te_1.jpg（掩膜只需标 te_0 一张，多视图共用）"),
            (["te_0", "te_1", "te_2"],
             {"te_0": "漫反射", "te_1": "形状图1", "te_2": "形状图2"},
             "命名约定：xxx_te_0.jpg(漫反射) / xxx_te_1.jpg(形状图1) / xxx_te_2.jpg(形状图2)；"
             "掩膜只需标 te_0 一张，三视图共用。"),
        ]
        view_combo = QComboBox()
        view_combo.addItems(["单张图像（1 视图 / 3 通道）",
                             "双视图（2 视图 / 6 通道）",
                             "三视图（3 视图 / 9 通道）"])
        view_combo.setCurrentIndex(2)   # 默认三视图：本项目主场景
        form.addRow("Input Type:", view_combo)
        view_hint = QLabel("")
        view_hint.setWordWrap(True)
        view_hint.setStyleSheet("color:#00E5FF;")
        form.addRow("", view_hint)

        def _sync_view_rows():
            is_seg = (task_combo.currentText() == "语义分割")
            for w in (view_combo, view_hint, form.labelForField(view_combo)):
                if w is not None:
                    w.setVisible(is_seg)
            view_hint.setText(_VIEW_PRESETS[view_combo.currentIndex()][2] if is_seg else "")
        task_combo.currentIndexChanged.connect(_sync_view_rows)
        view_combo.currentIndexChanged.connect(_sync_view_rows)
        _sync_view_rows()
        # Directory chooser
        dir_row = QHBoxLayout()
        dir_edit = QLineEdit()
        dir_edit.setPlaceholderText("Default: workspace")
        dir_edit.setReadOnly(True)
        dir_btn = QPushButton("Browse...")
        def choose_dir():
            d = QFileDialog.getExistingDirectory(dlg, "Select Project Directory",
                                                  str(Path.home() / "Documents"))
            if d:
                dir_edit.setText(d)
        dir_btn.clicked.connect(choose_dir)
        dir_row.addWidget(dir_edit)
        dir_row.addWidget(dir_btn)
        form.addRow("Location:", dir_row)
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
            task_type = task_combo.currentText()
            parent_dir = dir_edit.text().strip() or None
            # 仅语义分割且选了多视图时才声明；单张保持与既有工程完全一致
            mv_views, mv_names = None, None
            if task_type == "语义分割":
                vs, vn, _ = _VIEW_PRESETS[view_combo.currentIndex()]
                if len(vs) > 1:
                    mv_views, mv_names = vs, vn
            self.pm.create_project(name, task_type, parent_dir=parent_dir,
                                   views=mv_views, view_names=mv_names)
            if mv_views:
                self.log(f"工程 {name} 已声明为 {len(mv_views)} 视图（{3 * len(mv_views)} 通道）："
                         + " / ".join(f"{v}({mv_names[v]})" for v in mv_views))
                self.log("图片命名需符合：" + " ".join(f"xxx_{v}.jpg" for v in mv_views))
            # For mixed classification, show paired image loader first
            if task_type == "混合分类":
                dlg2 = MixedAddImagesDialog(self)
                if dlg2.exec_() != QDialog.Accepted:
                    # User cancelled - project already created, log it
                    self.log(f"Project created without images: {name}")
                else:
                    gray_paths, height_paths = dlg2.get_paired_paths()
                    img_dir = os.path.join(str(self.pm.get_project_dir(name)), "images")
                    for gp, hp in zip(gray_paths, height_paths):
                        shutil.copy2(gp, os.path.join(img_dir, os.path.basename(gp)))
                        shutil.copy2(hp, os.path.join(img_dir, os.path.basename(hp)))
                        # Create blank JSON for grayscale (not for tif)
                        json_path = os.path.join(img_dir, os.path.splitext(os.path.basename(gp))[0] + ".json")
                        if not os.path.exists(json_path):
                            from PyQt5.QtCore import QSize
                            try:
                                sz = Image.open(gp).size
                            except:
                                sz = (0, 0)
                            blank = {"version":"5.3.1","flags":{},"shapes":[],"imagePath":os.path.basename(gp),"imageData":None,"imageHeight":sz[1],"imageWidth":sz[0]}
                            import json as _json
                            with open(json_path, "w", encoding="utf-8") as jf:
                                _json.dump(blank, jf, indent=2, ensure_ascii=False)
                    self.log(f"Imported {len(gray_paths)} paired image sets")
            self._open_project_by_name(name)
            self._refresh_projects()
            self.log(f"Created project: {name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    def _open_project(self):
        from pathlib import Path
        # 弹窗选择工程目录（含 project.json 的文件夹），支持打开工作区之外的工程
        d = QFileDialog.getExistingDirectory(self, "Select Project Directory",
                                              str(Path.home()))
        if not d:
            return
        try:
            name = self.pm.register_external_project(d)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open project: {e}")
            return
        self._refresh_projects()
        self._open_project_by_name(name)

    def _open_project_by_name(self, name):
        try:
            # 作废上一个工程的扫描/预热线程：否则它们的 gen 仍等于当前值，
            # 会继续往缓存里写、并误报"预热完成"。
            self._project_gen += 1
            self._mask_stats_ready_gen = -1
            self.current_project = self.pm.open_project(name)
            self._project_dir = str(self.pm.get_project_dir(name))
            self.canvas.set_project(self.pm.get_project_dir(name))

            # Auto-detect task type BEFORE async image scan
            # (prevents race condition where scan completes before _mixed_cls_mode is set)
            self._load_classes()
            task_map = {"语义分割":"segmentation","目标检测":"detection","图像分类":"classification","混合分类":"mixed_classification","OCR文字识别":"ocr","OCV字符质检":"ocv",
                        # 兼容旧版英文 task_type
                        "semantic_segmentation":"segmentation","object_detection":"detection",
                        "image_classification":"classification","mixed_classification":"mixed_classification",
                        "ocr_recognition":"ocr","ocv":"ocv"}
            config_path = os.path.join(self._project_dir, "project.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    task_type = meta.get("task_type", "语义分割")
                    internal = task_map.get(task_type, "segmentation")
                    self._set_task(internal)
                except Exception:
                    pass

            # Restore classification training settings (Pre-Scale, etc.)
            if internal in ("classification", "mixed_classification"):
                self._load_cls_train_settings()
                self._load_cls_predictions()
                # Restore heatmap display mode if auto-heatmap was on and heatmaps exist
                if (not self._mixed_cls_mode
                        and hasattr(self, "cls_auto_heatmap") and self.cls_auto_heatmap.isChecked()):
                    hm_dir = os.path.join(self._project_dir, "outputs", "classification", "heatmaps")
                    if os.path.isdir(hm_dir) and any(
                            f.endswith("_heatmap.jpg") for f in os.listdir(hm_dir)):
                        self.canvas._show_heatmap = True
            # Default to the annotation panel on the right after opening a project
            # (task switching above may have moved it to inference/training).
            if hasattr(self, "_right_tabs"):
                self.show_right_tab("annot")
            # 左栏同理：默认落在「工程管理」了（维护者 2026-10-15），所以打开工程
            # 后**必须**切到「图像管理」，否则用户点开工程后还盯着一页空图片列表。
            # 与右栏那一行是同一个道理：右侧切标注页，左侧就切图片列表。
            if hasattr(self, "_left_stack"):
                self.show_left_tab("img")

            self._load_image_list_async()
            self.log(f"Opened: {name}")
            self.setWindowTitle(f"{APP_NAME} - {name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _close_project(self):
        # Save classification training settings before closing
        if self._project_dir and getattr(self, "_cls_mode", False):
            self._save_cls_train_settings()
        
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

    def _show_task_panel(self, infer_idx=None, train_idx=None, prefer=None):
        """切到「训练/推理」合并面板，并同步两个子栈。

        合并前这里是 `_show_task_panel(outer_idx, ...)`，调用点还要按"有没有模型"
        在 2（训练）/3（推理）之间挑页签。现在两段在同一个面板里同时可见，
        不需要再挑页签；高度也改成**按内容自动分配**（`_fit_ti_split`），
        所以 `prefer` 只作为兼容参数保留、不再决定比例。
        """
        self.show_right_tab("params")           # 训练/推理参数都在「参数管理」模块里
        if infer_idx is not None and hasattr(self, "_infer_stack"):
            self._infer_stack.setCurrentIndex(infer_idx)
        if train_idx is not None and hasattr(self, "_train_stack"):
            self._train_stack.setCurrentIndex(train_idx)

    def _task_model_exists(self, task):
        """Whether the current project already has trained weights for the task."""
        project_dir = getattr(self, "_project_dir", None)
        if not project_dir:
            return False
        if task == "detection":
            return os.path.exists(os.path.join(project_dir, "models", "detection", "best.pt"))
        if task in ("classification", "mixed_classification"):
            d = os.path.join(project_dir, "models", "classification")
            return os.path.isdir(d) and any(f.endswith(".pth") for f in os.listdir(d))
        if task == "segmentation":
            return (os.path.exists(os.path.join(project_dir, "models", "best_model.pth"))
                    or os.path.exists(os.path.join(project_dir, "models", "last_model.pth")))
        return False

    def _project_has_height(self):
        """工程里有没有高度图（TIF/TIFF）。混合分类工程一定要显示高度图参数。"""
        try:
            if getattr(self, "_mixed_cls_mode", False):
                return True
            pd = self._project_dir
            if not pd:
                return False
            img_dir = os.path.join(pd, "images")
            if os.path.isdir(img_dir):
                for fn in os.listdir(img_dir):
                    if fn.lower().endswith((".tif", ".tiff")):
                        return True
        except Exception:
            pass
        return False

    def _update_hm_section_visibility(self):
        """按工程实际情况显隐「高度图参数」区块（只在状态变化时才动控件）。"""
        try:
            sec = getattr(self, "_hm_section", None)
            if sec is None:
                return
            want = bool(self._project_has_height())
            if want != getattr(self, "_hm_section_visible", None):
                self._hm_section_visible = want
                sec.setVisible(want)
        except Exception:
            pass

    def _set_task(self, task):
        # 「检测工具」只对目标检测工程有意义：语义分割/分类/OCR 工程里那些
        # HBB/OBB 画框按钮点了也没用，不该占位置（用户要求）。
        try:
            if hasattr(self, "_det_section"):
                self._det_section.setVisible(task == "detection")
        except Exception:
            pass
        self._update_hm_section_visibility()
        """Switch between segmentation and detection task modes."""
        # Annotation toolbar (rect/circle/ellipse brush tools) is segmentation-only
        self.annotation_toolbar.setVisible(task == "segmentation")
        # Lightweight detection toolbar is detection-only
        self.det_toolbar.setVisible(task == "detection")
        # Mask editing is only for segmentation; other tasks must not draw or
        # save mask shapes (would pollute their JSON with polygons).
        self.canvas._mask_editing = (task == "segmentation")
        if task != "segmentation":
            self.canvas.set_mode(self.canvas.MODE_PAN)
        if task == "detection":
            self._detection_mode = True
            self._cls_mode = False
            self._ocr_mode = False
            self._ocv_mode = False
            self._mixed_cls_mode = False
            self.log("Task: Object Detection")
            from detection.box_manager import BoxManager
            from detection.box_overlay import DetectionOverlay
            classes = getattr(self, "_cached_classes", None) or ["background"]
            self._box_manager = BoxManager(categories=classes)
            self.canvas._det_overlay = DetectionOverlay(self.canvas, self._box_manager)
            self.annotation_toolbar.set_mode("pan")
            self.canvas._mode = self.canvas.MODE_PAN
            self._load_det_annotations()
            self._populate_det_class_list()
            self.det_toolbar.set_draw_active(True)
            self._sync_det_toolbar()
            self._infer_title.setText("推理 · 目标检测")
            self._train_title.setText("训练 · 目标检测")
            self._show_task_panel(infer_idx=0, train_idx=1,
                                  prefer="infer" if self._task_model_exists("detection") else "train")
        elif task == "classification":
            self._detection_mode = False
            self._cls_mode = True
            self._ocr_mode = False
            self._ocv_mode = False
            self._mixed_cls_mode = False
            self.canvas._mixed_cls_mode = False
            self._viewer_stack.setCurrentIndex(0)  # Show AnnotationCanvas
            self.log("Task: Image Classification")
            self.canvas._det_overlay = None
            self.canvas._cls_mode = True
            self.canvas.update()
            self._load_classification_labels()
            self._refresh_cls_model_list()
            self._filter_images(self.search_input.text())
            self._restore_confusion_matrix()
            self._infer_title.setText("推理 · 图像分类")
            self._train_title.setText("训练 · 图像分类")
            self._show_task_panel(infer_idx=1, train_idx=2,
                                  prefer="infer" if self._task_model_exists("classification") else "train")
        elif task == "mixed_classification":
            self._detection_mode = False
            self._cls_mode = True
            self._ocr_mode = False
            self._ocv_mode = False
            self._mixed_cls_mode = True
            self.canvas._mixed_cls_mode = True
            self.canvas._view_mode = 2
            self._viewer_stack.setCurrentIndex(1)  # Show MixedViewer
            self.log("Task: Mixed Classification (Gray+Height)")
            self.canvas._det_overlay = None
            self.canvas._cls_mode = True
            self.canvas.update()
            self._load_classification_labels()
            self._refresh_cls_model_list()
            self._filter_images(self.search_input.text())
            self._restore_confusion_matrix()
            self._infer_title.setText("推理 · 混合分类")
            self._train_title.setText("训练 · 混合分类")
            self._show_task_panel(infer_idx=1, train_idx=2,
                                  prefer="infer" if self._task_model_exists("mixed_classification") else "train")
        elif task == "ocr":
            self._detection_mode = False
            self._cls_mode = False
            self._ocr_mode = True
            self._ocv_mode = False
            self.log("Task: OCR Text Recognition")
            self.canvas._det_overlay = None
            self.canvas._cls_mode = False
            self.canvas._ocr_overlay = None
            self.canvas.update()
            self._infer_title.setText("推理 · OCR 文字识别")
            self._train_title.setText("训练 · OCR（无训练流程）")
            # OCR/OCV 没有训练流程（train_idx=3 是"无训练流程"提示页），
            # 所以合并面板里当然重点给推理段
            self._show_task_panel(infer_idx=2, train_idx=3, prefer="infer")
        elif task == "ocv":
            self._detection_mode = False
            self._cls_mode = False
            self._ocr_mode = False
            self._ocv_mode = True
            self.log("Task: OCV Quality Inspection")
            self.canvas._det_overlay = None
            self.canvas._cls_mode = False
            self.canvas.update()
            self._infer_title.setText("推理 · OCV 字符质检")
            self._train_title.setText("训练 · OCV（无训练流程）")
            self._show_task_panel(infer_idx=3, train_idx=3, prefer="infer")
        else:
            self._detection_mode = False
            self._cls_mode = False
            self._ocr_mode = False
            self._ocv_mode = False
            self.log("Task: Semantic Segmentation")
            self.canvas._det_overlay = None
            self.canvas._cls_mode = False
            self.canvas.clear_heatmap()
            self.canvas._ocr_overlay = None
            self.canvas.update()
            self._infer_title.setText("推理 · 语义分割")
            self._train_title.setText("训练 · 语义分割")
            self._show_task_panel(infer_idx=0, train_idx=0,
                                  prefer="infer" if self._task_model_exists("segmentation") else "train")

    def _load_classification_labels(self):
        """Load classification labels from class_labels.json."""
        if not self.current_project:
            return
        try:
            project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
            from classification.label_manager import LabelManager as ClsLabelManager
            self.class_list.set_project_dir(project_dir)
            lm = ClsLabelManager(project_dir)
            data = lm.load()
            labels = data.get("labels", [])
            if not labels:
                data = lm.auto_label_from_folders()
                labels = data.get("labels", [])
            if labels:
                self._cached_classes = labels
                self.class_list.set_classes(labels)
                self.log(f"Classification labels: {len(labels)} classes")
            else:
                self.log("No classification labels found")
        except Exception as e:
            self.log(f"Failed to load classification labels: {e}")


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
        proj_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        # 多视图工程走专用对话框（每视图一个槽位 + 绑定图片组 + 预览），
        # 单图工程保持原有"选文件即导入"的流程不变。
        try:
            from training.multiview_gui import is_multiview_project
            if is_multiview_project(proj_dir):
                return self._import_images_multiview(proj_dir)
        except Exception as e:
            self.log(f"多视图检测失败，按单图流程处理: {type(e).__name__}: {e}")
        files, _ = QFileDialog.getOpenFileNames(
            self, "Import Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if files:
            self.log(f"Importing {len(files)} images...")
            img_dir = os.path.join(
                self.pm.get_project_dir(self.current_project["name"]), "images")
            def copy_files():
                for i, f in enumerate(files):
                    dst = os.path.join(img_dir, os.path.basename(f))
                    if not os.path.exists(dst):
                        shutil.copy2(f, dst)
                    # Skip JSON creation for height map files (.tif/.tiff)
                    if os.path.splitext(dst)[1].lower() in (".tif", ".tiff"):
                        pass
                    else:
                        json_path = os.path.splitext(dst)[0] + ".json"
                        if not os.path.exists(json_path):
                            blank = {"version":"5.3.1","flags":{},"shapes":[],"imagePath":os.path.basename(dst),"imageData":None,"imageHeight":0,"imageWidth":0}
                            with open(json_path, "w", encoding="utf-8") as jf:
                                json.dump(blank, jf, indent=2, ensure_ascii=False)
                    if (i + 1) % 100 == 0:
                        self.log(f"Import progress: {i+1}/{len(files)}")
                self.log(f"Import complete: {len(files)} images")
                self._load_image_list_async()
            threading.Thread(target=copy_files, daemon=True).start()

    def _import_images_multiview(self, proj_dir):
        """多视图加图：每个视图一个槽位，按 _te_N 后缀绑定成组后整组导入。

        注意：**不创建空白 JSON**。平台的 SegmentationDataset 优先读同名的 LabelMe JSON，
        若生成空白 JSON 会把真正的掩膜 PNG 遮蔽掉（全变背景），标注就白做了。
        """
        from training.multiview_gui import project_views
        from ui.multiview_add_dialog import MultiViewAddDialog
        views, vnames = project_views(proj_dir)
        dlg = MultiViewAddDialog(self, views=views, view_names=vnames)
        if dlg.exec_() != QDialog.Accepted:
            self.log("已取消加图")
            return
        views, per_view = dlg.get_view_paths()
        n = min((len(x) for x in per_view), default=0)
        if n == 0:
            return QMessageBox.warning(self, "提示", "没有可导入的图片组")
        for info in dlg.dropped_info():
            self.log(f"未成组已剔除：{info}（缺少配对的其它视图）")
        img_dir = os.path.join(proj_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        total = n * len(views)
        self.log(f"导入多视图图片：{n} 组 × {len(views)} 视图（{'/'.join(views)}）= {total} 张")

        def copy_files():
            done, skip = 0, 0
            for i in range(n):
                for vi in range(len(views)):
                    src = per_view[vi][i]
                    dst = os.path.join(img_dir, os.path.basename(src))
                    if os.path.exists(dst):
                        skip += 1
                    else:
                        try:
                            shutil.copy2(src, dst)
                        except Exception as e:
                            self.log(f"复制失败 {os.path.basename(src)}: {e}")
                    done += 1
                    if done % 100 == 0:
                        self.log(f"导入进度 {done}/{total}")
            self.log(f"导入完成：{n} 组 × {len(views)} 视图 = {total} 张"
                     + (f"（已存在跳过 {skip} 张）" if skip else ""))
            self.log("下一步：在右侧图像列表点开任一 _te_0 图，用标注工具画掩膜并 Save"
                     "（掩膜多视图共用，只需标 te_0 一张）")
            self._load_image_list_async()

        threading.Thread(target=copy_files, daemon=True).start()

    def _load_image_list_async(self):
        if not self.current_project:
            return
        # 必须在起线程前就建好缓存：后台预热线程可能比主线程的 _filter_images 更早
        # 跑到，那时若属性还不存在，预热会直接放弃（首屏就白等了）。
        if not hasattr(self, "_mask_stat_cache"):
            self._mask_stat_cache = {}
        self.log("Scanning image list...")
        gen = self._project_gen
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        def scan():
            img_dir = os.path.join(project_dir, "images")
            images = []
            if os.path.exists(img_dir):
                for f in sorted(os.listdir(img_dir)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                        images.append(os.path.join(img_dir, f))
            try:
                self.image_list_signal.emit((gen, images))
            except RuntimeError:
                # 窗口已销毁（用户关了软件）—— 后台线程还在跑，直接收摊，别刷异常
                return
            # ★ 列表先显示出来，再在**同一个后台线程**里预热掩膜统计缓存。
            #   不预热的话，主线程 _filter_images 要为每张图解码 1500² PNG，
            #   1690 张 ≈ 7~17 秒界面冻结 —— 这就是"双击打开工程很卡"。
            self._prewarm_mask_stats(gen, project_dir, images)
        threading.Thread(target=scan, daemon=True).start()

    def _prewarm_mask_stats(self, gen, project_dir, images):
        """后台预热两类标注统计，填满缓存；算完发信号让主线程补刷列表。

        ① JSON 标签 —— 走 core.label_scan 的**进程级**缓存。这些 JSON 合计上 GB
           （单个最大 18MB，内容 99.9% 是顶点坐标），而这里只需要"有哪些 label"。
           首次全解析很慢（实测过 4641ms 的段），但它纯 CPU/IO，适合丢后台。
        ② 掩膜类别数 —— PIL.getcolors 统计，1500² 约 4.4ms/张。

        用户看到的效果：工程秒开，列表立刻可用，「标注」列数字稍后自动补齐。
        """
        # ★ 先等首屏画完再开工（600ms）。
        #   原来 emit 完列表就立刻开始解码/扫 JSON，正好和主线程的 _filter_images
        #   撞在一起抢 GIL —— 实测首屏 400ms 被拖到 870ms。
        #   降 OS 线程优先级没用（GIL 是 Python 层的锁，不看调度优先级），
        #   只能靠错峰。预热晚 0.6 秒开始，用户完全感觉不到；
        #   而首屏能实实在在快一倍。
        time.sleep(0.6)

        # 顺手把本线程降到 BELOW_NORMAL（对纯 IO 阶段仍有帮助）
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), -1)   # THREAD_PRIORITY_BELOW_NORMAL
        except Exception:
            pass

        ann_dir = os.path.join(project_dir, "annotations")
        filled = 0
        for p in images:
            if gen != self._project_gen:
                return                      # 已经切到别的工程，放弃这轮
            base = os.path.splitext(os.path.basename(p))[0]

            # ① 预热 JSON 标签缓存（标签表用于列表搜索和"标注名"筛选）
            jp = os.path.join(os.path.dirname(p), base + ".json")
            try:
                if os.path.exists(jp):
                    self._cached_json_labels(jp)
            except Exception:
                pass

            # ② 预热掩膜统计
            mp = os.path.join(ann_dir, base + "_mask.png")
            cache = getattr(self, "_mask_stat_cache", None)
            if cache is None:
                return
            if mp in cache:
                continue
            try:
                if not os.path.exists(mp):
                    continue
                mt = os.path.getmtime(mp)
                classes, has = _mask_stats_of(mp)
                cache[mp] = (mt, classes, has)
                filled += 1
            except Exception:
                continue                        # 单张坏了不影响整体
        # 即使一张都没补（工程本来就没有掩膜），也要通知一次：
        # 否则 _mask_stats_ready_gen 永远不置位，之后新建的掩膜得等下次开关工程。
        try:
            self.mask_stats_signal.emit(gen)
        except RuntimeError:
            pass          # 窗口已销毁，无需通知

    def _on_mask_stats_ready(self, gen):
        """后台掩膜统计就绪 → 置位 ready 并重刷一次列表（此时全命中缓存）。"""
        if gen != self._project_gen:
            return
        # ⚠️ 必须先置位：_filter_images 靠它判断"能不能自己解码"
        self._mask_stats_ready_gen = gen
        _t0 = time.perf_counter()
        try:
            keep = self.image_list_widget.currentRow()
        except Exception:
            keep = -1
        self._filter_images(self.search_input.text())
        _dt = (time.perf_counter() - _t0) * 1000.0
        try:
            if 0 <= keep < self.image_list_widget.rowCount():
                self._suppress_cell_change = True
                self.image_list_widget.selectRow(keep)
                self._suppress_cell_change = False
        except Exception:
            pass
        self.log("标注统计预热完成，列表回填 %.0f ms（已缓存 %d 张掩膜）"
                 % (_dt, len(self._mask_stat_cache)))

    def _populate_image_list(self, data):
        gen, image_paths = data
        if gen != self._project_gen:
            return
        self._all_images = image_paths
        self._suppress_cell_change = True
        self._filter_images("")
        self._suppress_cell_change = False
        # 多视图信息在_这里_打日志（不能在 _filter_images 里打，会重入崩溃）
        if getattr(self, "_mv_mode_list", False):
            _meta = getattr(self, "_mv_view_meta", []) or []
            _gn = getattr(self, '_mv_group_count', 0)
            # 说清"一行 = 一颗电芯的 N 张视图"，否则人会以为列表少图了
            # （2026-10-14 用户看到 33 行就去问"为什么只有 33 张"）
            # ⚠️ 日志面板是 QTextEdit，不认 Markdown：这里只能写纯文本，
            #    写 `**重点**` 会原样显示两个星号（截图里看到的）。
            self.log(f"多视图列表：{_gn} 组 × {len(_meta)} 视图"
                     f"（{' / '.join('%s %s' % (v, n) for v, n in _meta)}），已按组成行"
                     f" —— 一行 = 一颗电芯的 {len(_meta)} 张视图，"
                     f"共 {_gn * len(_meta)} 张图都在（展开看「图片名」列的多行文件名）")
        elif getattr(self, "_mv_list_note", ""):
            self.log(f"多视图识别/分组失败，按单图列表显示：{self._mv_list_note}")
        # Keep detection annotation state created by _set_task("detection")
        # for detection projects. The async image-list scan must not reset it,
        # otherwise the det overlay/BoxManager lose their loaded boxes and
        # Ctrl+Space can never display existing rectangles after opening a project.
        if not self._detection_mode:
            self._box_manager = None  # BoxManager for detection annotations
        # Async load output stats (pred counts + IoU scores)
        if self.current_project:
            pd = str(self.pm.get_project_dir(self.current_project["name"]))
            self._pred_count_cache_key = None  # force reload
            threading.Thread(target=self._load_output_stats_async, args=(pd,), daemon=True).start()
        # In mixed mode, count paired height maps
        pair_count = 0
        if self._mixed_cls_mode:
            for p in image_paths:
                base = os.path.splitext(p)[0]
                for ext in (".tif", ".tiff"):
                    if os.path.exists(base + ext):
                        pair_count += 1
                        break
            self.image_count_label.setText(f"Total: {len(image_paths)} images ({pair_count} pairs)")
        else:
            self.image_count_label.setText(f"Total: {len(image_paths)} images")
        self.log(f"Loaded {len(image_paths)} images in list")
        _ms = getattr(self, "_last_filter_ms", None)
        if _ms is not None:
            self.log("列表构建耗时 %.0f ms（%d 行）" % (_ms, getattr(self, "_last_filter_rows", 0)))
        if image_paths:
            self._suppress_cell_change = True
            self.image_list_widget.selectRow(0)
            self._suppress_cell_change = False
            self._load_image_by_index(0)
        self._refresh_class_ui()

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
            img_conf = {}          # base -> 平均置信度（0~1），来自 <base>_prob.png
            for f in os.listdir(out_dir):
                # Segmentation: _pred.png masks
                if f.endswith("_pred.png"):
                    base = f.replace("_pred.png", "")
                    try:
                        mask = _np.array(Image.open(os.path.join(out_dir, f)))
                        unique = _np.unique(mask)
                        # ★ 2026-10-22 用户要求：这里要**块数**（连通域个数），不是类别数
                        #   （原来 len(unique) 只有 2，而 Result 表有 39 行，两边对不上）
                        _n = 0
                        try:
                            import cv2 as _cv
                            for _c in unique:
                                if _c in (0, 255):
                                    continue
                                _nl, _ = _cv.connectedComponents((mask == _c).astype(_np.uint8), 8)
                                _n += max(0, _nl - 1)
                        except Exception:
                            _n = len([u for u in unique if u not in (0, 255)])
                        pred_count[base] = _n
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
                        # ★★ 2026-10-22 用户要求：列表「分数」列统一成**平均置信度**，
                        #   与 Result 表「分数」同一口径（模型概率，不是与标注的 IoU）。
                        #   置信度图由批量推理落盘（<base>_prob.png，uint8 = 概率×255）；
                        #   没有这张图（老结果）就留空，绝不拿 IoU 冒充 0 分。
                        pp = os.path.join(out_dir, base + "_prob.png")
                        if os.path.exists(pp):
                            try:
                                prob = _np.array(Image.open(pp)).astype(_np.float32) / 255.0
                                if prob.shape[:2] == mask.shape[:2]:
                                    fg = (mask != 0) & (mask != 255)
                                    if bool(fg.any()):
                                        img_conf[base] = round(float(prob[fg].mean()), 4)
                                    else:
                                        img_conf[base] = 0.0
                            except Exception:
                                pass
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
            self._img_conf = img_conf
            self._pred_count_cache_key = project_dir
            self.refresh_list_signal.emit()
        except Exception:
            pass

    def _filter_images(self, text):
        _t_fi0 = time.perf_counter()
        if not hasattr(self, "_mask_stat_cache"):
            # 掩膜统计缓存：key = 掩膜绝对路径, value = (mtime, 类别数, 是否有内容)
            # 见下方"标注统计缓存"注释 —— 这是打开工程不再卡的关键
            self._mask_stat_cache = {}
        self._suppress_cell_change = True
        self.image_list_widget.setRowCount(0)
        text_lower = text.lower().strip()
        # 标记是"工程级"数据，按工程缓存读一次即可（列表刷新是热路径）
        self._ensure_marks_loaded()
        # split 表缓存：每次刷新重建一次（从文件读一次即可），循环内只查表
        self._split_map_cache = None
        sf_text = self.split_filter.currentText()
        annotated_only = self.annotated_check.isChecked()
        # 标注名字筛选：选项跟随工程类别（懒同步）
        lf_text = ""
        try:
            classes = [c for c in (getattr(self, "_cached_classes", []) or [])
                       if c and str(c).lower() != "background"]
            want = ["All Labels"] + classes
            cur = [self.label_filter.itemText(i) for i in range(self.label_filter.count())]
            if want != cur:
                prev = self.label_filter.currentText()
                self.label_filter.blockSignals(True)
                self.label_filter.clear()
                self.label_filter.addItems(want)
                idx = self.label_filter.findText(prev)
                self.label_filter.setCurrentIndex(max(0, idx))
                self.label_filter.blockSignals(False)
            lf_text = self.label_filter.currentText()
        except Exception:
            lf_text = ""
        # Load prediction counts + IoU scores from outputs dir (triggered async by _populate_image_list)
        # Ensure fallback dicts exist
        if not hasattr(self, "_pred_count"):
            self._pred_count = {}
        if not hasattr(self, "_img_scores"):
            self._img_scores = {}
        total_train = 0
        total_val = 0
        total_test = 0
        total_annotated = 0
        total_shapes = 0
        ann_dir = os.path.join(os.path.dirname(self._all_images[0]) if self._all_images else "",
                               "..", "annotations")

        # ── 目录索引（2026-10-29 第三轮性能：清掉最后的文件系统开销）──────────
        # 原来循环里对每张图做 3 次 os.path.exists + 1 次 getmtime：
        #   1690 张 ⇒ 5112 次 exists + 3610 次 stat ≈ 340ms（cProfile 实测），
        #   占 _filter_images 剩余耗时的一半以上。
        # 改成先各扫一次目录（os.scandir 是批量读目录项，比逐个 stat 便宜），
        # 循环内只做内存查表。
        # 失效风险：遍历期间有人在外部增删标注文件 —— 实际不会发生（单机单实例操作）。
        _img_dir = os.path.dirname(self._all_images[0]) if self._all_images else ""
        _img_names = None          # None = 扫描失败，循环里回退到 exists
        _img_mtimes = {}
        _need_mtime = self.recent_check.isChecked()
        if _img_dir:
            try:
                _names = set()
                with os.scandir(_img_dir) as _it:
                    for _e in _it:
                        _names.add(_e.name)
                        if _need_mtime:
                            try:
                                _img_mtimes[_e.name] = _e.stat().st_mtime
                            except OSError:
                                _img_mtimes[_e.name] = 0.0
                _img_names = _names
            except OSError:
                _img_names = None

        _ann_index = None          # 文件名 -> mtime（掩膜只有 228 个，scan 很便宜）
        try:
            _ann = {}
            with os.scandir(ann_dir) as _it:
                for _e in _it:
                    try:
                        _ann[_e.name] = _e.stat().st_mtime
                    except OSError:
                        _ann[_e.name] = 0.0
            _ann_index = _ann
        except OSError:
            _ann_index = None

        # Sort by annotation date if "Recent" is checked
        if self.recent_check.isChecked():
            if _img_mtimes:
                def _json_mtime(p):
                    return _img_mtimes.get(
                        os.path.splitext(os.path.basename(p))[0] + ".json", 0.0)
            else:
                def _json_mtime(p):
                    base = os.path.splitext(os.path.basename(p))[0]
                    jp = os.path.join(os.path.dirname(p), base + ".json")
                    return os.path.getmtime(jp) if os.path.exists(jp) else 0
            self._all_images.sort(key=_json_mtime, reverse=True)
        # Classification mode: load class_labels.json mapping once
        cls_mapping = {}
        if self._cls_mode and self.current_project:
            try:
                from classification.label_manager import LabelManager as ClsLabelManager
                pd = str(self.pm.get_project_dir(self.current_project["name"]))
                cls_mapping = ClsLabelManager(pd).get_mapping()
            except Exception:
                pass
        # Build filtered data list (deferred rendering)
        self._filtered_data = []

        # Mixed classification: build paired image list
        _mixed = getattr(self, "_mixed_cls_mode", False)
        _image_sources = self._all_images
        if _mixed:
            _gray_map = {}
            _height_map = {}
            for p in self._all_images:
                b = os.path.splitext(os.path.basename(p))[0]
                e = os.path.splitext(p)[1].lower()
                if e in (".tif", ".tiff"):
                    _height_map[b] = p
                else:
                    _gray_map[b] = p
            _paired = []
            for base, gp in sorted(_gray_map.items()):
                hp = _height_map.get(base, "")
                _paired.append((gp, hp))
            _image_sources = _paired

        # Multi-view: group the N view files of one cell into a single row
        # （照 _mixed 的配对模式做：一行 = 一颗电芯，名称列显示 N 行文件名）
        # ⚠️ 这段运行在"_filter_images"这个会被频繁重建的热路径上，
        #    绝对不能在这里调 self.log() —— 日志面板会触发重入，
        #    递归撑爆 C 栈直接 0xC0000409 崩溃（已踩过）。信息先存属性，外面再打。
        _mv = False
        _mv_views = []
        _mv_names = {}
        _mv_err = ""
        if self.current_project:
            try:
                from training.multiview_gui import is_multiview_project, project_views
                _mvdir = str(self.pm.get_project_dir(self.current_project["name"]))
                if is_multiview_project(_mvdir):
                    _mv_views, _mv_names = project_views(_mvdir)
                    _mv = len(_mv_views) > 1
            except Exception as e:
                _mv_err = f"{type(e).__name__}: {e}"
                _mv = False
        self._mv_mode_list = _mv
        self._mv_view_meta = [(v, (_mv_names or {}).get(v, v)) for v in _mv_views] if _mv else []
        self._mv_list_note = _mv_err
        if _mv:
            try:
                _maps = {v: {} for v in _mv_views}
                for p in self._all_images:
                    _stem = os.path.splitext(os.path.basename(p))[0]
                    for v in _mv_views:
                        if _stem.endswith("_" + v):
                            _maps[v][_stem[: -(len(v) + 1)]] = p
                            break
                # 只保留 N 个视图都齐的组（缺视图无法成样本）
                _keys = sorted(set.intersection(*[set(m) for m in _maps.values()])) if _maps else []
                _image_sources = [tuple(_maps[v][k] for v in _mv_views) for k in _keys]
                self._mv_group_count = len(_keys)
            except Exception as e:
                self._mv_list_note = f"分组失败: {type(e).__name__}: {e}"
                self._mv_group_count = 0
                _mv = False
                self._mv_mode_list = False
                _image_sources = self._all_images

        for _img_entry in _image_sources:
            if _mixed:
                path = _img_entry[0]
                height_path = _img_entry[1]
            elif _mv:
                path = _img_entry[0]
                height_path = ""
            else:
                path = _img_entry
                height_path = ""
            fname = os.path.basename(path)
            base = os.path.splitext(fname)[0]

            # ── 标记 / 缺陷类型 筛选：放在最前面，被筛掉的图后面的形状统计就不白算了 ──
            mk_rec = marks_mod.get_mark(self._marks, base)
            defect = marks_mod.defect_of(base)
            # 缺陷类型用**文件名推断**（命名规范 {date}-{机号/料号}-{面?}-{缺陷}{N}_te_{n} 里
            # 就带缺陷段），不用第一条标注的类别名 —— 否则统计面板显示的名字和筛选时比的名字
            # 可能不是同一个（一个来自类别表、一个来自文件名），点了格子却筛不出东西。
            if self._mark_filter and mk_rec["mark"] != self._mark_filter:
                continue
            if self._defect_filter and defect != self._defect_filter:
                continue

            mask_path = os.path.join(ann_dir, base + "_mask.png")
            json_path = os.path.join(os.path.dirname(path), base + ".json")
            det_json_path = os.path.join(os.path.dirname(path), base + "_det.json")

            # ── 标注统计缓存（2026-10-29 性能修复：打开工程卡顿主因）──────────
            # 原来这里为**每张图**解码两次 1500² 的 PNG 掩膜：
            #   ① 下面 elif 分支里 np.unique 数类别
            #   ② 再下面"是否真标注过"又 Image.open 一遍
            # 单次解码 20~30ms × 2 × 1690 张 ≈ 一分多钟，而且全程在主线程，
            # 表现就是"双击打开工程很卡，界面像死了"。
            # （JSON 那一半 2026-10-28 已由 core.label_scan 的进程级缓存解决，
            #   掩膜这一半一直是漏的 —— 而掩膜比 JSON 大得多。）
            # 现在：① 只解一次，两个用途复用同一个数组
            #       ② 结果按掩膜 mtime 缓存，同一张图再次刷新直接命中，不碰磁盘
            _mask_classes = 0
            _mask_has_content = False
            _m_mtime = None
            # 用目录索引查 mtime（不再逐个 getmtime）；索引扫不到才回退到真实 IO
            _m_mtime = None
            if _ann_index is not None:
                _m_mtime = _ann_index.get(base + "_mask.png")
            else:
                try:
                    if os.path.exists(mask_path):
                        _m_mtime = os.path.getmtime(mask_path)
                except OSError:
                    _m_mtime = None

            # ★ 预热未完成时**只读缓存、绝不自己解码**。
            #   第一版没做这个判断，结果主线程和后台预热线程抢同一批 PNG，
            #   实测"列表构建 5979ms + 预热 2s"两头白烧（用户实测 2026-10-29）。
            #   现在的约定：解码只归后台预热线程；主线程拿不到就直接用占位，
            #   等 mask_stats_signal 回来再补刷 —— 那时全部命中缓存。
            _allow_decode = (self._mask_stats_ready_gen == self._project_gen)

            if _m_mtime is not None:
                _hit = self._mask_stat_cache.get(mask_path)
                if _hit is not None and _hit[0] == _m_mtime:
                    _mask_classes, _mask_has_content = _hit[1], _hit[2]
                elif _allow_decode:
                    try:
                        _mask_classes, _mask_has_content = _mask_stats_of(mask_path)
                    except Exception:
                        _mask_classes, _mask_has_content = 0, False
                    self._mask_stat_cache[mask_path] = (
                        _m_mtime, _mask_classes, _mask_has_content)
                # else: 预热中 → 保持 (0, False) 占位，稍后补刷

            # 同样走目录索引（两次 scandir 取代 3000+ 次 exists）
            if _img_names is not None:
                _det_exists = (base + "_det.json") in _img_names
                _json_exists = (base + ".json") in _img_names
            else:
                _det_exists = os.path.exists(det_json_path)
                _json_exists = os.path.exists(json_path)
            has_ann = _mask_has_content or _json_exists or _det_exists

            # Count shapes and collect labels for search
            shape_count = 0
            shape_labels = []
            # Classification mode: check class_labels.json mapping
            cls_id = None
            if self._cls_mode and cls_mapping:
                rel_path = f"images/{fname}"
                cls_id = cls_mapping.get(rel_path, cls_mapping.get(fname, None))
                if cls_id is not None:
                    has_ann = True
                    shape_count = 1
                    labels = getattr(self, "_cached_classes", []) or []
                    if int(cls_id) < len(labels):
                        shape_labels.append(labels[int(cls_id)])
            if _det_exists:
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
            if _json_exists:
                # 预热未完成时也跳过：这批 JSON 合计上 GB，主线程扫一遍就是几秒冻结。
                # shape_labels 只在预热期间为空，等补刷时全部回来（搜索/标注名筛选随之生效）。
                if _allow_decode:
                    try:
                        for lbl in self._cached_json_labels(json_path):
                            shape_count += 1
                            shape_labels.append(lbl)
                    except Exception:
                        pass
            elif _mask_has_content:
                shape_count = _mask_classes

            # ★★ 2026-10-23 修「标注筛选失效」：
            #   上面那句 `has_ann = 文件存在` 只说明"有个 json/mask 文件"，
            #   而平台保存过空标注（画完又删、或只建了文件）会留下**空 json** ⇒
            #   「Annotated Only」全通过（实测 46/46）、「Unannotated Only」永远是空。
            #   而列表「标注」列显示的是**缺陷块数**（0），两边天然矛盾 —— 用户实拍。
            #   现在收紧：必须真的有内容（缺陷块 / 掩膜非零含 ignore / 检测框 / 分类标签）。
            if shape_count == 0 and not _det_exists:
                # 复用上面已经解码的 _mask_has_content，别再开一次 PNG
                has_ann = _mask_has_content

            # CM cell click filter
            cm_filter = getattr(self, "_cm_filter_bases", None)
            if cm_filter is not None and base not in cm_filter:
                continue
            # Search: match filename OR class label
            if text_lower and text_lower not in fname.lower():
                # Check if any shape label matches
                label_match = any(text_lower in lbl.lower() for lbl in shape_labels)
                if not label_match:
                    continue
            # 标注名字下拉：只看含该标注类别的图片
            if lf_text and lf_text != "All Labels" and lf_text not in shape_labels:
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
                    # ★ 2026-10-23：split 表原来在**每张图**的循环里都 json.load 一次
                    #   （489 张 = 489 次读同一个文件，光这一项就几百 ms）。
                    #   现在循环外读一次、循环内只查表。
                    if self._split_map_cache is None:
                        pd = str(self.pm.get_project_dir(self.current_project["name"]))
                        sf = os.path.join(pd, "train_test_split.json")
                        if os.path.exists(sf):
                            has_split_file = True
                            with open(sf, "r", encoding="utf-8") as sfh:
                                self._split_map_cache = json.load(sfh) or {}
                        else:
                            self._split_map_cache = {}
                    if self._split_map_cache:
                        has_split_file = True
                        v = self._split_map_cache.get(base)
                        if v == "val":
                            split_status = " [Val]"
                        elif v == "test":
                            split_status = " [Test]"
                        elif v == "train":
                            split_status = " [Train]"
                        # ★ 注意：**没有** elif has_ann -> [Train] 了。
                        #   用户明确要求：训练集必须自己设置，标了不等于进训练集。
                        #   没在 train_test_split.json 里划分过的图 = 空集（split_status 保持 ""）。
            except Exception:
                pass

            # Split filter
            if sf_text == "Train Only" and split_status != " [Train]":
                continue
            if sf_text == "Val Only" and split_status != " [Val]":
                continue
            if sf_text == "Test Only" and split_status != " [Test]":
                continue
            # 以下两项对标友商筛选下拉（标注状态）
            if sf_text == "Annotated Only" and not has_ann:
                continue
            if sf_text == "Unannotated Only" and has_ann:
                continue
            # 注意：Has Result / No Result 放到下面 pred_cnt 算完之后再判 ——
            # pred_cnt 在本轮后面才赋值，写在这里会读到**上一轮的旧值**，
            # 不报错但筛选结果是错的（静默错误，比崩溃更难发现）。

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

            # Prediction count / classification result
            pred_cnt = getattr(self, "_pred_count", {}).get(base, 0)
            # 结果筛选必须放在 pred_cnt 赋值**之后**（见上面注释）
            if sf_text == "Has Result" and not pred_cnt:
                continue
            if sf_text == "No Result" and pred_cnt:
                continue
            cls_text = ""
            ocr_text = ""
            ocr_score = ""
            ocr_time = ""
            if self._ocr_mode:
                ocr_data = getattr(self, "_ocr_predictions", {}).get(fname, None)
                if ocr_data:
                    ocr_text = ocr_data.get("text", "")
                    ocr_score = f"{ocr_data.get('confidence', 0):.3f}"
                ocr_time = str(getattr(self, "_ocr_times", {}).get(fname, ""))
            if self._cls_mode:
                cls_data = getattr(self, "_cls_predictions", {}).get(path, None) or getattr(self, "_cls_predictions", {}).get(fname, None)
                if cls_data is not None:
                    if isinstance(cls_data, dict):
                        cls_text = cls_data.get("class", "")
                        conf = cls_data.get("confidence", None)
                        if conf is not None:
                            score_str = f"{conf:.2f}"

            # Inference time
            infer_ms = getattr(self, "_infer_times", {}).get(base, "")

            # Score = 该图所有检出块的**平均模型置信度**（与 Result 表「分数」同一口径）
            # ⚠️ 别退回用 IoU：IoU 是"和标注比"的指标，没标注的图恒为 0，
            #    摆在这一列会被读成"模型没把握"（2026-10-22 用户实拍）。
            _raw_conf = getattr(self, "_img_conf", {}).get(base, None)
            score_str = ("%.1f%%" % (_raw_conf * 100.0)) if _raw_conf is not None else ""
            if self._cls_mode and cls_data is not None:
                if isinstance(cls_data, dict) and "confidence" in cls_data:
                    score_str = f"{cls_data['confidence']:.2f}"

            rec = {
                'path': path, 'fname': fname, 'base': base,
                'img_w': img_w, 'img_h': img_h,
                'split_status': split_status, 'shape_count': shape_count,
                'shape_labels': shape_labels, 'has_ann': has_ann,
                'cls_text': cls_text, 'ocr_text': ocr_text,
                'ocr_score': ocr_score, 'ocr_time': ocr_time,
                'score_str': score_str, 'infer_ms': infer_ms,
                'pred_cnt': pred_cnt,
                # 标记（与 label 分离，不进训练集）
                'mark': mk_rec["mark"], 'mark_note': mk_rec["note"],
                'mark_by': mk_rec["by"], 'mark_at': mk_rec["at"],
                'defect': defect,
                'split': {"[Train]": "train", "[Val]": "val", "[Test]": "test"}.get(split_status, ""),
                'height_path': height_path,
                'height_fname': os.path.basename(height_path) if height_path else "",
                # 多视图：这一组的所有视图文件（第 0 个是主标注视图）
                'view_paths': list(_img_entry) if _mv else [],
                'view_fnames': [os.path.basename(x) for x in _img_entry] if _mv else [],
            }
            # 条目式表达式筛选放在**记录搭好之后**：它要读分数/耗时/结果数量，
            # 提前判会读到还没赋值的字段（不报错，但筛选结果是错的）。
            if not self._filter_expr.evaluate(rec):
                continue
            self._filtered_data.append(rec)

            if split_status == " [Train]":
                total_train += 1
            elif split_status == " [Val]":
                total_val += 1
            elif split_status == " [Test]":
                total_test += 1
            if has_ann:
                total_annotated += 1
                total_shapes += shape_count

        # Reset to page 0 and render
        self._current_page = 0
        self._render_page()

        # Update stats summary bar
        # ⚠️ 多视图工程里"一行 = 一颗电芯的 N 张视图"，**行数与张数不是一个量纲**。
        #    老写法拿行数直接去比 len(self._all_images)（文件数），未筛选时也会显示
        #    "Showing 33/99" —— 看上去像有 66 张图被藏掉了。用户 2026-10-14 就是被这行
        #    数字问住（"为什么我图片列表只能看到33张"），实际是 33 组 × 3 视图 = 99 张，一张没少。
        #    现在按 组/张 两个量纲分别报，并在计数后面带上单位"组"。
        # ── 第一行（分页条内，紧贴列表）：只讲【列表范围】 ──
        # ⚠️ 两行各司其职，别互相抄。2026-10-14 用户指出"这个地方信息冗余了"：
        #    分页条这行与下面状态栏那行都在报 Total / Train / Ann，数字一字不差。
        #    分工：这一行回答"我现在看到多少、被筛掉多少"（随筛选变化）
        #          下一行回答"这批数据的划分与标注情况"（Train/Val/Ann/Blocks，不重复总量）
        total = len(self._filtered_data)
        _mv_n = len(getattr(self, "_mv_view_meta", []) or [])
        _mv_on = bool(getattr(self, "_mv_mode_list", False)) and _mv_n > 1
        _f_all = len(self._all_images)
        _g_all = getattr(self, "_mv_group_count", 0) or total
        if _mv_on:
            _f_shown = sum(max(1, len(r.get("view_fnames") or [])) for r in self._filtered_data)
            if total < _g_all or _f_shown < _f_all:
                self.stats_summary.setText(f"筛选 {total}/{_g_all} 组（{_f_shown}/{_f_all} 张）")
            else:
                self.stats_summary.setText(f"共 {_g_all} 组（{_f_all} 张）")
        elif total < _f_all:
            self.stats_summary.setText(f"筛选 {total}/{_f_all} 张")
        else:
            self.stats_summary.setText(f"共 {_f_all} 张")

        # ── 第二行（独立一行）：只讲【数据集统计】 ──
        # 不再重复总量（上一行已经给了列表范围），只留划分 / 标注 / 标注块。
        _u = " 组" if _mv_on else ""      # 多视图下 Train/Val/Test/Ann 数的都是"组"
        parts2 = []
        if total_train:
            parts2.append(f"Train: {total_train}{_u}")
        if total_val:
            parts2.append(f"Val: {total_val}{_u}")
        if total_test:
            parts2.append(f"Test: {total_test}{_u}")
        parts2.append(f"Ann: {total_annotated}{_u}")
        if not self._cls_mode:
            parts2.append(f"Blocks: {total_shapes}")
        else:
            parts2.append(f"Classes: {len(getattr(self, '_cached_classes', []))}")
        self.list_status.setText(" | ".join(parts2))
        self._update_selection_count()

        # Update accordion header counts
        self._update_hm_section_visibility()      # 列表扫完后才能确知有没有 TIF
        # 2026-10-22：**不要**用列表级计数覆盖右侧 Result(n)/Annotation(n)
        #   这两个数表示"当前这张图"的块数与标注数，由 _update_stats /
        #   _load_image_by_index 单独维护。原来这里传 total_shapes/total_annotated，
        #   列表一刷新就把当前图的值盖成全列表的值（用户看到 Result(240) 但其实 Pred:OFF）。
        self._update_accordion_counts(img_count=len(self._all_images))

        # 标记相关：筛选条件回显 + 统计面板灌数据（统计用全量图，与筛选互不影响）
        self._sync_mark_filter_label()
        try:
            self.mark_stats.set_records(self._mark_stats_records())
        except Exception as e:
            self.log(f"标记统计刷新失败: {type(e).__name__}: {e}")

        self._suppress_cell_change = False
        # 记录刚被整批重建：信息栏的记录缓存作废，并跟着刷新一次
        # （「已标注 / 已检出 / 耗时」三项就是从这里那条记录取的）
        self._img_info_rec_cache = None
        self._refresh_img_info()
        # ⚠️ 这里**不能** self.log()（日志面板会重入，撑爆 C 栈，见上方注释）。
        #    耗时只存属性，由调用方（_populate_image_list）在安全位置打出来。
        self._last_filter_ms = (time.perf_counter() - _t_fi0) * 1000.0
        self._last_filter_rows = len(self._filtered_data or [])

    def _render_page(self):
        """Render current page from _filtered_data into the table."""
        self._suppress_cell_change = True
        self.image_list_widget.setRowCount(0)

        total = len(self._filtered_data)
        total_pages = max(1, (total + self._page_size - 1) // self._page_size)
        self._current_page = max(0, min(self._current_page, total_pages - 1))

        start = self._current_page * self._page_size
        end = min(start + self._page_size, total)

        for i, data in enumerate(self._filtered_data[start:end]):
            row = self.image_list_widget.rowCount()
            self.image_list_widget.insertRow(row)
            global_idx = start + i
            if getattr(self, "_mixed_cls_mode", False) and data.get('height_fname'):
                name_text = data['fname'] + '\n' + data['height_fname']
            elif data.get('view_fnames'):
                name_text = '\n'.join(data['view_fnames'])     # 多视图：N 行文件名
            else:
                name_text = data['fname']
            items_data = [
                (str(global_idx + 1), True),
                (name_text, False),
                (f"{data['img_w']}x{data['img_h']}" if data['img_w'] else "", True),
                (data['split_status'].replace("[", "").replace("]", "").strip(), True),
                (", ".join(data['shape_labels']) if self._cls_mode else (str(data['shape_count']) if data['shape_count'] > 0 else ""), True),
                (data['cls_text'] if self._cls_mode else (data['ocr_text'] if self._ocr_mode else (str(data['pred_cnt']) if data['pred_cnt'] > 0 else "")), True),
                (data['ocr_score'] if self._ocr_mode else data['score_str'], True),
                (data['ocr_time'] if self._ocr_mode else (str(data['infer_ms']) if data['infer_ms'] else ""), True),
                # 第 9 列 = 标记。⚠️ 必须在这里一起建格子：只 setColumnCount(9) 而下面
                #    再 item(row, 8) 取回来是 **None**（单元格从没被 insert 过），
                #    标记列会永远空白 —— 这是真窗口探针抓出来的真 bug（AST 核验看不见）。
                (marks_mod.kind_label(data.get('mark') or "") or "", True),
            ]
            for col, (val, center) in enumerate(items_data):
                item = QTableWidgetItem(val)
                if center:
                    item.setTextAlignment(Qt.AlignCenter)
                self.image_list_widget.setItem(row, col, item)
            # Store path in first column
            self.image_list_widget.item(row, 0).setData(Qt.UserRole, data['path'])
            # 标记列（第 8 列）：文字 + 该格底色 —— 行色标一眼能扫出来
            mkind = data.get('mark') or ""
            mit = self.image_list_widget.item(row, 8)
            if mit is not None:
                mit.setText(marks_mod.kind_label(mkind) or "")
                mit.setTextAlignment(Qt.AlignCenter)
                if mkind:
                    # 只给这一个格子上底色（不改整行）：整行底色会和交替行色/主题表打架，
                    # 而标记是"这一格的属性"，格子底色反而更好定位。
                    mit.setBackground(QColor(marks_mod.kind_color(mkind)))
                    mit.setForeground(QColor("#FFFFFF"))
                    tip = marks_mod.MARK_HINTS.get(mkind, "")
                    if data.get('mark_note'):
                        tip += "\n备注：" + str(data['mark_note'])
                    if data.get('mark_by'):
                        tip += "\n标记人：%s（%s）" % (data['mark_by'], data.get('mark_at', ''))
                    mit.setToolTip(tip)
                    # ⚠️ 标记列在最右侧：左栏默认宽度装不下 9 列，人**不横向滚动就看不见它**
                    #    （真窗口截图确认过）。序号列永远在可视区里，同步上同色底，
                    #    这样"扫一眼看出哪些图有标记"在不滚动的默认视野里也成立。
                    sit = self.image_list_widget.item(row, 0)
                    if sit is not None:
                        sit.setBackground(QColor(marks_mod.kind_color(mkind)))
                        sit.setForeground(QColor("#FFFFFF"))
                        sit.setToolTip(marks_mod.MARK_LABELS.get(mkind, "") +
                                       (("｜" + data['mark_note']) if data.get('mark_note') else ""))
            # 默认整行白色；「标注」「结果」两列的类别名用 Class 表对应的类别颜色
            if getattr(self, "_mixed_cls_mode", False) and data.get('height_fname'):
                self.image_list_widget.setRowHeight(row, 40)
            elif data.get('view_fnames'):
                self.image_list_widget.setRowHeight(row, 16 * len(data['view_fnames']) + 10)
            # ⚠️ 这里**不能**再给整行 setForeground(QColor(255,255,255))：
            #    内联前景色的优先级**高于应用级 QSS**，主题表改不动它 ——
            #    切浅色主题后就是"白字压白底"，整个列表一行都看不见
            #    （用户实测反馈：「换成浅色主题后包括图片列表等地方都看不清文字」）。
            #    默认行色改由主题表按 objectName=imageList 下发：
            #      深色主题 base 表给 #FFFFFF（与原来逐像素一致），浅色主题给深字；
            #    这样**切换主题时 QSS 立即生效**，不需要重刷列表。
            #    类别色仍走 _apply_class_color()（那是语义色，必须随类别走，保留内联）。
            if self._cls_mode:
                labels = getattr(self, "_cached_classes", []) or []
                # 标注列（第 4 列）
                if data.get('shape_labels'):
                    self._apply_class_color(row, 4, data['shape_labels'][0], labels)
                # 结果列（第 5 列）
                if data.get('cls_text'):
                    self._apply_class_color(row, 5, data['cls_text'], labels)

        # 自适应「图片名」列宽：长文件名/多视图多行文件名不再被截断，
        # 用户不需要手动拖列宽（多视图下这一列有 N 行文件名，尤其容易挤爆）。
        try:
            hdr = self.image_list_widget.horizontalHeader()
            self.image_list_widget.resizeColumnToContents(1)
            w = self.image_list_widget.columnWidth(1)
            lo = 180 if getattr(self, "_mv_mode_list", False) else 130
            self.image_list_widget.setColumnWidth(1, int(max(lo, min(w + 18, 520))))
            hdr.setSectionResizeMode(1, QHeaderView.Interactive)   # 仍可手动调
        except Exception:
            pass

        # Update pagination controls
        if getattr(self, "_page_size", 50) >= 10 ** 8:
            # 不分页 = 整批一页装下。这里不再报行数：左边那行已经写了"共 N 组（M 张）"，
            # 是同一个数字的第二遍；而且窄面板下会被截成"全部 3"（2026-10-14 用户截图
            # 上就是这个样子）。只保留"没分页"这个状态提示。
            self.page_info_label.setText("全部")
        else:
            self.page_info_label.setText(f"Page {self._current_page + 1}/{total_pages}")
        self.page_prev_btn.setEnabled(self._current_page > 0)
        self.page_next_btn.setEnabled(self._current_page < total_pages - 1)

        # Re-select the image currently being annotated so the user always
        # knows which row they are working on (even after list refresh).
        cur = getattr(self.canvas, "current_image_path", None)
        if cur:
            for r in range(self.image_list_widget.rowCount()):
                it = self.image_list_widget.item(r, 0)
                if it and it.data(Qt.UserRole) == cur:
                    self.image_list_widget.selectRow(r)
                    self.image_list_widget.scrollToItem(
                        it, self.image_list_widget.PositionAtCenter)
                    break

        self._suppress_cell_change = False
        # 行数变了 ⇒ 表格该多高也跟着变（行少时收成内容高度，不留大片空白）
        self._schedule_fit_list_height()

    def _apply_class_color(self, row, col, class_name, labels):
        """给某单元格按类别名着色为 Class 表对应的类别颜色。"""
        try:
            cid = labels.index(class_name)
        except ValueError:
            return
        if not (0 <= cid < len(labels)):
            return
        it = self.image_list_widget.item(row, col)
        if it is not None:
            it.setForeground(QColor(*CLASS_COLORS[cid % len(CLASS_COLORS)]))

    def _go_to_page(self, page):
        """Navigate to a specific page."""
        total = len(self._filtered_data)
        total_pages = max(1, (total + self._page_size - 1) // self._page_size)
        self._current_page = max(0, min(page, total_pages - 1))
        self._render_page()

    def _on_page_size_changed(self, text):
        """Handle page size change. 「全部」= 不分页。"""
        t = (text or "").strip()
        if t in ("全部", "All", "all", "所有"):
            self._page_size = 10 ** 9          # 不分页：一次渲染全部
        else:
            try:
                self._page_size = max(1, int(t))
            except ValueError:
                self._page_size = 50
        self._current_page = 0
        self._render_page()

    def _update_selection_count(self):
        """Update list_status with current selection count."""
        selected = len(set(idx.row() for idx in self.image_list_widget.selectedIndexes()))
        current_text = self.list_status.text()
        if " | Selected:" in current_text:
            current_text = current_text[:current_text.rfind(" | Selected:")]
        if selected > 0:
            current_text += f" | Selected: {selected}"
        self.list_status.setText(current_text)
        try:
            # 标记按钮的作用域是"当前选中的行"，选中数必须让用户看得见
            # （点了没反应的头号原因是没选中，而不是按钮坏了）
            self.mark_toolbar.set_scope_text(selected)
        except Exception:
            pass

    def _on_current_cell_changed(self, currentRow, currentColumn, previousRow, previousColumn):
        """Handle mouse click and keyboard navigation in image list."""
        if getattr(self, "_suppress_cell_change", False):
            return
        if currentRow >= 0 and currentRow != previousRow:
            item = self.image_list_widget.item(currentRow, 0)
            path = item.data(Qt.UserRole) if item else None
            if path and os.path.exists(path):
                self._load_image_by_index(currentRow)
        self._update_selection_count()


    def eventFilter(self, obj, event):
        """Catch Space key in classification mode before child widgets consume it."""
        from PyQt5.QtCore import QEvent
        if (event.type() == QEvent.KeyPress 
                and event.key() == Qt.Key_Space 
                and not event.isAutoRepeat()
                and getattr(self, "_cls_mode", False)):
            self._toggle_heatmap()
            return True
        return super().eventFilter(obj, event)

    def _toggle_heatmap(self):
        """Toggle Grad-CAM heatmap overlay (called by Space shortcut)."""
        from PyQt5.QtGui import QImage
        if not getattr(self, "_cls_mode", False):
            return
        current_path = getattr(self.canvas, "current_image_path", "")
        if not current_path or not self.current_project:
            return
        base = os.path.splitext(os.path.basename(current_path))[0]
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        hm_file = os.path.join(pd, "outputs", "classification", "heatmaps", base + "_heatmap.jpg")
        if self.canvas._show_heatmap:
            self.canvas._show_heatmap = False
            self.canvas.update()
            self.statusBar().showMessage("Heatmap: OFF", 2000)
        elif os.path.exists(hm_file):
            qimg = QImage(hm_file)
            if not qimg.isNull():
                iw, ih = self.canvas.image.size
                if qimg.width() != iw or qimg.height() != ih:
                    qimg = qimg.scaled(iw, ih)
                self.canvas._heatmap_qimage = qimg.copy()
                self.canvas._show_heatmap = True
                self.canvas.update()
                self.statusBar().showMessage("Heatmap: ON", 2000)
            else:
                self.statusBar().showMessage("Heatmap: failed to load", 2000)
        else:
            self.statusBar().showMessage("No heatmap file: " + os.path.basename(hm_file), 2000)

    def _auto_load_heatmap_for_current(self):
        """When heatmap display mode is ON, load heatmap for the current image.

        Called after switching images so the heatmap view persists across images.
        """
        from PyQt5.QtGui import QImage
        if not getattr(self, "_cls_mode", False):
            return
        current_path = getattr(self.canvas, "current_image_path", "")
        if not current_path or not self.current_project:
            return
        base = os.path.splitext(os.path.basename(current_path))[0]
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        hm_file = os.path.join(pd, "outputs", "classification", "heatmaps", base + "_heatmap.jpg")
        if not os.path.exists(hm_file):
            self.statusBar().showMessage("No heatmap for: " + os.path.basename(current_path), 2000)
            return
        qimg = QImage(hm_file)
        if qimg.isNull():
            return
        iw, ih = self.canvas.image.size
        if qimg.width() != iw or qimg.height() != ih:
            qimg = qimg.scaled(iw, ih)
        self.canvas._heatmap_qimage = qimg.copy()
        self.canvas._show_heatmap = True
        self.canvas.update()
        self.statusBar().showMessage("Heatmap: ON", 2000)


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
            self._fit_stat_columns(self.stats_table, headers, stretch_col=1)
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
                    # 列宽按**内容**重算一次：上面那次只按表头算，而「平均灰度」装的是三段并列值
                    self._fit_stat_columns(self.stats_table, headers, stretch_col=1)
            return

        # --- Segmentation mode (original) ---
        headers = ["序号", "类别", "分数", "面积", "宽度", "高度", "周长", "半径", "平均灰度", "交并比"]
        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setHorizontalHeaderLabels(headers)
        self._fit_stat_columns(self.stats_table, headers, stretch_col=1)
        self.stats_table.setRowCount(0)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)

        arr = pred_mask if pred_mask is not None else getattr(self.canvas, "_prediction_mask", None)
        mask = ann_mask if ann_mask is not None else (self.canvas.mask if self.canvas.image is not None else None)
        if arr is None:
            return
        pred_classes = getattr(self.canvas, "_pred_classes", None)

        import cv2 as _cv2
        import numpy as _np
        img_gray = getattr(self.canvas, "_cached_gray", None)

        # ⚠️ 现在**按连通域逐行**（以前是"一个类别一行"把整类像素合并统计）：
        #    推理参数里的"排序 / 最大检测个数"是按**块**定义的，一个类别一行的表
        #    根本体现不出来；友商的结果表也是逐块的。宽度/高度两列同时给出，
        #    方便直接对照推理参数里的宽度/高度区间。
        # ★★ 标注掩膜的 unique / 分类掩膜**提到循环外**只算一次（2026-10-22 卡顿）
        #   原来在 39 个连通域的循环里，每个都跑一次 np.unique(20MP mask) + 每类 (mask==ac)
        #   → py-spy 采样：np.unique 独占 54%，_update_stats 占 90%+ 主线程时间
        # ★★ 统计全部在**显示尺寸**（≤2048）上算（2026-10-22 选图卡顿）
        #   39 个连通域 × 20MP 的 (lab==i)/findContours/灰度，一次选图要好几秒。
        #   统计只用于填右侧表给人看，按 1/2.7 缩放算够用；面积/宽高再按 _k 还原。
        _k = 1.0
        try:
            _cq = getattr(self.canvas, "_cached_qimage", None)
            if _cq is not None and arr is not None:
                _dw, _dh = _cq.width(), _cq.height()
                if tuple(arr.shape[:2]) != (_dh, _dw):
                    _k = float(arr.shape[1]) / float(_dw)
                    arr = resize_mask_u(arr, (_dw, _dh))
                    if mask is not None:
                        mask = resize_mask_u(mask, (_dw, _dh))
        except Exception:
            _k = 1.0

        _ann_cls = []
        _ann_bin = {}
        if mask is not None and getattr(mask, "shape", None) == getattr(arr, "shape", None):
            _ann_cls = [int(v) for v in (set(_np.unique(mask).tolist()) - {0, 255})]
            for _ac in _ann_cls:
                _ann_bin[_ac] = (mask == _ac)

        # ★★ 2026-10-22：置信度图要在**循环外**取一次。
        #   以前循环里直接读 `canvas._pred_prob` —— 那是**单张推理**才赋值的；
        #   批量推理走 flow，从不写这个属性 ⇒ 批量跑完 Result 表「分数」整列 0.0%
        #   （用户实拍："推理结果应该都有分数的吧"）。
        #   现在退化顺序：内存概率图 → outputs/<base>_prob.png（批量推理产物）。
        _prob_map = getattr(self.canvas, "_pred_prob", None)
        if _prob_map is None or getattr(_prob_map, "shape", None) != arr.shape:
            _lp = self._load_prob_for_current(arr.shape[:2])
            if _lp is not None:
                _prob_map = _lp

        rows = []
        for cid in sorted(set(_np.unique(arr).tolist()) - {0, 255}):
            cls_name = (pred_classes[int(cid)] if pred_classes and int(cid) < len(pred_classes)
                        else "Class%d" % int(cid))
            binary = (arr == cid).astype(_np.uint8)
            n_lab, lab, stats, _cents = _cv2.connectedComponentsWithStats(binary, 8)
            for i in range(1, n_lab):
                x, y, bw, bh, area = (int(v) for v in stats[i])
                # 缩放过的话，把几何量还原成原图尺度（_k≥1）
                if _k > 1.0:
                    x, y = int(round(x * _k)), int(round(y * _k))
                    bw, bh = int(round(bw * _k)), int(round(bh * _k))
                    area = int(round(area * _k * _k))
                if area <= 0:
                    continue
                region = (lab == i)
                cnts, _ = _cv2.findContours((region.astype(_np.uint8) * 255),
                                            _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
                perimeter = sum(int(_cv2.arcLength(c, True)) for c in cnts)
                # ★ 2026-10-23：**别 int() 截断**。等效圆半径 sqrt(area/π) 保留 1 位小数，
                #   截断会把 19.9 显示成 19、小块相对误差可达 5%。
                #   公式保持不变 —— 推理参数里的「半径」过滤用的就是这个口径，
                #   换成外接圆会悄悄改变用户已调好的阈值语义。
                radius = float(_np.sqrt(area / _np.pi))
                avg_gray = 0.0
                if img_gray is not None and img_gray.shape[:2] == region.shape[:2]:
                    try:
                        avg_gray = float(img_gray[region].mean())
                    except IndexError:
                        avg_gray = 0.0
                # 三视图工程：同一块在**三张图**上的灰度并列显示（"，"隔开）
                gray_text = self._gray_cell_text(region, avg_gray)
                # IoU: 与标注里"最匹配的那个类"求交并比
                iou_val = 0.0
                if _ann_bin:
                    best = 0.0
                    # 只在**当前连通域的包围盒**里算（原来每次都是整幅 20MP 与/或）
                    _sub = region[y:y + bh, x:x + bw]
                    for ac, annb in _ann_bin.items():
                        _ann_sub = annb[y:y + bh, x:x + bw]
                        inter = (_sub & _ann_sub).sum()
                        uni = (_sub | _ann_sub).sum()
                        if uni > 0:
                            ciou = inter / uni
                            if ciou > best:
                                best = ciou
                    iou_val = best
                # 分数：优先用**模型置信度**（概率图里该块的平均值），
                # 没有概率图时退回"与标注的 IoU"（并在表头体现不出来，只能靠数值判断）
                prob_mean = None
                prob_map = _prob_map
                if prob_map is not None and getattr(prob_map, "shape", None) == region.shape:
                    try:
                        prob_mean = float(prob_map[region].mean())
                    except Exception:
                        prob_mean = None
                rows.append((cls_name, iou_val, area, bw, bh, perimeter, radius, gray_text,
                             iou_val, prob_mean))

        # 排序：跟随推理参数面板里的"排序"（默认面积降序）
        try:
            how = ((self.infer_params.current_params() or {}).get("global") or {}).get("sort", "面积降序")
        except Exception:
            how = "面积降序"
        if how == "面积升序":
            rows.sort(key=lambda r: r[2])
        elif how == "分数降序":
            rows.sort(key=lambda r: -r[1])
        elif how == "宽度降序":
            rows.sort(key=lambda r: -r[3])
        elif how != "无":
            rows.sort(key=lambda r: -r[2])

        self.stats_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            (cls_name, sc, area, bw, bh, perim, rad, gray, iou, prob_mean) = r
            sc_txt = ("%.1f%%" % (prob_mean * 100)) if prob_mean is not None else ("%.1f%%" % (sc * 100))
            items = [str(i + 1), cls_name, sc_txt, str(area), str(bw), str(bh),
                     str(perim), ("%.1f" % rad), gray, "%.3f" % iou]
            for j, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.stats_table.setItem(i, j, item)
        # ★★ 列宽只在**所有行填完之后**算一次（2026-10-22 白屏转圈真凶）
        #   原来它被写在逐行循环里面 ⇒ 每填一行就把整张表重算一遍：
        #   39 行 → 39 × (39 行 × 10 列读 item + 10 次 setSectionResizeMode/setColumnWidth)
        #   ≈ 15000 次 item 访问 + 390 次重排 ⇒ 主线程被占死，界面白屏转圈。
        #   （py-spy dump 抓到的现场就是 MainThread 停在 _fit_stat_columns 里）
        self._fit_stat_columns(self.stats_table, headers, stretch_col=1)
        # ★ 2026-10-22：Result(n) 用**当前这张图**的块数（原来用整个列表的 total_shapes）
        try:
            self._update_block_counts(result_count=len(rows))
        except Exception:
            pass

    # ────────── 统计表的列宽策略（三张表共用） ──────────
    def _fit_stat_columns(self, table, headers, stretch_col=None, min_extra=10):
        """按"**内容 + 表头谁宽用谁**"定列宽，只有名称列吃剩余宽度。

        为什么不能再"全部 Stretch"（原实现）：
          Result 表 10 列挤在右栏 ~590px ⇒ 每列才59px，而「平均灰度」要装
          `114.5，188.2，68.3` 这种三视图并列值（约 120px）⇒ **被截成"114.5，…"**，
          功能等于白做；表头"平均灰度/交并比"四个字本身也放不下。
        为什么用 ResizeToContents 而不是"按表头文字宽 + 余量"：
          **只按表头定宽会漏掉内容** —— 「平均灰度」表头 4 字约 58px 就够，
          但单元格里是三段并列值（120px+）⇒ 照样被截。ResizeToContents 同时看表头与单元格。
        stretch_col（类别名）用 Stretch 吃余量；列宽总和超过表格宽度时 Qt 自动出横向
        滚动条 —— **可滚动好过被截断**。
        """
        from PyQt5.QtGui import QFontMetrics
        hdr = table.horizontalHeader()
        fm = QFontMetrics(hdr.font())
        hdr.setMinimumSectionSize(max(24, fm.horizontalAdvance("序号") + 8))
        for i, text in enumerate(headers):
            if stretch_col is not None and i == stretch_col:
                hdr.setSectionResizeMode(i, QHeaderView.Stretch)
            else:
                # Interactive + 手动给"内容宽"：ResizeToContents 在每次填数据后都要
                # 重算全部行，89 行 Result 表在切图时会明显卡；这里按当前内容算一次即可。
                need = fm.horizontalAdvance(str(text)) + min_extra
                for r in range(table.rowCount()):
                    it = table.item(r, i)
                    if it is not None:
                        need = max(need, fm.horizontalAdvance(it.text()) + min_extra)
                hdr.setSectionResizeMode(i, QHeaderView.Interactive)
                table.setColumnWidth(i, int(need))

    # ────────── 「平均灰度」：三视图工程按视图并列（2026-10-15）──────────
    def _view_paths_of_current_image(self):
        """当前图所在**组**的全部视图路径（单视图工程 = 只有当前这一张）。

        数据源：`_filtered_data` 每条记录带 `view_paths`（第 0 个是主标注视图）。
        为什么不在这里现去 glob：同一组的三视图名是 `xxx_te_0/1/2`，
        现 glob 会把"别的组同名前缀"也捞进来。
        """
        path = getattr(self.canvas, "current_image_path", "") or ""
        if not path:
            return []
        base = os.path.splitext(os.path.basename(path))[0]
        for r in getattr(self, "_filtered_data", []) or []:
            vps = list(r.get("view_paths") or [])
            if not vps:
                continue
            for vp in vps:
                if os.path.splitext(os.path.basename(vp))[0] == base:
                    return vps
            # 画布拿到的可能是组名而不是某一视图名（如 xxx 对应 xxx_te_0）
            gpre = base.split("_te_")[0]
            if any(os.path.splitext(os.path.basename(v))[0].split("_te_")[0] == gpre
                   for v in vps):
                return vps
        return [path]

    def _load_gray_cached(self, path):
        """读一张图的灰度图（np.float64），**按 (路径, mtime, 大小) 缓存**。

        为什么必须缓存：这两张统计表在切图/改标注时都会刷新，而三视图工程
        一次刷新要读 3 张 1160×3706 的图 —— 不缓存就是每刷一次读 3 张全尺寸图，
        页面上会明显卡。缓存键带 mtime/size ⇒ 用户替换了同名图也不会读到旧的。
        """
        import numpy as _np
        try:
            st = os.stat(path)
            key = (path, int(st.st_mtime), int(st.st_size))
        except Exception:
            return None
        cache = getattr(self, "_gray_cache", None)
        if cache is None:
            cache = {}
            self._gray_cache = cache
        hit = cache.get(key)
        if hit is not None:
            return hit
        try:
            from PIL import Image
            arr = _np.array(Image.open(path).convert("RGB"))
            # ⚠️ 口径必须与 `mask_editor._cached_gray` **完全一致**：
            #    那里是 `np.mean(arr, axis=2)`（三通道等权平均），不是 PIL 的 `convert("L")`
            #    （那是 ITU-R 601-2 亮度加权）。两处口径不同 ⇒ 同一块区域算出的灰度
            #    差几级，**用户会拿着这一列的数字去卡灰度差阈值**（现场标准是 15-20），
            #    口径一漂，阈值就废了。所以这里也走等权平均。
            arr = (_np.mean(arr.astype(_np.float64), axis=2)
                   if arr.ndim == 3 else arr.astype(_np.float64))
        except Exception:
            return None
        if len(cache) > 12:                      # 简单上限，别让缓存无脑长
            cache.clear()
        cache[key] = arr
        return arr

    def _load_prob_for_current(self, ref_shape):
        """读当前图对应的**置信度图**（批量推理保存的 `outputs/<base>_prob.png`）。

        为什么需要：`canvas._pred_prob` 只在**单张推理**时被赋值；批量推理走 flow，
        结果只落盘（mask/overlay），内存里没有概率图 ⇒ Result 表「分数」退化成 IoU，
        没标注的图上就整列 0.0%（用户实拍："推理结果应该都有分数的吧"）。

        `_prob.png` 按 uint8 存的（概率 ×255），这里还原成 0~1 并缩放到统计用的显示尺寸。
        """
        try:
            import numpy as _np
            from PIL import Image as _Image
            cur = getattr(self.canvas, "current_image_path", "") or ""
            if not cur or not self.current_project:
                return None
            pd_ = str(self.pm.get_project_dir(self.current_project["name"]))
            base = os.path.splitext(os.path.basename(cur))[0]
            p = os.path.join(pd_, "outputs", base + "_prob.png")
            if not os.path.exists(p):
                return None
            arr = _np.array(_Image.open(p))
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.shape[:2] != tuple(ref_shape):
                arr = _np.array(_Image.fromarray(arr).resize(
                    (int(ref_shape[1]), int(ref_shape[0])), _Image.BILINEAR))
            return arr.astype(_np.float32) / 255.0
        except Exception:
            return None

    def _sync_mask_png_for_current(self):
        """把当前图的标注掩膜同步写回 `annotations/<base>_mask.png`。

        JSON 是源、mask.png 是派生产物。两者不同步的后果很隐蔽：
        训练读 JSON 所以看不出问题，但评估/统计读 mask.png 会拿到**缺了缺陷**的真值。
        """
        try:
            import numpy as _np
            from PIL import Image as _I
            cur = getattr(self.canvas, "current_image_path", "") or ""
            pd_ = str(self._project_dir or "")
            m = getattr(self.canvas, "mask", None)
            if not (cur and pd_ and m is not None):
                return
            img = getattr(self.canvas, "image", None)
            if img is not None and (m.shape[0], m.shape[1]) != (img.size[1], img.size[0]):
                return
            ann_dir = os.path.join(pd_, "annotations")
            os.makedirs(ann_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(cur))[0]
            _I.fromarray(_np.ascontiguousarray(m).astype(_np.uint8)).save(
                os.path.join(ann_dir, base + "_mask.png"))
        except Exception as e:
            self.log(f"[掩膜同步] 失败（不影响 JSON 已保存）: {type(e).__name__}: {e}")

    def _gray_cell_text(self, region, avg_gray):
        """两张统计表共用的「平均灰度」单元格文本。

        ★★ 2026-10-22 用户实测反馈：**20MP 大图上整列显示「—」**。
           原因是 `_gray_text_for_region()` 在"视图尺寸与 region 不一致"时返回的是
           字符串 `"—"`（不是空串），而调用处写的是 `if not gray_text.strip()` ——
           `"—"` 非空 ⇒ 回退永远不触发 ⇒ 明明 `avg_gray` 已经按显示尺寸的灰度算好了，
           却整列不显示。

           判空必须把 `"—"` 也算进去。两张表都走这里，避免又一次各写一套判断。
        """
        txt = self._gray_text_for_region(region)
        if not txt or not txt.strip() or txt == "—":
            return "%.1f" % (avg_gray or 0.0)
        return txt

    def _gray_text_for_region(self, region, view_paths=None):
        """一个区域在各视图上的平均灰度，**用「，」并列**（单视图就一个值）。

        · 区域（region）来自当前视图的掩膜/预测；同一组三视图分辨率一致，
          所以**同一个 region 直接套到另外两张图上**算灰度（这正是用户要的：
          同一块缺陷在漫反射 / 形状图1 / 形状图2 上各是多少灰）。
        · 尺寸对不上的视图**跳过**而不是填 0 —— 填 0 会被误读成"纯黑区域"。
        · 尺寸对不上时返回单值（退化成从前行为），不抛错。

        ★★ 2026-10-22 切图卡顿：**先用文件头判断尺寸，别急着解码**。
           统计表是在**显示尺寸**（≤2048）上算 region 的，而这些视图是原图尺寸
           （现场那张是 4096×9320）⇒ 尺寸必然对不上、每个视图都会被 skip。
           旧代码是"先 `_load_gray_cached()` 解码 38MB 图（实测 0.6s）→ 再发现
           形状不匹配 → 丢掉"，一次切图白扔 0.6 秒。
           `Image.open(...).size` 只读文件头（毫秒级），先比尺寸、对上了才解码。
        """
        import numpy as _np
        if view_paths is None:
            view_paths = self._view_paths_of_current_image()
        vals = []
        ref_shape = region.shape[:2]
        for p in view_paths:
            # ① 只读文件头拿尺寸（不解码）
            try:
                from PIL import Image as _Image
                with _Image.open(p) as _im:
                    _sz = _im.size
                if (_sz[1], _sz[0]) != ref_shape:
                    continue
            except Exception:
                continue
            # ② 尺寸对得上才真正解码
            g = self._load_gray_cached(p)
            if g is None or g.shape[:2] != ref_shape:
                continue
            try:
                vals.append(float(g[region].mean()))
            except Exception:
                continue
        if not vals:
            return "—"
        return "，".join("%.1f" % v for v in vals)

    def _update_ann_stats(self):
        """Update per-annotation-block statistics table (segmentation or detection)."""
        # --- Detection mode: show annotation boxes from BoxManager ---
        if self._detection_mode and self._box_manager is not None:
            bm = self._box_manager
            headers = ["序号", "类别", "面积", "宽度", "高度", "x1", "y1"]
            self.ann_table.setColumnCount(len(headers))
            self.ann_table.setHorizontalHeaderLabels(headers)
            self._fit_stat_columns(self.ann_table, headers, stretch_col=1)
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
                    # 列宽按**内容**重算一次：上面那次只按表头算，而「平均灰度」装的是三段并列值
                    self._fit_stat_columns(self.ann_table, headers, stretch_col=1)
            return

        # --- Segmentation mode (original) ---
        headers = ["序号", "类别", "面积", "周长", "半径", "平均灰度"]
        self.ann_table.setColumnCount(len(headers))
        self.ann_table.setHorizontalHeaderLabels(headers)
        self._fit_stat_columns(self.ann_table, headers, stretch_col=1)
        self.ann_table.setRowCount(0)
        self.ann_table.setEditTriggers(QTableWidget.NoEditTriggers)

        mask = self.canvas.mask if self.canvas.image is not None else None
        if mask is None:
            return
        import numpy as _np
        import cv2

        classes = (getattr(self, "_cached_classes", None) or ["background"])
        h, w = mask.shape[:2]
        rows = []

        # ★★ 2026-10-22 修「标了 40 个 Hy，Annotation 表只有 1 行」：
        #    原来这里是 `for cid in sorted(set(np.unique(mask)) - {0, 255})`
        #    —— **一个类别一行**，把同类 40 块的面积/周长**求和**成一行
        #    （实测那行 面积 228936 = 40 块之和，半径 269 也是按总面积反推的，
        #     一块缺陷的真实尺寸全被抹掉了）。而「已标注 40」是 JSON 里的 shape 数，
        #    两边口径不同 ⇒ 数字对不上，用户以为是丢标注。
        #
        #    现在逐**块**出行，且优先用 labelme JSON 的 shape 列表：
        #    图片列表「标注」列、底栏「已标注 N」都是 JSON shape 数
        #    （见 _filter_images 的 shape_count），表格必须与它们**同源**才可能永远相等。
        #    没有 JSON 时（只有掩膜的老工程）退回"一个连通域一行"，
        #    与右侧 Result 表的口径一致。
        shapes = self._current_json_shapes()
        if shapes:
            rows = self._ann_rows_from_shapes(shapes, mask, classes)
        if not rows:
            rows = self._ann_rows_from_mask(mask, classes)

        self.ann_table.setRowCount(len(rows))
        for i, (cls_name, area, perim, rad, gray) in enumerate(rows):
            items = [str(i + 1), cls_name, str(area), str(perim), ("%.1f" % rad), gray]
            for j, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.ann_table.setItem(i, j, item)
        # ★ 列宽只在**所有行填完**之后算一次（逐行重算会把主线程占死，见 Result 表那次白屏）
        self._fit_stat_columns(self.ann_table, headers, stretch_col=1)
        # 表头计数直接取行数：表格与标题再也不可能各说各话
        try:
            self._update_block_counts(ann_count=len(rows))
        except Exception:
            pass

    # ────────── 标注统计的两种取数来源 ──────────

    @staticmethod
    def _is_real_shape(sh):
        """算一条真实标注：跳过 background / ignore / 空 label。"""
        lab = str((sh or {}).get("label", "") or "")
        if not lab or lab == "background":
            return False
        low = lab.lower()
        if "ignore" in low or "gnore" in low:
            return False
        return True

    def _cached_json_labels(self, json_path):
        """json 里**非 ignore / 非 background** 的标签列表（每个 shape 一项，含重复）。

        ★★ 2026-10-28：改走 `core.label_scan`（bytes 正则 + **进程级共享缓存**）。
           这里原来是自建缓存 + `json.load` 整个文件，问题在于那份缓存是**本窗口私有**的：
           首次打开工程（或换工程后缓存还是空的）时，489 个 JSON 仍要全解析一遍
           —— 实测 `_filter_images` 里这一段就吃 **4641ms**。
           而这些 JSON 合计 **1GB**（单个最大 18MB，内容 99.9% 是 shape 的顶点坐标），
           本函数只需要"有哪些 label"。
           现在与 `discover_labels()`、`ClassTableView._seg_class_ids()` 共用同一份缓存，
           **整个工程只真扫一次**，其余全部命中。
        """
        from core.label_scan import scan_labels
        info = scan_labels(json_path)
        if info is None:
            return []
        out = []
        for lbl in info[0]:
            low = str(lbl).lower()
            if lbl and lbl != "background" and "ignore" not in low and "gnore" not in low:
                out.append(lbl)
        return out

    def _current_json_shapes(self):
        """当前图 labelme JSON 里的真实 shape 列表（没有就返回 []）。"""
        try:
            from annotation.labelme_io import load_labelme_json, get_json_path
            cur = getattr(self.canvas, "current_image_path", "") or ""
            if not cur:
                return []
            jp = get_json_path(cur)
            if not jp or not os.path.exists(jp):
                return []
            # ★ 2026-10-23：按 (路径, mtime, size) 缓存。撤销/改标注时这张表会重算，
            #   而 4.7MP 工程的 json 有几十个 shape，光 json.load 就 45ms（实测 profile）。
            try:
                st = os.stat(jp)
                _k = (jp, int(st.st_mtime), int(st.st_size))
            except Exception:
                _k = None
            _cache = getattr(self, "_json_shapes_cache", None)
            if _cache is None:
                _cache = {}
                self._json_shapes_cache = _cache
            if _k is not None and _k in _cache:
                return _cache[_k]
            shapes, _info = load_labelme_json(jp)
            out = [s for s in (shapes or []) if self._is_real_shape(s)]
            if _k is not None:
                if len(_cache) > 8:
                    _cache.clear()
                _cache[_k] = out
            return out
        except Exception:
            return []

    def _gray_text_for_crop(self, x, y, sub, view_paths=None):
        """一块区域内各视图的平均灰度（按 bbox 裁切后取均值）。

        `_gray_text_for_region(region)` 要一整幅同尺寸的布尔图，逐块调用会为 40 块 ×
        3 视图各扫一遍整幅图（几千万像素，必卡）。这里只用 bbox 那一小块，
        语义不变：**尺寸对不上的视图跳过**，不填 0（填 0 会被误读成"纯黑区域"）。
        """
        if view_paths is None:
            view_paths = self._view_paths_of_current_image()
        vals = []
        for p in view_paths:
            g = self._load_gray_cached(p)
            if g is None:
                continue
            try:
                gs = g[y:y + sub.shape[0], x:x + sub.shape[1]]
                if gs.shape[:2] != sub.shape[:2] or not sub.any():
                    continue
                vals.append(float(gs[sub].mean()))
            except Exception:
                continue
        return "，".join("%.1f" % v for v in vals) if vals else "—"

    def _gray_from_display(self, x, y, sub):
        """大图（显示图已被降采样）时，按**显示图灰度**算一块的均值。

        为什么不去读原图：38MP 的原图解码要 0.6s，而这一列只是给人看的量级参考。
        显示图灰度就是 `canvas._cached_gray`（= np.mean(显示图 RGB)），
        与 `_update_stats` 里 avg_gray 用的是**同一份数据、同一套口径**。
        """
        import numpy as _np
        g = getattr(self.canvas, "_cached_gray", None)
        sc = float(getattr(self.canvas, "_cached_qimage_scale", 1.0) or 1.0)
        if g is None or sc <= 0 or sub is None or sub.size == 0:
            return None
        h, w = sub.shape[:2]
        x0, y0 = int(round(x * sc)), int(round(y * sc))
        wd, hd = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
        if x0 < 0 or y0 < 0 or y0 + hd > g.shape[0] or x0 + wd > g.shape[1]:
            return None
        try:
            from PIL import Image as _I
            subd = _np.array(_I.fromarray(sub.astype(_np.uint8)).resize(
                (wd, hd), _I.NEAREST)) > 0
            if not subd.any():
                return None
            return float(g[y0:y0 + hd, x0:x0 + wd][subd].mean())
        except Exception:
            return None

    def _ann_rows_from_shapes(self, shapes, mask, classes):
        """一个 shape 一行：面积/周长/半径/灰度 全部按**这一块**算。

        一次 `shapes_to_mask` 就把所有 shape 光栅化成**各自独立的编号**
        （临时类别名 `__s0/__s1/...`），所以只需要一次全图绘制，不用逐块重绘。
        重叠时后来的覆盖先前的 —— 与保存掩膜时的规则完全一致。
        """
        import numpy as _np
        import cv2
        from annotation.labelme_io import shapes_to_mask

        h, w = mask.shape[:2]
        tmp = []
        uniq = ["background"]
        for i, sh in enumerate(shapes):
            d = dict(sh)
            d["label"] = "__s%d" % i
            tmp.append(d)
            uniq.append("__s%d" % i)
        try:
            idmap = shapes_to_mask(tmp, uniq, (h, w))
        except Exception:
            return []
        if idmap is None:
            return []

        # bbox 一次算全：只扫一遍全图取非零像素，再按编号分组求 min/max。
        # ⚠️ **不能用 scipy.ndimage.find_objects**（跑了才知道 aitp_venv 里没装 scipy），
        #    也**不能逐块 np.nonzero(idmap == cid)** —— 那是每块扫一遍 6MP 全图，
        #    40 块 = 2.4 亿次比较，实测 0.45s，切一张图就卡一下。
        flat = idmap.reshape(-1)
        nz = _np.flatnonzero(flat)
        boxes = {}
        if nz.size:
            ids_flat = flat[nz]
            ys_flat, xs_flat = _np.divmod(nz, w)
            order = _np.argsort(ids_flat, kind="stable")
            ids_flat = ids_flat[order]
            ys_flat = ys_flat[order]
            xs_flat = xs_flat[order]
            bounds = _np.searchsorted(ids_flat, _np.arange(1, len(shapes) + 2))
            for i in range(len(shapes)):
                lo, hi = int(bounds[i]), int(bounds[i + 1])
                if hi > lo:
                    boxes[i + 1] = (int(xs_flat[lo:hi].min()), int(ys_flat[lo:hi].min()),
                                    int(xs_flat[lo:hi].max()) + 1, int(ys_flat[lo:hi].max()) + 1)

        rows = []
        for i, sh in enumerate(shapes):
            cid = i + 1
            box = boxes.get(cid)
            if box is None:
                # 点 / 线这类没有面积的 shape：仍然出一行（计数要跟"我标了几个"对上）
                rows.append((str(sh.get("label", "")), 0, 0, 0, "—"))
                continue
            x0, y0, x1, y1 = box
            sub = (idmap[y0:y1, x0:x1] == cid)
            area = int(sub.sum())
            if area <= 0:
                rows.append((str(sh.get("label", "")), 0, 0, 0, "—"))
                continue
            cnts, _ = cv2.findContours((sub.astype(_np.uint8) * 255),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = sum(int(cv2.arcLength(c, True)) for c in cnts)
            radius = float(_np.sqrt(area / _np.pi))
            gray = self._gray_text_for_crop(x0, y0, sub)
            if not gray or gray == "—":
                # 大图被降采样显示时，视图尺寸与这一块对不上 ⇒ 用显示图灰度兜底
                # （以前这里去解码原图全尺寸灰度，38MB 一张要 0.6s，纯属找死）
                gd = self._gray_from_display(x0, y0, sub)
                gray = ("%.1f" % gd) if gd is not None else "—"
            rows.append((str(sh.get("label", "")), area, perimeter, radius, gray))
        return rows

    def _ann_rows_from_mask(self, mask, classes):
        """没有 JSON 时的兜底：一个连通域一行（与 Result 表同口径）。"""
        import numpy as _np
        import cv2

        rows = []
        for cid in sorted(set(_np.unique(mask).tolist()) - {0, 255}):
            cls_name = classes[int(cid)] if int(cid) < len(classes) else "Class%d" % int(cid)
            binary = (mask == cid).astype(_np.uint8)
            n_lab, _lab, stats, _cents = cv2.connectedComponentsWithStats(binary, 8)
            for i in range(1, n_lab):
                x, y, bw, bh, area = (int(v) for v in stats[i])
                if area <= 0:
                    continue
                sub = (_lab[y:y + bh, x:x + bw] == i)
                cnts, _ = cv2.findContours((sub.astype(_np.uint8) * 255),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter = sum(int(cv2.arcLength(c, True)) for c in cnts)
                # ★ 2026-10-23：**别 int() 截断**。等效圆半径 sqrt(area/π) 保留 1 位小数，
                #   截断会把 19.9 显示成 19、小块相对误差可达 5%。
                #   公式保持不变 —— 推理参数里的「半径」过滤用的就是这个口径，
                #   换成外接圆会悄悄改变用户已调好的阈值语义。
                radius = float(_np.sqrt(area / _np.pi))
                gray = self._gray_text_for_crop(x, y, sub)
                if not gray or gray == "—":
                    gd = self._gray_from_display(x, y, sub)
                    gray = ("%.1f" % gd) if gd is not None else "—"
                rows.append((cls_name, area, perimeter, radius, gray or "—"))
        return rows


    def _toggle_detection_mode(self):
        """Toggle between segmentation and detection annotation modes."""
        self._detection_mode = not self._detection_mode
        self.canvas._mask_editing = not self._detection_mode
        if self._detection_mode:
            self.log("Switched to Detection mode")
            from detection.box_manager import BoxManager
            from detection.box_overlay import DetectionOverlay
            classes = getattr(self, "_cached_classes", None) or ["background"]
            self._box_manager = BoxManager(categories=classes)
            self.canvas._det_overlay = DetectionOverlay(self.canvas, self._box_manager)
            self.annotation_toolbar.set_mode("pan")
            self.canvas._mode = self.canvas.MODE_PAN
            self.det_toolbar.setVisible(True)
            self.det_toolbar.set_draw_active(True)
            self._load_det_annotations()
            self._sync_det_toolbar()
        else:
            self.log("Switched to Segmentation mode")
            self.canvas._det_overlay = None
            self.det_toolbar.setVisible(False)
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
            _mixed = getattr(self, "_mixed_cls_mode", False)
            _hp = ""
            if _mixed and hasattr(self, "_filtered_data"):
                _global_idx = getattr(self, "_current_page", 0) * getattr(self, "_page_size", 50) + idx
                if 0 <= _global_idx < len(self._filtered_data):
                    _hp = self._filtered_data[_global_idx].get("height_path", "")
            if path and os.path.exists(path):
                if _mixed:
                    self.log(f"[Mixed] Loading pair for: {os.path.basename(path)}")
                    self.mixed_viewer.set_pair(path, _hp if _hp else None)
                    self._viewer_stack.setCurrentIndex(1)
                else:
                    self.canvas.load_image(path)
                    self._viewer_stack.setCurrentIndex(0)
                    # 多视图：把这一组的其余视图并排显示（默认 N*1 横排）
                    _vp = []
                    if hasattr(self, "_filtered_data") and self._filtered_data:
                        _gi = (getattr(self, "_current_page", 0) * getattr(self, "_page_size", 50) + idx)
                        if 0 <= _gi < len(self._filtered_data):
                            _vp = self._filtered_data[_gi].get("view_paths") or []
                    if len(_vp) > 1:
                        _meta = getattr(self, "_mv_view_meta", []) or []
                        _items = []
                        for _k, _p in enumerate(_vp):
                            _nm = _meta[_k][1] if _k < len(_meta) else os.path.basename(_p)
                            _items.append((_p, _nm))
                        self.mv_preview.set_views(_items)
                    else:
                        self.mv_preview.clear_views()
                    # Keep heatmap display state when switching images
                    if getattr(self.canvas, "_show_heatmap", False):
                        self._auto_load_heatmap_for_current()
                self._update_height_controls()
                self._refresh_versions()
                # 这里原来往**临时**状态栏打一条 "Image: xxx.jpg | WxH"（几秒后自己消失）。
                # 它与新的底部信息栏讲的是同一件事，且那个尺寸还是**图像自身**的、
                # 与列表「大小」列可能不同源。删掉，只留底栏这一份。
                self._refresh_img_info()
                # ★ 2026-10-22：Annotation(n) 用**当前这张图**的标注形状数
                # Annotation(n) 用**列表里同一行的「标注」列**，保证和左栏数字一致
                # （左栏的标注数是平台自己扫出来的，可能来自 json 也可能是 mask.png，
                #   这里不重复实现，直接读同一份结果）
                try:
                    _ns = 0
                    _it = self.image_list_widget.item(idx, 4)
                    if _it is not None and _it.text().strip():
                        _ns = int(str(_it.text()).strip())
                    self._update_block_counts(ann_count=_ns)
                except Exception:
                    pass
                # 换图后视图模式可能刚变（单图 <-> 多视图），底栏右端要重算一次：
                # 「视图布局」按钮只在多视图出现、−/滑块/+ 要看还放不放得下。
                self._schedule_fit_img_info_bar()
                self._stats_timer.start(150)
                self._load_det_annotations()


    # ============ Annotation Modes ============

    def _set_annotation_mode(self, mode):
        """Set the annotation tool mode (brush/pan/polygon/line/eraser/rect/circle/ellipse/sam)."""
        mode_map = {
            "pan": self.canvas.MODE_PAN,
            "brush": self.canvas.MODE_BRUSH,
            "polygon": self.canvas.MODE_POLYGON,
            "line": self.canvas.MODE_LINE,
            "eraser": self.canvas.MODE_ERASER,
            "ignore": self.canvas.MODE_IGNORE,
            "sam": self.canvas.MODE_SAM,
            "rect": self.canvas.MODE_RECT,
            "circle": self.canvas.MODE_CIRCLE,
            "ellipse": self.canvas.MODE_ELLIPSE,
        }
        if mode in mode_map:
            self.canvas.set_mode(mode_map[mode])
            if mode == "sam":
                self.log("SAM Assist: Left-click=foreground, Right-click=background, Enter=accept, Esc=cancel")

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
        # ── 条目式筛选（原为面板上的「筛选…」按钮，2026-10-15 收进这里） ──
        a_filter = menu.addAction("筛选（条件构建器）…")
        a_filter.setToolTip("条目式筛选：字段 / 条件 / 值 + AND·OR（含「标记」字段）")
        a_filter.triggered.connect(self._open_filter_builder)
        a_clear = menu.addAction("清除筛选")
        a_clear.triggered.connect(self._clear_mark_filters)
        menu.addSeparator()
        # ── 标记（四态）：右键直接改，支持多选批量 ──
        m_menu = menu.addMenu("标记")
        for k in marks_mod.MARK_KINDS:
            a = m_menu.addAction("标记为「%s」" % marks_mod.MARK_LABELS[k])
            a.setToolTip(marks_mod.MARK_HINTS[k])
            a.triggered.connect(lambda _c=False, kk=k: self._on_mark_requested(kk))
        m_menu.addSeparator()
        a_note = m_menu.addAction("写备注…")
        a_note.triggered.connect(self._on_mark_note)
        a_clr = m_menu.addAction("清空标记")
        a_clr.triggered.connect(lambda: self._on_mark_requested(""))
        m_menu.addSeparator()
        a_conv = m_menu.addAction("结果转标注（按标记闭环）…")
        a_conv.setToolTip("过杀→扣成背景（负样本）；漏杀→重建掩膜（正样本）；正常→固化 AI 检出")
        a_conv.triggered.connect(self._run_convert_by_mark)
        menu.addSeparator()
        for label, key in [("Set as Train", "train"), ("Set as Val", "val"), ("Set as Test", "test")]:
            action = menu.addAction(label)
            action.triggered.connect(lambda checked, k=key: self._set_selected_split(selected_rows, k, project_dir, split_map))

        action4 = menu.addAction("Open JSON Location")
        action4.triggered.connect(lambda: self._open_json_location(selected_rows, project_dir))
        # 「输出目录」原先在右栏「图像管理」页里（该页 2026-10-14 删除），挂到这里 ——
        # 与上面那条同属"打开某个目录"，放一起最顺。
        a_out = menu.addAction("打开输出目录")
        a_out.triggered.connect(lambda: self._open_output_dir(project_dir))
        # 「管理划分…」：原为左栏图片列表底部的一个按钮，维护者 2026-10-14 晚判定
        # 面板上那行冗余 ⇒ 删按钮、动作挂到这里（和上面的 Set as Train/Val/Test 同源，
        # 都是"划分"；也补上了它作为**唯一入口**不能丢的那条命）。
        a_split = menu.addAction("管理数据集划分…")
        a_split.setToolTip("按比例 / 手动把当前工程的图片划进 训练 / 验证 / 测试")
        a_split.triggered.connect(self._manage_split)
        menu.addSeparator()
        # 全选（列表已默认不分页，全选后可一次性批量操作/批量导入 JSON）
        act_all = menu.addAction("Select All (全选)")
        act_all.setShortcut(QKeySequence.SelectAll)
        act_all.triggered.connect(self.image_list_widget.selectAll)
        menu.addSeparator()
        # 第 5 项：把选中的这批图片的 JSON 从外部目录批量导进来
        # （多视图工程会连同该组的 N 个视图 JSON 一起导，命名可带 _te_N 后缀）
        action5 = menu.addAction("Import JSON for Selected...")
        action5.setToolTip("从指定文件夹导入这批选中图片的 JSON 标注（按文件名匹配）")
        action5.triggered.connect(lambda: self._import_json_for_selected(selected_rows, project_dir))
        # 导出/清理：原来在「标注操作」区块里（该区块已去掉），改挂到这里 ——
        # 这几个都是"对这批图做点什么"，和上面的导入 JSON 是同一类操作。
        menu.addSeparator()
        a_exp = menu.addAction("导出 JSON...")
        a_exp.triggered.connect(self._export_json)
        a_exp2 = menu.addAction("导出标注")
        a_exp2.triggered.connect(self._export_annotations)
        a_tiny = menu.addAction("清理碎标注 (<10k px)...")
        a_tiny.setToolTip("把面积小于 1 万像素的标注块整块删掉")
        a_tiny.triggered.connect(self._clean_tiny_ann)
        # 删除 / 撤销：破坏性操作单独一组放最后，避免误点。
        # 删除走 core.file_ops（移到工程外备份目录 → 可撤销；索引同步清理）。
        menu.addSeparator()
        from core import file_ops as _fo
        try:
            n_bak = len(_fo.list_backups(project_dir))
        except Exception:
            n_bak = 0
        act_del = menu.addAction("🗑  删除选中图片…")
        act_del.setToolTip("删除选中图片（多视图按**整组**）：图片 + 标注 JSON + 掩膜一起移走，"
                           "并同步清理 train_test_split / class_labels / tile_index")
        act_del.triggered.connect(lambda: self._delete_selected_images(selected_rows, project_dir))
        act_undo = menu.addAction("↩  撤销上次删除" + (f"（可撤销 {n_bak} 次）" if n_bak else ""))
        act_undo.setEnabled(n_bak > 0)
        act_undo.setToolTip("把最近一次删除的文件与索引还原回去" if n_bak else "没有可撤销的删除记录")
        act_undo.triggered.connect(lambda: self._undo_last_delete(project_dir))
        menu.exec_(self.image_list_widget.viewport().mapToGlobal(pos))

    # ---------- 删除 / 撤销（见 core/file_ops.py） ----------

    def _row_bases(self, selected_rows):
        """把选中的行号解析成 base 名（多视图行的 base 可能是任一视图）。"""
        out = []
        for r in sorted(selected_rows):
            item = self.image_list_widget.item(r, 0)
            p = item.data(Qt.UserRole) if item else None
            if p:
                out.append(os.path.splitext(os.path.basename(p))[0])
        return out

    def _delete_selected_images(self, selected_rows, project_dir):
        """删除选中行（多视图整组）：文件 → 索引 → 刷新，可撤销。"""
        from core import file_ops as fo
        bases = self._row_bases(selected_rows)
        if not bases:
            return
        tgt = fo.collect_targets(project_dir, bases)
        if not tgt["files"]:
            QMessageBox.information(self, "删除", "选中项没有可删除的文件（可能已删除）。")
            return
        preview = "\n".join("  • " + os.path.relpath(p, project_dir) for p in tgt["files"][:5])
        more = len(tgt["files"]) - 5
        if more > 0:
            preview += f"\n  … 还有 {more} 个"
        msg = (f"将删除以下文件：\n\n{preview}\n\n"
               f"共 {len(tgt['files'])} 个文件（{len(tgt['groups'])} 组 / {len(tgt['bases'])} 个视图）。\n"
               f"图片、标注 JSON、掩膜会一起移走，划分与类别索引同步清理。\n"
               f"删除后可右键「撤销上次删除」还原。")
        if QMessageBox.question(self, "确认删除", msg,
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        rep = fo.delete_data(project_dir, bases, log=self.log)
        self._refresh_after_delete(rep)
        if rep.failed:
            QMessageBox.warning(
                self, "部分文件未能删除",
                "有 %d 个文件删除失败（可能正被推理/训练占用）：\n%s" % (
                    len(rep.failed),
                    "\n".join(f"  • {p}（{e}）" for p, e in rep.failed[:5])))

    def _undo_last_delete(self, project_dir):
        """撤销最近一次删除（文件 + 索引一起还原）。"""
        from core import file_ops as fo
        if not fo.list_backups(project_dir):
            QMessageBox.information(self, "撤销", "没有可撤销的删除记录。")
            return
        if QMessageBox.question(self, "撤销上次删除",
                                "将还原最近一次删除的文件与索引，继续？",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes) != QMessageBox.Yes:
            return
        r = fo.undo_last(project_dir, log=self.log)
        try:
            self._pred_count_cache_key = None
        except Exception:
            pass
        self._refresh_after_annotation_import()
        QMessageBox.information(self, "撤销", "已还原 %d 个文件。" % r.get("restored", 0))

    def _refresh_after_delete(self, rep=None):
        """删除后刷新：图片列表 / 画布 / Class 表 / 统计。

        复用 `_refresh_after_annotation_import()`（导入 JSON 用的就是那套刷新），
        额外做两件删除专有的事：清 per-image 缓存、当前图被删时切到下一张。
        """
        deleted = set(getattr(rep, "bases", []) or [])
        try:
            self._pred_count_cache_key = None
            for attr in ("_pred_count", "_infer_times"):
                d = getattr(self, attr, None)
                if isinstance(d, dict):
                    for k in list(d.keys()):
                        kb = os.path.splitext(os.path.basename(str(k)))[0]
                        if kb in deleted:
                            d.pop(k, None)
        except Exception as e:
            self.log(f"清理删除缓存失败: {type(e).__name__}: {e}")
        try:
            cur = getattr(self.canvas, "current_image_path", None)
            if cur and not os.path.exists(cur):
                for r in range(self.image_list_widget.rowCount()):
                    it = self.image_list_widget.item(r, 0)
                    p = it.data(Qt.UserRole) if it else None
                    if p and os.path.exists(p):
                        self.canvas.load_image(p)
                        break
        except Exception as e:
            self.log(f"删除后切换当前图失败: {type(e).__name__}: {e}")
        self._refresh_after_annotation_import()

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
        self._refresh_class_ui()
        self.log(f"Set {len(selected_rows)} image(s) to {split_key}")

    # ---------- 标记（四态：正常/过杀/漏杀/存疑，见 core/marks.py） ----------

    def _marks_dir(self):
        """当前工程目录（标记的落盘位置）。没打开工程返回 ''。"""
        if not self.current_project:
            return ""
        try:
            return str(self.pm.get_project_dir(self.current_project["name"]))
        except Exception:
            return ""

    def _ensure_marks_loaded(self, force=False):
        """按工程缓存标记，避免每次刷新列表都读盘（列表刷新是热路径）。"""
        pd = self._marks_dir()
        if not pd:
            self._marks, self._marks_project = {}, None
            return self._marks
        if force or self._marks_project != pd:
            self._marks = marks_mod.load_marks(pd)
            self._marks_project = pd
        return self._marks

    def _flush_marks(self):
        pd = self._marks_project or self._marks_dir()
        if pd:
            marks_mod.save_marks(pd, self._marks)
            self._marks_project = pd
        return pd

    def _selected_bases(self):
        """当前选中行对应的 base 名列表（多视图行取该组主视图的 base）。"""
        rows = sorted(set(i.row() for i in self.image_list_widget.selectedIndexes()))
        return self._row_bases(rows)

    def _refresh_after_mark_change(self):
        """标记改动后只刷列表与统计（不重扫磁盘 —— 标记不影响图片/标注文件）。"""
        self._filter_images(self.search_input.text())

    def _on_mark_requested(self, kind):
        if not self.current_project:
            return QMessageBox.information(self, "标记", "先打开一个工程。")
        bases = self._selected_bases()
        if not bases:
            return QMessageBox.information(self, "标记", "先在图片列表里选中一张或多张图。")
        self._ensure_marks_loaded()
        if kind:
            marks_mod.set_mark(self._marks, bases, kind)
            self.log(f"标记「{marks_mod.kind_label(kind)}」：{len(bases)} 张")
        else:
            n = marks_mod.clear_marks(self._marks, bases)
            self.log(f"清空标记：{n} 张")
        self._flush_marks()
        self._refresh_after_mark_change()

    def _on_mark_note(self):
        if not self.current_project:
            return QMessageBox.information(self, "备注", "先打开一个工程。")
        bases = self._selected_bases()
        if not bases:
            return QMessageBox.information(self, "备注", "先选中图片。")
        self._ensure_marks_loaded()
        marked = [b for b in bases if marks_mod.get_mark(self._marks, b)["mark"]]
        if not marked:
            return QMessageBox.information(
                self, "备注",
                "选中的图还没有标记。\n\n先打标记再写理由 —— 没有标记的备注会成为孤儿数据，"
                "统计和筛选中都看不到它。")
        cur = marks_mod.get_mark(self._marks, marked[0])["note"] if len(marked) == 1 else ""
        title = "标记备注" + ("" if len(marked) == 1 else "（将应用到 %d 张）" % len(marked))
        text, ok = QInputDialog.getMultiLineText(
            self, title, "理由（现场经验最难传承，写清楚为什么这么判）：", cur)
        if not ok:
            return
        n = marks_mod.set_note(self._marks, marked, text)
        self._flush_marks()
        self._refresh_after_mark_change()
        self.log(f"写入标记备注：{n} 张")

    def _on_mark_kind_filter(self, kind):
        """统计面板点四态计数按钮 -> 只看该标记。"""
        self._mark_filter = marks_mod.normalize_kind(kind)
        self._defect_filter = ""
        self._switch_accordion(0)
        self._refresh_after_mark_change()

    def _on_mark_cell_filter(self, defect, kind):
        """统计面板点交叉表格子 -> 只看「这个缺陷 + 这个标记」。"""
        self._defect_filter = "" if defect in ("", "(无缺陷名)") else str(defect)
        self._mark_filter = marks_mod.normalize_kind(kind)
        self._switch_accordion(0)
        self._refresh_after_mark_change()

    def _clear_mark_filters(self):
        self._mark_filter = ""
        self._defect_filter = ""
        self._filter_expr = FilterExpr()
        self._refresh_after_mark_change()
        self.log("已清除标记/表达式筛选")

    def _open_filter_builder(self):
        dlg = FilterBuilderDialog(self, expr=self._filter_expr)
        if dlg.exec_() != QDialog.Accepted:
            return
        self._filter_expr = dlg.result_expr()
        self._refresh_after_mark_change()
        t = self._filter_expr.describe()
        self.log("筛选条件：" + (t if t else "（空）"))

    def _sync_mark_filter_label(self):
        """把当前生效的筛选条件写在列表上方（不显示的话用户会以为图丢了）。"""
        bits = []
        if self._defect_filter:
            bits.append("缺陷类型=%s" % self._defect_filter)
        if self._mark_filter:
            bits.append("标记=%s" % marks_mod.kind_label(self._mark_filter))
        t = self._filter_expr.describe()
        if t:
            bits.append(t)
        if bits:
            self.mark_filter_label.setText("筛选中：" + " 且 ".join(bits) + "（点「清除筛选」复位）")
            self.mark_filter_label.setVisible(True)
        else:
            self.mark_filter_label.setVisible(False)

    def _mark_stats_records(self):
        """给统计面板喂数据：用**全量**图片（不是筛选后的），否则筛选一开统计就自己变了。"""
        self._ensure_marks_loaded()
        # ★★ 2026-10-29 性能（打开工程卡顿的真正大头）：
        #   这里原来对**每张图**调用 _defect_of_base()，而它内部是
        #       for r in self._filtered_data:      # 线性查找
        #           if r.get("base") == base: ...
        #   于是 1690 张 × 1690 项 ≈ **285 万次比较**。
        #   cProfile 实测这一段 963ms，占整个 _filter_images 的 94%
        #   （profile 里那 289 万次 dict.get 就是它）。
        #   改成先建一次 base→标签 索引，循环里 O(1) 查 —— 从 O(n²) 降到 O(n)。
        _lbl_by_base = {}
        for r in (getattr(self, "_filtered_data", []) or []):
            _b = r.get("base")
            if _b and r.get("shape_labels") and _b not in _lbl_by_base:
                _lbl_by_base[_b] = str(r["shape_labels"][0])

        recs = []
        for p in (self._all_images or []):
            base = os.path.splitext(os.path.basename(p))[0]
            m = marks_mod.get_mark(self._marks, base)
            _lbl = _lbl_by_base.get(base)
            recs.append({"base": base, "mark": m["mark"], "note": m["note"],
                         "defect": _lbl if _lbl is not None
                         else marks_mod.defect_of(base)})
        return recs

    def _defect_of_base(self, base):
        """缺陷类型：优先用类别表里的标注类别，退回从文件名猜（命名规范里有缺陷段）。"""
        for r in getattr(self, "_filtered_data", []) or []:
            if r.get("base") == base and r.get("shape_labels"):
                return str(r["shape_labels"][0])
        return marks_mod.defect_of(base)

    def _run_convert_by_mark(self):
        """难例闭环一键入口：把标记过的图按语义转成训练数据。"""
        pd = self._marks_dir()
        if not pd:
            return QMessageBox.information(self, "结果转标注", "先打开一个工程。")
        self._ensure_marks_loaded()
        bases = self._selected_bases()
        if not bases:
            # 没选就作用于当前筛选出来的全部
            bases = [r.get("base") for r in (self._filtered_data or []) if r.get("base")]
        bases = [b for b in bases if marks_mod.get_mark(self._marks, b)["mark"]]
        if not bases:
            return QMessageBox.information(
                self, "结果转标注",
                "选中范围里没有带标记的图。\n\n先给图打标记（正常/过杀/漏杀），再执行这个动作。")
        cnt = marks_mod.count_marks({b: self._marks[b] for b in bases})
        msg = ("将对 %d 张带标记的图执行闭环动作：\n\n"
               "  · 过杀 %d 张 → 把与 AI 输出重叠的自有标注扣掉，该区域回退成**背景**（负样本）\n"
               "  · 漏杀 %d 张 → 由标注 JSON 重建掩膜，并输出漏检区域图（确保正样本完整）\n"
               "  · 正常 %d 张 → 把 AI 检出固化成标注（省人工，幂等）\n"
               "  · 存疑 %d 张 → 跳过（按设计不进训练集）\n\n"
               "处理完会把参与的图归入训练集（已在 Val/Test 的不动）。\n"
               "原标注会先经版本快照，可回滚。是否继续？"
               % (len(bases), cnt.get("overkill", 0), cnt.get("escape", 0),
                  cnt.get("normal", 0), cnt.get("doubt", 0)))
        if QMessageBox.question(self, "确认结果转标注", msg,
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes) != QMessageBox.Yes:
            return
        try:
            from annotation.convert import convert_by_mark, summarize_report
        except Exception as e:
            return QMessageBox.critical(self, "结果转标注", "模块加载失败：%s" % e)
        src = "segment" if not self._detection_mode else "detect"
        rep = convert_by_mark(pd, bases, self._marks, source=src)
        self.log("[结果转标注] " + summarize_report(rep))
        for r in rep.get("results", []):
            if r.get("detail"):
                self.log("  · %s：%s" % (r.get("base"), r["detail"]))
        self._pred_count_cache_key = None
        self._refresh_after_annotation_import()
        title = "完成"
        body = summarize_report(rep)
        if rep.get("failures"):
            body += "\n\n失败 %d 张：\n%s" % (
                len(rep["failures"]),
                "\n".join("  · %s：%s" % (f["base"], f["detail"]) for f in rep["failures"][:8]))
            title = "完成（部分失败）"
        QMessageBox.information(self, title, body)

    def _import_json_for_selected(self, selected_rows, project_dir):
        """把选中图片（多视图则含该组全部视图）的 JSON 从外部目录批量导入。

        匹配规则：按**文件名**（不含扩展名）匹配，目录递归查找。
        落地位置：与图片同目录（images/<base>.json），这是平台约定，
        标注编辑器与训练数据集都按这个位置读取。
        """
        img_dir = os.path.join(project_dir, "images")
        if not os.path.isdir(img_dir):
            return QMessageBox.warning(self, "Warning", "No images directory found")
        last = getattr(self, "_last_json_import_dir", "") or ""
        folder = QFileDialog.getExistingDirectory(self, "选择 JSON 所在文件夹", last)
        if not folder:
            return
        self._last_json_import_dir = folder

        # 收集目标 base 名（多视图：该组的每个视图各要一个 JSON）
        targets = []
        for r in sorted(selected_rows):
            gi = self._current_page * self._page_size + r
            d = self._filtered_data[gi] if hasattr(self, "_filtered_data") and 0 <= gi < len(self._filtered_data) else None
            if not d:
                continue
            vps = d.get("view_paths") or []
            if vps:
                targets += [os.path.splitext(os.path.basename(p))[0] for p in vps]
            elif d.get("base"):
                targets.append(d["base"])
        if not targets:
            return QMessageBox.information(self, "Hint", "没有可导入的图片")
        targets = list(dict.fromkeys(targets))     # 去重保序

        # 索引外部目录里的 JSON（递归，按文件名匹配）
        src = {}
        try:
            for root, _dirs, files in os.walk(folder):
                for fn in files:
                    if fn.lower().endswith(".json"):
                        src.setdefault(os.path.splitext(fn)[0], os.path.join(root, fn))
        except Exception as e:
            return QMessageBox.warning(self, "Error", f"扫描目录失败: {e}")

        self.log(f"导入 JSON：选中 {len(selected_rows)} 行 / 需要 {len(targets)} 个标注文件，"
                 f"来源 {folder}")
        self.log(f"  该目录下共找到 {len(src)} 个 JSON")

        copied, missing = 0, []
        for base in targets:
            sp = src.get(base)
            if not sp:
                missing.append(base)
                continue
            dst = os.path.join(img_dir, base + ".json")
            try:
                shutil.copy2(sp, dst)
                copied += 1
            except Exception as e:
                self.log(f"  复制失败 {base}: {e}")
                missing.append(base)

        self.log(f"导入完成：成功 {copied} 个" + (f"，未找到 {len(missing)} 个" if missing else ""))
        if missing:
            self.log("  未找到的（前 5 个）：" + "、".join(missing[:5]))
            QMessageBox.information(
                self, "导入结果",
                f"成功导入 {copied} 个 JSON。\n\n有 {len(missing)} 个在所选目录里没找到同名文件：\n"
                + "、".join(missing[:5]) + ("..." if len(missing) > 5 else ""))
        else:
            QMessageBox.information(self, "导入结果", f"成功导入 {copied} 个 JSON。")
        # 导入的 JSON 里可能有工程还没登记的类别 —— 类别表的真源就是这些 JSON，
        # 必须先补类别表，再刷新列表/预览/Class 表，否则 Class 表还是空的。
        if copied:
            try:
                self._load_classes()
            except Exception as e:
                self.log(f"[Class] 导入后补类别失败: {type(e).__name__}: {e}")
            self._rebuild_masks_for_imported(targets)
        self._refresh_after_annotation_import()

    def _rebuild_masks_for_imported(self, bases):
        """导入 JSON 后顺手把掩膜 PNG 生成出来。

        平台约定 `annotations/<组名>_te_0_mask.png` 是训练要读的文件，而画布只在
        "打开某张图"时才会顺手从 JSON 生成掩膜 —— 批量导完不生成的话，
        训练会直接报缺掩膜（用户还得一张张点开）。这里按 JSON 一次性补齐。
        多视图只生成 te_0（三视图像素对齐、共用一张掩膜），单图工程按原名生成。
        """
        try:
            from annotation.labelme_io import load_labelme_json, shapes_to_mask
            from core.class_discovery import read_project_classes
            pd = self._project_dir
            classes = list(getattr(self, "_cached_classes", []) or read_project_classes(pd))
            ann_dir = os.path.join(pd, "annotations")
            os.makedirs(ann_dir, exist_ok=True)
            mv = bool(getattr(self, "_mv_mode_list", False))
            made = skipped = 0
            for base in bases:
                if mv and not base.endswith("_te_0"):
                    continue                      # 多视图：只要 te_0 的掩膜
                jp = os.path.join(pd, "images", base + ".json")
                if not os.path.exists(jp):
                    continue
                shapes, info = load_labelme_json(jp)
                if not shapes:
                    skipped += 1
                    continue
                h = w = 0
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    ip = os.path.join(pd, "images", base + ext)
                    if os.path.exists(ip):
                        try:
                            from PIL import Image as _I
                            w, h = _I.open(ip).size
                        except Exception:
                            h = w = 0
                        break
                if not (h and w) and info:
                    h, w = info.get("imageHeight", 0), info.get("imageWidth", 0)
                if not (h and w):
                    continue
                unknown = set()
                mask = shapes_to_mask(shapes, classes, (h, w), unknown_out=unknown)
                from PIL import Image as _I
                _I.fromarray(mask.astype("uint8")).save(os.path.join(ann_dir, base + "_mask.png"))
                made += 1
                if unknown:
                    self.log(f"  ⚠️ {base}: JSON 里有类别表之外的标签 {sorted(unknown)}（已按背景处理）")
            self.log(f"掩膜 PNG 生成：{made} 个" + (f"，空标注跳过 {skipped} 个" if skipped else ""))
        except Exception as e:
            self.log(f"生成掩膜失败（不影响 JSON 已导入）: {type(e).__name__}: {e}")

    def _refresh_after_annotation_import(self):
        """导入标注后同步刷新三处：图片列表 / 预览画布（含掩膜叠加）/ Class 表。"""
        # ① 图片列表：标注列、形状数、统计条
        try:
            self._pred_count_cache_key = None
            self._filter_images(self.search_input.text())
        except Exception as e:
            self.log(f"刷新图片列表失败: {type(e).__name__}: {e}")
        # ② 预览画布：重新载入当前图，掩膜叠加立刻反映新导入的标注
        try:
            cur = getattr(self.canvas, "current_image_path", None)
            if cur and os.path.exists(cur):
                self.canvas.load_image(cur)
                if getattr(self.canvas, "_show_heatmap", False):
                    self._auto_load_heatmap_for_current()
                if self.mv_preview.is_multiview():
                    self.mv_preview._sync_from_canvas(force=True)
            self._refresh_versions()
        except Exception as e:
            self.log(f"刷新预览失败: {type(e).__name__}: {e}")
        # ③ Class 表 + 右侧统计
        for fn in ("_refresh_class_ui", "_refresh_stats", "_update_stats"):
            try:
                f = getattr(self, fn, None)
                if callable(f):
                    f()
            except Exception:
                pass
        try:
            self._stats_timer.start(150)
        except Exception:
            pass
        self.log("已刷新：图片列表 / 预览 / Class 表")

    def _open_json_location(self, selected_rows, project_dir):
        """Open folder in Explorer and highlight the JSON file."""
        import subprocess
        for r in selected_rows:
            item = self.image_list_widget.item(r, 0)
            p = item.data(Qt.UserRole) if item else None
            if not p:
                continue
            json_path = os.path.splitext(p)[0] + ".json"
            if not os.path.exists(json_path):
                blank = {"version":"5.3.1","flags":{},"shapes":[],"imagePath":os.path.basename(p),"imageData":None,"imageHeight":0,"imageWidth":0}
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(blank, jf, indent=2, ensure_ascii=False)
            subprocess.Popen(f'explorer /select,"{os.path.abspath(json_path)}"', shell=True)
            break

    def _refresh_versions(self):
        """Refresh version history list for current image."""
        if not hasattr(self, "version_list"):
            return
        # 顺带刷新「模型管理」的模型列表 ——
        # 这个函数已经在切图、保存标注、打开工程时被调用，挂在这里不用再加触发点。
        self._refresh_image_info()
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
        """Load class list from project config.

        类别表的真源是**图片的标注 JSON**：project.json 里的 classes 只是缓存。
        所以这里读完 project.json 还要扫一遍 JSON，把没登记过的类别补进去，
        否则会出现"JSON 导进来了、图片列表能看标注，但 Class 表一行都没有、
        手工标注也选不到类别"（真实踩过：工程 classes 是 []）。
        """
        if not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        from core.class_discovery import merge_classes, read_project_classes, discover_labels
        try:
            # ⚠️ 试过"用 JSON 指纹跳过全量扫描"的缓存，实测**反而更慢**
            #    （同进程内第 2/3 次 _load_classes 从 1.2s 涨到 8~10s），已回退。
            #    大头不在这里，而在下面的 _compute_counts（遍历 402 个文件逐个读 JSON/掩膜）。
            discovered = discover_labels(pd)
            classes, added = merge_classes(pd, discovered)
            if added:
                self.log(f"[Class] 从标注 JSON 补入 {len(added)} 个类别: {', '.join(added)}")
        except Exception as e:
            self.log(f"[Class] 扫描标注 JSON 失败，仅用 project.json: {type(e).__name__}: {e}")
            classes = read_project_classes(pd)
        self._cached_classes = classes
        self._reload_canvas_classes()
        self._refresh_class_ui()

    def _reload_canvas_classes(self):
        """把类别表推给画布（LabelManager 决定掩膜像素值，必须与类别表同源）。"""
        try:
            lm = getattr(self.canvas, "label_manager", None)
            if lm is not None:
                lm.load()
                # 类别变多/变少后，当前选中类别要夹到合法范围
                n = len(getattr(lm, "classes", []) or [])
                if n and getattr(self.canvas, "current_class_id", 0) >= n:
                    self.canvas.set_class(n - 1)
        except Exception as e:
            self.log(f"[Class] 画布类别表刷新失败: {type(e).__name__}: {e}")

    def _on_class_changed(self, current):
        """Handle class selection change in class list."""
        if current < 0 or not hasattr(self.canvas, "label_manager"):
            return
        if current >= 0:
            self.canvas.set_class(current)

    def _on_class_cell_filter(self, cls_name, split_key):
        """点击 Class 表的「训练集/测试集/验证集/空集」数字格子 -> 图片列表过滤出这批图。"""
        matched = self.class_list.bases_for(cls_name, split_key)
        if not matched:
            self.log(f"[Class] {cls_name} / {split_key} -> 0 images")
            return
        self._cm_filter_bases = set(matched)
        # 复位搜索框与数据集下拉，避免与这批过滤冲突
        self.search_input.blockSignals(True)
        self.search_input.setText("")
        self.search_input.blockSignals(False)
        if self.split_filter.currentIndex() != 0:
            self.split_filter.blockSignals(True)
            self.split_filter.setCurrentIndex(0)
            self.split_filter.blockSignals(False)
        self._filter_images("")
        self._cm_filter_bases = None
        self.log(f"[Class] {cls_name} / {split_key} -> {len(matched)} images")



    def _on_canvas_width_changed(self, v):
        """工具栏宽度微调框 -> 画布笔刷大小。"""
        self.canvas.brush_size = max(1, int(v))

    def _on_toolbar_reload(self):
        """刷新视图：重置缩放/平移。"""
        self.canvas.reset_view()
        self.canvas.update()

    def _on_canvas_class_select(self, cid):
        """数字键 1-9 选择类别。"""
        if cid < 0 or cid >= self.class_list.count():
            return
        self.class_list.setCurrentRow(cid, emit=False)
        self.canvas.set_class(cid)

    def _add_class(self):
        """Add a new class to the project."""
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        classes = list(self._cached_classes)
        if name in classes:
            self.log(f"[Class] Duplicate: {name}")
            return
        classes.append(name)
        self._cached_classes = classes
        self._save_classes_to_project(classes)
        self._refresh_class_ui()
        self.log(f"[Class] Added: {name}")

    def _delete_class(self):
        """删除选中的类别。

        ⚠️ 掩膜里的类别是**按编号**存的（= 类别表里的位置），不是按名字存的
        —— 见 `annotation/convert.py`：`background` 占 0 号位，`255` = ignore 不占号。
        实测这个工程也严格对得上：编号 1..4 的掩膜文件数分别 3/28/2/5，
        正是类别表里 Zw/Ys/Zz/Hh 各自的数量。

        由此这个动作有两件事必须做对，少一件都是**静默错**：

        1. **删完必须让画布重读类别表**（`_reload_canvas_classes`）。
           画布的 `label_manager.classes` 只是 `load()` 时抓到的一份普通 list，
           不重读就一直停在旧表上。若用户刚才选中的正是被删的那个类别，
           `current_class_id` 就成了一个**已经不存在的编号**，继续画会把
           未定义的类别号写进掩膜（2026-10-14 实测：删掉 `Atd` 后画布仍是 6 张表、
           `current_class_id` 仍停在 5，而 project.json 只剩 0..4）。
        2. **弹窗必须说实话**。删最后一个类别：谁都不用重新编号，安全；
           删中间那个：它后面所有类别前移一位，而已有掩膜里的数字不会跟着改
           ⇒ 那些区域此后被解释成**别的类别**。旧文案一律写
           "Existing annotations will keep their class index" ——
           对后者这句话恰好把最危险的后果说成了"没事"。
        """
        row = self.class_list.currentRow()
        if row < 0:
            # 什么都没选中时**不能**说成"background 不能删" —— 那是两回事，
            # 而删除是静默 no-op 的话，用户只会觉得"点了没反应"。
            self.log("[Class] 没有选中要删除的类别（先在类别表里点一行）")
            return
        if row == 0:  # protect background (index 0)
            self.log("[Class] Cannot delete background")
            return
        classes = list(self._cached_classes)
        if row >= len(classes):
            return
        name = classes[row]
        n_tail = len(classes) - row - 1
        if n_tail == 0:
            warn = ("『%s』是最后一个类别，其余类别的编号一个都不变，"
                    "已有标注不受影响。" % name)
        else:
            warn = ("⚠️ 『%s』后面还有 %d 个类别，删掉它们会**整体前移一位**；"
                    "而已有标注是按编号存的、不会跟着改，"
                    "编号 ＞ %d 的区域此后会被当成别的类别（界面上看不出来）。"
                    % (name, n_tail, row))
        reply = QMessageBox.question(self, "删除类别",
            "确定删除类别『%s』？\n\n%s\n\n（掩膜文件本身不会被改动；background 不能删）"
            % (name, warn),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        # 先记下当前选中类别：删除会让编号前移，它得跟着挪
        cid = int(getattr(self.canvas, "current_class_id", 1) or 0)
        del classes[row]
        self._cached_classes = classes
        self._save_classes_to_project(classes)
        self._reload_canvas_classes()   # 画布重读类别表（含越界夹取）
        # 被删的在选中项之前 ⇒ 选中项编号减 1；再夹回合法范围（含"删的就是它自己"）
        if cid > row:
            cid -= 1
        self.canvas.set_class(max(0, min(cid, len(classes) - 1)))
        # 推理参数里那条同名记录要跟着摘掉，否则 JSON 里留一个点不开的残项
        try:
            from inference.params import drop_class_params
            pd = self._project_dir_or_none()
            if pd and drop_class_params(pd, name):
                self.log(f"[Class] 清理推理参数里的残项: {name}")
        except Exception as e:
            self.log(f"[Class] 推理参数清理失败: {type(e).__name__}: {e}")
        self._refresh_class_ui()
        self.log(f"[Class] Deleted: {name}")

    def _rename_class(self):
        """Rename selected class."""
        row = self.class_list.currentRow()
        if row < 0:
            return
        classes = list(self._cached_classes)
        if row >= len(classes):
            return
        old_name = classes[row]
        new_name, ok = QInputDialog.getText(self, "Rename Class", "New name:", text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        if new_name in classes:
            self.log(f"[Class] Duplicate: {new_name}")
            return
        classes[row] = new_name
        self._cached_classes = classes
        self._save_classes_to_project(classes)
        # 推理参数那条记录要**跟着搬家**：面板按名字查参数，不搬的话新名字查不到
        # ⇒ 悄悄给一组默认值，用户调好的八维阈值就没了（静默错）。
        try:
            from inference.params import rename_class_params
            pd = self._project_dir_or_none()
            if pd and rename_class_params(pd, old_name, new_name):
                self.log(f"[Class] 推理参数已迁移: {old_name} -> {new_name}")
        except Exception as e:
            self.log(f"[Class] 推理参数迁移失败: {type(e).__name__}: {e}")
        # 画布的 label_manager 拿着的是**名字**（编号 -> 名字的映射），也必须重读，
        # 否则工具栏与画布会各说各的（一个显示新名字、一个显示旧名字）。
        self._reload_canvas_classes()
        self._refresh_class_ui()
        self.log(f"[Class] Renamed: {old_name} -> {new_name}")

    def _save_classes_to_project(self, classes):
        """Persist class list to project.json."""
        if not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        config_path = os.path.join(pd, "project.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["classes"] = classes
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
            except Exception as e:
                self.log(f"[Class] Save error: {e}")

    def _project_dir_or_none(self):
        """Return the current project directory, or None."""
        if not self.current_project:
            return None
        try:
            return str(self.pm.get_project_dir(self.current_project["name"]))
        except Exception:
            return None

    def _move_class(self, delta):
        """Move the selected class up/down in the class list."""
        row = self.class_list.currentRow()
        if row < 0:
            return
        new_row = row + delta
        classes = list(self._cached_classes)
        if not (0 <= new_row < len(classes)):
            return
        classes[row], classes[new_row] = classes[new_row], classes[row]
        self._cached_classes = classes
        self._save_classes_to_project(classes)
        # ⚠️ 上移/下移换的是**位置**，而位置就是掩膜的类别号 —— 已有标注的编号
        #    不会跟着走（它们此后会被解释成别的类别）。这一点目前只在文档/日志里
        #    交代，没有拦人；但画布的 label_manager 必须重读，否则它手里的
        #    "编号 -> 名字"还是旧的那套（工具栏与画布会各说各的）。
        self._reload_canvas_classes()
        self._refresh_class_ui()
        if 0 <= new_row < self.class_list.count():
            self.class_list.setCurrentRow(new_row, emit=False)
            if hasattr(self.canvas, "label_manager"):
                self.canvas.set_class(new_row)
        self.log(f"[Class] Moved '{classes[new_row]}' {'up' if delta < 0 else 'down'}")

    def _refresh_class_stats(self):
        """Recompute class table counts from project files.

        注意：这里必须走 `_load_classes()`（重读 project.json + 重扫标注 JSON），
        不能只 `_refresh_class_ui()`（那只是把**内存里**的类别表重画一遍）。
        用户在外部补了 JSON 标注、或类别表还是空的时，按"刷新"就是要重新发现类别。
        """
        self._load_classes()
        self._refresh_class_ui()
        self.log("[Class] 已重读 project.json 并重扫标注 JSON，统计数据已更新")

    def _refresh_class_ui(self):
        """Refresh class list and palette from _cached_classes."""
        classes = self._cached_classes
        if hasattr(self, "class_list"):
            self.class_list.set_project_dir(self._project_dir_or_none())
            # ★★ 2026-10-29 性能：先把表建出来，数字列留 0，
            #   把 _compute_counts（遍历标注文件，1690 张工程实测 559ms）推到下一拍。
            #   打开工程时这段原来卡在主线程上，是"双击之后干等"的第二个大头
            #   （第一个是 _mark_stats_records 的 O(n²)）。
            #   singleShot(0) 在下一个事件循环就执行，用户感知不到延迟。
            self.class_list.set_classes(classes, with_counts=False)
            QTimer.singleShot(0, lambda: self._deferred_class_counts(classes))
        # ★ 选中行也要跟上画布的当前类别（2026-10-15 补）：
        #    工具栏那格「类别:xx」删掉之后，**右栏这张表的选中行就是"当前类别"的唯一显示处**
        #    （单图模式画布状态行也显示，但多视图模式画布叠加是关的）。
        #    实测原来这里不同步：打开工程后 `class_list.currentRow()` 是 -1，
        #    而 canvas.current_class_id 已经是 1 —— 表里空着选中行，多视图下就没法
        #    知道正在标哪个类别（静默不同步，不报任何错）。
        #    emit=False：不让它反过来发 currentRowChanged 去改画布（那是键盘那条路的事）。
        try:
            if hasattr(self, "class_list"):
                _cur = getattr(self.canvas, "current_class_id", None)
                _n = len(classes or [])
                self.class_list.setCurrentRow(
                    _cur if (isinstance(_cur, int) and 0 <= _cur < _n) else -1, emit=False)
        except Exception:
            pass
        self._update_class_count_label(len(classes or []))
        # 推理参数面板按类别组织：类别表一变，它的下拉也要跟着重建
        try:
            if hasattr(self, "infer_params"):
                self.infer_params.set_project(self._project_dir_or_none(), classes)
        except Exception:
            pass

    def _deferred_class_counts(self, classes):
        """延后补算 Class 表的 train/test/val/empty 数字列。

        `_refresh_class_ui` 里用 QTimer.singleShot(0) 排到这里，
        让表先显示出来、再算统计（见那边的注释）。
        期间如果类别表被换掉（切工程、改类别），对不上就放弃这一拍。
        """
        try:
            if list(getattr(self, "_cached_classes", []) or []) != list(classes or []):
                return
            if hasattr(self, "class_list"):
                self.class_list.refresh_counts()
        except Exception:
            pass
        # 工具栏那个「类别」格 2026-10-15 已删（当前类别由**右栏类别表的选中行**承担，
        # 选类别走数字键 1-9 / 右栏类别表）。这里因此**不再**需要重建工具栏菜单 ——
        # 原来那段留着会一直给一个不存在的控件喂数据，看着正常其实空转。
        # 右栏类别表自己的 set_classes(classes) 已在上面调用，它的选中行由
        # `_on_canvas_class_select` 同步（见那里的 setCurrentRow）。

    # ============ UI Panel Toggles ============


    # ============ Classification ============

    def _on_cls_scale_link_toggled(self, checked):
        """Sync W/H scale spinboxes when link is on."""
        if checked and self.cls_scale_h_spin.value() > 0:
            self._cls_scale_ratio = self.cls_scale_w_spin.value() / self.cls_scale_h_spin.value()

    def _sync_cls_scale(self, changed):
        """When link is on, keep the other scale spinbox in sync."""
        if not self.cls_scale_link_btn.isChecked():
            return
        ratio = getattr(self, "_cls_scale_ratio", 1.0)
        if changed == "w":
            h = round(self.cls_scale_w_spin.value() / max(0.001, ratio), 2)
            h = max(0.1, min(1.0, h))
            self.cls_scale_h_spin.blockSignals(True)
            self.cls_scale_h_spin.setValue(h)
            self.cls_scale_h_spin.blockSignals(False)
        else:
            w = round(self.cls_scale_h_spin.value() * ratio, 2)
            w = max(0.1, min(1.0, w))
            self.cls_scale_w_spin.blockSignals(True)
            self.cls_scale_w_spin.setValue(w)
            self.cls_scale_w_spin.blockSignals(False)

    _CLS_PARAM_SPECS = [
        ("model", "cls_model_combo", "text"),
        ("epochs", "cls_epochs_spin", "int"),
        ("batch_size", "cls_batch_spin", "int"),
        ("optimizer", "cls_optim_combo", "text"),
        ("lr", "cls_lr_spin", "float"),
        ("loss", "cls_loss_combo", "text"),
        ("scale_w", "cls_scale_w_spin", "float"),
        ("scale_h", "cls_scale_h_spin", "float"),
        ("input_w", "cls_input_w_spin", "int"),
        ("input_h", "cls_input_h_spin", "int"),
        ("keep_aspect", "cls_keep_aspect_check", "bool"),
        ("amp", "cls_amp_check", "bool"),
        ("k_folds", "cls_kfold_spin", "int"),
        ("resume", "cls_resume_check", "bool"),
        ("class_balanced", "cls_balanced_check", "bool"),
        ("ema", "cls_ema_check", "bool"),
        ("weight_decay", "cls_wd_spin", "float"),
        ("momentum", "cls_momentum_spin", "float"),
        ("label_smoothing", "cls_ls_spin", "float"),
        ("focal_gamma", "cls_focal_gamma_spin", "float"),
        ("dropout", "cls_dropout_spin", "float"),
        ("lr_scheduler", "cls_scheduler_combo", "text"),
        ("warmup_epochs", "cls_warmup_spin", "int"),
        ("early_stop_patience", "cls_early_stop_spin", "int"),
        ("select_metric", "cls_select_metric_combo", "text"),
    ]

    # ---------- 训练参数持久化（面板 <-> .train_settings.json）----------
    def _train_settings_path(self):
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            ".train_settings.json")

    def _load_panel_train_settings(self):
        """把 .train_settings.json 的参数灌回面板控件（重启后保留用户设置）。"""
        try:
            import json as _json
            _p = self._train_settings_path()
            if not os.path.exists(_p):
                return
            with open(_p, "r", encoding="utf-8") as _f:
                s = _json.load(_f) or {}
            if hasattr(self, "epochs_spin"):
                self.epochs_spin.setValue(int(s.get("epochs", 200)))
            if hasattr(self, "early_stop_spin"):
                self.early_stop_spin.setValue(int(s.get("early_stop_patience", 0)))
            if hasattr(self, "batch_spin"):
                self.batch_spin.setValue(int(s.get("batch_size", 8)))
            if hasattr(self, "kfold_spin"):
                self.kfold_spin.setValue(int(s.get("k_folds", 1)))
            if hasattr(self, "resume_check"):
                self.resume_check.setChecked(bool(s.get("resume", False)))
            if hasattr(self, "tr_model_combo"):
                _m = str(s.get("model", "deeplabv3"))
                if self.tr_model_combo.findText(_m) < 0:
                    self.tr_model_combo.addItem(_m)
                self.tr_model_combo.setCurrentText(_m)
            if hasattr(self, "tr_size_combo"):
                self.tr_size_combo.setCurrentText(str(s.get("image_size", 256)))
            if hasattr(self, "tr_loss_combo"):
                _l = str(s.get("loss", "ce+lovasz"))
                if self.tr_loss_combo.findText(_l) < 0:
                    self.tr_loss_combo.addItem(_l)
                self.tr_loss_combo.setCurrentText(_l)
            if hasattr(self, "tr_mode_combo"):
                self.tr_mode_combo.setCurrentIndex(
                    0 if str(s.get("infer_mode", "fine")) == "fine" else 1)
            if hasattr(self, "tr_tile_check"):
                self.tr_tile_check.setChecked(bool(s.get("tile", False)))
            if hasattr(self, "tr_tile_spin"):
                self.tr_tile_spin.setValue(int(s.get("tile_size", 256)))
            if hasattr(self, "tr_tile_stride_spin"):
                self.tr_tile_stride_spin.setValue(int(s.get("tile_stride", 75)))
        except Exception:
            pass

    def _save_panel_train_settings(self):
        """把面板上的参数写回 .train_settings.json（关窗 / 开始训练时调用）。"""
        try:
            import json as _json
            _p = self._train_settings_path()
            s = {}
            if os.path.exists(_p):
                try:
                    with open(_p, "r", encoding="utf-8") as _f:
                        s = _json.load(_f) or {}
                except Exception:
                    s = {}
            if hasattr(self, "epochs_spin"):
                s["epochs"] = int(self.epochs_spin.value())
            if hasattr(self, "early_stop_spin"):
                s["early_stop_patience"] = int(self.early_stop_spin.value())
            if hasattr(self, "batch_spin"):
                s["batch_size"] = int(self.batch_spin.value())
            if hasattr(self, "kfold_spin"):
                s["k_folds"] = int(self.kfold_spin.value())
            if hasattr(self, "resume_check"):
                s["resume"] = bool(self.resume_check.isChecked())
            if hasattr(self, "tr_model_combo"):
                s["model"] = self.tr_model_combo.currentText()
            if hasattr(self, "tr_size_combo"):
                try:
                    s["image_size"] = int(self.tr_size_combo.currentText())
                except Exception:
                    pass
            if hasattr(self, "tr_loss_combo"):
                s["loss"] = self.tr_loss_combo.currentText()
            if hasattr(self, "tr_mode_combo"):
                s["infer_mode"] = ("fast" if "快速" in self.tr_mode_combo.currentText()
                                   else "fine")
            if hasattr(self, "tr_tile_check"):
                s["tile"] = bool(self.tr_tile_check.isChecked())
            if hasattr(self, "tr_tile_spin"):
                s["tile_size"] = int(self.tr_tile_spin.value())
            if hasattr(self, "tr_tile_stride_spin"):
                s["tile_stride"] = int(self.tr_tile_stride_spin.value())
            with open(_p, "w", encoding="utf-8") as _f:
                _json.dump(s, _f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def closeEvent(self, event):
        """关窗前把训练参数存盘（下次打开保留）。"""
        try:
            self._save_panel_train_settings()
        except Exception:
            pass
        super().closeEvent(event)

    def _save_cls_train_settings(self):
        """Save ALL classification training settings to project dir (.cls_train_settings.json)."""
        if not self._project_dir:
            return
        import json as _json, os as _os
        settings_path = _os.path.join(self._project_dir, ".cls_train_settings.json")
        settings = {}
        if _os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as _f:
                    settings = _json.load(_f)
            except Exception:
                pass
        for key, attr, kind in self._CLS_PARAM_SPECS:
            w = getattr(self, attr, None)
            if w is None:
                continue
            try:
                if kind == "text":
                    settings[key] = w.currentText()
                elif kind == "bool":
                    settings[key] = w.isChecked()
                elif kind == "int":
                    settings[key] = int(w.value())
                else:
                    settings[key] = float(w.value())
            except Exception:
                pass
        try:
            with open(settings_path, "w", encoding="utf-8") as _f:
                _json.dump(settings, _f, indent=2)
        except Exception:
            pass

    def _load_cls_train_settings(self):
        """Restore ALL classification training settings from project dir."""
        if not self._project_dir:
            return
        import json as _json, os as _os
        settings_path = _os.path.join(self._project_dir, ".cls_train_settings.json")
        if not _os.path.exists(settings_path):
            return
        try:
            with open(settings_path, "r", encoding="utf-8") as _f:
                settings = _json.load(_f)
            for key, attr, kind in self._CLS_PARAM_SPECS:
                if key not in settings:
                    continue
                w = getattr(self, attr, None)
                if w is None:
                    continue
                try:
                    w.blockSignals(True)
                    if kind == "text":
                        idx = w.findText(str(settings[key]))
                        if idx >= 0:
                            w.setCurrentIndex(idx)
                    elif kind == "bool":
                        w.setChecked(bool(settings[key]))
                    elif kind == "int":
                        w.setValue(int(settings[key]))
                    else:
                        w.setValue(float(settings[key]))
                except Exception:
                    pass
                finally:
                    w.blockSignals(False)
        except Exception:
            pass

    def _load_cls_predictions(self):
        """Restore classification inference results from classification_results.json.

        Called when opening a project so the Result column and heatmap state survive restarts.
        """
        if not self.current_project or not self._project_dir:
            return
        out_path = os.path.join(self._project_dir, "outputs", "classification", "classification_results.json")
        if not os.path.exists(out_path):
            return
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            return
        if not isinstance(results, dict) or not results:
            return
        self._cls_predictions = {}
        img_dir = os.path.join(self._project_dir, "images")
        for base, info in results.items():
            if not isinstance(info, dict):
                continue
            found = False
            for ext in (".bmp", ".png", ".jpg", ".jpeg"):
                img_path = os.path.join(img_dir, base + ext)
                if os.path.exists(img_path):
                    self._cls_predictions[img_path] = info
                    found = True
                    break
            self._cls_predictions[base] = info
        self.log(f"Restored {len(results)} classification result(s)")

    def _open_cls_recommend(self):
        """打开智能推荐训练参数弹窗。"""
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        try:
            from ui.recommend_dialog import RecommendationDialog
        except Exception as e:
            import traceback
            self.log_signal.emit(f"推荐弹窗打开失败: {e}")
            self.log_signal.emit(traceback.format_exc())
            return
        dlg = RecommendationDialog(project_dir, self)
        dlg.apply_requested.connect(self._apply_cls_recommend)
        dlg.exec_()

    def _apply_cls_recommend(self, params, start):
        """把推荐参数套用到训练面板控件。"""
        try:
            self.cls_model_combo.setCurrentText(params["model"])
            self.cls_scale_w_spin.setValue(params["scale_w"])
            self.cls_scale_h_spin.setValue(params["scale_h"])
            self.cls_epochs_spin.setValue(params["epochs"])
            self.cls_batch_spin.setValue(params["batch_size"])
            self.cls_lr_spin.setValue(params["lr"])
            self.cls_optim_combo.setCurrentText(params["optimizer"])
            self.cls_loss_combo.setCurrentText(params["loss"])
            self._set_augment_level(params["augment"])
            self.cls_amp_check.setChecked(params["use_amp"])
            self.cls_kfold_spin.setValue(params["k_folds"])
            self.cls_wd_spin.setValue(params["weight_decay"])
            self.cls_momentum_spin.setValue(params["momentum"])
            self.cls_ls_spin.setValue(params["label_smoothing"])
            self.cls_focal_gamma_spin.setValue(params["focal_gamma"])
            self.cls_dropout_spin.setValue(params["dropout"])
            self.cls_scheduler_combo.setCurrentText(params["lr_scheduler"])
            self.cls_warmup_spin.setValue(params["warmup_epochs"])
            self.cls_early_stop_spin.setValue(params["early_stop_patience"])
            self.cls_resume_check.setChecked(params["resume"])
            self.cls_balanced_check.setChecked(bool(params.get("class_balanced", False)))
            self.cls_ema_check.setChecked(bool(params.get("ema", True)))
            self.cls_tta_check.setChecked(bool(params.get("tta", False)))
            self.log_signal.emit(
                f"[智能推荐] 已套用: {params['model']}, {params['epochs']}ep, "
                f"batch={params['batch_size']}, lr={params['lr']}, loss={params['loss']}, "
                f"scale={params['scale_w']}x{params['scale_h']}, "
                f"类平衡={params.get('class_balanced', False)}")
        except Exception as e:
            import traceback
            self.log_signal.emit(f"[智能推荐] 套用失败: {e}")
            self.log_signal.emit(traceback.format_exc())
            return
        if start:
            self._start_cls_training()

    def _apply_cls_native_params(self):
        """一键复位为最原生基线参数（无增强/EMA/类平衡/调度，早停20）。"""
        try:
            self.cls_model_combo.setCurrentText("resnet18")
            self.cls_epochs_spin.setValue(120)
            self.cls_batch_spin.setValue(32)
            self.cls_optim_combo.setCurrentText("AdamW")
            self.cls_lr_spin.setValue(0.001)
            self.cls_loss_combo.setCurrentText("cross_entropy")
            self.cls_scale_w_spin.setValue(1.0)
            self.cls_scale_h_spin.setValue(1.0)
            self.cls_scale_link_btn.setChecked(True)
            self.cls_amp_check.setChecked(False)
            self.cls_kfold_spin.setValue(1)
            self.cls_resume_check.setChecked(False)
            self.cls_balanced_check.setChecked(False)
            self.cls_ema_check.setChecked(False)
            self.cls_wd_spin.setValue(0.0001)
            self.cls_momentum_spin.setValue(0.9)
            self.cls_ls_spin.setValue(0.0)
            self.cls_focal_gamma_spin.setValue(2.0)
            self.cls_dropout_spin.setValue(0.0)
            self.cls_scheduler_combo.setCurrentText("none")
            self.cls_warmup_spin.setValue(0)
            self.cls_early_stop_spin.setValue(20)
            if hasattr(self, "cls_select_metric_combo"):
                self.cls_select_metric_combo.setCurrentText("acc")
            self._set_augment_level("none")
            self.cls_tta_check.setChecked(False)
            self.log_signal.emit(
                "[原生基线] 已套用: resnet18, 120ep, batch=32, lr=0.001, CE, "
                "scale=1.0, 无增强/EMA/类平衡/调度, 早停20")
        except Exception as e:
            import traceback
            self.log_signal.emit(f"[原生基线] 套用失败: {e}")
            self.log_signal.emit(traceback.format_exc())

    def _start_cls_training(self):
        """Start classification model training."""
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        model_name = self.cls_model_combo.currentText()
        epochs = self.cls_epochs_spin.value()
        batch_size = self.cls_batch_spin.value()
        scale_w = self.cls_scale_w_spin.value()
        scale_h = self.cls_scale_h_spin.value()
        scale_factor = (scale_w, scale_h) if scale_w != 1.0 or scale_h != 1.0 else 1.0
        self._save_cls_train_settings()  # persist settings (incl. input size / keep_aspect)
        mixed_mode = self.cls_mixed_check.isChecked()
        keep_aspect = self.cls_keep_aspect_check.isChecked()
        if mixed_mode:
            image_size = None  # 双输入模式保持原行为
        else:
            # 固定输入尺寸 + letterbox（保持宽高比），避免长条图被硬压成正方形
            image_size = (self.cls_input_w_spin.value(), self.cls_input_h_spin.value())
        lr = self.cls_lr_spin.value()
        optim_name = self.cls_optim_combo.currentText()
        loss_name = self.cls_loss_combo.currentText()
        augment = self._get_augment_flags()
        use_amp = self.cls_amp_check.isChecked()
        k_folds = self.cls_kfold_spin.value()
        resume = self.cls_resume_check.isChecked()
        weight_decay = self.cls_wd_spin.value()
        momentum_val = self.cls_momentum_spin.value()
        label_smoothing = self.cls_ls_spin.value()
        focal_gamma = self.cls_focal_gamma_spin.value()
        dropout = self.cls_dropout_spin.value()
        lr_scheduler = self.cls_scheduler_combo.currentText()
        warmup_epochs = self.cls_warmup_spin.value()
        early_stop_patience = self.cls_early_stop_spin.value()
        # ★ 这三项原本写在 run() 里 train(...) 的调用参数中现读控件，
        #   而 run() 跑在子线程 —— 子线程读控件和写控件一样是未定义行为。
        #   统一在**主线程**先读成局部快照，线程里只认这些值。
        cls_balanced_flag = self.cls_balanced_check.isChecked()
        cls_ema_flag = self.cls_ema_check.isChecked()
        cls_select_metric = self.cls_select_metric_combo.currentText()

        mode_label = "Mixed Classification" if mixed_mode else "Classification"
        self.log(f"{mode_label}: {model_name}, {epochs} epochs, batch={batch_size}, lr={lr}")
        self.cls_train_progress.setVisible(True)
        self.cls_train_progress.setValue(0)
        self.cls_train_status.setText("Preparing...")
        self._cls_stop_btn.setEnabled(True)
        self._cls_start_btn.setEnabled(False)
        self._stop_flag = False
        # Auto-pop the independent Loss/Acc window when training starts
        if hasattr(self, "_cls_loss_window"):
            self._cls_loss_window.show()
            self._cls_loss_window.raise_()
            self._cls_loss_window.activateWindow()
        self._cls_scale_ratio = 1.0

        def run():
            try:
                if mixed_mode:
                    from classification.trainer import DualClassificationTrainer
                    self._cls_trainer = DualClassificationTrainer(project_dir, model_name=model_name)
                else:
                    from classification.trainer import ClassificationTrainer
                    self._cls_trainer = ClassificationTrainer(project_dir, model_name=model_name)
                def epoch_cb(epoch, total, train_loss=0, val_acc=0):
                    if self._stop_flag:
                        self._cls_trainer._stop_flag = True
                    self.progress_signal.emit(epoch, total)
                    ui_call(self.cls_train_status.setText, f"Epoch {epoch}/{total}")
                def stop_check():
                    return self._stop_flag
                self._cls_trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    optimizer=optim_name, loss_func=loss_name,
                    image_size=image_size, scale_factor=scale_factor, use_amp=use_amp,
                    augment=augment, k_folds=k_folds, resume=resume,
                    weight_decay=weight_decay, momentum=momentum_val,
                    label_smoothing=label_smoothing, focal_gamma=focal_gamma,
                    dropout=dropout,
                    lr_scheduler=lr_scheduler, warmup_epochs=warmup_epochs,
                    early_stop_patience=early_stop_patience,
                    class_balanced=cls_balanced_flag,
                    ema=cls_ema_flag,
                    keep_aspect=keep_aspect,
                    select_metric=cls_select_metric,
                    progress_callback=epoch_cb, stop_check=stop_check,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                    plot_callback=lambda h: self.chart_signal.emit(h),
                )
                ui_call(self._refresh_cls_model_list)
                ui_call(self._refresh_cls_versions)
                ui_call(self._update_cls_loss_curve)
                self.log_signal.emit(f"{mode_label} training complete!")
            except Exception as e:
                import traceback
                self.log_signal.emit(f"{mode_label} training error: {e}")
                self.log_signal.emit(traceback.format_exc())
            finally:
                # 子线程不碰控件：统一排队回主线程执行
                ui_call(self.cls_train_progress.setVisible, False)
                ui_call(self.cls_train_status.setText, "Ready")
                ui_call(self._cls_stop_btn.setEnabled, False)
                ui_call(self._cls_start_btn.setEnabled, True)
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _on_cls_mixed_toggled(self, checked):
        """Handle mixed mode checkbox toggle."""
        self._mixed_cls_mode = checked
        self.canvas._mixed_cls_mode = checked
        # Show/hide mixed view buttons in canvas header
        if hasattr(self, "_mixed_view_group"):
            for btn in self._mixed_view_group:
                btn.setVisible(checked)
        if checked:
            self.log("Mixed mode ON: using grayscale + height map pairs")
            # Reload current image as mixed pair (load height map)
            current_path = getattr(self.canvas, "current_image_path", None)
            if current_path and os.path.exists(current_path):
                row = self.image_list_widget.currentRow()
                if row >= 0:
                    self._load_image_by_index(row)
        """Handle mixed mode checkbox toggle."""
        self._mixed_cls_mode = checked
        if checked:
            self.log("Mixed mode ON: using grayscale + height map pairs")

    def _update_cls_loss_curve(self):
        """Update classification loss/acc curve in the independent window."""
        if not hasattr(self, "_cls_trainer") or self._cls_trainer is None:
            return
        history = getattr(self._cls_trainer, "history", None)
        if history is None:
            return
        self._ensure_loss_window().update_curve(history)

    def _show_cls_loss_window(self):
        """Reopen the Loss/Acc curve window (after user closed it)."""
        win = self._ensure_loss_window()
        win.show()
        win.raise_()
        win.activateWindow()

    def _refresh_cls_model_list(self):
        """Refresh classification model list for inference (legacy .pth + versions/V{n})."""
        if not self.current_project:
            return
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        models_dir = os.path.join(project_dir, "models", "classification")
        prev = self.cls_infer_model_combo.currentText()
        self.cls_infer_model_combo.clear()
        if os.path.exists(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.endswith(".pth"):
                    self.cls_infer_model_combo.addItem(f)
            versions_dir = os.path.join(models_dir, "versions")
            if os.path.isdir(versions_dir):
                vers = []
                for v in os.listdir(versions_dir):
                    if v.startswith("V") and os.path.isfile(os.path.join(versions_dir, v, "model.pth")):
                        vers.append(v)
                for v in sorted(vers, key=lambda s: (len(s), s)):
                    self.cls_infer_model_combo.addItem(v)
        if self.cls_infer_model_combo.count() == 0:
            self.cls_infer_model_combo.addItem("(no models found)")
        else:
            idx = self.cls_infer_model_combo.findText(prev)
            if idx >= 0:
                self.cls_infer_model_combo.setCurrentIndex(idx)

    def _start_cls_inference(self):
        """Run classification inference on current image."""
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        self._refresh_cls_model_list()
        model_file = self.cls_infer_model_combo.currentText()
        if not model_file or "(no models" in model_file:
            return QMessageBox.warning(self, "Error", "No classification model found.")
        current_path = getattr(self.canvas, "current_image_path", None)
        if not current_path or not os.path.exists(current_path):
            return QMessageBox.warning(self, "Warning", "No image selected")
        top_k = self.cls_topk_spin.value()
        mixed_mode = self.cls_mixed_check.isChecked()
        # ★ run() 跑在子线程，里面对控件的读取/写入都要先在主线程取快照
        tta_on = self.cls_tta_check.isChecked()
        auto_heatmap_on = (hasattr(self, "cls_auto_heatmap")
                           and self.cls_auto_heatmap.isChecked())
        self.cls_infer_status.setText("Running...")
        def run():
            import time
            try:
                if mixed_mode:
                    from classification.trainer import DualClassificationTrainer
                    ct = DualClassificationTrainer(project_dir)
                    base = os.path.splitext(current_path)[0]
                    height_path = None
                    for ext in [".tif", ".tiff"]:
                        candidate = base + ext
                        if os.path.exists(candidate):
                            height_path = candidate
                            break
                    if not height_path:
                        self.log_signal.emit("Mixed inference: no paired height map found")
                        ui_call(self.cls_result_label.setText, "No paired height map")
                        return
                    t0 = time.time()
                    result = ct.predict(current_path, height_path)
                    elapsed = time.time() - t0
                    if result:
                        if not hasattr(self, "_cls_predictions"):
                            self._cls_predictions = {}
                        self._cls_predictions[current_path] = result
                        self.log_signal.emit(
                            f"Mixed: {result['class']} ({result['confidence']:.2%}) in {elapsed:.2f}s")
                        ui_call(self.cls_result_label.setText,
                                f"{result['class']}: {result['confidence']:.4f}")
                    else:
                        ui_call(self.cls_result_label.setText, "No result")
                        self.log_signal.emit("Mixed: no result")
                else:
                    from classification.trainer import ClassificationTrainer
                    ct = ClassificationTrainer(project_dir)
                    t0 = time.time()
                    result = ct.predict_single(current_path, model_file, top_k=top_k,
                                              tta=tta_on)
                    elapsed = time.time() - t0
                    if result and result.get("predictions"):
                        if not hasattr(self, "_cls_predictions"):
                            self._cls_predictions = {}
                        self._cls_predictions[current_path] = result
                        lines = [f"{p['class']}: {p['confidence']:.4f}" for p in result["predictions"]]
                        self.log_signal.emit(
                            f"Classification: {result['predictions'][0]['class']} "
                            f"({result['predictions'][0]['confidence']:.2%}) in {elapsed:.2f}s")
                        ui_call(self.cls_result_label.setText, "\n".join(lines))
                        if auto_heatmap_on and result and result.get("predictions"):
                            self._generate_cls_heatmap()
                    else:
                        ui_call(self.cls_result_label.setText, "No result")
                        self.log_signal.emit("Classification: no result")
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Classification inference error: {e}")
                ui_call(self.cls_result_label.setText, f"Error: {e}")
            finally:
                ui_call(self.cls_infer_status.setText, "Ready")
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _start_cls_batch_inference(self):
        """Run batch classification inference on all images."""
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        self._refresh_cls_model_list()
        model_file = self.cls_infer_model_combo.currentText()
        if not model_file or "(no models" in model_file:
            return QMessageBox.warning(self, "Error", "No classification model found.")
        img_dir = os.path.join(project_dir, "images")
        out_dir = os.path.join(project_dir, "outputs", "classification")
        os.makedirs(out_dir, exist_ok=True)
        mixed_mode = self.cls_mixed_check.isChecked()

        if mixed_mode:
            from classification.dual_dataset import _find_pairs
            pairs = _find_pairs(project_dir, "all", {})
            if not pairs:
                return QMessageBox.warning(self, "Warning", "No paired gray+height images found")
            images = [(gray, height) for gray, height, _ in pairs]
        else:
            images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                      if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))]
            if not images:
                return QMessageBox.warning(self, "Warning", "No images found")
        total = len(images)
        top_k = self.cls_topk_spin.value()
        # ★ run() 跑在子线程，里面不能再读控件：这里先取快照。
        tta_on = self.cls_tta_check.isChecked()
        auto_heatmap_on = (hasattr(self, "cls_auto_heatmap")
                           and self.cls_auto_heatmap.isChecked())
        self._stop_flag = False
        self.cls_infer_progress.setVisible(True)
        self.cls_infer_progress.setMaximum(total)
        self.cls_infer_progress.setValue(0)
        self.cls_infer_status.setText("Running...")
        self.log(f"Batch classification on {total} images...")
        def run():
            import time, json as _json
            results = {}
            infer_times = []
            try:
                if mixed_mode:
                    from classification.trainer import DualClassificationTrainer
                    ct = DualClassificationTrainer(project_dir)
                    ct.load_model(model_file)
                    for i, (gray_path, height_path) in enumerate(images):
                        if self._stop_flag:
                            self.log_signal.emit(f"Batch classification stopped at {i}/{total}")
                            break
                        gray_path, height_path = images[i]
                        base = os.path.splitext(os.path.basename(gray_path))[0]
                        t0 = time.time()
                        try:
                            r = ct.predict(gray_path, height_path)
                            elapsed = time.time() - t0
                            infer_times.append(elapsed)
                            if r:
                                results[base] = {"class": r["class"], "confidence": r["confidence"]}
                                self.log_signal.emit(
                                    f"[{i+1}/{total}] {base}: {r['class']} ({r['confidence']:.2%}, {elapsed:.2f}s)")
                            else:
                                self.log_signal.emit(f"[{i+1}/{total}] {base}: no result")
                        except Exception as ex:
                            self.log_signal.emit(f"[{i+1}/{total}] {base}: FAILED - {ex}")
                        self.infer_progress_signal.emit(i + 1, total)
                else:
                    from classification.trainer import ClassificationTrainer
                    ct = ClassificationTrainer(project_dir)
                    ct.load_model(model_file)
                    self.log_signal.emit(
                        f"Batch inference: device={ct.device}, TTA={tta_on}, "
                        f"keep_aspect={getattr(ct, '_keep_aspect', False)}, model={model_file}")
                    for i, img_path in enumerate(images):
                        if self._stop_flag:
                            self.log_signal.emit(f"Batch classification stopped at {i}/{total}")
                            break
                        base = os.path.splitext(os.path.basename(img_path))[0]
                        t0 = time.time()
                        try:
                            result = ct.predict_single(img_path, model_file, top_k=top_k, tta=tta_on)
                            elapsed = time.time() - t0
                            infer_times.append(elapsed)
                            if result and result.get("predictions"):
                                pred = result["predictions"][0]
                                results[base] = {"class": pred["class"], "confidence": pred["confidence"]}
                                self.log_signal.emit(
                                    f"[{i+1}/{total}] {base}: {pred['class']} ({pred['confidence']:.2%}, {elapsed:.2f}s)")
                            else:
                                self.log_signal.emit(f"[{i+1}/{total}] {base}: no result")
                        except Exception as ex:
                            self.log_signal.emit(f"[{i+1}/{total}] {base}: FAILED - {ex}")
                        self.infer_progress_signal.emit(i + 1, total)
                out_path = os.path.join(out_dir, "classification_results.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    _json.dump(results, f, indent=2, ensure_ascii=False)
                self.log_signal.emit(f"Batch classification complete! {len(results)} results saved to {out_path}")
                self._cls_predictions = {}
                for base, info in results.items():
                    for ext in (".bmp", ".png", ".jpg", ".jpeg"):
                        img_path = os.path.join(img_dir, base + ext)
                        if os.path.exists(img_path):
                            self._cls_predictions[img_path] = info
                            break
                    if base not in [os.path.splitext(k)[0] for k in self._cls_predictions]:
                        self._cls_predictions[base] = info
                self.refresh_list_signal.emit()
                # Auto-generate heatmaps for ALL images after batch inference
                if not mixed_mode and auto_heatmap_on:
                    self.log_signal.emit("Generating heatmaps for all images...")
                    hm_dir = os.path.join(out_dir, "heatmaps")
                    os.makedirs(hm_dir, exist_ok=True)
                    try:
                        from PIL import Image
                        import numpy as np
                        from classification.trainer import ClassificationTrainer
                        from classification.gradcam import generate_gradcam_heatmap
                        ct2 = ClassificationTrainer(project_dir)
                        ct2.load_model(model_file)
                        rsz = ct2._resolve_inference_resize()
                        model_nm = getattr(ct2, "model_name", "resnet18")
                        hm_total = len(images)
                        hm_done = 0
                        for img_path in images:
                            if self._stop_flag:
                                break
                            base = os.path.splitext(os.path.basename(img_path))[0]
                            hm_file = os.path.join(hm_dir, base + "_heatmap.jpg")
                            try:
                                img = Image.open(img_path).convert("RGB")
                                img_np = np.array(img)
                                # Find target class from predictions
                                target_cls = None
                                info = results.get(base, {})
                                cls_name = info.get("class", "") if isinstance(info, dict) else ""
                                if cls_name and hasattr(ct2, "class_names"):
                                    try:
                                        target_cls = ct2.class_names.index(cls_name)
                                    except ValueError:
                                        pass
                                overlay = generate_gradcam_heatmap(
                                    ct2.model, img_np,
                                    model_name=model_nm,
                                    target_class=target_cls,
                                    resize_size=rsz,
                                    keep_aspect=getattr(ct2, "_keep_aspect", False),
                                )
                                if overlay is not None:
                                    Image.fromarray(overlay).save(hm_file, quality=85)
                                hm_done += 1
                            except Exception:
                                pass
                        self.log_signal.emit(f"Heatmaps saved: {hm_done}/{hm_total} to {hm_dir}")
                        # Load heatmap for current image
                        current_base = os.path.splitext(os.path.basename(
                            getattr(self.canvas, "current_image_path", "")))[0]
                        current_hm = os.path.join(hm_dir, current_base + "_heatmap.jpg")
                        self.canvas.load_heatmap_from_file(current_hm)
                    except Exception as e:
                        import traceback
                        self.log_signal.emit(f"Heatmap batch error: {e}")
                        traceback.print_exc()
                ui_call(self.cls_infer_status.setText, "Complete")
                # Compute confusion matrix from batch results
                try:
                    from classification.label_manager import LabelManager as ClsLabelManager
                    lm = ClsLabelManager(project_dir)
                    label_data = lm.load()
                    gt_mapping = label_data.get("mapping", {})
                    class_names = label_data.get("labels", [])
                    if len(class_names) < 2:
                        class_names = ["NG", "OK"]
                    self.log_signal.emit(
                        f"[批推] 模型类别顺序={getattr(ct, 'class_names', '?')}  标签表类别顺序={class_names}")
                    n = len(class_names)
                    cm = [[0]*n for _ in range(n)]
                    matched = 0
                    gt_map = {}  # base_name -> gt class name
                    pred_map = {}  # base_name -> pred class name
                    for base, info in results.items():
                        pred_cls = info.get("class", "")
                        # Find ground truth
                        gt_cls = None
                        for rel_path, cls_id in gt_mapping.items():
                            if base in rel_path or rel_path.endswith("/" + base):
                                gt_cls = class_names[cls_id] if cls_id < n else None
                                break
                        if gt_cls is None:
                            continue
                        pred_idx = class_names.index(pred_cls) if pred_cls in class_names else -1
                        gt_idx = class_names.index(gt_cls)
                        if pred_idx >= 0:
                            cm[gt_idx][pred_idx] += 1
                            if pred_idx == gt_idx:
                                matched += 1
                        gt_map[base] = gt_cls
                        pred_map[base] = pred_cls
                    total_matched = sum(sum(row) for row in cm)
                    acc = matched / total_matched if total_matched > 0 else 0
                    # Update table
                    # 先做纯计算（display_order / cm_display / row_sums 都不碰控件，留在子线程即可）
                    display_order = list(range(n))
                    if "OK" in class_names and "NG" in class_names:
                        ok_idx = class_names.index("OK")
                        ng_idx = class_names.index("NG")
                        if ok_idx > ng_idx:
                            display_order = [ok_idx, ng_idx] if n == 2 else sorted(range(n), key=lambda i: (class_names[i] != "OK", class_names[i] != "NG", class_names[i]))
                    display_names = [class_names[i] for i in display_order]
                    # Reorder cm matrix
                    cm_display = [[cm[display_order[i]][display_order[j]] for j in range(n)] for i in range(n)]
                    headers = display_names + ["Total"]
                    row_sums = [sum(row) for row in cm_display]
                    col_sums = [sum(cm_display[r][c] for r in range(n)) for c in range(n)]
                    cm_total = sum(row_sums)

                    # 填表必须在主线程：setRowCount / setItem 全是 GUI 操作。
                    # 这里用 ui_call_sync 而不是 ui_call —— 下面 _cls_cm_data 紧接着就要用到，
                    # 异步的话主线程还没填完，子线程已经往下跑了。
                    def _render_cm():
                        self.cls_cm_table.setRowCount(n + 1)
                        self.cls_cm_table.setColumnCount(n + 1)
                        self.cls_cm_table.setHorizontalHeaderLabels(headers)
                        self.cls_cm_table.setVerticalHeaderLabels(display_names + ["Total"])
                        for i in range(n):
                            for j in range(n):
                                item = QTableWidgetItem(str(cm_display[i][j]))
                                item.setTextAlignment(Qt.AlignCenter)
                                if i == j:
                                    item.setBackground(QColor(180, 255, 180)); item.setForeground(QColor("#0A0E17"))
                                self.cls_cm_table.setItem(i, j, item)
                            item = QTableWidgetItem(str(row_sums[i]))
                            item.setTextAlignment(Qt.AlignCenter)
                            self.cls_cm_table.setItem(i, n, item)
                        for j in range(n):
                            item = QTableWidgetItem(str(col_sums[j]))
                            item.setTextAlignment(Qt.AlignCenter)
                            self.cls_cm_table.setItem(n, j, item)
                        item = QTableWidgetItem(str(cm_total))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.cls_cm_table.setItem(n, n, item)
                        self._size_cm_table()
                        self.cls_eval_status.setText("Acc: {:.2%}  ({}/{} matched)".format(acc, matched, total_matched))

                    ui_call_sync(_render_cm)
                    # Store per-image mapping for click-to-filter
                    self._cls_cm_data = {"cm": cm_display, "names": display_names, "gt_map": gt_map, "pred_map": pred_map}
                    # Persist to disk
                    cm_path = os.path.join(out_dir, "confusion_matrix.json")
                    try:
                        with open(cm_path, "w", encoding="utf-8") as cmf:
                            json.dump(self._cls_cm_data, cmf, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                    # 评估结果随版本入包
                    try:
                        self._write_cls_version_result(
                            model_file, cm_display, display_names, gt_map, pred_map,
                            acc, matched, total_matched, infer_times)
                    except Exception:
                        pass
                    ui_call(self._refresh_cls_versions)

                    # 重新绑定单元格点击：disconnect/connect 都是 GUI 操作，得回主线程
                    def _rebind_cm_click():
                        try:
                            self.cls_cm_table.cellClicked.disconnect()
                        except TypeError:
                            pass      # 本来就没连，忽略
                        self.cls_cm_table.cellClicked.connect(self._on_cm_cell_clicked)

                    ui_call(_rebind_cm_click)
                except Exception as ex:
                    self.log_signal.emit("CM error: " + str(ex))
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Batch classification error: {e}")
                self.log_signal.emit(traceback.format_exc())
                ui_call(self.cls_infer_status.setText, "Failed")
            finally:
                ui_call(self.cls_infer_progress.setVisible, False)
        import threading
        threading.Thread(target=run, daemon=True).start()




    def _generate_cls_heatmap(self):
        """Generate Grad-CAM heatmap for current classification result."""
        if not self.current_project:
            return
        current_path = getattr(self.canvas, "current_image_path", None)
        if not current_path or not os.path.exists(current_path):
            return
        self.cls_infer_status.setText("Generating heatmap...")
        import threading
        threading.Thread(target=self._do_generate_heatmap, daemon=True).start()


    def _do_generate_heatmap(self):
        """Core heatmap generation (synchronous, call from any thread)."""
        current_path = getattr(self.canvas, "current_image_path", None)
        if not current_path or not os.path.exists(current_path):
            self.log_signal.emit("Heatmap: no current image")
            return
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        # Scan disk directly (thread-safe) instead of reading UI combo
        models_dir = os.path.join(project_dir, "models", "classification")
        model_file = None
        if os.path.exists(models_dir):
            pth_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pth")])
            if pth_files:
                model_file = pth_files[-1]  # Use most recent model
        if not model_file:
            self.log_signal.emit("Heatmap: no model found")
            return
        try:
            from PIL import Image
            import numpy as np
            from classification.trainer import ClassificationTrainer
            from classification.gradcam import generate_gradcam_heatmap
            from PyQt5.QtGui import QImage
            img = Image.open(current_path).convert("RGB")
            img_np = np.array(img)
            ct = ClassificationTrainer(project_dir)
            ct.load_model(model_file)
            if ct.model is None:
                self.log_signal.emit("Heatmap: model not loaded")
                return
            rsz = ct._resolve_inference_resize()
            cls_preds = getattr(self, "_cls_predictions", {}).get(current_path, None)
            target_class = None
            if cls_preds is not None and isinstance(cls_preds, dict):
                top_pred = cls_preds.get("predictions", [{}])[0] if cls_preds.get("predictions") else {}
                class_name = top_pred.get("class", "")
                if class_name and hasattr(ct, "class_names"):
                    try:
                        target_class = ct.class_names.index(class_name)
                    except ValueError:
                        pass
            overlay = generate_gradcam_heatmap(
                ct.model, img_np,
                model_name=getattr(ct, "model_name", "resnet18"),
                target_class=target_class,
                resize_size=rsz,
                keep_aspect=getattr(ct, "_keep_aspect", False),
            )
            if overlay is not None:
                h, w, ch = overlay.shape
                qimg = QImage(overlay.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
                self.canvas.set_heatmap(qimg)
                self.canvas._show_heatmap = True
                self.log_signal.emit("Heatmap generated (Space to toggle)")
            else:
                self.log_signal.emit("Heatmap: model unsupported (ViT?) or failed")
        except Exception as e:
            import traceback
            self.log_signal.emit(f"Heatmap error: {e}")
            traceback.print_exc()

    def _size_cm_table(self):
        """Compact confusion-matrix layout: clamp column widths (cell + header),
        wrap long class names, fit row heights, and highlight Total row/col."""
        table = self.cls_cm_table
        fm = table.fontMetrics()
        pad = 14
        min_w, max_w = 34, 170
        for col in range(table.columnCount()):
            hdr = table.horizontalHeaderItem(col)
            if hdr is not None:
                hdr.setToolTip(hdr.text())
            w = fm.horizontalAdvance(hdr.text()) if hdr is not None else 0
            for row in range(table.rowCount()):
                item = table.item(row, col)
                if item is not None:
                    w = max(w, fm.horizontalAdvance(item.text()))
            table.setColumnWidth(col, max(min_w, min(w + pad, max_w)))
        table.resizeRowsToContents()
        # Highlight the Total row/column
        n = table.rowCount() - 1
        if n >= 1 and table.rowCount() == table.columnCount():
            bold = QFont(table.font()); bold.setBold(True)
            hl = QColor("#1B2740")  # 深色主题：汇总行/列用略亮深底，浅色字清晰
            for j in range(table.columnCount()):
                item = table.item(n, j)
                if item is not None:
                    item.setFont(bold); item.setBackground(hl)
            for i in range(table.rowCount()):
                item = table.item(i, n)
                if item is not None:
                    item.setFont(bold); item.setBackground(hl)

    def _selected_cls_version(self):
        """返回版本表当前选中行的版本名，无则 None。"""
        if not hasattr(self, "cls_version_table"):
            return None
        row = self.cls_version_table.currentRow()
        if row < 0:
            return None
        item = self.cls_version_table.item(row, 0)
        return item.text() if item else None

    def _refresh_cls_versions(self):
        """刷新模型版本表：版本 / 时间 / 验证Acc / 批推Acc / 平均耗时 / 模型。"""
        if not hasattr(self, "cls_version_table"):
            return
        self.cls_version_table.setRowCount(0)
        if not self.current_project:
            return
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        versions_dir = os.path.join(project_dir, "models", "classification", "versions")
        if not os.path.isdir(versions_dir):
            return
        rows = []
        for v in os.listdir(versions_dir):
            vdir = os.path.join(versions_dir, v)
            if not os.path.isdir(vdir) or not os.path.isfile(os.path.join(vdir, "model.pth")):
                continue
            meta, result = {}, {}
            mp = os.path.join(vdir, "meta.json")
            if os.path.exists(mp):
                try:
                    with open(mp, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            rp = os.path.join(vdir, "result.json")
            if os.path.exists(rp):
                try:
                    with open(rp, "r", encoding="utf-8") as f:
                        result = json.load(f)
                except Exception:
                    pass
            rows.append((v, meta, result))
        rows.sort(key=lambda r: (len(r[0]), r[0]))
        self.cls_version_table.setRowCount(len(rows))
        for i, (v, meta, result) in enumerate(rows):
            created = str(meta.get("created", ""))[:16]
            val_acc = meta.get("best_val_acc")
            val_s = f"{val_acc:.2%}" if isinstance(val_acc, (int, float)) else "-"
            acc = result.get("accuracy")
            acc_s = f"{acc:.2%}" if isinstance(acc, (int, float)) else "-"
            avg_t = result.get("avg_infer_time_ms")
            t_s = f"{avg_t:.1f}ms" if isinstance(avg_t, (int, float)) else "-"
            model = str(meta.get("model_name", ""))
            for c, txt in enumerate([v, created, val_s, acc_s, t_s, model]):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                self.cls_version_table.setItem(i, c, item)
        self.cls_version_table.resizeColumnsToContents()

    def _cls_version_activate(self):
        """把选中版本复制为 best_model.pth 并设为推理下拉的当前项（回滚）。"""
        v = self._selected_cls_version()
        if not v:
            return QMessageBox.warning(self, "提示", "请先在版本表选择一行")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        src = os.path.join(project_dir, "models", "classification", "versions", v, "model.pth")
        dst = os.path.join(project_dir, "models", "classification", "best_model.pth")
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            return QMessageBox.warning(self, "错误", f"设为当前失败: {e}")
        idx = self.cls_infer_model_combo.findText(v)
        if idx >= 0:
            self.cls_infer_model_combo.setCurrentIndex(idx)
        self.log_signal.emit(f"[模型版本] {v} 已复制为 best_model.pth")
        QMessageBox.information(self, "完成", f"{v} 已设为当前最佳模型（best_model.pth）。")

    def _cls_version_export(self):
        """把选中版本打包为部署 zip（模型 + meta + history + 评估结果）。"""
        v = self._selected_cls_version()
        if not v:
            return QMessageBox.warning(self, "提示", "请先在版本表选择一行")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        vdir = os.path.join(project_dir, "models", "classification", "versions", v)
        if not os.path.isdir(vdir):
            return QMessageBox.warning(self, "错误", f"版本目录不存在: {vdir}")
        default_name = os.path.join(project_dir, f"{os.path.basename(project_dir)}_{v}.zip")
        path, _ = QFileDialog.getSaveFileName(self, "导出部署包", default_name, "ZIP 文件 (*.zip)")
        if not path:
            return
        try:
            import zipfile
            meta, result = {}, {}
            mp = os.path.join(vdir, "meta.json")
            if os.path.exists(mp):
                with open(mp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            rp = os.path.join(vdir, "result.json")
            if os.path.exists(rp):
                with open(rp, "r", encoding="utf-8") as f:
                    result = json.load(f)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fn in sorted(os.listdir(vdir)):
                    fp = os.path.join(vdir, fn)
                    if os.path.isfile(fp):
                        zf.write(fp, f"{v}/{fn}")
                readme = self._build_version_readme(v, meta, result)
                zf.writestr(f"{v}/部署说明.txt", readme.encode("utf-8"))
        except Exception as e:
            import traceback
            self.log_signal.emit(traceback.format_exc())
            return QMessageBox.warning(self, "错误", f"导出失败: {e}")
        self.log_signal.emit(f"[模型版本] 部署包已导出: {path}")
        QMessageBox.information(self, "完成", f"部署包已导出:\n{path}")

    def _build_version_readme(self, v, meta, result):
        """生成部署包内的说明文本。"""
        lines = [
            f"模型版本: {v}",
            f"创建时间: {meta.get('created', '-')}",
            f"模型: {meta.get('model_name', '-')}  类别: {meta.get('class_names', '-')}",
            f"输入尺寸: {meta.get('image_size', '-')}  缩放: {meta.get('scale_factor', '-')}",
            f"验证准确率: {meta.get('best_val_acc', '-')}",
        ]
        for k in ["epochs", "batch_size", "lr", "optimizer", "loss", "class_balanced", "ema", "augment"]:
            if k in meta:
                lines.append(f"{k}: {meta[k]}")
        if result:
            lines.append("")
            lines.append(f"批量推理准确率: {result.get('accuracy', '-')}  ({result.get('matched', 0)}/{result.get('total', 0)})")
            if "avg_infer_time_ms" in result:
                lines.append(f"平均推理耗时: {result['avg_infer_time_ms']} ms")
            for name, m in (result.get("per_class") or {}).items():
                lines.append(f"  {name}: precision={m.get('precision')} recall={m.get('recall')} f1={m.get('f1')} samples={m.get('samples')}")
        return "\n".join(lines)

    def _write_cls_version_result(self, model_ref, cm_display, display_names, gt_map, pred_map,
                                  acc, matched, total, infer_times=None):
        """把批量推理/评估结果写入对应模型版本的 result.json（P/R/F1/耗时/CM）。"""
        if not self.current_project:
            return
        ref = str(model_ref or "")
        if not re.fullmatch(r"V\d+", ref, re.IGNORECASE):
            return  # 只归档到版本目录
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        vdir = os.path.join(project_dir, "models", "classification", "versions", ref)
        if not os.path.isdir(vdir):
            return
        n = len(display_names)
        per_class = {}
        for i, name in enumerate(display_names):
            tp = cm_display[i][i]
            fp = sum(cm_display[r][i] for r in range(n)) - tp
            fn = sum(cm_display[i][c] for c in range(n)) - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[name] = {
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(f1, 4), "samples": int(sum(cm_display[r][i] for r in range(n))),
            }
        result = {
            "version": ref,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": round(float(acc), 4),
            "matched": int(matched), "total": int(total),
            "cm": cm_display, "names": display_names,
            "per_class": per_class,
        }
        if gt_map:
            result["gt_map"] = gt_map
        if pred_map:
            result["pred_map"] = pred_map
        if infer_times:
            times = sorted(infer_times)
            result["avg_infer_time_ms"] = round(sum(times) / len(times) * 1000.0, 2)
            result["median_infer_time_ms"] = round(times[len(times) // 2] * 1000.0, 2)
            result["num_infer"] = len(times)
        rp = os.path.join(vdir, "result.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self.log_signal.emit(f"[模型版本] 评估结果已写入 {rp}")

    def _eval_cls_model(self):
        """Evaluate classification model and show confusion matrix."""
        if not self.current_project:
            return
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        self._refresh_cls_model_list()
        model_file = self.cls_infer_model_combo.currentText()
        if not model_file or "(no models" in model_file:
            return QMessageBox.warning(self, "Error", "No classification model found.")
        mixed_mode = self.cls_mixed_check.isChecked()
        self.cls_eval_status.setText("Evaluating...")
        def run():
            try:
                if mixed_mode:
                    self._eval_cls_model_mixed(project_dir, model_file)
                    return
                from classification.trainer import ClassificationTrainer
                ct = ClassificationTrainer(project_dir)
                ct.load_model(model_file)
                from classification.eval import evaluate_classification
                result = evaluate_classification(
                    project_dir, split="val",
                    model_path="classification/" + model_file,
                    batch_size=32, image_size=ct._image_size,
                    compute_confusion=True, top_k=(1, min(5, ct.num_classes)),
                    keep_aspect=ct._keep_aspect,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                )
                cm = result.get("confusion_matrix")
                class_names = result.get("class_names", [])
                top1 = result.get("top_k_results", {}).get("top1_acc", 0)
                top5 = result.get("top_k_results", {}).get("top5_acc", 0)
                self.log_signal.emit("Evaluation: Top-1={:.4f}, Top-5={:.4f}".format(top1, top5))
                if cm and len(class_names) > 0:
                    n = len(class_names)
                    # Reorder: OK first, NG second for display
                    display_order = list(range(n))
                    if "OK" in class_names and "NG" in class_names:
                        ok_idx = class_names.index("OK")
                        ng_idx = class_names.index("NG")
                        if ok_idx > ng_idx:
                            display_order = [ok_idx, ng_idx] if n == 2 else sorted(range(n), key=lambda i: (class_names[i] != "OK", class_names[i] != "NG", class_names[i]))
                    display_names = [class_names[i] for i in display_order]
                    # Reorder cm matrix
                    cm_display = [[cm[display_order[i]][display_order[j]] for j in range(n)] for i in range(n)]
                    headers = display_names + ["Total"]
                    row_sums = [sum(row) for row in cm_display]
                    col_sums = [sum(cm_display[r][c] for r in range(n)) for c in range(n)]
                    cm_total = sum(row_sums)

                    # 填表必须回主线程（setRowCount/setItem 都是 GUI 操作）
                    def _render_cm_eval():
                        self.cls_cm_table.setRowCount(n + 1)
                        self.cls_cm_table.setColumnCount(n + 1)
                        self.cls_cm_table.setHorizontalHeaderLabels(headers)
                        self.cls_cm_table.setVerticalHeaderLabels(display_names + ["Total"])
                        for i in range(n):
                            for j in range(n):
                                item = QTableWidgetItem(str(cm_display[i][j]))
                                item.setTextAlignment(Qt.AlignCenter)
                                if i == j:
                                    item.setBackground(QColor(180, 255, 180)); item.setForeground(QColor("#0A0E17"))
                                self.cls_cm_table.setItem(i, j, item)
                            item = QTableWidgetItem(str(row_sums[i]))
                            item.setTextAlignment(Qt.AlignCenter)
                            self.cls_cm_table.setItem(i, n, item)
                        for j in range(n):
                            item = QTableWidgetItem(str(col_sums[j]))
                            item.setTextAlignment(Qt.AlignCenter)
                            self.cls_cm_table.setItem(n, j, item)
                        item = QTableWidgetItem(str(cm_total))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.cls_cm_table.setItem(n, n, item)
                        self._size_cm_table()

                    ui_call_sync(_render_cm_eval)
                    try:
                        self._write_cls_version_result(
                            model_file, cm_display, display_names, {}, {}, top1, 0, 0, None)
                    except Exception:
                        pass
                    self._refresh_cls_versions()
                ui_call(self.cls_eval_status.setText,
                        "Top-1: {:.2%}  Top-5: {:.2%}".format(top1, top5))
            except Exception as e:
                import traceback
                self.log_signal.emit("Evaluation error: " + str(e))
                ui_call(self.cls_eval_status.setText, "Error: " + str(e))
        import threading
        threading.Thread(target=run, daemon=True).start()



    def _eval_cls_model_mixed(self, project_dir, model_file):
        """Evaluate mixed classification model on validation set."""
        import torch
        from classification.trainer import DualClassificationTrainer
        from classification.dual_dataset import MixedClassificationDataset
        ct = DualClassificationTrainer(project_dir)
        ct.load_model(model_file)
        val_ds = MixedClassificationDataset(project_dir, split="val", image_size=getattr(ct, "_image_size", None) or 224)
        if len(val_ds) == 0:
            self.log_signal.emit("Mixed eval: no validation samples")
            return
        loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
        ct.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for (gray, height), labels in loader:
                gray = gray.to(ct.device)
                height = height.to(ct.device)
                labels = labels.to(ct.device)
                inputs = torch.cat([gray, height], dim=1)
                outputs = ct.model(inputs)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        acc = correct / total if total > 0 else 0
        self.log_signal.emit(f"Mixed eval: Top-1={acc:.4f} ({correct}/{total})")
        class_names = getattr(ct, "class_names", ["NG", "OK"])
        n = len(class_names)
        cm_matrix = [[0]*n for _ in range(n)]
        for gt, pred in zip(all_labels, all_preds):
            if 0 <= gt < n and 0 <= pred < n:
                cm_matrix[gt][pred] += 1
        display_order = list(range(n))
        if "OK" in class_names and "NG" in class_names:
            ok_idx = class_names.index("OK")
            ng_idx = class_names.index("NG")
            if ok_idx > ng_idx:
                display_order = [ok_idx, ng_idx]
        display_names = [class_names[i] for i in display_order]
        cm_display = [[cm_matrix[display_order[i]][display_order[j]] for j in range(n)] for i in range(n)]
        headers = display_names + ["Total"]
        self.cls_cm_table.setRowCount(n + 1)
        self.cls_cm_table.setColumnCount(n + 1)
        self.cls_cm_table.setHorizontalHeaderLabels(headers)
        self.cls_cm_table.setVerticalHeaderLabels(display_names + ["Total"])
        row_sums = [sum(row) for row in cm_display]
        col_sums = [sum(cm_display[r][c] for r in range(n)) for c in range(n)]
        cm_total = sum(row_sums)
        from PyQt5.QtGui import QColor
        from PyQt5.QtCore import Qt
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem(str(cm_display[i][j]))
                item.setTextAlignment(Qt.AlignCenter)
                if i == j:
                    item.setBackground(QColor(180, 255, 180)); item.setForeground(QColor("#0A0E17"))
                self.cls_cm_table.setItem(i, j, item)
            item = QTableWidgetItem(str(row_sums[i]))
            item.setTextAlignment(Qt.AlignCenter)
            self.cls_cm_table.setItem(i, n, item)
        for j in range(n):
            item = QTableWidgetItem(str(col_sums[j]))
            item.setTextAlignment(Qt.AlignCenter)
            self.cls_cm_table.setItem(n, j, item)
        item = QTableWidgetItem(str(cm_total))
        item.setTextAlignment(Qt.AlignCenter)
        self.cls_cm_table.setItem(n, n, item)
        self._size_cm_table()
        self.cls_eval_status.setText(f"Top-1: {acc:.2%}")

    def _restore_confusion_matrix(self):
        """Restore confusion matrix from disk after project open."""
        if not self.current_project:
            return
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        cm_path = os.path.join(project_dir, "outputs", "classification", "confusion_matrix.json")
        if not os.path.exists(cm_path):
            return
        try:
            with open(cm_path, "r", encoding="utf-8") as cmf:
                data = json.load(cmf)
            cm = data.get("cm", [])
            names = data.get("names", [])
            gt_map = data.get("gt_map", {})
            pred_map = data.get("pred_map", {})
            if not cm or not names:
                return
            n = len(names)
            self.cls_cm_table.setRowCount(n + 1)
            self.cls_cm_table.setColumnCount(n + 1)
            headers = names + ["Total"]
            self.cls_cm_table.setHorizontalHeaderLabels(headers)
            self.cls_cm_table.setVerticalHeaderLabels(names + ["Total"])
            row_sums = [sum(row) for row in cm]
            col_sums = [sum(cm[r][c] for r in range(n)) for c in range(n)]
            cm_total = sum(row_sums)
            for i in range(n):
                for j in range(n):
                    item = QTableWidgetItem(str(cm[i][j]))
                    item.setTextAlignment(Qt.AlignCenter)
                    if i == j:
                        item.setBackground(QColor(180, 255, 180)); item.setForeground(QColor("#0A0E17"))
                    self.cls_cm_table.setItem(i, j, item)
                item = QTableWidgetItem(str(row_sums[i]))
                item.setTextAlignment(Qt.AlignCenter)
                self.cls_cm_table.setItem(i, n, item)
            for j in range(n):
                item = QTableWidgetItem(str(col_sums[j]))
                item.setTextAlignment(Qt.AlignCenter)
                self.cls_cm_table.setItem(n, j, item)
            item = QTableWidgetItem(str(cm_total))
            item.setTextAlignment(Qt.AlignCenter)
            self.cls_cm_table.setItem(n, n, item)
            self._size_cm_table()
            self._cls_cm_data = data
            self.cls_cm_table.cellClicked.disconnect() if self.cls_cm_table.receivers(self.cls_cm_table.cellClicked) > 0 else None
            self.cls_cm_table.cellClicked.connect(self._on_cm_cell_clicked)
            matched = sum(cm[i][i] for i in range(n))
            acc = matched / cm_total if cm_total > 0 else 0
            self.cls_eval_status.setText("Acc: {:.2%}  ({}/{} matched)".format(acc, matched, cm_total))
        except Exception:
            pass

    def _on_cm_cell_clicked(self, row, col):
        """Filter image list when clicking a confusion matrix cell."""
        import os
        data = getattr(self, "_cls_cm_data", None)
        if not data or row >= len(data["names"]) or col >= len(data["names"]):
            return
        gt_cls = data["names"][row]
        pred_cls = data["names"][col]
        matched = []
        for base, gt in data["gt_map"].items():
            pd = data["pred_map"].get(base, "")
            if gt == gt_cls and pd == pred_cls:
                for ext in (".bmp", ".png", ".jpg", ".jpeg"):
                    p = os.path.join(getattr(self, "_project_dir", ""), "images", base + ext)
                    if os.path.exists(p):
                        matched.append(base)
                        break
        if matched:
            self._cm_filter_bases = set(matched)
            self.search_input.blockSignals(True)
            self.search_input.setText("")
            self.search_input.blockSignals(False)
            self._filter_images("")
            self._cm_filter_bases = None
        self.log(f"CM click: {gt_cls}->{pred_cls} = {len(matched)} images")

    def _toggle_panel(self, side):
        """折叠/展开左栏或右栏。

        ★ 2026-10-15 改版（维护者 两轮反馈）：
          第一版问题：折叠按钮在面板**里面**，一按跟着消失 ⇒ 只能去 View 菜单展开。
          第二版问题：折叠 = 收窄成 26px 把手条 ⇒ 画布**没占满整宽**，两侧各留一条空白
                      （维护者："折叠后，预览窗口直接占满左右栏就好"）。
          现在：**折叠 = 整个面板藏起来**（宽度归零，画布吃满），恢复按钮是**贴在画布
          边缘的浮层小按钮**（不进布局 ⇒ 不占宽度），面板内那枚箭头只在展开态可见。
        """
        panel = self._left_panel if side == "left" else self._right_panel
        edge = getattr(self, "_edge_left_btn" if side == "left" else "_edge_right_btn", None)
        inner = getattr(self, "_left_toggle_btn" if side == "left"
                        else "_right_toggle_btn", None)
        expand = not panel.isVisible()
        panel.setVisible(expand)
        if inner is not None:
            inner.setVisible(expand)             # 展开态才显示面板内那枚箭头
        if edge is not None:
            edge.setVisible(not expand)          # 折叠态显示边缘恢复按钮
        self._save_ui_setting("%s_panel_visible" % side, expand)
        # 画布宽度变了：底栏读数、图片信息栏排版都要重量
        # （延后一轮 —— 点击处理中布局几何还没落定，立刻量拿到的是旧值）
        QTimer.singleShot(0, lambda: (self._layout_edge_buttons(),
                                      self._on_toolbar_collapsed(expand)))
        self._on_toolbar_collapsed(expand)








    def _update_page_label(self):
        sp = getattr(self.canvas, "_single_page", 0)
        self._page_label.setText(f"{sp + 1}/2")

    def _update_page_label(self):
        sp = getattr(self.canvas, "_single_page", 0)
        self._page_label.setText(f"{sp + 1}/2")

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

        # ★★ 2026-10-22：已经有训练在排队/执行时，别再起第二个。
        #   平台只有一个 self.trainer 槽位，同时开两个必然互相顶掉
        #   （用户实拍：点停止后下一个任务报 NoneType ... has no attribute 'train'）。
        #   与其让它半路炸，不如在这里说清楚。
        _q0 = getattr(self, "_task_queue", None)
        if _q0 is not None:
            try:
                from ui.task_queue import TaskKind as _TK0
                _busy = [t for t in _q0.active_tasks()
                         if getattr(t, "kind", None) == _TK0.TRAIN]
            except Exception:
                _busy = []
            if _busy:
                return QMessageBox.information(
                    self, "已有训练任务",
                    "训练任务「%s」还在排队或执行中。\n\n"
                    "平台同一时间只挂一个训练器，同时开两个会互相顶掉"
                    "（表现就是 NoneType object has no attribute train）。\n\n"
                    "等它跑完，或先在「任务队列」里取消它，再开始新的训练。"
                    % _busy[0].title)
        # ★ 不再弹「Training Settings」窗（用户要求）：参数直接读面板上那几个控件。
        #   要回退成弹窗模式：把下面这段换回
        #   dlg = TrainSettingsDialog(self, project_dir=project_dir)
        #   if dlg.exec_() != dlg.Accepted: return
        #   settings = dlg.get_settings()
        try:
            from training.multiview_gui import is_multiview_project
            _is_mv_here = bool(is_multiview_project(project_dir))
        except Exception:
            _is_mv_here = False
        # ★ 按工程类型校正模型下拉：多视图才有 unet/lraspp，单图只有 deeplabv3/fcn。
        #   （不校正的话，单图工程选到 unet 会 ValueError: Unknown model: unet）
        _want = (["unet", "lraspp", "deeplabv3", "fcn"] if _is_mv_here
                 else ["deeplabv3", "fcn", "unet"])
        _cur = [self.tr_model_combo.itemText(i) for i in range(self.tr_model_combo.count())]
        if _cur != _want:
            _keep = self.tr_model_combo.currentText()
            self.tr_model_combo.blockSignals(True)
            self.tr_model_combo.clear()
            self.tr_model_combo.addItems(_want)
            self.tr_model_combo.setCurrentText(_keep if _keep in _want else _want[0])
            self.tr_model_combo.blockSignals(False)

        try:
            self._save_panel_train_settings()
        except Exception:
            pass
        settings = {
            "epochs": self.epochs_spin.value(),
            "early_stop_patience": self.early_stop_spin.value(),
            "batch_size": self.batch_spin.value(),
            "model": self.tr_model_combo.currentText(),
            "image_size": int(self.tr_size_combo.currentText()),
            "loss": self.tr_loss_combo.currentText(),
            "k_folds": self.kfold_spin.value(),
            "resume": self.resume_check.isChecked(),
            "is_multiview": _is_mv_here,
            "tile": self.tr_tile_check.isChecked(),
            "tile_size": self.tr_tile_spin.value(),
            "tile_stride": self.tr_tile_stride_spin.value(),
            "tile_mode": "slide",
        }
        epochs = settings.get("epochs", 200)
        batch_size = settings.get("batch_size", 8)
        model_name = settings.get("model", "deeplabv3")
        image_size = settings.get("image_size", 512)
        loss_name = settings.get("loss", "cross_entropy")
        augment = self._get_augment_flags()
        cv_folds = settings.get("k_folds", 1)

        self._stop_flag = False
        self.btn_train_stop.setEnabled(True)
        self.train_status.setText("Training...")
        self.train_progress.setVisible(True)
        self.train_progress.setValue(0)

        from training.trainer import Trainer
        # ── 三视图工程走 9 通道训练器 ──
        # 两者 train() 接口完全一致，所以下面的线程/进度/停止逻辑不用改一行。
        is_mv = bool(settings.get("is_multiview", False))
        if is_mv:
            from training.multiview_gui import MultiViewTrainer
            # 分块参数来自训练设置对话框（「分块」勾选框 + 滑窗大小 + 步长% + 取样方式）
            _tile = int(settings.get("tile_size", 256)) if settings.get("tile", False) else 0
            self.trainer = MultiViewTrainer(
                project_dir, model_name=model_name,
                unet_base=settings.get("unet_base", 16),
                unet_depth=settings.get("unet_depth", 4),
                tile_size=_tile,
                tile_mode=settings.get("tile_mode", "slide"),
                tile_stride_pct=settings.get("tile_stride", 75))
            if _tile:
                self.log(f"分块训练：滑窗 {_tile}px，步长 {settings.get('tile_stride', 75)}%"
                         f"（{'均匀滑窗遍历' if settings.get('tile_mode', 'slide') == 'slide' else '缺陷居中+负样本'}）"
                         f" —— 原分辨率裁剪，不做整图缩放")
            else:
                self.log("⚠️ 未勾选分块：整图会被缩放到训练尺寸，大图上的小缺陷可能被缩小抹掉")
        else:
            # ★★ 2026-10-29：单视图也把「分块」这三个参数接上（之前是直接丢掉的，
            #   界面上填了滑窗 768 / 步长 75% 也永远不生效，实际恒按 Image Size 无重叠切）。
            _tile = int(settings.get("tile_size", 256)) if settings.get("tile", False) else 0
            _pct = int(settings.get("tile_stride", 75))
            self.trainer = Trainer(project_dir, model_name=model_name,
                                   tile_size=_tile, tile_stride_pct=_pct,
                                   tile_mode=settings.get("tile_mode", "slide"))
            if _tile:
                self.log(f"分块训练：滑窗 {_tile}px（= 网络输入尺寸），步长 {_pct}%"
                         f"（{100 - _pct}% 重叠）—— 原分辨率裁剪，不缩放整图")
            else:
                self.log(f"整图模式：按 Image Size {image_size} 切块（无重叠）"
                         f" —— 想开重叠请勾「分块」并设滑窗大小")

        # ★★ 2026-10-22 修「点停止后报 'NoneType' object has no attribute 'train'」：
        #    `self.trainer` 是**共享属性**，而"上一个任务收尾清空它"和"下一个任务开始用它"
        #    是两条并发路径。用户在一次训练还没收尾时又起了/停了任务，收尾那句
        #    `self.trainer = None` 就把**新任务刚建好的实例**抹掉了，
        #    新任务线程走到 `self.trainer.train()` 时自然是 None。
        #    ⇒ 线程里必须用**局部快照**，不能到调用那一刻再去读 self.trainer。
        _trainer_ref = self.trainer

        resume = settings.get("resume", False)
        self.log(f"Starting training: {model_name}, {epochs} epochs, batch={batch_size}"
                 + (f", {cv_folds}-fold CV" if cv_folds > 1 else "")
                 + f", loss={loss_name}, aug={self._augment_level_text(augment)}"
                 + (f", resume" if resume else "")
                 + ("  [三视图 9 通道]"
                    + (f" base={settings.get('unet_base', 16)} depth={settings.get('unet_depth', 4)}"
                       if model_name == "unet" else "") if is_mv else ""))
        # 独立 Loss/Acc 窗口随训练启动自动弹出（与分类训练一致）
        try:
            win = self._ensure_loss_window()
            win.show()
            win.raise_()
            win.activateWindow()
        except Exception:
            pass
        def _queue_run_progress(ctx):
            """队列适配层：跑真正的训练，并把 epoch 进度回灌给队列行。

            为什么要有这层：trainer 的 progress_callback 原来只喂 UI 进度条，
            队列要的是**同一个**数字（两处各算一套必然对不上）。
            """
            _qctx = ctx

            def _cb(c, t):
                try:
                    _qctx.progress(c, t)
                except Exception:
                    pass
                self.progress_signal.emit(c, t)

            run(_qctx)

        def run(_qctx=None):
            _cb = (lambda c, t: (_qctx.progress(c, t), self.progress_signal.emit(c, t))) \
                if _qctx is not None else (lambda c, t: self.progress_signal.emit(c, t))
            try:
                if _trainer_ref is None:
                    raise RuntimeError(
                        "训练器未就绪（被上一次任务的收尾清理掉了）。请重新点「开始训练」。")
                _trainer_ref.train(
                    epochs=epochs, batch_size=batch_size,
                    image_size=image_size, loss_name=loss_name,
                    augment=augment, k_folds=cv_folds, resume=resume,
                    early_stop_patience=int(settings.get("early_stop_patience", 0) or 0),
                    stop_check=lambda: self._stop_flag,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                    progress_callback=_cb,
                    plot_callback=lambda h: self.chart_signal.emit(h),
                    batch_callback=lambda b, t, l: (self.batch_signal.emit(b, t, l), self._log_batch_progress(b, t, l)),
                    # 被队列取消时置回标志，trainer 在下个 epoch 边界会停下
                    )
                # 注意：trainer 自己已经打过一条 "Training complete!"（含早停摘要之后），
                # 这里再打一条会出现两行一模一样的日志（实拍）。只更新状态栏即可。
                ui_call(self.train_status.setText, "Complete")
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Training error: {e}")
                self.log_signal.emit(traceback.format_exc())
                ui_call(self.train_status.setText, "Failed")
            finally:
                # ★ 复位 _stop_flag：否则"取消一次"之后**后面所有任务都会秒停**
                #   （标志是全局的，原先只靠停止按钮路径复位，队列路径漏了）
                self._stop_flag = False
                # ⚠️ 这里是子线程：直接碰控件会随机崩，统一排队回主线程
                ui_call(self.btn_train_stop.setEnabled, False)
                ui_call(self.train_progress.setVisible, False)
                # ★ 带上本次任务的实例：别把"下一个任务刚建好的 trainer"清掉
                self._cleanup_trainer(_trainer_ref)

        # ★ 进任务队列（2026-10-16）：队列串行执行，因此"再点一次训练"不会并发抢显存，
        #   而是排队；每行可单独取消。原来是直接起线程 -> 第二个任务把 self.trainer 顶掉。
        _q = getattr(self, "_task_queue", None)
        if _q is not None:
            _stamp = time.strftime("%Y%m%d_%H%M%S")
            from ui.task_queue import TaskKind as _TK
            _q.submit(_TK.TRAIN,
                      "%s [%s]" % (model_name, _stamp),
                      project=os.path.basename(project_dir),
                      total=int(epochs or 0),
                      detail="%d epochs · batch=%d" % (epochs, batch_size),
                      run=_queue_run_progress)
        else:
            threading.Thread(target=run, daemon=True).start()

    def _on_task_cancel(self, task_id):
        """队列里点×：置取消标志（不强杀线程，见 task_queue 里的说明）。"""
        q = getattr(self, "_task_queue", None)
        if q is None:
            return
        t = q.get(task_id)
        if t is not None:
            # 训练/推理共用_stop_flag，正在跑的那个也要真的停下来
            self._stop_flag = True
            q.request_cancel(task_id)
            self.log("已请求取消任务：%s" % t.title)

    def _on_task_clear(self):
        q = getattr(self, "_task_queue", None)
        if q is not None:
            q.clear_finished()
            self.log("已清空任务队列的历史记录")

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
                    augment=self._get_augment_flags(),
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

    def _cleanup_trainer(self, trainer=None):
        """收尾：释放训练器与显存。

        ★★ 2026-10-22：必须**能指定是哪个实例**。原来无条件 `self.trainer = None`，
        而"停止后收尾"和"下一个任务开始"并发 —— 上一个任务的 finally 会把
        下一个任务刚建好的 trainer 清成 None，下一个任务随即
        `'NoneType' object has no attribute 'train'`（用户实拍）。
        只有"当前挂着的还是我这个实例"时才置空，别人的不动。
        """
        target = trainer if trainer is not None else self.trainer
        if target is not None and hasattr(target, "cleanup"):
            try:
                target.cleanup()
            except Exception as e:
                self.log(f"[Train] 训练器清理出错（忽略）: {type(e).__name__}: {e}")
        if self.trainer is target:
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
        if self._cls_mode:
            return self._start_cls_inference()
        if self._ocr_mode:
            return self._run_ocr_single()
        if self.canvas.image is None:
            return QMessageBox.warning(self, "Warning", "Please load an image first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        if self._detection_mode:
            self._run_det_inference_single(project_dir)
        else:
            self._run_seg_inference_single(project_dir)
    # ────────── 多视图推理（三视图 9 通道）──────────
    def _mv_key_of(self, image_path):
        """从 <组名>_te_N.jpg 反推组名；不是多视图命名就返回原文件名。"""
        import re as _re
        stem = os.path.splitext(os.path.basename(image_path or ""))[0]
        return _re.sub(r"_te_\d+$", "", stem)

    def _mv_group_paths(self, project_dir, key):
        """找出某一组的三视图路径 {view: path}；缺视图返回空（调用方报错）。"""
        from training.multiview_gui import project_views
        views, _names = project_views(project_dir)
        img_dir = os.path.join(project_dir, "images")
        out = {}
        for v in views:
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = os.path.join(img_dir, f"{key}_{v}{ext}")
                if os.path.exists(p):
                    out[v] = p
                    break
        return out

    def _mv_infer_one(self, predictor, project_dir, image_path, tile=None):
        """跑一组三视图推理，返回 (pred, overlay)。"""
        from PIL import Image as _I
        key = self._mv_key_of(image_path)
        paths = self._mv_group_paths(project_dir, key)
        missing = [v for v in predictor.views if v not in paths]
        if missing:
            raise RuntimeError(f"组 {key} 缺少视图 {missing}（三视图必须齐全）")
        t = int(tile or self._mv_tile(project_dir))
        imgs = {v: _I.open(paths[v]) for v in predictor.views}
        pred, overlay = predictor.predict_multiview(
            imgs, tile=t, stride=max(1, int(t * 0.75)), return_overlay=True)
        self.log(f"[多视图推理] {key}：{len(predictor.views)} 视图 → {len(predictor.views)*3} 通道，"
                 f"滑窗 {t}（原分辨率分块）")
        return pred, overlay

    def _mv_tile(self, project_dir):
        """推理滑窗大小：优先用刚才训练设置里保存的，其次 256。"""
        try:
            return int(self._load_ui_setting("tile_size", 256)) or 256
        except Exception:
            return 256

    def _run_seg_batch_inference_mv(self, project_dir, model_file, out_dir):
        """多视图批处理：**按组遍历**，每组一次 9 通道推理，写出与单图流程同格式的结果。"""
        import threading
        from inference.predictor import Predictor
        from flow.frame import Frame
        from flow.registry import create_node
        from training.multiview_gui import project_views
        views, _names = project_views(project_dir)
        keys = []
        img_dir = os.path.join(project_dir, "images")
        for fn in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            if not stem.endswith("_" + views[0]) or stem in keys:
                continue
            key = stem[: -(len(views[0]) + 1)]
            if all(any(os.path.exists(os.path.join(img_dir, f"{key}_{v}{e}"))
                       for e in (".jpg", ".jpeg", ".png", ".bmp")) for v in views):
                keys.append(key)
        if not keys:
            return QMessageBox.warning(self, "Warning", "没有三视图齐全的组")
        total = len(keys)
        self._stop_flag = False
        self.btn_infer_stop.setEnabled(True)
        self.infer_progress.setVisible(True)
        self.infer_progress.setMaximum(total); self.infer_progress.setValue(0)
        self.infer_status.setText("Running...")
        self.log(f"多视图批处理：{total} 组（每组一次 {len(views)*3} 通道推理，按组不按文件）")

        def run():
            done = fail = 0
            try:
                pr = Predictor(project_dir, model_file, backend=self._backend_key())
                writer = create_node("seg_result_writer", node_id="out", out_dir=out_dir)
                from PIL import Image as _I
                import numpy as _np
                t = self._mv_tile(project_dir)
                for i, key in enumerate(keys, 1):
                    _t0 = time.time()
                    if self._stop_flag:
                        self.log_signal.emit("已停止（完成 %d/%d 组）" % (done, total)); break
                    try:
                        paths = self._mv_group_paths(project_dir, key)
                        imgs = {v: _I.open(paths[v]) for v in pr.views}
                        pred, overlay = pr.predict_multiview(
                            imgs, tile=t, stride=max(1, int(t * 0.75)), return_overlay=True)
                        fr = Frame(image=_np.array(imgs[pr.views[0]]), path=paths[pr.views[0]],
                                   result={"mask": pred, "overlay": overlay,
                                           "classes": list(pr.classes)},
                                   meta={"project_dir": project_dir})
                        writer.run(fr)
                        done += 1
                        # ⚠️ 每组都要打一行进度：原来只在**失败**时打日志，
                        #    成功时一声不吭 -> 用户看到"点了没动静"（实测反馈）。
                        #    134 组 × 8s ≈ 18 分钟，中间没有任何输出等于让人干等。
                        _dt = time.time() - _t0
                        self.log_signal.emit("[%d/%d] %s 完成 (%.1fs)" % (i, total, key, _dt))
                        ui_call(self.infer_status.setText, "Running... %d/%d" % (i, total))
                    except Exception as e:
                        fail += 1
                        self.log_signal.emit("[%d/%d] %s: FAILED - %s: %s" % (i, total, key, type(e).__name__, e))
                    self.infer_progress_signal.emit(i, total)
                self.log_signal.emit("多视图批处理完成：成功 %d / 失败 %d，输出 %s" % (done, fail, out_dir))
                ui_call(self.infer_status.setText, "Complete")
                self._pred_count_cache_key = None
                threading.Thread(target=self._load_output_stats_async, args=(project_dir,), daemon=True).start()
            except Exception as e:
                import traceback
                self.log_signal.emit(f"多视图批处理错误: {e}")
                self.log_signal.emit(traceback.format_exc())
                ui_call(self.infer_status.setText, "Failed")
            finally:
                # 子线程不碰控件：统一排队回主线程执行
                ui_call(self.btn_infer_stop.setEnabled, False)
                ui_call(self.infer_progress.setVisible, False)

        threading.Thread(target=run, daemon=True).start()

    def _run_seg_inference_single(self, project_dir):
        model_file = self._resolve_model_for_use(self.model_combo.currentText())
        model_path = os.path.join(project_dir, "models", model_file)
        if not os.path.exists(model_path):
            return QMessageBox.warning(self, "Error",
                f"Model not found: {model_path}\nPlease train a model first.")
        self.log(f"Running inference: {model_file}")
        _t_infer = time.time()          # 单张推理耗时（回填列表「耗时」列）
        try:
            from inference.predictor import Predictor
            predictor = Predictor(project_dir, model_file, backend=self._backend_key())
            # ★ 智能推理模式（快速/精细）
            try:
                predictor._智能模式 = ("fast" if "快速" in self.tr_mode_combo.currentText()
                                       else "fine")
            except Exception:
                predictor._智能模式 = "fine"
            if getattr(predictor, "views", ()):
                # 多视图工程：三视图按 views 顺序拼 N*3 通道，走和训练/工具菜单**同一条**路
                pred, overlay = self._mv_infer_one(predictor, project_dir,
                                                   getattr(self.canvas, "current_image_path", ""))
            else:
                # tiled=None ⇒ 由 Predictor 按**训练模式**自动决定（见 _resolve_train_mode）
                pred, overlay = predictor.predict(self.canvas.image, return_overlay=True,
                    tiled=None, scale=self.scale_slider.value() / 100.0)

            # ── 推理参数过滤（面积/宽度/高度/角度/灰度/分数 + 孔洞填充/最大个数）──
            #    以前这一步**完全没有**：模型 argmax 出来是什么就显示什么，
            #    碎点、孔洞、超限块全都直接进结果表。
            prob = getattr(predictor, "last_prob", None)
            self.canvas._pred_prob = prob
            pred, report = self._apply_infer_params(pred, predictor, project_dir, prob=prob)
            if any(report.get("dropped", {}).values()):
                overlay = predictor._create_overlay(self.canvas.image, pred)

            h, w, ch = overlay.shape
            qimg = QImage(overlay.data, w, h, 3 * w, QImage.Format_RGB888)
            self.canvas._prediction_qimage = qimg.copy()
            self.canvas._prediction_mask = pred.copy()
            self.canvas._pred_stats_cache = {}
            self.canvas._pred_classes = predictor.classes
            self.canvas.update()
            self.log("Inference complete - Space to view")
            # ── 推理完必须刷两处（用户实测反馈：Result 表和图片列表都没动）──
            #    ① Result 表：按连通域逐行（_update_stats 里做）
            self._update_stats()
            #    ② 图片列表的「结果 / 分数 / 耗时」列：这两列读的是 _pred_count / _infer_times，
            #       单张推理以前只写了画布的预测层，从没回填这两张表 -> 列表一直显示 0
            try:
                import numpy as _np, cv2 as _cv2
                base = os.path.splitext(os.path.basename(
                    getattr(self.canvas, "current_image_path", "") or ""))[0]
                pm = getattr(self.canvas, "_prediction_mask", None)
                n_blocks = 0
                if pm is not None and base:
                    for cid in set(_np.unique(pm).tolist()) - {0, 255}:
                        n, _lab, _st, _c = _cv2.connectedComponentsWithStats(
                            (pm == cid).astype(_np.uint8), 8)
                        n_blocks += max(0, n - 1)
                if base:
                    if not hasattr(self, "_pred_count") or self._pred_count is None:
                        self._pred_count = {}
                    self._pred_count[base] = n_blocks
                    if not hasattr(self, "_infer_times") or self._infer_times is None:
                        self._infer_times = {}
                    # ⚠️ 单位必须是**毫秒**：这里以前存的是"秒"（round(x, 2)），
                    #    而消费它的两端都按毫秒理解 —— 筛选字段名 infer_ms / 中文名「推理耗时」
                    #    （core/filter_expr.py），列表列头「耗时」又不带单位。后果是很静默的：
                    #    按 "推理耗时 < 1000"（1000ms = 1s）筛，**每一张图都命中**
                    #    （值是 1.23 这种），既不报错也看不出错。
                    self._infer_times[base] = round((time.time() - _t_infer) * 1000.0, 1)
                self.log(f"本图预测块数：{n_blocks}（已回填图片列表「结果」列）")
                self._pred_count_cache_key = None
                self._filter_images(self.search_input.text())
            except Exception as e:
                self.log(f"[刷新列表] 失败: {type(e).__name__}: {e}")
        except Exception as e:
            import traceback
            self.log(traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Inference failed: {e}")

    def _apply_infer_params(self, pred, predictor=None, project_dir=None, prob=None):
        """按「推理参数」过滤分割结果，并把结果写日志/状态栏。

        返回 (过滤后的掩膜, report)。任何异常都不影响出图 —— 过滤失败就原样返回。
        """
        try:
            from inference.params import filter_prediction, summarize
            project_dir = project_dir or self._project_dir
            lm = getattr(self.canvas, "label_manager", None)
            classes = list(getattr(lm, "classes", []) or [])
            panel = getattr(self, "infer_params", None)
            params = panel.current_params() if panel is not None else None
            if params is None:
                from inference.params import load_params
                params = load_params(project_dir, classes)
            gray = None
            try:
                gray = np.asarray(self.canvas.image.convert("L"), dtype=np.float32)
            except Exception:
                gray = None
            out, report = filter_prediction(pred, classes=classes, params=params,
                                            gray=gray, prob=prob, image=self.canvas.image)
            txt = summarize(report)
            if txt:
                self.log("[推理参数] " + txt)
                try:
                    self.infer_result_label.setText(txt[:80])
                except Exception:
                    pass
            return out, report
        except Exception as e:
            self.log(f"[推理参数] 过滤失败，已按原结果输出: {type(e).__name__}: {e}")
            return pred, {}
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
        """批量推理入口：**先在任务队列登记一行**，再由队列串行执行真正的推理。

        为什么要登记而不是直接跑：批量推理动辄几十秒~几分钟，原来只有一个全局进度条，
        连跑两次就互相覆盖，也**没法单独取消**某一次。
        登记后：进度/剩余时间在「任务队列」页可见、×可单独取消、串行保证不抢显存。

        ⚠️ 进度回灌**不在这里 monkey-patch 信号的 emit**（PyQt 绑定信号改属性很脆弱），
           而是在既有的 `_update_infer_progress` 槽里顺手推给队列 —— 一个数字两处用，
           不各算一套（否则进度条与队列行会对不上）。
        """
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        q = getattr(self, "_task_queue", None)
        if q is None:                       # 没有队列控件时按老路走（保持能用）
            return self._dispatch_batch_inference(project_dir)

        from ui.task_queue import TaskKind as _TK
        kindname = ("分类" if self._cls_mode else "OCR" if self._ocr_mode
                    else "检测" if self._detection_mode else "分割")

        def _run(ctx):
            self._stop_flag = False
            self._active_infer_ctx = ctx
            try:
                self._dispatch_batch_inference(project_dir)
                if not ctx.canceled:
                    ctx.progress(max(1, ctx.task.total or 1), ctx.task.total or 1)
            finally:
                self._active_infer_ctx = None
                self._stop_flag = False

        q.submit(_TK.INFER,
                 "批量推理·%s [%s]" % (kindname, time.strftime("%H%M%S")),
                 project=os.path.basename(project_dir), total=0,
                 detail="排队中…", run=_run)

    def _dispatch_batch_inference(self, project_dir):
        """按当前工程类型分派到对应的批量推理实现（队列只管"排队/取消/进度"）。"""
        if self._cls_mode:
            return self._start_cls_batch_inference()
        if self._ocr_mode:
            return self._run_ocr_batch()
        if self._detection_mode:
            return self._start_det_batch_inference()
        return self._run_seg_batch_inference(project_dir)

    def _run_seg_batch_inference(self, project_dir):
        model_file = self._resolve_model_for_use(self.model_combo.currentText())
        model_path = os.path.join(project_dir, "models", model_file)
        if not os.path.exists(model_path):
            return QMessageBox.warning(self, "Error",
                "Model not found: %s\nPlease train a model first." % model_path)
        out_dir = os.path.join(project_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        from flow.batch import BatchRunner, build_seg_flow, list_images
        # 多视图工程：**按"组"遍历**（一颗电芯一次推理）。
        # 原来按 402 个文件遍历 -> 三视图各算一次、结果重复，且模型是 9 通道根本喂不进去。
        try:
            from training.multiview_gui import is_multiview_project
            if is_multiview_project(project_dir):
                return self._run_seg_batch_inference_mv(project_dir, model_file, out_dir)
        except Exception as e:
            self.log(f"[多视图批处理] 判定失败，退回单图流程: {type(e).__name__}: {e}")
        images = list_images(project_dir)
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
        flow = build_seg_flow(
            project_dir, model_file, out_dir=out_dir,
            backend=self._backend_key(),
            tiled=self.tiled_check.isChecked(),
            scale=self.scale_slider.value() / 100.0,
        )
        def run():
            try:
                if not hasattr(self, "_infer_times") or self._infer_times is None:
                    self._infer_times = {}
                # ★ 批量推理原来**不记耗时**：只初始化了 _infer_times，从没往里写 →
                #   列表「耗时ms」列整列空白（单图推理那条路 8043 行是有写的）。
                #   这里用"相邻两次回调的时间差"当每张图的耗时 —— 回调是每张处理完才发一次，
                #   所以差值就是这张图跑完的时间（含读图/预处理/推理，和单图口径一致）。
                _t_prev = [time.time()]
                def on_progress(i, total, path, frame, err):
                    base = os.path.splitext(os.path.basename(path))[0]
                    _now = time.time()
                    _ms = (_now - _t_prev[0]) * 1000.0
                    _t_prev[0] = _now
                    if err is None:
                        # ★ 列表「耗时ms」用**纯推理**（和友商同一口径）；
                        #   总耗时（含读图/存盘）只写进日志，方便判断瓶颈在哪。
                        _meta = getattr(frame, "meta", None) or {}
                        _pure = _meta.get("infer_ms")
                        _total = _meta.get("total_ms")
                        self._infer_times[base] = round(float(_pure), 1) if _pure is not None else round(_ms, 1)
                        self.log_signal.emit("[%d/%d] %s  推理 %sms / 总 %sms" % (
                            i, total, base,
                            ("%.1f" % _pure) if _pure is not None else "?",
                            ("%.1f" % _total) if _total is not None else "?"))
                    else:
                        self.log_signal.emit("[%d/%d] %s: FAILED - %s" % (i, total, base, err))
                    self.infer_progress_signal.emit(i, total)
                br = BatchRunner(flow, project_dir=project_dir, stop_check=lambda: self._stop_flag)
                br.run(images, on_progress=on_progress)
                self.log_signal.emit("Batch inference complete! Results: %s" % out_dir)
                ui_call(self.infer_status.setText, "Complete")
                self._pred_count_cache_key = None  # force re-scan outputs
                # ★ 刷新图片列表：让「耗时ms」（和重新扫描出来的「结果/分数」）立刻显示出来
                self.refresh_list_signal.emit()   # 类里定义的信号（见 428 行）
                threading.Thread(target=self._load_output_stats_async, args=(project_dir,), daemon=True).start()
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Batch inference error: {e}")
                self.log_signal.emit(traceback.format_exc())
                ui_call(self.infer_status.setText, "Failed")
            finally:
                # 子线程不碰控件：统一排队回主线程执行
                ui_call(self.btn_infer_stop.setEnabled, False)
                ui_call(self.infer_progress.setVisible, False)
        threading.Thread(target=run, daemon=True).start()
    def _run_det_batch_inference(self, project_dir):
        """Batch detection inference on all images."""
        model_path = os.path.join(project_dir, "models", "detection", "best.pt")
        if not os.path.exists(model_path):
            return QMessageBox.warning(self, "Error",
                "Detection model not found: %s\nPlease train a detection model first." % model_path)
        out_dir = os.path.join(project_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        from flow.batch import BatchRunner, build_det_flow, list_images
        images = list_images(project_dir)
        if not images:
            return QMessageBox.warning(self, "Warning", "No images found")
        classes = getattr(self, "_cached_classes", None) or ["background"]
        total = len(images)
        self._stop_flag = False
        self.btn_infer_stop.setEnabled(True)
        self.infer_progress.setVisible(True)
        self.infer_progress.setMaximum(total)
        self.infer_progress.setValue(0)
        self.infer_status.setText("Running...")
        self.log(f"Batch detection inference on {total} images...")
        flow = build_det_flow(project_dir, out_dir=out_dir, classes=",".join(classes))
        def run():
            try:
                ok_cnt = 0
                def on_progress(i, total, path, frame, err):
                    nonlocal ok_cnt
                    base = os.path.splitext(os.path.basename(path))[0]
                    if err is None:
                        ok_cnt += 1
                        self.log_signal.emit("[%d/%d] %s: %d boxes" % (i, total, base, frame.result.get("count", 0)))
                    else:
                        self.log_signal.emit("[%d/%d] %s: FAILED - %s" % (i, total, base, err))
                    self.infer_progress_signal.emit(i, total)
                br = BatchRunner(flow, project_dir=project_dir, stop_check=lambda: self._stop_flag)
                br.run(images, on_progress=on_progress)
                self.log_signal.emit("Batch detection complete! %d images, results: %s" % (ok_cnt, out_dir))
                ui_call(self.infer_status.setText, "Complete")
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
                ui_call(self.infer_status.setText, "Failed")
            finally:
                # 子线程不碰控件：统一排队回主线程执行
                ui_call(self.btn_infer_stop.setEnabled, False)
                ui_call(self.infer_progress.setVisible, False)
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
        label_map = {}       # image_name -> category
        split_map = {}        # image_name -> train/val/test
        img_sizes = {}        # image_name -> (w, h)
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
                    base = os.path.splitext(img_name)[0]
                    # Extract classification label and split from 友商平台Flow format
                    # ★ 有标注才允许分训练/测试集；没标注的一律不分（= 空集，不参与训练）
                    _shapes = data.get("shapes") or []
                    if not _shapes:
                        base_data = []
                    else:
                        base_data = data.get("baseData", [])
                    if base_data and isinstance(base_data, list):
                        cat = base_data[0].get("category", "")
                        ds = base_data[0].get("dataSet", "")
                        if cat:
                            label_map[f"images/{img_name}"] = cat
                        if ds:
                            split_map[base] = ds.lower()
                        else:
                            split_map[base] = "test"  # unassigned -> test set
                    # Image dimensions
                    iw = data.get("imageWidth", 0)
                    ih = data.get("imageHeight", 0)
                    if iw and ih:
                        img_sizes[base] = (iw, ih)
                except Exception:
                    pass
        
        ann_dir = os.path.join(project_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        
        # Generate class_labels.json
        if label_map:
            labels_list = sorted(set(label_map.values()))
            # Map image_name -> label_id
            mapping = {}
            for k, v in label_map.items():
                mapping[k] = labels_list.index(v)
                # Also add without "images/" prefix
                short_key = os.path.basename(k)
                mapping[short_key] = labels_list.index(v)
            lbl_path = os.path.join(ann_dir, "class_labels.json")
            with open(lbl_path, "w", encoding="utf-8") as lf:
                json.dump({"labels": labels_list, "mapping": mapping}, lf, indent=2, ensure_ascii=False)
            self.log(f"Updated class_labels.json: {labels_list} ({len(label_map)} images)")
        
        # Generate train_test_split.json
        if split_map:
            split_path = os.path.join(project_dir, "train_test_split.json")
            with open(split_path, "w", encoding="utf-8") as sf:
                json.dump(split_map, sf, indent=2, ensure_ascii=False)
            train_cnt = sum(1 for v in split_map.values() if v == "train")
            val_cnt = sum(1 for v in split_map.values() if v == "val")
            self.log(f"Updated train_test_split.json: {train_cnt} train, {val_cnt} val")
        
        self.log(f"Batch imported {count} JSON files")
        self._load_classification_labels()  # refresh _cached_classes
        self._load_image_list_async()

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
            self._refresh_class_ui()

    def _manage_classes(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        from ui.class_dialog import ClassDialog
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        dlg = ClassDialog(pd, self)
        if dlg.exec_():
            # ★ 这里**不能**用 `_load_classes()`：它会"从标注 JSON 补类别"
            #   （`merge_classes`）。用户在对话框里删掉的类别，只要标注 JSON 里
            #   还写着那个 label，就会被原样补回来 —— 表现是"删除无效"，
            #   而且日志里轻描淡写一句「从标注 JSON 补入 1 个类别」，根本看不出是它干的。
            #   对话框已经把最终表落盘了，这里只吃它交回来的那份。
            self._cached_classes = list(getattr(dlg, "_classes", None) or self._cached_classes)
            self._reload_canvas_classes()
            self._refresh_class_ui()

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
        self._sync_det_toolbar()
        if hasattr(self, "class_list"):
            self.class_list.refresh()

    def _refresh_annotation_views(self):
        """标注变更后刷新列表「标注」列与 Class 表统计。"""
        if getattr(self, "_all_images", None) and self.current_project:
            try:
                self._filter_images(self.search_input.text())
            except Exception:
                pass
        if hasattr(self, "class_list"):
            self.class_list.refresh()

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

    def _set_det_draw_mode(self, checked):
        """画框开关：点亮=可画新框；熄灭=仅选中/拖动/缩放已有框。"""
        ov = getattr(self.canvas, "_det_overlay", None)
        if ov is not None:
            ov.set_draw_enabled(checked)
        # 2026-10-28：画框时也用箭头（用户明确要求"不要十字架，要箭头"）。
        # 箭头热点就在尖端，起点仍然准；检测框本来也不是靠十字线对齐的。
        self.canvas.setCursor(QCursor(Qt.ArrowCursor if checked else Qt.OpenHandCursor))

    def _det_cancel(self):
        """取消当前画到一半的框（不撤销已完成的操作）。"""
        ov = getattr(self.canvas, "_det_overlay", None)
        if ov is not None:
            ov.cancel_drawing()
            self.canvas.update()

    def _det_undo(self):
        """撤销上一次框操作。"""
        if self._box_manager is not None and self._box_manager.undo():
            self.canvas.update()
            self.canvas.det_boxes_changed.emit()

    def _det_redo(self):
        """重做被撤销的框操作。"""
        if self._box_manager is not None and self._box_manager.redo():
            self.canvas.update()
            self.canvas.det_boxes_changed.emit()

    def _sync_det_toolbar(self):
        """同步检测工具栏按钮可用状态（撤销/重做/删除选中）。"""
        bm = getattr(self, "_box_manager", None)
        ov = getattr(self.canvas, "_det_overlay", None)
        self.det_toolbar.set_can_undo(bool(bm) and bm.can_undo())
        self.det_toolbar.set_can_redo(bool(bm) and bm.can_redo())
        self.det_toolbar.set_has_selection(bool(ov) and ov.has_active_box())

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



    def _open_screen_ocr(self):
        """Tools 菜单：屏幕截图 -> 平台 OCR 识别。"""
        # 从其他应用触发全局热键时，先把窗口带到前台，确保询问对话框可见
        try:
            if self.isMinimized():
                self.showNormal()
            self.raise_()
            self.activateWindow()
        except Exception:
            pass
        try:
            screen_ocr_and_show(self)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "截图 OCR", f"启动失败: {e}")

    def _register_screen_ocr_hotkey(self):
        """注册系统级全局热键 Shift+X。

        注册成功则移除窗口内快捷键（避免重复触发）；
        失败（被其他软件占用）则回退为仅窗口聚焦时生效的快捷键。
        """
        self._screen_ocr_global = False
        self._screen_ocr_hotkey_id = 0x5358  # "SX"
        try:
            hwnd = int(self.winId())
            ok = register_global_hotkey(hwnd, self._screen_ocr_hotkey_id)
        except Exception as e:  # noqa: BLE001
            ok = False
            self.log(f"[热键] 全局热键注册异常: {e}")
        if ok:
            self._screen_ocr_global = True
            self._act_screen_ocr.setShortcut(QKeySequence())
            self.log("[热键] Shift+X 全局截图 OCR 已启用（窗口未聚焦也可触发）")
        else:
            self._act_screen_ocr.setShortcut(QKeySequence("Shift+X"))
            self.log("[热键] Shift+X 全局注册失败（可能被其他软件占用），"
                     "已回退为窗口内快捷键（需聚焦平台窗口）")

    def nativeEvent(self, eventType, message):
        """接收 Windows 全局热键消息 WM_HOTKEY。"""
        if getattr(self, "_screen_ocr_global", False) and eventType == b"windows_generic_MSG":
            try:
                msg = wintypes.MSG.from_address(int(message))
                if int(msg.message) == WM_HOTKEY and int(msg.wParam) == self._screen_ocr_hotkey_id:
                    QTimer.singleShot(0, self._open_screen_ocr)
                    return True, 0
            except Exception:  # noqa: BLE001
                pass
        return super().nativeEvent(eventType, message)

    def _open_resize_tool(self):
        """Open the image resize tool dialog."""
        dlg = ResizeToolDialog(self)
        dlg.exec_()

    def _open_heatmap_analyzer(self):
        """Tools 菜单：批量热力图区域分析（辅助 NG 样本种类/个数统计）。"""
        dlg = HeatmapAnalyzerDialog(self)
        dlg.resize(1180, 760)
        dlg.exec_()

    def _open_threshold_panel(self):
        """Tools 菜单：分类判定阈值调优（过杀率 / 漏检率 权衡）。"""
        try:
            from ui.threshold_panel import ThresholdPanelDialog
        except Exception as e:
            import traceback
            self.log_signal.emit(f"阈值调优面板打开失败: {e}")
            self.log_signal.emit(traceback.format_exc())
            return
        project_dir = None
        try:
            if self.current_project:
                project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        except Exception:
            project_dir = None
        try:
            dlg = ThresholdPanelDialog(self, project_dir=project_dir)
            dlg.exec_()
        except Exception as e:
            import traceback
            self.log_signal.emit(f"阈值调优面板异常: {e}")
            self.log_signal.emit(traceback.format_exc())
    def _open_flow_editor(self):
        """打开可视化流程编辑器（Vision Flow）。"""
        try:
            from ui.flow_editor import FlowEditorDialog
        except Exception as e:
            import traceback
            self.log_signal.emit(f"Flow Editor 打开失败: {e}")
            self.log_signal.emit(traceback.format_exc())
            return
        try:
            dlg = FlowEditorDialog(self)
            dlg.resize(1280, 820)
            dlg.exec_()
        except Exception as e:
            import traceback
            self.log_signal.emit(f"Flow Editor 异常: {e}")
            self.log_signal.emit(traceback.format_exc())
            try:
                crash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          os.pardir, "crash_log.txt")
                with open(crash_path, "a", encoding="utf-8") as f:
                    f.write("\n===== FLOW EDITOR @ %s =====\n"
                            % __import__("datetime").datetime.now())
                    traceback.print_exc(file=f)
            except Exception:
                pass

    def _open_crop_tool(self):
        """Open the image crop tool dialog."""
        dlg = CropToolDialog(self)
        dlg.exec_()
    def _about(self):
        """Show about dialog."""
        QMessageBox.about(self, "AI Training Platform",
            "AI Training Platform v0.1.0\n\n"
            "Semantic Segmentation + Object Detection\n"
            "Built with PyTorch + PyQt5")

    def _save_current_mask(self):
        """Save current annotation mask to JSON."""
        if not getattr(self.canvas, "_mask_editing", True):
            self.log("Mask saving is disabled in this task mode")
            return
        if self.canvas.image is None:
            return
        path = getattr(self.canvas, "current_image_path", "")
        if not path:
            return
        try:
            from annotation.labelme_io import save_mask_to_json
            # ⚠️ 参数顺序：save_mask_to_json(mask, image_path, project_dir, class_names)
            #    以前这里写成了 (path, mask, classes, project) —— 每次保存都抛异常，
            #    被 except 吞掉只在日志里留一句 "Save failed"，按钮看着能点其实没保存。
            classes = []
            try:
                classes = list(self.canvas.label_manager.classes)
            except Exception:
                classes = list(getattr(self, "_cached_classes", []) or ["background"])
            save_mask_to_json(self.canvas.mask, path, self._project_dir, classes)
            self._refresh_class_ui()
            self._ann_refresh_timer.start()
            self.log("Annotation saved")
        except Exception as e:
            self.log(f"Save failed: {e}")

    def _clear_current_mask(self):
        """Clear current annotation mask and SAM state."""
        if self.canvas.image is None:
            return
        reply = QMessageBox.question(self, "Clear Annotation",
            "Clear all annotations for this image?", 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            h, w = self.canvas.mask.shape
            self.canvas.mask = np.zeros((h, w), dtype=np.uint8)
            self.canvas._overlay_dirty = True
            # Clear SAM state if active
            self.canvas._sam_points = []
            self.canvas._sam_labels = []
            self.canvas._sam_mask = None
            self.canvas._sam_score = 0.0
            self.canvas.update()
            self.canvas.mask_changed.emit()
            self._refresh_class_ui()
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
