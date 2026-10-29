"""AI训练平台 - 主入口"""
import sys
import os
import traceback

# pythonw 无控制台时 sys.stdout/sys.stderr 为 None，
# torch/tqdm 等第三方库直接 write 会 AttributeError 崩溃。
# 统一重定向到空设备，避免 'NoneType' object has no attribute 'write'。
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# 关键：在 sys.path 修改前先导入 torch，避免工作目录 DLL 干扰
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui.main_window import MainWindow


def _write_crash(exc_type, exc_value, exc_tb):
    """pythonw 无控制台，未捕获异常会静默退出：把 traceback 写到 crash_log.txt。"""
    try:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n===== CRASH @ %s =====\n" % __import__("datetime").datetime.now())
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        # 尽量通知 GUI
        try:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, "AI训练平台 - 崩溃", 
                "程序遇到未处理异常，详情已写入：\n" + log_path)
        except Exception:
            pass
    except Exception:
        pass

def main():
    # ===== 高 DPI 缩放（幻16 等 2.5K/4K 高分屏，否则界面字体像蚂蚁一样小）=====
    # PyQt5 默认不开启高 DPI 缩放，必须显式设置，且必须在 QApplication 创建之前。
    from PyQt5.QtCore import Qt
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    try:
        # PassThrough：按系统实际缩放比例（如 150% -> 1.5x）精确缩放，避免四舍五入过大/过小
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass

    # 修复 PyQt5 在 Windows 上 "Could not find the Qt platform plugin windows" 的问题：
    # 拆分包（PyQt5 + PyQt5-Qt5）在部分环境不会自动解析插件路径，这里显式指定。
    try:
        import PyQt5
        _plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
        if os.path.isdir(_plugins):
            os.environ["QT_PLUGIN_PATH"] = _plugins
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.pathsep.join(
                [_plugins, os.path.join(_plugins, "platforms")]
            )
    except Exception:
        pass

    # ★ 任务栏图标：Windows 默认按宿主进程（pythonw.exe）取图标 → 会显示 Python 的蛇。
    #   给进程一个独立 AppUserModelID，任务栏就会独立成组、改用窗口自己的图标。
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'Arthur.AITrainingPlatform.v1')
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 必须在主线程、QApplication 之后立刻初始化：把「后台线程回主线程更新 UI」
    # 的转发器建在主线程上，否则子线程里 ui_call() 无处投递。
    from ui.qt_thread import init_ui_invoker as _init_ui_invoker
    _init_ui_invoker()
    # 滚轮守卫：数字框/下拉框不再被滚轮改值，滚轮改为滚动外层面板
    from ui.wheel_guard import install as _install_wheel_guard
    _install_wheel_guard(app)

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # ===== 赛博朋克主题 · 经典霓虹 =====
    app.setStyleSheet("""
        QWidget {
            background-color: #0A0E17;
            color: #E6F0FF;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            font-size: 13px;
        }
        QMainWindow { background-color: #0A0E17; }
        QDialog { background-color: #0D1220; }

        QLabel { color: #E6F0FF; background: transparent; }
        QLabel:disabled { color: #55607A; }

        /* ---------- 按钮 ---------- */
        QPushButton {
            background-color: #131B2E;
            color: #D6E4FF;
            border: 1px solid #2A3A5C;
            border-radius: 4px;
            padding: 5px 12px;
        }
        QPushButton:hover { background-color: #1B2740; border-color: #00E5FF; color: #00E5FF; }
        QPushButton:pressed { background-color: #0E1830; }
        QPushButton:checked { background-color: #0E3A52; border-color: #00E5FF; color: #00E5FF; }
        QPushButton:disabled { color: #55607A; border-color: #1C2740; background-color: #0E1420; }

        QToolButton {
            background: transparent;
            color: #D6E4FF;
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 3px;
        }
        QToolButton:hover { border-color: #00E5FF; background: #131B2E; }
        QToolButton:pressed, QToolButton:checked { background: #0E3A52; border-color: #00E5FF; }

        /* ---------- 输入控件 ---------- */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
            background-color: #0E1420;
            color: #E6F0FF;
            border: 1px solid #2A3A5C;
            border-radius: 4px;
            padding: 3px 6px;
            selection-background-color: #0E5A6E;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
        QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus { border-color: #00E5FF; }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background-color: #0E1420;
            color: #E6F0FF;
            border: 1px solid #2A3A5C;
            selection-background-color: #0E3A52;
            selection-color: #00E5FF;
        }

        /* ---------- 列表 / 树 / 表格 ---------- */
        QAbstractItemView {
            background-color: #0C1220;
            alternate-background-color: #101828;
            color: #D6E4FF;
            border: 1px solid #22304A;
            gridline-color: #1A2440;
            selection-background-color: #0E3A52;
            selection-color: #00E5FF;
        }
        QListWidget::item, QTreeWidget::item { padding: 3px 4px; }
        QListWidget::item:hover, QTreeWidget::item:hover { background: #131B2E; }

        QHeaderView::section {
            background-color: #0F1728;
            color: #00E5FF;
            font-weight: bold;
            border: none;
            border-right: 1px solid #22304A;
            border-bottom: 1px solid #2A3A5C;
            padding: 5px 4px;
        }
        QTableCornerButton::section { background-color: #0F1728; border: none; }

        /* 图片列表的**默认行文字色**：故意用 ID 选择器（优先级高于任何类型选择器），
           以便主题表——而不是 main_window._render_page 里的 setForeground(白)——决定行色。
           内联前景色优先级高于 QSS，之前深色主题写死纯白，切浅色后就成了"白字白底"。 */
        QTableWidget#imageList { color: #FFFFFF; }

        /* ---------- 分组框 ---------- */
        QGroupBox {
            color: #00E5FF;
            font-weight: bold;
            border: 1px solid #2A3A5C;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 12px;
            background-color: #0C1220;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #00E5FF;
        }

        /* ---------- 菜单 ---------- */
        QMenuBar { background-color: #0A0E17; color: #D6E4FF; border-bottom: 1px solid #1A2440; }
        QMenuBar::item { padding: 4px 10px; background: transparent; }
        QMenuBar::item:selected { background: #131B2E; color: #00E5FF; }
        QMenu { background-color: #0E1420; color: #D6E4FF; border: 1px solid #2A3A5C; }
        QMenu::item { padding: 5px 24px; background: transparent; }
        QMenu::item:selected { background-color: #0E3A52; color: #00E5FF; }
        QMenu::separator { height: 1px; background: #1A2440; margin: 4px 8px; }

        /* ---------- 进度条 ---------- */
        QProgressBar {
            background-color: #0E1420;
            border: 1px solid #2A3A5C;
            border-radius: 4px;
            text-align: center;
            color: #E6F0FF;
            height: 16px;
        }
        QProgressBar::chunk { background-color: #00E5FF; border-radius: 3px; }

        /* ---------- 滚动条 ---------- */
        QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
        QScrollBar::handle:vertical { background: #2A3A5C; border-radius: 5px; min-height: 24px; }
        QScrollBar::handle:vertical:hover { background: #00E5FF; }
        QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
        QScrollBar::handle:horizontal { background: #2A3A5C; border-radius: 5px; min-width: 24px; }
        QScrollBar::handle:horizontal:hover { background: #00E5FF; }
        QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
        QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

        /* ---------- 勾选框 / 单选框 ---------- */
        QCheckBox, QRadioButton { color: #D6E4FF; background: transparent; spacing: 6px; }

        /* ---------- 标签页 ---------- */
        QTabWidget::pane { border: 1px solid #2A3A5C; }
        QTabBar::tab { background: #0C1220; color: #8A93A6; padding: 6px 14px; border: 1px solid #2A3A5C; }
        QTabBar::tab:selected { background: #0E3A52; color: #00E5FF; }

        /* ---------- 其它 ---------- */
        QToolTip { background-color: #0E1420; color: #00E5FF; border: 1px solid #00E5FF; padding: 4px; }
        QStatusBar { background-color: #0A0E17; color: #8A93A6; }
        QSplitter::handle { background-color: #1A2440; }
        QSplitter::handle:hover { background-color: #00E5FF; }
        QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #22304A; }
    """)

    # ===== 主题：基础表登记下来，然后按用户上次选择应用（深色/浅色）=====
    from ui.themes import set_base_qss, apply_theme
    set_base_qss(app.styleSheet())
    try:
        import json as _json, os as _os
        _ui = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".ui_settings.json")
        _th = (_json.load(open(_ui, encoding="utf-8")).get("theme", "dark")
               if _os.path.exists(_ui) else "dark")
    except Exception:
        _th = "dark"
    apply_theme(app, _th)
    # ⚠️ 清扫必须等**主窗口构造完**再跑：main 这里调 apply_theme 时一个控件都还没有，
    #    那一轮清扫等于空转（已踩：启动日志里一条 swept 都没有）。
    #    用 singleShot(0) 排到事件循环第一拍，那时 UI 已经全部建好。
    from ui.themes import sweep_inline_styles as _sweep

    sys.excepthook = _write_crash
    window = MainWindow()
    # ★★ 2026-10-22 修「启动一闪一闪」：原来用 QTimer 在 0ms 和 1200ms 各扫一次，
    #   1200ms 那次是在**窗口已经显示之后**重新给所有控件套一遍样式 ⇒ 整窗重绘，
    #   用户看到的就是"一闪一闪"。改成在 show() **之前**同步扫一次 ——
    #   此时 MainWindow 已构造完、所有控件都在，扫完再显示，用户完全看不见。
    try:
        _sweep(app, restore=False)
    except Exception:
        pass
    # ★★ 2026-10-22 修「双击 exe 后屏幕中间一闪一闪」：
    #   窗口 show() 之后，异步加载（图片列表 / 统计 / 各面板刷新）会陆续到达，
    #   每一次都触发重排 ⇒ 用户看到中间那块在闪。先把窗口重绘关掉，
    #   等这些首屏加载跑完再打开，用户看到的就是"一次成型"。
    window.setUpdatesEnabled(False)
    window.show()
    from PyQt5.QtCore import QTimer as _QT
    _QT.singleShot(280, lambda: window.setUpdatesEnabled(True))
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
