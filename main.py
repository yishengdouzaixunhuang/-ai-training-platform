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

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # ===== 赛博朋克主题 · 经典霓虹 =====
    app.setStyleSheet("""
        QWidget {
            background-color: #0A0E17;
            color: #E6F0FF;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            font-size: 12px;
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
    
    sys.excepthook = _write_crash
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
