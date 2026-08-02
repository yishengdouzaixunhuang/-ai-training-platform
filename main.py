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
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    app.setStyleSheet("""
        QMainWindow { background: #f5f5f5; }
        QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 4px; margin-top: 8px; padding-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QPushButton { padding: 5px 12px; border: 1px solid #bbb; border-radius: 3px; background: #fff; }
        QPushButton:hover { background: #e0e0e0; }
        QPushButton:checked { background: #cce5ff; border-color: #004085; }
        QListWidget { border: 1px solid #ccc; }
        QComboBox { padding: 3px; }
        QTreeWidget { border: 1px solid #ccc; }
    """)
    
    sys.excepthook = _write_crash
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
