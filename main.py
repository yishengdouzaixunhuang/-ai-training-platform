"""AI训练平台 - 主入口"""
import sys
import os

# 关键：在 sys.path 修改前先导入 torch，避免工作目录 DLL 干扰
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
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
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
