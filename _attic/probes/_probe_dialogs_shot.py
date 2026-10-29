# -*- coding: utf-8 -*-
"""把两个对话框真渲染成 PNG，用于肉眼看版面（文字被截、控件重叠、列宽不合理）。

用法：
    "C:/aitp_venv/Scripts/python.exe" _probe_dialogs_shot.py
产物：_dlg_shots/split_dialog.png / class_dialog.png
"""
import importlib.util
import os
import sys
import tempfile

import torch  # noqa: F401
from PyQt5.QtWidgets import QApplication

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 复用 _probe_dialogs 里的造数函数（顺便借用它的 QMessageBox 打桩）
spec = importlib.util.spec_from_file_location("pd_probe", os.path.join(HERE, "_probe_dialogs.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def shot(widget, path):
    widget.show()
    QApplication.instance().processEvents()
    QApplication.instance().processEvents()
    pix = widget.grab()
    pix.save(path)
    print("%s  %dx%d" % (path, pix.width(), pix.height()))
    widget.close()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    root = tempfile.mkdtemp(prefix="aitp_shot_")
    out = os.path.join(HERE, "_dlg_shots")
    os.makedirs(out, exist_ok=True)

    pd_mv = P.make_multiview_project(root)
    pd_flat = P.make_flat_project(root, 60)

    from ui.split_dialog import SplitDialog
    dlg = SplitDialog(pd_mv)
    # 造一点"各状态都有"的数据，便于看配色与对齐
    dlg._rows[0]["split"] = "train"
    dlg._rows[1]["split"] = "val"
    dlg._rows[2]["split"] = "test"
    dlg._fill_table()
    shot(dlg, os.path.join(out, "split_dialog_mv.png"))

    dlg2 = SplitDialog(pd_flat)
    shot(dlg2, os.path.join(out, "split_dialog_flat.png"))

    from ui.class_dialog import ClassDialog
    dlg3 = ClassDialog(pd_mv)
    shot(dlg3, os.path.join(out, "class_dialog.png"))


if __name__ == "__main__":
    main()
