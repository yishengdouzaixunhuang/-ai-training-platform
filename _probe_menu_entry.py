# -*- coding: utf-8 -*-
"""冒烟：菜单「数据集划分管理…」「类别管理…」真的能打开（原来这两条是 ModuleNotFoundError）。

不是构造对话框就算过 —— 走的是 main_window 里那两个槽函数的**原文**：
`_manage_split` / `_manage_classes`。把 `exec_()` 打桩成 0，跳过模态等待。
"""
import os
import sys
import tempfile

import torch  # noqa: F401
from PyQt5.QtWidgets import QApplication

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

app = QApplication(sys.argv)
root = tempfile.mkdtemp(prefix="aitp_menu_")
pd = os.path.join(root, "proj")
os.makedirs(os.path.join(pd, "images"))
os.makedirs(os.path.join(pd, "annotations"))
import json
json.dump({"classes": ["background", "A"]},
          open(os.path.join(pd, "project.json"), "w", encoding="utf-8"))
open(os.path.join(pd, "images", "a.png"), "wb").write(b"")

from ui.main_window import MainWindow
win = MainWindow()
win.current_project = {"name": "proj"}
win.pm.get_project_dir = lambda name=None: pd

ok = []
for name in ("_manage_split", "_manage_classes"):
    try:
        mod = __import__("ui.split_dialog", fromlist=["SplitDialog"]) if name == "_manage_split" \
            else __import__("ui.class_dialog", fromlist=["ClassDialog"])
        cls = getattr(mod, "SplitDialog" if name == "_manage_split" else "ClassDialog")
        calls = []
        orig = cls.exec_
        cls.exec_ = lambda self, *a, **k: (calls.append(1), 0)[1]
        try:
            getattr(win, name)()
        finally:
            cls.exec_ = orig
        ok.append((name, len(calls) == 1, "对话框被打开 %d 次" % len(calls)))
    except Exception as e:
        ok.append((name, False, "%s: %s" % (type(e).__name__, e)))

for n, good, detail in ok:
    print(("PASS " if good else "FAIL ") + n + "  " + detail)
print("RESULT: " + ("PASS" if all(g for _n, g, _d in ok) else "FAIL"))
os._exit(0 if all(g for _n, g, _d in ok) else 1)
