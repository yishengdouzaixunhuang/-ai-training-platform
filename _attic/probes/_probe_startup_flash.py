# -*- coding: utf-8 -*-
"""启动闪烁回归探针 —— 记录启动全过程里「顶层窗口 SHOW/HIDE」与各窗口重绘次数。

判据：MainWindow 自己 SHOW 之前，不应出现任何**其它顶层窗口**的 SHOW。
（2026-10-22：右栏 7 个区块的 body=QWidget() 无父 + setVisible(True) ⇒ 屏幕中间连闪 7 下）

用法：
    "C:/aitp_venv/Scripts/python.exe" _probe_startup_flash.py
结果落盘 _probe_flash.log；末行 PASS / FAIL。
"""
import os
import sys
import time
import traceback

import torch  # noqa: F401  # 地雷 2：torch 必须先于 PyQt5

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.join(HERE, "_probe_flash.log")
_f = open(LOG, "w", encoding="utf-8")
T0 = time.perf_counter()
STRAY = []          # MainWindow 出现前的杂散顶层 SHOW
MAIN_WIN_ID = [None]


def log(msg):
    _f.write("[%7.3fs] %s\n" % (time.perf_counter() - T0, msg))
    _f.flush()


from PyQt5.QtCore import QObject, QEvent, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402


class FlashSpy(QObject):
    def __init__(self):
        super().__init__()
        self.paint = {}

    @staticmethod
    def _key(w):
        try:
            wid = w.winId()
        except Exception:
            wid = None
        if wid is None:
            return "%s@pyid%x" % (type(w).__name__, id(w))
        return "%s#%x" % (type(w).__name__, int(wid))

    def eventFilter(self, obj, ev):
        t = ev.type()
        try:
            if hasattr(obj, "isWindow") and obj.isWindow():
                k = self._key(obj)
                if t == QEvent.Show:
                    g = obj.geometry()
                    log("SHOW %s geo=(%d,%d %dx%d)" % (k, g.x(), g.y(), g.width(), g.height()))
                    if type(obj).__name__ == "MainWindow":
                        MAIN_WIN_ID[0] = int(obj.winId())
                    elif MAIN_WIN_ID[0] is None:
                        STRAY.append("%s geo=(%d,%d %dx%d)" % (k, g.x(), g.y(), g.width(), g.height()))
                elif t == QEvent.Hide:
                    log("HIDE %s" % k)
                elif t == QEvent.Paint:
                    self.paint[k] = self.paint.get(k, 0) + 1
        except Exception as e:
            log("spy error: %r" % (e,))
        return False

    def dump(self):
        log("---- 各顶层窗口重绘次数 ----")
        for k, v in sorted(self.paint.items(), key=lambda x: -x[1]):
            log("  %-40s repaint=%d" % (k, v))
        if STRAY:
            log("---- 杂散顶层窗口 %d 个（= 会闪几下）----" % len(STRAY))
            for s in STRAY:
                log("  " + s)
            log("RESULT: FAIL")
        else:
            log("RESULT: PASS  杂散顶层窗口 = 0")


def main():
    from PyQt5.QtCore import Qt
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    spy = FlashSpy()
    app.installEventFilter(spy)

    from ui.main_window import MainWindow
    log("building MainWindow ...")
    w = MainWindow()
    log("MainWindow built size=%dx%d" % (w.width(), w.height()))
    w.show()
    log("MainWindow.show() returned")

    def finish():
        spy.dump()
        _f.close()
        app.quit()

    QTimer.singleShot(5000, finish)
    app.exec_()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("CRASH:\n" + traceback.format_exc())
        _f.close()
        os._exit(1)
