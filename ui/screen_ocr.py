# -*- coding: utf-8 -*-
"""屏幕截图 OCR 工具。

用法（挂到主窗口 Tools 菜单）:
    from ui.screen_ocr import screen_ocr_and_show
    screen_ocr_and_show(main_window)

流程: 隐藏主窗口 -> 全屏截图 -> 拖拽框选区域 -> 后台 RapidOCR 识别
       -> 弹出结果对话框（文本按位置排序，可一键复制全部）。
"""
import os
import tempfile
import time

from PyQt5.QtCore import QRect, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QMessageBox,
    QProgressDialog, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)


def grab_virtual_desktop():
    """拼接所有屏幕的截图，返回 (QPixmap, 虚拟桌面 QRect)。"""
    app = QApplication.instance()
    screens = app.screens()
    left = min(s.geometry().left() for s in screens)
    top = min(s.geometry().top() for s in screens)
    right = max(s.geometry().right() for s in screens)
    bottom = max(s.geometry().bottom() for s in screens)
    vg = QRect(left, top, right - left + 1, bottom - top + 1)
    pix = QPixmap(vg.width(), vg.height())
    pix.fill(Qt.black)
    p = QPainter(pix)
    for s in screens:
        g = s.geometry()
        shot = s.grabWindow(0)
        p.drawPixmap(QRect(g.x() - left, g.y() - top, g.width(), g.height()), shot)
    p.end()
    return pix, vg


def pixmap_region_to_pil(pix, rect):
    """把截图区域转为 PIL RGB 图像。"""
    img = pix.copy(rect).toImage()
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        img.save(tmp, "PNG")
        from PIL import Image
        pil = Image.open(tmp).convert("RGB")
        pil.load()
        return pil
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


class ScreenRegionSelector(QDialog):
    """全屏遮罩选区窗口：拖拽选择识别区域，Esc 取消。"""

    def __init__(self, screenshot, virtual_geo, parent=None):
        super().__init__(parent,
                         Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self._shot = screenshot
        self._start = None
        self._cur = None
        self._sel = None
        self.setGeometry(virtual_geo)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def selected_rect(self):
        return self._sel

    def paintEvent(self, event):
        p = QPainter(self)
        p.drawPixmap(self.rect(), self._shot)
        dim = QColor(0, 0, 0, 110)
        if self._start is not None and self._cur is not None:
            sel = QRect(self._start, self._cur).normalized()
            shade = [
                QRect(0, 0, self.width(), sel.top()),
                QRect(0, sel.bottom() + 1, self.width(), self.height() - sel.bottom() - 1),
                QRect(0, sel.top(), sel.left(), sel.height()),
                QRect(sel.right() + 1, sel.top(), self.width() - sel.right() - 1, sel.height()),
            ]
            for r in shade:
                if r.width() > 0 and r.height() > 0:
                    p.fillRect(r, dim)
            p.setPen(QPen(QColor(0, 170, 255), 2))
            p.drawRect(sel)
            p.setPen(Qt.white)
            info = f"{sel.width()} x {sel.height()}"
            p.drawText(sel.left(), max(0, sel.top() - 6), info)
        else:
            p.fillRect(self.rect(), dim)
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._start = e.pos()
            self._cur = e.pos()
            self.update()

    def mouseMoveEvent(self, e):
        if self._start is not None:
            self._cur = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._start is not None:
            self._cur = e.pos()
            sel = QRect(self._start, self._cur).normalized()
            if sel.width() >= 5 and sel.height() >= 5:
                self._sel = sel
                self.accept()
            else:
                self._start = None
                self._cur = None
                self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(e)


class OcrTask(QThread):
    """后台 OCR 识别任务（模型加载/推理不阻塞 UI）。"""

    done = pyqtSignal(object)

    def __init__(self, pil_img, parent=None):
        super().__init__(parent)
        self._img = pil_img

    def run(self):
        t0 = time.time()
        try:
            from ocr.engine import get_ocr_engine
            results = get_ocr_engine().detect_and_recognize(self._img)
            self.done.emit({"ok": True, "results": results, "elapsed": time.time() - t0})
        except Exception as e:  # noqa: BLE001
            self.done.emit({"ok": False, "error": str(e), "elapsed": time.time() - t0})


def compose_text(results, line_gap=15):
    """按位置排序识别结果：同行（y 相近）按 x 拼接，跨行换行。"""
    items = sorted(results, key=lambda r: (r["y"], r["x"]))
    lines = []
    buf = []
    base_y = None
    for r in items:
        if base_y is None or abs(r["y"] - base_y) <= line_gap:
            buf.append((r["x"], r["text"]))
            if base_y is None:
                base_y = r["y"]
        else:
            lines.append(" ".join(t for _, t in sorted(buf)))
            buf = [(r["x"], r["text"])]
            base_y = r["y"]
    if buf:
        lines.append(" ".join(t for _, t in sorted(buf)))
    return "\n".join(lines)


class ScreenOcrResultDialog(QDialog):
    """识别结果展示：文本可选中，一键复制全部。"""

    def __init__(self, results, elapsed, parent=None):
        super().__init__(parent)
        self.setWindowTitle("截图 OCR 结果")
        self.resize(560, 420)
        lay = QVBoxLayout(self)
        text = compose_text(results)
        self._edit = QTextEdit()
        self._edit.setPlainText(text)
        self._edit.setReadOnly(True)
        lay.addWidget(self._edit)
        info = QLabel(f"识别到 {len(results)} 块文本，耗时 {elapsed:.1f}s"
                      "（文本按位置从上到下、从左到右排列）")
        lay.addWidget(info)
        btns = QHBoxLayout()
        btn_copy = QPushButton("复制全部")
        btn_close = QPushButton("关闭")
        btns.addStretch()
        btns.addWidget(btn_copy)
        btns.addWidget(btn_close)
        lay.addLayout(btns)
        btn_copy.clicked.connect(self._copy_all)
        btn_close.clicked.connect(self.accept)

    def _copy_all(self):
        QApplication.clipboard().setText(self._edit.toPlainText())
        QMessageBox.information(self, "截图 OCR", "已复制全部识别文本到剪贴板")


def screen_ocr_and_show(main_win):
    """入口：询问是否隐藏主窗口 -> 截图框选 -> OCR -> 结果对话框。"""
    # 全局热键可能被连按，避免重复启动截图会话
    if getattr(main_win, "_screen_ocr_busy", False):
        return
    main_win._screen_ocr_busy = True
    try:
        ans = QMessageBox.question(
            main_win, "截图 OCR",
            "截图前是否隐藏主窗口？\n（选\"是\"可避免平台窗口出现在截图画面里）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    except Exception:
        main_win._screen_ocr_busy = False
        raise
    if ans == QMessageBox.Yes:
        main_win.hide()
        QTimer.singleShot(200, lambda: _capture_and_ocr(main_win))
    else:
        _capture_and_ocr(main_win)


def _capture_and_ocr(main_win):
    try:
        try:
            pix, vg = grab_virtual_desktop()
            sel = ScreenRegionSelector(pix, vg, main_win)
            ok = sel.exec_()
            sel_rect = sel.selected_rect()
        finally:
            main_win.show()
        if not ok or sel_rect is None:
            main_win._screen_ocr_busy = False
            return

        try:
            pil_img = pixmap_region_to_pil(pix, sel_rect)
        except Exception as e:  # noqa: BLE001
            main_win._screen_ocr_busy = False
            QMessageBox.warning(main_win, "截图 OCR", f"截取区域失败: {e}")
            return
    except Exception as e:  # noqa: BLE001
        main_win._screen_ocr_busy = False
        QMessageBox.warning(main_win, "截图 OCR", f"截图失败: {e}")
        return

    wait = QProgressDialog("OCR 识别中，请稍候…（首次加载模型较慢）", None, 0, 0, main_win)
    wait.setWindowTitle("截图 OCR")
    wait.setCancelButton(None)
    wait.setWindowModality(Qt.WindowModal)
    wait.setMinimumDuration(200)
    wait.show()

    task = OcrTask(pil_img, main_win)
    main_win._screen_ocr_task = task  # 保持引用，防 GC

    def on_done(res):
        wait.close()
        try:
            if not res.get("ok"):
                QMessageBox.warning(main_win, "截图 OCR",
                                    f"识别失败: {res.get('error')}")
                return
            dlg = ScreenOcrResultDialog(res.get("results", []),
                                        res.get("elapsed", 0.0), main_win)
            dlg.exec_()
        finally:
            main_win._screen_ocr_busy = False

    task.done.connect(on_done)
    task.start()
