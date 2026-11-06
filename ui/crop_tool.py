"""Image crop tool - sliding window + ROI Crop, batch support."""
from core.imgio import imread_u, imwrite_u
import os, cv2, json
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox,
    QFileDialog, QMessageBox, QListWidget, QProgressBar, QTextEdit,
    QGroupBox, QCheckBox, QWidget, QSplitter
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QRect, QPoint


class ROICanvas(QLabel):
    """Simple canvas for drawing a ROI Crop."""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        # ⚠️ 深色底 + **主题**字色 = 浅色主题下"深字压深底"（这段样式含 #444 三位色，
        #    不在主题色白名单里，不会被 themes.py 清扫）。文字色必须显式给，
        #    否则「Drop image here or click Browse」在浅色主题下看不见。
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444; color: #D6E4FF;")
        self.setText("Drop image here or click Browse")
        self.setFont(QFont("Sans", 12))
        self._orig_pixmap = None
        self._scaled_pixmap = None
        self._roi = None  # (x, y, w, h) in image coords
        self._drawing = False
        self._start = None

    def load_image(self, path):
        self._orig_pixmap = QPixmap(path)
        self._roi = None
        self._fit_to_view()
        self.update()

    def _fit_to_view(self):
        if self._orig_pixmap is None:
            return
        self._scaled_pixmap = self._orig_pixmap.scaled(
            self.width() - 10, self.height() - 10,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_to_view()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._scaled_pixmap is None:
            return
        painter = QPainter(self)
        x = (self.width() - self._scaled_pixmap.width()) // 2
        y = (self.height() - self._scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, self._scaled_pixmap)

        if self._roi is not None:
            rx, ry, rw, rh = self._image_to_widget_rect(self._roi)
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
            painter.setBrush(QColor(0, 255, 0, 30))
            painter.drawRect(rx, ry, rw, rh)

        painter.end()

    def _widget_to_image(self, wx, wy):
        if self._scaled_pixmap is None or self._orig_pixmap is None:
            return None, None
        sx = (self.width() - self._scaled_pixmap.width()) // 2
        sy = (self.height() - self._scaled_pixmap.height()) // 2
        scale = self._orig_pixmap.width() / self._scaled_pixmap.width()
        ix = int((wx - sx) * scale)
        iy = int((wy - sy) * scale)
        return max(0, min(ix, self._orig_pixmap.width() - 1)), max(0, min(iy, self._orig_pixmap.height() - 1))

    def _image_to_widget_rect(self, roi):
        if self._scaled_pixmap is None or self._orig_pixmap is None:
            return 0, 0, 0, 0
        sx = (self.width() - self._scaled_pixmap.width()) // 2
        sy = (self.height() - self._scaled_pixmap.height()) // 2
        scale = self._scaled_pixmap.width() / self._orig_pixmap.width()
        return int(sx + roi[0] * scale), int(sy + roi[1] * scale), int(roi[2] * scale), int(roi[3] * scale)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._orig_pixmap:
            ix, iy = self._widget_to_image(event.pos().x(), event.pos().y())
            if ix is not None:
                self._drawing = True
                self._start = (ix, iy)
                self._roi = None
                self.update()

    def mouseMoveEvent(self, event):
        if self._drawing and self._start:
            ix, iy = self._widget_to_image(event.pos().x(), event.pos().y())
            if ix is not None:
                x = min(self._start[0], ix)
                y = min(self._start[1], iy)
                w = abs(ix - self._start[0])
                h = abs(iy - self._start[1])
                self._roi = (x, y, w, h)
                self.update()

    def mouseReleaseEvent(self, event):
        self._drawing = False

    def get_roi(self):
        return self._roi


class CropToolDialog(QDialog):
    """Dual-mode image crop dialog."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Crop Tool")
        self.setMinimumSize(900, 650)
        self._files = []
        self._output_dir = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # File selection
        fl = QHBoxLayout()
        fl.addWidget(QLabel("Images:"))
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(80)
        fl.addWidget(self._file_list)
        btn_add = QPushButton("+"); btn_add.setFixedWidth(28)
        btn_add.clicked.connect(self._add_files)
        fl.addWidget(btn_add)
        btn_clr = QPushButton("x"); btn_clr.setFixedWidth(28)
        btn_clr.clicked.connect(lambda: (self._file_list.clear(), self._files.clear()))
        fl.addWidget(btn_clr)
        layout.addLayout(fl)

        # Output dir
        ol = QHBoxLayout()
        ol.addWidget(QLabel("Output:"))
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Select output directory...")
        ol.addWidget(self._out_edit)
        btn_out = QPushButton("..."); btn_out.setFixedWidth(28)
        btn_out.clicked.connect(self._pick_output)
        ol.addWidget(btn_out)
        # 输出格式：默认跟随原图，避免 bmp 进来 png 出去
        ol.addWidget(QLabel("Format:"))
        self._out_fmt = QComboBox()
        self._out_fmt.addItems(["Same as source", "PNG", "BMP", "JPG"])
        self._out_fmt.setFixedWidth(120)
        ol.addWidget(self._out_fmt)
        layout.addLayout(ol)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._create_sliding_tab(), "Sliding Window")
        self._tabs.addTab(self._create_roi_tab(), "ROI Crop")
        layout.addWidget(self._tabs)

        # Go button
        bl = QHBoxLayout()
        bl.addStretch()
        btn_go = QPushButton("Start Crop")
        btn_go.setStyleSheet("QPushButton { padding: 8px 24px; font-weight: bold; background: #2d6a4f; color: white; }")
        btn_go.clicked.connect(self._start_crop)
        bl.addWidget(btn_go)
        layout.addLayout(bl)

    def _create_sliding_tab(self):
        w = QWidget()
        l = QFormLayout(w); l.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self._sw_win_w = QSpinBox(); self._sw_win_w.setRange(64, 4096); self._sw_win_w.setValue(512)
        self._sw_win_h = QSpinBox(); self._sw_win_h.setRange(64, 4096); self._sw_win_h.setValue(512)
        l.addRow("Window W:", self._sw_win_w)
        l.addRow("Window H:", self._sw_win_h)
        self._sw_stride_x = QSpinBox(); self._sw_stride_x.setRange(1, 4096); self._sw_stride_x.setValue(256)
        self._sw_stride_y = QSpinBox(); self._sw_stride_y.setRange(1, 4096); self._sw_stride_y.setValue(256)
        l.addRow("Stride X:", self._sw_stride_x)
        l.addRow("Stride Y:", self._sw_stride_y)
        # 重叠率由 stride / window 决定，这里只读显示。
        # 原来是个可输入的 QDoubleSpinBox，但全文件从来没读过它的值 ——
        # 填了 0.5 也不生效，纯属骗人（2026-11-06 修）。
        self._sw_overlap_label = QLabel("--")
        self._sw_overlap_label.setStyleSheet("color: #555;")
        l.addRow("Overlap (computed):", self._sw_overlap_label)
        for sb in (self._sw_win_w, self._sw_win_h,
                   self._sw_stride_x, self._sw_stride_y):
            sb.valueChanged.connect(self._update_overlap_label)
        self._update_overlap_label()
        self._sw_discard = QCheckBox("Discard edge tiles smaller than window")
        self._sw_discard.setChecked(True)
        l.addRow(self._sw_discard)
        return w

    def _create_roi_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        self._roi_canvas = ROICanvas()
        vl.addWidget(self._roi_canvas)
        hl = QHBoxLayout()
        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self._load_roi_preview)
        hl.addWidget(btn_load)
        btn_clear = QPushButton("Clear ROI")
        btn_clear.clicked.connect(lambda: (setattr(self._roi_canvas, '_roi', None), self._roi_canvas.update()))
        hl.addWidget(btn_clear)
        self._roi_label = QLabel("ROI: --"); hl.addWidget(self._roi_label); hl.addStretch()
        vl.addLayout(hl)
        return w

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        for f in files:
            if f not in self._files:
                self._files.append(f)
                self._file_list.addItem(os.path.basename(f))

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_dir = d
            self._out_edit.setText(d)

    def _update_overlap_label(self):
        """按当前 window / stride 反算重叠率，只做显示。"""
        try:
            ow = self._sw_win_w.value()
            oh = self._sw_win_h.value()
            sx = self._sw_stride_x.value()
            sy = self._sw_stride_y.value()
            rx = (1.0 - sx / ow) * 100.0 if ow else 0.0
            ry = (1.0 - sy / oh) * 100.0 if oh else 0.0
            txt = "X %.1f%%  /  Y %.1f%%" % (rx, ry)
            if rx < 0 or ry < 0:
                txt += "   (负值 = 块之间有间隙)"
            self._sw_overlap_label.setText(txt)
        except Exception:
            pass

    def _out_ext(self, src_path):
        """输出扩展名。默认跟随源文件，避免 bmp 进 png 出。

        注意：这里返回的扩展名同时决定编码器，imwrite_u 内部按扩展名选
        cv2.imencode 的格式 —— 所以不需要另写一份映射。
        """
        idx = self._out_fmt.currentIndex() if hasattr(self, "_out_fmt") else 0
        if idx == 0:                                   # Same as source
            ext = os.path.splitext(src_path)[1].lower()
            return ext if ext in (".bmp", ".png", ".jpg", ".jpeg",
                                  ".tif", ".tiff") else ".png"
        return {1: ".png", 2: ".bmp", 3: ".jpg"}.get(idx, ".png")

    def _load_roi_preview(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Preview", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if f:
            self._roi_canvas.load_image(f)
            self._roi_label.setText("Draw rectangle on image...")

    def _start_crop(self):
        if not self._files:
            QMessageBox.warning(self, "Warning", "Please add images first")
            return
        if not self._output_dir:
            QMessageBox.warning(self, "Warning", "Please select output directory")
            return

        if self._tabs.currentIndex() == 0:
            self._crop_sliding()
        else:
            self._crop_roi()

    def _crop_sliding(self):
        win_w = self._sw_win_w.value()
        win_h = self._sw_win_h.value()
        stride_x = self._sw_stride_x.value()
        stride_y = self._sw_stride_y.value()
        discard = self._sw_discard.isChecked()

        def _positions(size, win, stride):
            """切窗位置：末尾回吸到边界，避免最后一块只有几个像素。"""
            if win >= size:
                return [0]
            pos = list(range(0, size, stride))
            last = size - win
            if pos[-1] + win > size and pos[-1] != last:
                pos[-1] = last
            elif pos[-1] + win < size:
                pos.append(last)
            # 回吸后仍会切出不足窗口的，去掉（除非就是边界那一块）
            pos = [p for p in pos if p + win <= size or p == last]
            return sorted(set(pos))

        total = 0
        skipped = []
        for path in self._files:
            # ⚠️ 必须用 IMREAD_UNCHANGED：原来用 IMREAD_COLOR 会把单通道灰度图
            # 硬转成三通道 BGR，灰度值复制成 3 份。数据没丢，但通道数变了、
            # 体积涨 2 倍（实测 512x512：单通道 bmp 257KB -> 三通道 png 548KB）。
            img = imread_u(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                skipped.append((os.path.basename(path), "读不出来"))
                continue
            h, w = img.shape[:2]
            if win_w > w or win_h > h:
                skipped.append((os.path.basename(path),
                                f"{w}x{h} 小于窗口 {win_w}x{win_h}"))
                continue
            base = os.path.splitext(os.path.basename(path))[0]
            ext = self._out_ext(path)

            xs = _positions(w, win_w, stride_x)
            ys = _positions(h, win_h, stride_y)

            for y in ys:
                for x in xs:
                    x2, y2 = min(x + win_w, w), min(y + win_h, h)
                    tile = img[y:y2, x:x2]
                    if discard and (tile.shape[1] < win_w or tile.shape[0] < win_h):
                        continue
                    out_name = (f"{base}_x{x}_y{y}"
                                f"_w{tile.shape[1]}x{tile.shape[0]}{ext}")
                    if not imwrite_u(os.path.join(self._output_dir, out_name), tile):
                        skipped.append((out_name, "写入失败"))
                        continue
                    total += 1

        msg = f"Sliding window: {total} tiles saved to:\n{self._output_dir}"
        msg += f"\n\n输出格式: {self._fmt_desc()}"
        if skipped:
            head = "、".join(s[0] for s in skipped[:3])
            msg += f"\n\n跳过 {len(skipped)} 项：{head}" + (" ..." if len(skipped) > 3 else "")
        QMessageBox.information(self, "Done", msg)

    def _fmt_desc(self):
        idx = self._out_fmt.currentIndex() if hasattr(self, "_out_fmt") else 0
        return {0: "跟随源文件", 1: "PNG", 2: "BMP", 3: "JPG"}.get(idx, "PNG")
    def _crop_roi(self):
        roi = self._roi_canvas.get_roi()
        if roi is None:
            QMessageBox.warning(self, "Warning", "Please draw a ROI rectangle first")
            return
        rx, ry, rw, rh = roi
        if rw < 2 or rh < 2:
            QMessageBox.warning(self, "Warning", "ROI too small")
            return

        total = 0
        skipped = []
        for path in self._files:
            # 同 _crop_sliding：IMREAD_UNCHANGED 保住单通道，别把灰度复制成 3 份
            img = imread_u(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                skipped.append((os.path.basename(path), "读不出来"))
                continue
            h, w = img.shape[:2]
            if rx >= w or ry >= h:
                skipped.append((os.path.basename(path),
                                f"ROI 在 {w}x{h} 之外"))
                continue
            rx2, ry2 = min(rx + rw, w), min(ry + rh, h)
            crop = img[ry:ry2, rx:rx2]
            if crop.size == 0:
                skipped.append((os.path.basename(path), "裁出空图"))
                continue
            base = os.path.splitext(os.path.basename(path))[0]
            ext = self._out_ext(path)
            out_name = f"{base}_roi_{rx}_{ry}_{rw}x{rh}{ext}"
            if not imwrite_u(os.path.join(self._output_dir, out_name), crop):
                skipped.append((out_name, "写入失败"))
                continue
            total += 1

        # If ROI was drawn on the preview image, save ROI params as JSON for reproducibility
        roi_json = os.path.join(self._output_dir, "_roi_params.json")
        with open(roi_json, "w") as f:
            json.dump({"x": rx, "y": ry, "w": rw, "h": rh}, f, indent=2)

        msg = f"ROI crop: {total} images saved to:\n{self._output_dir}"
        msg += f"\n\n输出格式: {self._fmt_desc()}"
        if skipped:
            head = "、".join(s[0] for s in skipped[:3])
            msg += f"\n\n跳过 {len(skipped)} 项：{head}" + (" ..." if len(skipped) > 3 else "")
        QMessageBox.information(self, "Done", msg)
