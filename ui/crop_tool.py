"""Image crop tool - sliding window + ROI Crop, batch support."""
import os, cv2, json
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QLineEdit,
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
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444;")
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
        self._sw_overlap = QDoubleSpinBox(); self._sw_overlap.setRange(0, 0.9); self._sw_overlap.setValue(0.0); self._sw_overlap.setSingleStep(0.1)
        l.addRow("Overlap:", self._sw_overlap)
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

        total = 0
        for path in self._files:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            base = os.path.splitext(os.path.basename(path))[0]

            # Compute positions with edge snap-back
            def _positions(size, win, stride):
                pos = list(range(0, size, stride))
                if not pos:
                    return [0]
                last = size - win
                if pos[-1] + win > size and pos[-1] != last:
                    pos[-1] = last
                elif pos[-1] + win < size:
                    pos.append(last)
                # Remove positions that produce undersized tiles after snap-back
                pos = [p for p in pos if p + win <= size or p == last]
                return sorted(set(pos))
                return sorted(set(pos))
            xs = _positions(w, win_w, stride_x)
            ys = _positions(h, win_h, stride_y)

            for y in ys:
                for x in xs:
                    x2, y2 = min(x + win_w, w), min(y + win_h, h)
                    tile = img[y:y2, x:x2]
                    if discard and (tile.shape[1] < win_w or tile.shape[0] < win_h):
                        continue
                    out_name = f"{base}_x{x}_y{y}_w{tile.shape[1]}x{tile.shape[0]}.png"
                    out_path = os.path.join(self._output_dir, out_name)
                    _, buf = cv2.imencode(".png", tile)
                    buf.tofile(out_path)
                    total += 1
        QMessageBox.information(self, "Done", f"Sliding window: {total} tiles saved to:\n{self._output_dir}")
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
        for path in self._files:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            rx2, ry2 = min(rx + rw, w), min(ry + rh, h)
            crop = img[ry:ry2, rx:rx2]
            base = os.path.splitext(os.path.basename(path))[0]
            out_name = f"{base}_roi_{rx}_{ry}_{rw}x{rh}.png"
            out_path = os.path.join(self._output_dir, out_name)
            _, buf = cv2.imencode(".png", crop)
            buf.tofile(out_path)
            total += 1

        roi_json = os.path.join(self._output_dir, "_roi_params.json")
        with open(roi_json, "w") as f:
            json.dump({"x": rx, "y": ry, "w": rw, "h": rh}, f, indent=2)

        QMessageBox.information(self, "Done", f"ROI crop: {total} images saved to:\n{self._output_dir}")
        QMessageBox.information(self, "Done", f"Sliding window: {total} tiles saved to:\n{self._output_dir}")
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
        for path in self._files:
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            rx2, ry2 = min(rx + rw, w), min(ry + rh, h)
            crop = img[ry:ry2, rx:rx2]
            base = os.path.splitext(os.path.basename(path))[0]
            out_name = f"{base}_roi_{rx}_{ry}_{rw}x{rh}.png"
            cv2.imwrite(os.path.join(self._output_dir, out_name), crop)
            total += 1

        # If ROI was drawn on the preview image, save ROI params as JSON for reproducibility
        roi_json = os.path.join(self._output_dir, "_roi_params.json")
        with open(roi_json, "w") as f:
            json.dump({"x": rx, "y": ry, "w": rw, "h": rh}, f, indent=2)

        QMessageBox.information(self, "Done", f"ROI crop: {total} images saved to:\n{self._output_dir}")
