import os, json, numpy as np, cv2
from PIL import Image
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QCursor, QPen, QColor, QBrush, QPolygonF
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QPoint

from core.config import CLASS_COLORS
from annotation.label_manager import LabelManager

class AnnotationCanvas(QWidget):
    mask_changed = pyqtSignal()
    status_message = pyqtSignal(str)

    MODE_BRUSH = \"brush\"
    MODE_POLYGON = \"polygon\"
    MODE_LINE = \"line\"
    MODE_ERASER = \"eraser\"
    MODE_PAN = \"pan\"

    MIN_ZOOM = 0.05
    MAX_ZOOM = 30.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.image = None
        self.mask = None
        self._cached_qimage = None
        self._cached_overlay_qimage = None
        self._overlay_dirty = True

        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self._fit_zoom = 1.0

        self._mode = self.MODE_BRUSH
        self.current_class_id = 1
        self.brush_size = 15

        self._drawing = False
        self._drag_start_pos = None

        self._panning = False
        self._pan_start_mouse = None
        self._pan_start_offset = None

        self._polygon_points = []
        self._hover_point = None
        self._polygon_close_threshold = 12

        self._line_start = None
        self._line_end = None

        self.label_manager = None
        self.current_image_path = None
        self.project_dir = None

        self.setCursor(QCursor(Qt.CrossCursor))

    def set_project(self, project_dir):
        self.project_dir = project_dir
        self.label_manager = LabelManager(project_dir)

    def load_image(self, image_path):
        self.current_image_path = image_path
        self.image = Image.open(image_path).convert(\"RGB\")
        self._cache_qimage()
        base = os.path.splitext(os.path.basename(image_path))[0]
        mp = os.path.join(self.project_dir, \"annotations\", base + \"_mask.png\")
        w, h = self.image.size
        if os.path.exists(mp):
            loaded = np.array(Image.open(mp))
            if loaded.shape[:2] == (h, w):
                self.mask = loaded.astype(np.uint8)
            else:
                self.mask = np.zeros((h, w), dtype=np.uint8)
        else:
            self.mask = np.zeros((h, w), dtype=np.uint8)
        self._overlay_dirty = True
        self._fit_to_window()
        self.update()

    def _fit_to_window(self):
        if self.image is None:
            return
        iw, ih = self.image.size
        ww, wh = self.width(), self.height()
        if ww < 10 or wh < 10:
            return
        self._fit_zoom = min(ww / iw, wh / ih) * 0.92
        self.zoom_level = self._fit_zoom
        self.pan_offset_x = 0
        self.pan_offset_y = 0

    def _image_to_widget_rect(self):
        if self.image is None:
            return QRectF()
        iw, ih = self.image.size
        dw = iw * self.zoom_level
        dh = ih * self.zoom_level
        x = (self.width() - dw) / 2.0 + self.pan_offset_x
        y = (self.height() - dh) / 2.0 + self.pan_offset_y
        return QRectF(x, y, dw, dh)

    def _window_to_image(self, pos):
        if self.image is None:
            return None
        rect = self._image_to_widget_rect()
        ix = (pos.x() - rect.x()) / self.zoom_level
        iy = (pos.y() - rect.y()) / self.zoom_level
        iw, ih = self.image.size
        if ix < 0 or iy < 0 or ix >= iw or iy >= ih:
            return None
        return int(ix), int(iy)

    def _image_to_widget(self, ix, iy):
        rect = self._image_to_widget_rect()
        return QPointF(rect.x() + ix * self.zoom_level, rect.y() + iy * self.zoom_level)

    def _cache_qimage(self):
        if self.image is None:
            return
        arr = np.array(self.image)
        h, w, ch = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        self._cached_qimage = qimg.copy()

    def _build_overlay(self):
        if self.image is None or self.mask is None:
            self._cached_overlay_qimage = None
            return
        if not self._overlay_dirty and self._cached_overlay_qimage is not None:
            return
        h, w = self.mask.shape
        img_arr = np.array(self.image).astype(np.float32)
        overlay = img_arr.copy()
        alpha = 0.45
        max_class = self.mask.max()
        for cid in range(1, int(max_class) + 1):
            region = (self.mask == cid)
            if not region.any():
                continue
            color = CLASS_COLORS[cid % len(CLASS_COLORS)]
            for ch in range(3):
                overlay[:, :, ch][region] = (1 - alpha) * img_arr[:, :, ch][region] + alpha * color[ch]
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        qimg = QImage(overlay.data, w, h, 3 * w, QImage.Format_RGB888)
        self._cached_overlay_qimage = qimg.copy()
        self._overlay_dirty = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        if self.image is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, \"No image loaded\")
            return
        rect = self._image_to_widget_rect()
        if self._cached_qimage is not None:
            painter.drawImage(rect, self._cached_qimage)
        self._build_overlay()
        if self._cached_overlay_qimage is not None:
            painter.drawImage(rect, self._cached_overlay_qimage)
        if self._mode == self.MODE_POLYGON and self._polygon_points:
            self._draw_polygon_preview(painter)
        if self._mode == self.MODE_LINE and self._line_start is not None:
            self._draw_line_preview(painter)
        if self._mode in (self.MODE_BRUSH, self.MODE_ERASER):
            self._draw_brush_cursor(painter)
        painter.setPen(QColor(220, 220, 220))
        zoom_pct = int(self.zoom_level / self._fit_zoom * 100)
        mode_names = {self.MODE_BRUSH: \"Brush\", self.MODE_ERASER: \"Eraser\",
                       self.MODE_POLYGON: \"Polygon\", self.MODE_LINE: \"Line\",
                       self.MODE_PAN: \"Pan\"}
        painter.drawText(8, self.height() - 8,
                          f\"{mode_names.get(self._mode, self._mode)} | Zoom: {zoom_pct}% | Brush: {self.brush_size}\")
