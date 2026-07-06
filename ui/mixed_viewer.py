"""Multi-image viewer for mixed classification (grayscale + height map side-by-side)."""
import os
import numpy as np
from PIL import Image
import cv2
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QImage, QColor, QPen, QFont


def _height_to_rainbow_qimage(height_image, vmin=None, vmax=None, colormap=None):
    """Convert a PIL height map (16-bit/float) to a rainbow-colored QImage."""
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    arr = np.array(height_image, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    valid_mask = (arr > -100) & (arr < 100) & (arr != 0)
    if vmin is not None and vmax is not None:
        pass
    else:
        if valid_mask.any():
            vmin, vmax = np.percentile(arr[valid_mask], (0.5, 99.5))
        else:
            vmin, vmax = arr.min(), arr.max()
        if vmax <= vmin:
            vmin, vmax = arr.min(), arr.max()
        if vmax <= vmin:
            vmax = vmin + 1
    arr_clipped = np.clip(arr, vmin, vmax)
    arr_norm = ((arr_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    arr_inv = 255 - arr_norm
    color = cv2.applyColorMap(arr_inv, colormap)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid_mask] = [40, 40, 40]
    color = np.ascontiguousarray(color)
    h, w, _ = color.shape
    qimg = QImage(color.data, w, h, 3 * w, QImage.Format_RGB888)
    return qimg.copy(), vmin, vmax


class MixedViewer(QWidget):
    """Side-by-side viewer: left=grayscale, right=rainbow height map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gray_image = None
        self._height_image = None
        self._gray_qimage = None
        self._height_qimage = None
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._dragging = False
        self._drag_start = QPointF()
        self._height_vmin = None
        self._height_vmax = None
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)

    def set_pair(self, gray_path, height_path=None):
        """Load grayscale + height map pair. Auto-discovers .tif if height_path is None."""
        self._gray_image = Image.open(gray_path).convert("RGB")
        data = self._gray_image.tobytes("raw", "RGBA") if self._gray_image.mode != "RGBA" else self._gray_image.tobytes("raw", "RGBA")
        if self._gray_image.mode != "RGBA":
            img_rgba = self._gray_image.convert("RGBA")
            data = img_rgba.tobytes("raw", "RGBA")
            w, h = img_rgba.size
        else:
            data = self._gray_image.tobytes("raw", "RGBA")
            w, h = self._gray_image.size
        self._gray_qimage = QImage(data, w, h, QImage.Format_RGBA8888).copy()

        # Load height map
        if height_path and os.path.exists(height_path):
            self._height_image = Image.open(height_path)
        else:
            base = os.path.splitext(gray_path)[0]
            for ext in (".tif", ".tiff"):
                candidate = base + ext
                if os.path.exists(candidate):
                    self._height_image = Image.open(candidate)
                    break
        if self._height_image is not None:
            self._height_qimage, vmin, vmax = _height_to_rainbow_qimage(
                self._height_image, self._height_vmin, self._height_vmax)
            if self._height_vmin is None:
                self._height_vmin = vmin
                self._height_vmax = vmax

        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._fit_to_window()
        self.update()

    def _fit_to_window(self):
        """Auto-fit both images side by side."""
        if self._gray_image is None:
            return
        iw, ih = self._gray_image.size
        hiw, hih = self._height_image.size if self._height_image else (0, 0)
        gap = 10
        total_w = iw + hiw + gap
        total_h = max(ih, hih)
        if total_w > 0 and total_h > 0:
            self._zoom = min(self.width() / total_w, self.height() / total_h)
            self._pan_x = 0
            self._pan_y = 0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_to_window()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        if self._gray_image is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return

        iw, ih = self._gray_image.size
        if self._height_image is not None:
            hiw, hih = self._height_image.size
        else:
            hiw, hih = 0, 0
        gap = 10
        total_w = iw + hiw + gap
        total_h = max(ih, hih)
        scale = self._zoom

        # Center in widget
        cx = (self.width() - total_w * scale) / 2.0 + self._pan_x
        cy = (self.height() - total_h * scale) / 2.0 + self._pan_y

        # Gray image on left
        gray_rect = QRectF(cx, cy + (total_h - ih) * scale / 2.0,
                           iw * scale, ih * scale)
        painter.drawImage(gray_rect, self._gray_qimage)

        if self._height_qimage is not None and hiw > 0:
            # Height image on right
            height_rect = QRectF(cx + iw * scale + gap * scale,
                                 cy + (total_h - hih) * scale / 2.0,
                                 hiw * scale, hih * scale)
            painter.drawImage(height_rect, self._height_qimage)

            # Divider line
            div_x = cx + iw * scale + gap * scale / 2.0
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            painter.drawLine(QPointF(div_x, cy),
                             QPointF(div_x, cy + total_h * scale))

            # Labels
            font = QFont("Consolas", max(8, int(10 * scale)))
            painter.setFont(font)
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(gray_rect, Qt.AlignBottom | Qt.AlignHCenter, "Grayscale")
            painter.setPen(QColor(200, 180, 100))
            painter.drawText(height_rect, Qt.AlignBottom | Qt.AlignHCenter, "Height Map")

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom *= factor
        self._zoom = max(0.1, min(50.0, self._zoom))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.pos() - self._drag_start
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._drag_start = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)

    def reset_view(self):
        self._fit_to_window()
        self.update()

    @property
    def image(self):
        return self._gray_image

    @property
    def _mixed_cls_mode(self):
        return True
