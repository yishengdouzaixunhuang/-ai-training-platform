"""Semantic segmentation annotation canvas with zoom, pan, brush, polygon, line tools."""
import os, json, re, numpy as np, cv2
from PIL import Image
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QCursor, QPen, QColor, QBrush, QPolygonF, QFont, QFontMetrics
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QPoint

from core.config import CLASS_COLORS, IGNORE_CLASS_ID, IGNORE_CLASS_COLOR
from annotation.label_manager import LabelManager
from annotation.labelme_io import load_mask_from_json, save_mask_to_json, load_labelme_json, shapes_to_mask, get_json_path

class AnnotationCanvas(QWidget):
    mask_changed = pyqtSignal()
    det_boxes_changed = pyqtSignal()
    status_message = pyqtSignal(str)

    MODE_BRUSH = "brush"
    MODE_POLYGON = "polygon"
    MODE_LINE = "line"
    MODE_ERASER = "eraser"
    MODE_IGNORE = "ignore"
    MODE_PAN = "pan"
    MODE_SAM = "sam"
    MODE_SAM = "sam"

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
        self._cached_gray = None  # cached grayscale numpy array
        self._det_overlay = None  # detection box overlay (attached by main_window)
        self._cached_overlay_qimage = None
        self._overlay_dirty = True
        self._show_annotation = True
        self._show_prediction = False
        self._show_annotation_boxes = True   # Ctrl+Space: toggle annotation boxes
        self._show_pred_boxes = True         # Space: toggle prediction boxes
        self._pred_overlay_cache = {}  # key -> QImage
        self._pred_stats_cache = {}  # class_id -> (area, avg_gray, radius, iou)
        self.setMouseTracking(True)

        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self._fit_zoom = 1.0

        self._mode = self.MODE_PAN
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
        # SAM state
        self._sam_points = []
        self._sam_labels = []
        self._sam_mask = None
        self._sam_score = 0.0
        self._sam_predicting = False
        # SAM state
        self._sam_points = []
        self._sam_labels = []
        self._sam_mask = None
        self._sam_score = 0.0
        self._sam_predicting = False

        self.label_manager = None
        self.current_image_path = None
        self.project_dir = None

        self.setCursor(QCursor(Qt.CrossCursor))
        self._overlay_visible = True
        self._undo_stack = []  # (mask_copy, action_desc)
        self._max_undo = 50
        self._prediction_qimage = None  # Separate storage for inference results
        self._heatmap_qimage = None     # Grad-CAM heatmap overlay
        self._show_heatmap = False      # Space toggles heatmap in cls mode
        self._ocr_overlay = None          # OCR detection overlay
        self._heatmap_qimage = None     # Grad-CAM heatmap overlay
        self._show_heatmap = False      # Space toggles heatmap in cls mode
        
        self.mask_changed.connect(self._auto_save_json)

    def set_project(self, project_dir):
        self.project_dir = project_dir
        self.label_manager = LabelManager(project_dir)

    def load_image(self, image_path):
        self.current_image_path = image_path
        self.image = Image.open(image_path).convert("RGB")
        self._cache_qimage()
        self._heatmap_qimage = None
        self._ocr_overlay = None
        self._heatmap_qimage = None
        w, h = self.image.size

        # Load class names for label mapping
        class_names = self.label_manager.classes if self.label_manager else ["background"]

        # Try JSON first, fall back to PNG mask
        loaded_mask = load_mask_from_json(image_path, self.project_dir, class_names)
        if loaded_mask is not None:
            self.mask = loaded_mask
        else:
            self.mask = np.zeros((h, w), dtype=np.uint8)

        self._overlay_dirty = True
        self._fit_to_window()
        self._load_prediction_overlay(image_path)
        self.update()

    def _fit_to_window(self):
        if self.image is None:
            return
        iw, ih = self.image.size
        ww, wh = self.width(), self.height()
        if ww < 10 or wh < 10:
            return
        self._fit_zoom = min(ww / iw, wh / ih) * 0.99
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


    def _window_to_image_unclamped(self, pos):
        """Convert widget pos to image coords, allowing outside-image values."""
        if self.image is None:
            return None
        rect = self._image_to_widget_rect()
        ix = (pos.x() - rect.x()) / self.zoom_level
        iy = (pos.y() - rect.y()) / self.zoom_level
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
        self._cached_gray = np.mean(arr.astype(np.float32), axis=2)

    def _build_overlay(self):
        """Build annotation overlay with HATCH pattern (distinct from solid prediction fill)."""
        if self.image is None or self.mask is None:
            self._cached_overlay_qimage = None
            return
        if not self._overlay_dirty and self._cached_overlay_qimage is not None:
            return
        h, w = self.mask.shape
        img_arr = np.array(self.image).astype(np.float32)
        ih, iw = img_arr.shape[:2]
        if h != ih or w != iw:
            self._cached_overlay_qimage = None
            return
        overlay = img_arr.copy()
        import cv2
        # Diagonal hatch pattern: 3px line every 12px
        yy, xx = np.mgrid[0:h, 0:w]
        hatch = ((xx + yy) % 12 < 3).astype(np.float32)
        unique_ids = np.unique(self.mask)
        for cid in unique_ids:
            if cid == 0:
                continue
            region = (self.mask == cid)
            if not region.any():
                continue
            if int(cid) == IGNORE_CLASS_ID:
                color = IGNORE_CLASS_COLOR
            else:
                color = CLASS_COLORS[int(cid) % len(CLASS_COLORS)]
            # Only draw color on hatch lines within the region
            hatch_region = region & (hatch > 0)
            for ch in range(3):
                overlay[:, :, ch][hatch_region] = color[ch]
            # Thin border around each region
            binary = region.astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (int(color[0]), int(color[1]), int(color[2])), 1)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        qimg = QImage(overlay.data, w, h, 3 * w, QImage.Format_RGB888)
        self._cached_overlay_qimage = qimg.copy()
        self._overlay_dirty = False

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.zoom_level < 1.0:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        if self.image is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return
        rect = self._image_to_widget_rect()
        if self._cached_qimage is not None:
            painter.drawImage(rect, self._cached_qimage)
        self._build_overlay()
        # Draw annotation overlay (hatch pattern)
        if self._show_annotation and self._cached_overlay_qimage is not None:
            painter.drawImage(rect, self._cached_overlay_qimage)
        # Draw prediction overlay on top
        if self._show_prediction:
            with_bg = not self._show_annotation  # transparent when annotation is also visible
            pred_arr = self._build_prediction_overlay(with_background=with_bg)
            if pred_arr is not None:
                h, w = pred_arr.shape[:2]
                if pred_arr.shape[2] == 4:
                    qimg = QImage(pred_arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
                else:
                    qimg = QImage(pred_arr.data, w, h, 3 * w, QImage.Format_RGB888)
                painter.drawImage(rect, qimg.copy())
            elif self._prediction_qimage is not None:
                pw, ph = self._prediction_qimage.width(), self._prediction_qimage.height()
                iw, ih = self.image.size
                if pw == iw and ph == ih:
                    if not with_bg:
                        painter.setOpacity(0.5)
                    painter.drawImage(rect, self._prediction_qimage)
                    painter.setOpacity(1.0)

        # Draw Grad-CAM heatmap overlay (classification mode)
        if (getattr(self, "_cls_mode", False) and self._show_heatmap
                and self._heatmap_qimage is not None):
            pw = self._heatmap_qimage.width()
            ph = self._heatmap_qimage.height()
            iw, ih = self.image.size
            if pw == iw and ph == ih:
                painter.setOpacity(0.55)
                painter.drawImage(rect, self._heatmap_qimage)
                painter.setOpacity(1.0)

        if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and self._polygon_points:
            painter.setClipping(False)
            self._draw_polygon_preview(painter)
            painter.setClipping(True)
        if self._mode == self.MODE_LINE and self._line_start is not None:
            painter.setClipping(False)
            self._draw_line_preview(painter)
            painter.setClipping(True)
        if self._mode == self.MODE_SAM:
            self._draw_sam_overlay(painter)
        if self._mode in (self.MODE_BRUSH, self.MODE_ERASER):
            self._draw_brush_cursor(painter)

        # Draw detection bounding boxes
        if self._det_overlay is not None:
            scale = self.zoom_level
            self._det_overlay.paint(painter, scale, rect.x(), rect.y(),
                                    show_annotations=self._show_annotation_boxes,
                                    show_predictions=self._show_pred_boxes)

            # Draw preview rect while drawing new box
            preview = self._det_overlay.get_drawing_preview_rect()
            if preview:
                x, y, w, h = [v * scale for v in preview]
                x += rect.x(); y += rect.y()
                pen = QPen(QColor(255, 255, 255, 150), 1, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(255, 255, 255, 30)))
                painter.drawRect(QRectF(x, y, w, h))

        zoom_pct = int(self.zoom_level / self._fit_zoom * 100)
        mode_names = {self.MODE_BRUSH: 'Brush', self.MODE_ERASER: 'Eraser', self.MODE_IGNORE: 'Ignore',
                       self.MODE_POLYGON: "Polygon", self.MODE_LINE: "Line",
                       self.MODE_PAN: "Pan", self.MODE_SAM: "SAM"}
        ann_status = "Annot:ON" if self._show_annotation else "Annot:OFF"
        if self._det_overlay is not None:
            pred_status = "Pred:ON" if self._show_pred_boxes else "Pred:OFF"
        else:
            pred_status = "Pred:ON" if self._show_prediction else "Pred:OFF"
        res = "%dx%d" % (self.image.size[0], self.image.size[1]) if self.image else "N/A"
        info = "%s | %s | %s | %s | Zoom: %d%% | Brush: %d" % (
            res,
            mode_names.get(self._mode, self._mode),
            ann_status, pred_status, zoom_pct, self.brush_size)
        self._draw_class_labels(painter)
        # Semi-transparent background so boxes don't obscure status text
        font = QFont("Sans", 9)
        painter.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(info) + 16
        th = fm.height() + 8
        bg_rect = QRectF(0, self.height() - th - 4, tw, th)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(bg_rect, 4, 4)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(8, self.height() - 8, info)


    def _build_prediction_overlay(self, with_background=True):
        """Build prediction overlay. Cached for performance."""
        mask = getattr(self, "_prediction_mask", None)
        if mask is None or self.image is None:
            return None
        h, w = mask.shape
        iw, ih = self.image.size
        if h != ih or w != iw:
            return None
        cache_key = (id(mask), with_background, self.current_image_path)
        if cache_key in self._pred_overlay_cache:
            return self._pred_overlay_cache[cache_key]
        import cv2
        from core.config import CLASS_COLORS
        if with_background:
            img = np.array(self.image).astype(np.float32).copy()
            alpha = 0.45
        else:
            # Transparent RGBA: annotation shows through
            img = np.zeros((h, w, 4), dtype=np.float32)
            alpha = 0.55
        for c in range(1, 255):
            region = (mask == c)
            if not region.any():
                continue
            color = np.array(CLASS_COLORS[c % len(CLASS_COLORS)])
            if with_background:
                for ch in range(3):
                    img[:, :, ch][region] = (1 - 0.45) * img[:, :, ch][region] + 0.45 * color[ch]
                # Draw thin border
                binary = region.astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, color.tolist() if hasattr(color, 'tolist') else (int(color[0]), int(color[1]), int(color[2])), 1)
            else:
                for ch in range(3):
                    img[:, :, ch][region] = color[ch]
                img[:, :, 3][region] = int(alpha * 255)
            # Draw class label
            if with_background:
                binary = region.astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        label = getattr(self, "_pred_classes", ["bg"] + [f"c{i}" for i in range(1, 256)])[c] if hasattr(self, "_pred_classes") and c < len(self._pred_classes) else f"c{c}"
                        cv2.putText(img, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        result = np.clip(img, 0, 255).astype(np.uint8)
        self._pred_overlay_cache[cache_key] = result
        return result

    def _compute_pred_stats(self):
        """Precompute per-class stats for hover tooltip (called once when prediction loads)."""
        mask = getattr(self, "_prediction_mask", None)
        if mask is None:
            self._pred_stats_cache = {}
            return
        import numpy as _np
        stats = {}
        for cid in _np.unique(mask):
            if cid == 0:
                continue
            region = (mask == cid)
            area = int(region.sum())
            avg_gray = 0.0
            if self.image is not None:
                img_arr = _np.array(self.image)
                gray = _np.mean(img_arr, axis=2) if img_arr.ndim == 3 else img_arr.astype(_np.float64)
                if gray.shape[:2] == region.shape[:2]:
                    try:
                        avg_gray = float(gray[region].mean())
                    except IndexError:
                        pass
            radius = int(_np.sqrt(area / _np.pi))
            # IoU: find best-matching annotation class (not all classes combined)
            iou_val = "?"
            if self.mask is not None and self.mask.shape == mask.shape:
                best_iou = 0.0
                for ac in _np.unique(self.mask):
                    if ac == 0:
                        continue
                    ann_region = (self.mask == ac)
                    inter = (region & ann_region).sum()
                    union = (region | ann_region).sum()
                    if union > 0:
                        ciou = inter / union
                        if ciou > best_iou:
                            best_iou = ciou
                if best_iou > 0:
                    iou_val = "%.3f" % best_iou
            class_names = getattr(self, "_pred_classes", None)
            cls_name = (class_names[int(cid)] if class_names and int(cid) < len(class_names) else f"Class{int(cid)}")
            stats[int(cid)] = (cls_name, area, avg_gray, radius, iou_val)
        self._pred_stats_cache = stats

    def _draw_class_labels(self, painter):
        """Draw class name labels at top-left corner of each annotation region."""
        if not self._show_annotation:
            return
        if not self.current_image_path or not self.label_manager:
            return
        try:
            json_path = os.path.splitext(self.current_image_path)[0] + ".json"
            shapes, _ = load_labelme_json(json_path)
            if not shapes:
                return
            for shape in shapes:
                try:
                    label = shape.get("label", "")
                    points = shape.get("points", [])
                    if not points or len(points) < 2:
                        continue
                    if "ignore" in label.lower() or "gnore" in label.lower():
                        # Draw gray "IGNORE" label instead of skipping
                        pts = np.array(points, dtype=np.float32)
                        cx = int(pts[:, 0].min())
                        cy = int(pts[:, 1].min())
                        pt = self._image_to_widget(cx, cy)
                        wx, wy = pt.x(), pt.y()
                        font_size = max(8, int(10 * self.zoom_level))
                        font = QFont("Sans", font_size, QFont.Bold)
                        painter.setFont(font)
                        fm = QFontMetrics(font)
                        label_text = "IGNORE"
                        tw = fm.horizontalAdvance(label_text) + 8
                        th = fm.height() + 4
                        bg_rect = QRectF(wx - 2, wy - th, tw, th)
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QColor(80, 80, 80, 180))
                        painter.drawRoundedRect(bg_rect, 3, 3)
                        painter.setPen(QColor(200, 200, 200))
                        painter.drawText(bg_rect, Qt.AlignCenter, label_text)
                        continue
                    # Use first point as top-left anchor
                    pts = np.array(points, dtype=np.float32)
                    cx = int(pts[:, 0].min())
                    cy = int(pts[:, 1].min())
                    pt = self._image_to_widget(cx, cy)
                    wx, wy = pt.x(), pt.y()
                    font_size = max(8, int(10 * self.zoom_level))
                    font = QFont("Sans", font_size, QFont.Bold)
                    painter.setFont(font)
                    fm = QFontMetrics(font)
                    tw = fm.horizontalAdvance(label) + 8
                    th = fm.height() + 4
                    bg_rect = QRectF(wx - 2, wy - th, tw, th)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(0, 0, 0, 170))
                    painter.drawRoundedRect(bg_rect, 3, 3)
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(bg_rect, Qt.AlignCenter, label)
                except Exception:
                    pass
        except Exception:
            pass

    def _draw_brush_cursor(self, painter):
        pos = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(pos):
            return
        pt = self._window_to_image(pos)
        if pt is None:
            return
        wp = self._image_to_widget(pt[0], pt[1])
        r = self.brush_size * self.zoom_level
        if self._mode == self.MODE_ERASER:
            color = QColor(255, 0, 0, 100)
        else:
            color = QColor(0, 200, 255, 100)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
        painter.drawEllipse(wp, r, r)

    def _draw_polygon_preview(self, painter):
        if len(self._polygon_points) < 1:
            return
        pts = self._polygon_points
        if self._mode == self.MODE_IGNORE:
            color = (128, 128, 128)  # gray for ignore
        else:
            color = CLASS_COLORS[self.current_class_id % len(CLASS_COLORS)]
        qcolor = QColor(*color)
        qcolor = QColor(*color)

        # Draw solid polyline (no fill, to keep image visible)
        painter.setBrush(Qt.NoBrush)
        if len(pts) >= 2:
            wpts = [self._image_to_widget(p[0], p[1]) for p in pts]
            # Outer glow for visibility
            pen_glow = QPen(QColor(255, 255, 255, 180), 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen_glow)
            painter.drawPolyline(QPolygonF(wpts))
            # Colored line
            pen_line = QPen(qcolor, 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen_line)
            painter.drawPolyline(QPolygonF(wpts))

        # Draw vertices as filled circles with border
        for i, p in enumerate(pts):
            wp = self._image_to_widget(p[0], p[1])
            r = 5 if i == 0 else 4
            # White border
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(qcolor if i > 0 else QColor(0, 255, 0)))
            painter.drawEllipse(wp, r, r)

        # Draw line from last vertex to cursor
        pos = self.mapFromGlobal(QCursor.pos())
        if self.rect().contains(pos):
            hpt = self._window_to_image_unclamped(pos)
            if hpt is not None:
                last = pts[-1]
                lw = self._image_to_widget(last[0], last[1])
                hw = self._image_to_widget(hpt[0], hpt[1])
                # Dashed guide line from last vertex to cursor
                pen_guide = QPen(QColor(255, 255, 255, 150), 1.5, Qt.DashLine)
                painter.setPen(pen_guide)
                painter.drawLine(lw, hw)

                # Check if hovering near start point for closing
                fp = pts[0]
                fw = self._image_to_widget(fp[0], fp[1])
                dist = ((hw.x() - fw.x())**2 + (hw.y() - fw.y())**2)**0.5
                if dist < self._polygon_close_threshold and len(pts) >= 3:
                    # Highlight start point to indicate can close
                    painter.setPen(QPen(QColor(0, 255, 0), 3))
                    painter.setBrush(QBrush(QColor(0, 255, 0, 80)))
                    painter.drawEllipse(fw, 8, 8)

    def _draw_line_preview(self, painter):
        if self._line_start is None:
            return
        pos = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(pos):
            return
        end_pt = self._window_to_image_unclamped(pos)
        if end_pt is None:
            return
        sp = self._image_to_widget(self._line_start[0], self._line_start[1])
        ep = self._image_to_widget(end_pt[0], end_pt[1])
        color = CLASS_COLORS[self.current_class_id % len(CLASS_COLORS)]
        pen = QPen(QColor(*color), self.brush_size * self.zoom_level, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(sp, ep)

    # ======================== Mouse Events ========================

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start_mouse = event.pos()
            self._pan_start_offset = (self.pan_offset_x, self.pan_offset_y)
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            return
        if self.image is None:
            return
        if event.button() == Qt.RightButton:
            if self._mode == self.MODE_SAM:
                self._handle_sam_click(event)
                return
            if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and self._polygon_points:
                self._polygon_points = []
                self.update()
                self.status_message.emit("Polygon cancelled")
            return
        # Detection mode: delegate to overlay
        if self._det_overlay is not None and self._mode == self.MODE_PAN:
            pt = self._window_to_image(event.pos())
            if pt is not None and self._det_overlay.mouse_press(pt, event.modifiers()):
                self.update()
                return

            return
        pt = self._window_to_image(event.pos())
        if pt is None and self._mode not in (self.MODE_POLYGON, self.MODE_IGNORE, self.MODE_LINE, self.MODE_SAM):
            return
        if self._mode == self.MODE_PAN:
            self._panning = True
            self._pan_start_mouse = event.pos()
            self._pan_start_offset = (self.pan_offset_x, self.pan_offset_y)
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif self._mode == self.MODE_BRUSH:
            self._push_undo("brush")
            self._drawing = True
            self._drag_start_pos = pt
            self._draw_brush_at(event.pos())
        elif self._mode == self.MODE_ERASER:
            self._push_undo("eraser")
            self._drawing = True
            self._drag_start_pos = pt
            self._draw_eraser_at(event.pos())
        elif self._mode in (self.MODE_POLYGON, self.MODE_IGNORE):
            if not self._polygon_points:
                self._push_undo("polygon")
            self._add_polygon_point(self._window_to_image_unclamped(event.pos()))
        elif self._mode == self.MODE_LINE:
            self._push_undo("line")
            self._line_start = self._window_to_image_unclamped(event.pos())
            self._line_end = self._window_to_image_unclamped(event.pos())
            self.update()
        elif self._mode == self.MODE_SAM:
            self._handle_sam_click(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            if self._pan_start_mouse is None:
                return
            dx = event.pos().x() - self._pan_start_mouse.x()
            dy = event.pos().y() - self._pan_start_mouse.y()
            self.pan_offset_x = self._pan_start_offset[0] + dx
            self.pan_offset_y = self._pan_start_offset[1] + dy
            self.update()
            return
        # Detection mode: delegate to overlay
        if self._det_overlay is not None:
            pt = self._window_to_image(event.pos())
            if pt is not None and self._det_overlay.mouse_move(pt):
                self.update()
                return

            return
        if self._drawing and self._mode == self.MODE_BRUSH:
            self._draw_brush_at(event.pos())
        elif self._drawing and self._mode == self.MODE_ERASER:
            self._draw_eraser_at(event.pos())
        if self._mode in (self.MODE_BRUSH, self.MODE_ERASER, self.MODE_POLYGON, self.MODE_IGNORE, self.MODE_LINE):
            self.update()
        # Prediction hover tooltip (uses precomputed stats, O(1) per move)
        if self._show_prediction and hasattr(self, "_prediction_mask") and self._prediction_mask is not None:
            pt = self._window_to_image(event.pos())
            if pt is not None:
                px, py = int(pt[0]), int(pt[1])
                h, w = self._prediction_mask.shape
                if 0 <= px < w and 0 <= py < h:
                    cid = int(self._prediction_mask[py, px])
                    if cid > 0:
                        if not self._pred_stats_cache:
                            self._compute_pred_stats()
                        s = self._pred_stats_cache.get(cid)
                        if s:
                            cls_name, area, avg_gray, radius, iou_str = s
                            tip = f"{cls_name} | Area:{area}px | Gray:{avg_gray:.0f} | R:{radius}px | IoU:{iou_str}"
                            self.setToolTip(tip)
                            return
        self.setToolTip("")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton or (self._panning and event.button() == Qt.LeftButton):
            self._panning = False
            self._update_mode_cursor()
            self.update()
            return
        if self._det_overlay is not None:
            pt = self._window_to_image(event.pos())
            if pt is not None:
                self._det_overlay.mouse_release(pt)
                self.det_boxes_changed.emit()
            self.update()

            self._drawing = False
            self._overlay_dirty = True
            self.update()
            self.mask_changed.emit()
        if self._mode == self.MODE_LINE and self._line_start is not None:
            pt = self._window_to_image_unclamped(event.pos())
            if pt is not None:
                self._line_end = pt
                self._apply_line()
            self._line_start = None
            self._line_end = None
            self._overlay_dirty = True
            self.update()
            self.mask_changed.emit()

    def mouseDoubleClickEvent(self, event):
        if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and len(self._polygon_points) >= 3:
            self._close_polygon()

    def wheelEvent(self, event):
        if self.image is None:
            return
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 1 / 1.12
        new_zoom = self.zoom_level * factor
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        actual_factor = new_zoom / self.zoom_level
        self.zoom_level = new_zoom
        self.pan_offset_x = event.pos().x() - actual_factor * (event.pos().x() - self.pan_offset_x)
        self.pan_offset_y = event.pos().y() - actual_factor * (event.pos().y() - self.pan_offset_y)
        self.update()

    def keyPressEvent(self, event):
        # SAM mode: Enter=accept, Esc=cancel, Backspace=undo last point
        if self._mode == self.MODE_SAM:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self._accept_sam_mask()
                return
            elif event.key() == Qt.Key_Escape:
                self._cancel_sam()
                return
            elif event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
                if self._sam_points:
                    self._sam_points.pop()
                    self._sam_labels.pop()
                    if self._sam_points:
                        self._run_sam_prediction()
                    else:
                        self._sam_mask = None
                        self.update()
                        self.status_message.emit("SAM: all points removed")
                return
        # SAM mode: Enter=accept, Esc=cancel, Backspace=undo last point
        if self._mode == self.MODE_SAM:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self._accept_sam_mask()
                return
            elif event.key() == Qt.Key_Escape:
                self._cancel_sam()
                return
            elif event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
                if self._sam_points:
                    self._sam_points.pop()
                    self._sam_labels.pop()
                    if self._sam_points:
                        self._run_sam_prediction()
                    else:
                        self._sam_mask = None
                        self.update()
                        self.status_message.emit("SAM: all points removed")
                return
        # Detection mode: Delete = remove selected box
        if self._det_overlay is not None and (event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace):
            self._det_overlay.delete_active_box()
            self.update()
            self.det_boxes_changed.emit()
            return

        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            # Ctrl+Z: undo last mask edit
            if self._undo_stack:
                self.mask, desc = self._undo_stack.pop()
                self._overlay_dirty = True
                self.update()
                self.mask_changed.emit()
                self.status_message.emit("Undo: %s (%d left)" % (desc, len(self._undo_stack)))
            else:
                self.status_message.emit("Nothing to undo")
        elif event.key() == Qt.Key_Escape:
            if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and self._polygon_points:
                self._polygon_points = []
                self.update()
                self.status_message.emit("Polygon cancelled")
            self._line_start = None
            self._line_end = None
            self.update()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and len(self._polygon_points) >= 3:
                self._close_polygon()
        elif event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
            if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and self._polygon_points:
                self._polygon_points.pop()
                self.update()
                self.status_message.emit("Removed point, %d remaining" % len(self._polygon_points))
        elif event.key() == Qt.Key_Space and event.modifiers() == Qt.ControlModifier:
            if self._det_overlay is not None:
                # Detection mode: Ctrl+Space -> toggle annotation boxes
                self._show_annotation_boxes = not self._show_annotation_boxes
                self.status_message.emit("Annotation boxes: %s" % ("ON" if self._show_annotation_boxes else "OFF"))
            else:
                # Segmentation mode: toggle annotation overlay
                self._show_annotation = not self._show_annotation
                self._overlay_dirty = True
                self.status_message.emit("Annotation: %s" % ("ON" if self._show_annotation else "OFF"))
            self.update()
            event.accept()
        elif event.key() == Qt.Key_Space:
            if self._det_overlay is not None:
                # Detection mode: Space -> toggle prediction boxes
                self._show_pred_boxes = not self._show_pred_boxes
                self.status_message.emit("Prediction boxes: %s" % ("ON" if self._show_pred_boxes else "OFF"))
            elif getattr(self, "_cls_mode", False):
                # Classification mode: Space -> toggle Grad-CAM heatmap
                self._show_heatmap = not self._show_heatmap
                self.status_message.emit("Heatmap: %s" % ("ON" if self._show_heatmap else "OFF"))
            else:
                # Segmentation mode: toggle prediction overlay
                self._show_prediction = not self._show_prediction
                self.status_message.emit("Prediction: %s" % ("ON" if self._show_prediction else "OFF"))
            self.update()
            event.accept()

    # ======================== Annotation Operations ========================

    def _draw_brush_at(self, pos):
        pt = self._window_to_image(pos)
        if pt is None:
            return
        px, py = pt
        if self.zoom_level > 1.0:
            r = int(self.brush_size / self.zoom_level)
        else:
            r = max(1, self.brush_size)
        r = max(1, r)
        h, w = self.mask.shape
        x1, y1 = max(0, px - r), max(0, py - r)
        x2, y2 = min(w, px + r), min(h, py + r)
        self.mask[y1:y2, x1:x2] = self.current_class_id
        self._overlay_dirty = True
        self.update()

    def _draw_eraser_at(self, pos):
        pt = self._window_to_image(pos)
        if pt is None:
            return
        px, py = pt
        if self.zoom_level > 1.0:
            r = int(self.brush_size / self.zoom_level)
        else:
            r = max(1, self.brush_size)
        r = max(1, r)
        h, w = self.mask.shape
        x1, y1 = max(0, px - r), max(0, py - r)
        x2, y2 = min(w, px + r), min(h, py + r)
        self.mask[y1:y2, x1:x2] = 0
        self._overlay_dirty = True
        self.update()

    def _add_polygon_point(self, pt):
        if len(self._polygon_points) >= 3:
            fp = self._polygon_points[0]
            fw = self._image_to_widget(fp[0], fp[1])
            pw = self._image_to_widget(pt[0], pt[1])
            if ((fw.x() - pw.x())**2 + (fw.y() - pw.y())**2)**0.5 < self._polygon_close_threshold:
                self._close_polygon()
                return
        self._polygon_points.append(pt)
        self.status_message.emit("Polygon: %d points" % len(self._polygon_points))
        self.update()

    def _close_polygon(self):
        if len(self._polygon_points) < 3:
            return
        pts = np.array(self._polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        fill_value = 255 if self._mode == self.MODE_IGNORE else self.current_class_id
        cv2.fillPoly(self.mask, [pts], color=fill_value)
        self._polygon_points = []
        self._overlay_dirty = True
        self.update()
        self.mask_changed.emit()
        self.status_message.emit("Polygon filled")

    def _apply_line(self):
        if self._line_start is None or self._line_end is None:
            return
        x1, y1 = self._line_start
        x2, y2 = self._line_end
        if self.zoom_level > 1.0:
            thickness = max(1, int(self.brush_size / self.zoom_level))
        else:
            thickness = max(1, self.brush_size)
        cv2.line(self.mask, (x1, y1), (x2, y2), color=self.current_class_id, thickness=thickness)
        self.status_message.emit("Line drawn (width=%d)" % thickness)

    # ======================== SAM Assist ========================

    def _handle_sam_click(self, event):
        """Add a SAM prompt point on click."""
        pt = self._window_to_image(event.pos())
        if pt is None:
            return
        if event.button() == Qt.LeftButton:
            self._sam_points.append(pt)
            self._sam_labels.append(1)  # foreground
        elif event.button() == Qt.RightButton:
            self._sam_points.append(pt)
            self._sam_labels.append(0)  # background
        else:
            return
        self._run_sam_prediction()

    def _run_sam_prediction(self):
        """Run SAM inference with current points."""
        if self._sam_predicting:
            return
        if not self._sam_points:
            return
        self._sam_predicting = True
        self.status_message.emit("SAM: predicting...")
        self.update()
        import threading
        def _predict():
            try:
                from annotation.sam_assist import sam_predict
                img = self.image
                result = sam_predict(img, self._sam_points, self._sam_labels)
                if result:
                    self._sam_mask = result["mask"]
                    self._sam_score = result["score"]
                    self.status_message.emit(
                        "SAM: score=%.3f | Enter=accept, Esc=cancel, +click=refine" % self._sam_score)
                else:
                    self.status_message.emit("SAM: prediction returned no mask")
            except ImportError as e:
                self.status_message.emit("SAM: segment-anything not installed. Run: pip install segment-anything")
            except Exception as e:
                self.status_message.emit("SAM error: %s" % str(e))
            finally:
                self._sam_predicting = False
                self.update()
        threading.Thread(target=_predict, daemon=True).start()

    def _draw_sam_overlay(self, painter):
        """Draw SAM prompt points and mask preview."""
        rect = self._image_to_widget_rect()
        scale = self.zoom_level

        # Draw mask preview
        if self._sam_mask is not None and self._sam_mask.any():
            painter.setClipping(True)
            painter.setClipRect(rect)
            h, w = self._sam_mask.shape
            arr = np.zeros((h, w, 4), dtype=np.uint8)
            color = list(CLASS_COLORS[self.current_class_id % len(CLASS_COLORS)])
            arr[self._sam_mask, :3] = color
            arr[self._sam_mask, 3] = 120
            qimg = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
            painter.drawImage(rect, qimg)
            painter.setClipping(False)

        # Draw prompt points
        for i, pt in enumerate(self._sam_points):
            wp = self._image_to_widget(pt[0], pt[1])
            if self._sam_labels[i] == 1:
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.setBrush(QBrush(QColor(0, 255, 0, 180)))
            else:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.setBrush(QBrush(QColor(255, 0, 0, 180)))
            painter.drawEllipse(wp, 5, 5)

    def _accept_sam_mask(self):
        """Accept SAM mask: fill into current class mask."""
        if self._sam_mask is None or not self._sam_mask.any():
            self.status_message.emit("SAM: no mask to accept")
            return
        self._push_undo("sam")
        self.mask[self._sam_mask] = self.current_class_id
        self._overlay_dirty = True
        self._sam_points = []
        self._sam_labels = []
        self._sam_mask = None
        self._sam_score = 0.0
        self.update()
        self.mask_changed.emit()
        self.status_message.emit("SAM mask applied")

    def _cancel_sam(self):
        """Cancel SAM: clear all points and mask."""
        self._sam_points = []
        self._sam_labels = []
        self._sam_mask = None
        self._sam_score = 0.0
        self.update()
        self.status_message.emit("SAM: cancelled")

    # ======================== SAM Assist ========================

    def _handle_sam_click(self, event):
        """Add a SAM prompt point on click."""
        pt = self._window_to_image(event.pos())
        if pt is None:
            return
        if event.button() == Qt.LeftButton:
            self._sam_points.append(pt)
            self._sam_labels.append(1)  # foreground
        elif event.button() == Qt.RightButton:
            self._sam_points.append(pt)
            self._sam_labels.append(0)  # background
        else:
            return
        self._run_sam_prediction()

    def _run_sam_prediction(self):
        """Run SAM inference with current points."""
        if self._sam_predicting:
            return
        if not self._sam_points:
            return
        self._sam_predicting = True
        self.status_message.emit("SAM: predicting...")
        self.update()
        import threading
        def _predict():
            try:
                from annotation.sam_assist import sam_predict
                img = self.image
                result = sam_predict(img, self._sam_points, self._sam_labels)
                if result:
                    self._sam_mask = result["mask"]
                    self._sam_score = result["score"]
                    self.status_message.emit(
                        "SAM: score=%.3f | Enter=accept, Esc=cancel, +click=refine" % self._sam_score)
                else:
                    self.status_message.emit("SAM: prediction returned no mask")
            except ImportError as e:
                self.status_message.emit("SAM: segment-anything not installed. Run: pip install segment-anything")
            except Exception as e:
                self.status_message.emit("SAM error: %s" % str(e))
            finally:
                self._sam_predicting = False
                self.update()
        threading.Thread(target=_predict, daemon=True).start()

    def _draw_sam_overlay(self, painter):
        """Draw SAM prompt points and mask preview."""
        rect = self._image_to_widget_rect()
        scale = self.zoom_level

        if self._sam_mask is not None and self._sam_mask.any():
            painter.setClipping(True)
            painter.setClipRect(rect)
            h, w = self._sam_mask.shape
            arr = np.zeros((h, w, 4), dtype=np.uint8)
            color = list(CLASS_COLORS[self.current_class_id % len(CLASS_COLORS)])
            arr[self._sam_mask, :3] = color
            arr[self._sam_mask, 3] = 120
            qimg = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
            painter.drawImage(rect, qimg)
            painter.setClipping(False)

        for i, pt in enumerate(self._sam_points):
            wp = self._image_to_widget(pt[0], pt[1])
            if self._sam_labels[i] == 1:
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.setBrush(QBrush(QColor(0, 255, 0, 180)))
            else:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.setBrush(QBrush(QColor(255, 0, 0, 180)))
            painter.drawEllipse(wp, 5, 5)

    def _accept_sam_mask(self):
        """Accept SAM mask: fill into current class mask."""
        if self._sam_mask is None or not self._sam_mask.any():
            self.status_message.emit("SAM: no mask to accept")
            return
        self._push_undo("sam")
        self.mask[self._sam_mask] = self.current_class_id
        self._overlay_dirty = True
        self._sam_points = []
        self._sam_labels = []
        self._sam_mask = None
        self._sam_score = 0.0
        self.update()
        self.mask_changed.emit()
        self.status_message.emit("SAM mask applied")

    def _cancel_sam(self):
        """Cancel SAM: clear all points and mask."""
        self._sam_points = []
        self._sam_labels = []
        self._sam_mask = None
        self._sam_score = 0.0
        self.update()
        self.status_message.emit("SAM: cancelled")

    # ======================== Mode & State ========================

    def set_mode(self, mode):
        if self._mode == self.MODE_SAM and mode != self.MODE_SAM:
            self._sam_points = []
            self._sam_labels = []
            self._sam_mask = None
        if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and self._polygon_points:
            self._polygon_points = []
        self._line_start = None
        self._line_end = None
        self._drawing = False
        self._panning = False
        self._mode = mode
        self._update_mode_cursor()
        self.update()

    def _update_mode_cursor(self):
        cursors = {
            self.MODE_BRUSH: Qt.CrossCursor,
            self.MODE_ERASER: Qt.CrossCursor,
            self.MODE_IGNORE: Qt.CrossCursor,
            self.MODE_POLYGON: Qt.CrossCursor,
            self.MODE_LINE: Qt.CrossCursor,
            self.MODE_PAN: Qt.OpenHandCursor,
            self.MODE_SAM: Qt.CrossCursor,
        }
        self.setCursor(QCursor(cursors.get(self._mode, Qt.CrossCursor)))

    def set_class(self, cid):
        self.current_class_id = cid
        if self._mode == self.MODE_ERASER:
            self._mode = self.MODE_PAN

    def set_eraser(self):
        self._mode = self.MODE_ERASER
        self._update_mode_cursor()
        self.update()

    def set_ignore(self):
        self._mode = self.MODE_IGNORE
        self._update_mode_cursor()
        self.update()


    def set_heatmap(self, qimage):
        """Set Grad-CAM heatmap overlay QImage (classification mode)."""
        self._heatmap_qimage = qimage.copy() if qimage else None
        self.update()


    def load_heatmap_from_file(self, filepath):
        """Load heatmap overlay from a saved image file."""
        import os
        if not os.path.exists(filepath):
            self._heatmap_qimage = None
            return
        from PyQt5.QtGui import QImage
        qimg = QImage(filepath)
        if qimg.isNull():
            self._heatmap_qimage = None
            return
        # Resize to match current image if needed
        if self.image is not None:
            iw, ih = self.image.size
            if qimg.width() != iw or qimg.height() != ih:
                qimg = qimg.scaled(iw, ih)
        self._heatmap_qimage = qimg.copy()
        self._show_heatmap = True
        self.update()

    def clear_heatmap(self):
        """Clear Grad-CAM heatmap overlay."""
        self._heatmap_qimage = None
        self._show_heatmap = False
        self.update()

    def set_heatmap(self, qimage):
        self._heatmap_qimage = qimage.copy() if qimage else None
        self.update()

    def clear_heatmap(self):
        self._heatmap_qimage = None
        self._show_heatmap = False
        self.update()

    def load_heatmap_from_file(self, filepath):
        import os
        if not os.path.exists(filepath):
            self._heatmap_qimage = None; return
        from PyQt5.QtGui import QImage
        qimg = QImage(filepath)
        if qimg.isNull():
            self._heatmap_qimage = None; return
        if self.image is not None:
            iw, ih = self.image.size
            if qimg.width() != iw or qimg.height() != ih:
                qimg = qimg.scaled(iw, ih)
        self._heatmap_qimage = qimg.copy()
        self._show_heatmap = True
        self.update()

    def _load_prediction_overlay(self, image_path):
        """Auto-load prediction overlay from outputs/ folder if exists."""
        self._prediction_qimage = None
        if not self.project_dir:
            return
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(os.path.dirname(self.project_dir),
                               os.path.basename(self.project_dir), "outputs")
        overlay_path = os.path.join(out_dir, base + "_overlay.jpg")
        mask_path = os.path.join(out_dir, base + "_pred.png")
        if os.path.exists(overlay_path):
            overlay_img = Image.open(overlay_path).convert("RGB")
            arr = np.array(overlay_img)
            h, w, ch = arr.shape
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
            self._prediction_qimage = qimg.copy()
            # Load mask for outline rendering
            if os.path.exists(mask_path):
                self._prediction_mask = np.array(Image.open(mask_path))
            # Load class names from project
            try:
                import json as _json
                proj_json = os.path.join(self.project_dir, "project.json")
                if os.path.exists(proj_json):
                    with open(proj_json, "r", encoding="utf-8") as f:
                        meta = _json.load(f)
                    raw = meta.get("classes", ["background"])
                    self._pred_classes = [c for c in raw if c == "background" or ("ignore" not in c.lower() and "gnore" not in c.lower())]
            except Exception:
                pass
            self._pred_stats_cache = {}
            self._pred_overlay_cache.clear()
            self.status_message.emit("Prediction loaded (Space to view)")

    def _push_undo(self, desc="draw"):
        """Save current mask state for undo."""
        if self.mask is not None:
            self._undo_stack.append((self.mask.copy(), desc))
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)

    def _auto_save_json(self):
        """Auto-save current mask as LabelMe JSON after each edit."""
        if not self.current_image_path or not self.project_dir:
            return
        if self.label_manager is None:
            return
        try:
            save_mask_to_json(self.mask, self.current_image_path,
                              self.project_dir, self.label_manager.classes)
        except Exception:
            pass  # Silent fail for auto-save

    def save_mask(self):
        if self.current_image_path and self.project_dir:
            self._auto_save_json()
            # Also save PNG for backward compat
            base = os.path.splitext(os.path.basename(self.current_image_path))[0]
            ann_dir = os.path.join(self.project_dir, "annotations")
            os.makedirs(ann_dir, exist_ok=True)
            mp = os.path.join(ann_dir, base + "_mask.png")
            Image.fromarray(self.mask.astype(np.uint8)).save(mp)

    def import_labelme_json(self, json_path):
        """Import annotations from an external LabelMe JSON file.
        Merges shapes from the JSON into the current mask."""
        if self.image is None or self.label_manager is None:
            return False
        shapes, info = load_labelme_json(json_path)
        if shapes is None or len(shapes) == 0:
            return False
        h, w = self.image.size[1], self.image.size[0]
        # Auto-register new classes from imported JSON (skip ignore/auto-generated)
        for s in shapes:
            lbl = s.get("label", "")
            if not lbl or lbl == "background":
                continue
            low = lbl.lower()
            if "ignore" in low or "gnore" in low:
                continue
            if re.match(r"^class[_\s]?\d+$", lbl, re.IGNORECASE):
                continue  # Skip auto-generated names like "class_1", "Class1"
            if lbl not in self.label_manager.classes:
                self.label_manager.add_class(lbl)

        imported_mask = shapes_to_mask(shapes, self.label_manager.classes, (h, w))
        self.mask = np.maximum(self.mask, imported_mask)  # Merge: higher class overrides
        self._overlay_dirty = True
        self._auto_save_json()
        self.update()
        self.mask_changed.emit()
        return True

    def export_labelme_json(self, output_path):
        """Export current annotations to a LabelMe JSON file."""
        if self.image is None or self.label_manager is None:
            return False
        h, w = self.image.size[1], self.image.size[0]
        save_mask_to_json(self.mask, self.current_image_path or output_path,
                          self.project_dir, self.label_manager.classes,
                          simplify_epsilon=1.0)
        return True

    def clear_mask(self):
        if self.image:
            h, w = self.image.size[1], self.image.size[0]
            self.mask = np.zeros((h, w), dtype=np.uint8)
            self._overlay_dirty = True
            self.update()
            self.mask_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image is not None:
            self._fit_to_window()

    def reset_view(self):
        if self.image is not None:
            self._fit_to_window()
            self.update()
