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
    det_selection_changed = pyqtSignal()
    status_message = pyqtSignal(str)
    mode_changed = pyqtSignal(str)
    width_changed = pyqtSignal(int)
    class_select_requested = pyqtSignal(int)
    save_requested = pyqtSignal()
    # 视图状态变化（缩放/平移/标注显示开关）。多视图预览靠它即时同步参考窗，
    # 不再依赖轮询：轮询最快也要等到下一个周期，看着就是"窗口 2/3 慢一拍"。
    view_changed = pyqtSignal()
    # 光标在**图像上**的像素坐标（x, y）。底部「图片信息栏」的"坐标/RGB"两项靠它。
    # 只在整数像素真的变了才发：鼠标每动一下都发信号，等于让信息栏跟着重排，
    # 比重绘画布还贵。光标不在图内（出图 / 离开控件）发 (-1, -1) 表示"无坐标"。
    cursor_moved = pyqtSignal(int, int)

    MODE_BRUSH = "brush"
    MODE_POLYGON = "polygon"
    MODE_LINE = "line"
    MODE_ERASER = "eraser"
    MODE_IGNORE = "ignore"
    MODE_PAN = "pan"
    MODE_SAM = "sam"
    MODE_RECT = "rect"
    MODE_CIRCLE = "circle"
    MODE_ELLIPSE = "ellipse"

    MIN_ZOOM = 0.05
    MAX_ZOOM = 30.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.image = None
        self.mask = None
        self._cursor_pt = None      # 上次上报的光标像素坐标（节流用，None = 不在图内）
        self._cached_qimage = None
        self._cached_gray = None  # cached grayscale numpy array
        # Mixed classification dual-image support
        self._mixed_cls_mode = False
        self._height_image = None      # PIL Image for height map
        self._cached_height_qimage = None
        self._height_path = None
        self._view_mode = 2           # 0/1=1x1, 2=1x2, 3=2x1
        self._single_page = 0        # 0=gray, 1=height (for 1x1 mode)
        self._height_vmin = None       # user-specified min for height colormap (None=auto)
        self._height_vmax = None       # user-specified max for height colormap
        self._height_colormap = cv2.COLORMAP_JET  # default colormap
        self._cached_height_pil = None      # PIL Image of rainbow for standard path
        self._gray_image = None       # cached original grayscale PIL Image
        self._det_overlay = None  # detection box overlay (attached by main_window)
        self._mask_editing = True  # False in classification/detection/OCR modes
        self._cached_overlay_qimage = None
        self._overlay_dirty = True
        self._show_annotation = True
        # Ctrl+Alt：单独控制忽略区(__ignore__, 255)的显示。
        # 忽略区常常占满整颗电芯（背面/边缘等不关注区域），跟缺陷类别混在一个开关里
        # 要么"想只看缺陷却被忽略区糊住"，要么"想看忽略区却把缺陷也带出来"。
        self._show_ignore = True
        self._show_prediction = False
        self._show_annotation_boxes = True   # Ctrl+Space: toggle annotation boxes
        self._show_pred_boxes = True         # Space: toggle prediction boxes
        # 状态信息默认叠加画在图像上（单图模式沿用原样）。多视图模式下由顶部信息栏
        # 显示同一份文本，这里关掉，免得那行字压住图像左下角。
        self._status_overlay = True
        # 画布底色：多视图模式下会被改成和参考窗同色，三个窗口看起来才是一套
        self._bg_color = QColor(40, 40, 40)
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
        self._shape_start = None   # 矩形/圆/椭圆 拖拽起点（图像坐标）
        self._shape_cur = None     # 拖拽当前点
        self._redo_stack = []      # 重做栈
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
        
        self.mask_changed.connect(self._auto_save_json)

    def set_project(self, project_dir):
        self.project_dir = project_dir
        self.label_manager = LabelManager(project_dir)

    def load_image(self, image_path):
        self.current_image_path = image_path
        pil_img = Image.open(image_path)
        ext = os.path.splitext(image_path)[1].lower()
        # Auto-detect height map: any .tif/.tiff
        if ext in (".tif", ".tiff"):
            self._height_image = pil_img  # keep native float32 for rainbow
            self._height_path = image_path
            # Normalize float32 height data to 8-bit grayscale for the "gray" view
            arr = np.array(pil_img, dtype=np.float64)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            vmin, vmax = np.percentile(arr, (0.5, 99.5))
            if vmax <= vmin:
                vmin, vmax = arr.min(), arr.max()
            if vmax <= vmin:
                vmax = vmin + 1
            arr_8u = np.clip((arr - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            gray_pil = Image.fromarray(arr_8u, mode="L").convert("RGB")
            self._gray_image = gray_pil
            self.image = gray_pil
            self._cache_height_qimage()
            # Use cached PIL rainbow image directly (avoids QImage->PIL conversion)
            if self._cached_height_pil is not None:
                self.image = self._cached_height_pil
            else:
                pass
            self._cache_qimage()
            self._view_mode = 0
        else:
            self.image = pil_img.convert("RGB")
            self._cache_qimage()
            self._height_image = None
            self._cached_height_qimage = None
            self._view_mode = 0
        self._height_vmin = None
        self._height_vmax = None
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
        # In side-by-side mixed mode, use combined image size for zoom
        _mixed = getattr(self, "_mixed_cls_mode", False)
        _hi = getattr(self, "_height_image", None)
        if _mixed and _hi is not None:
            hiw, hih = _hi.size
            iw = iw + hiw + 10  # gap
            ih = max(ih, hih)
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
        _vm = getattr(self, "_view_mode", 2)
        if self._mixed_cls_mode and self._height_image is not None:
            gap = 10
            hiw, hih = self._height_image.size
            if _vm == 3:  # 2x1 stacked
                total_w = max(iw, hiw)
                total_h = ih + hih + gap
            elif _vm >= 2:  # 1x2 (or any other dual mode)
                total_w = iw + hiw + gap
                total_h = max(ih, hih)
            else:  # 1x1
                total_w = iw
                total_h = ih
            dw = total_w * self.zoom_level
            dh = total_h * self.zoom_level
        else:
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
        # Side-by-side: only left half (grayscale) is annotatable
        side_by_side = (getattr(self, "_mixed_cls_mode", False)
                        and self._height_image is not None)
        if side_by_side:
            iw_total = self.image.size[0] + self._height_image.size[0] + 10
            ih_total = max(self.image.size[1], self._height_image.size[1])
            if ix < 0 or iy < 0 or ix >= iw_total or iy >= ih_total:
                return None
            # Only map clicks on grayscale half (left)
            if ix >= self.image.size[0]:
                return None  # height map half - no annotation
            return int(ix), int(iy)
        else:
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
        # Side-by-side: clamp to grayscale image bounds for annotation
        side_by_side = (getattr(self, "_mixed_cls_mode", False)
                        and self._height_image is not None)
        if side_by_side:
            # Clamp to grayscale image
            iw_gray = self.image.size[0]
            ix = max(0, min(ix, iw_gray - 1))
            ih_total = max(self.image.size[1], self._height_image.size[1])
            iy = max(0, min(iy, ih_total - 1))
        return int(ix), int(iy)
    def _image_to_widget(self, ix, iy):
        rect = self._image_to_widget_rect()
        return QPointF(rect.x() + ix * self.zoom_level, rect.y() + iy * self.zoom_level)

    def set_mixed_pair(self, gray_path, height_path=None):
        """Set paired grayscale + height images for mixed classification."""
        self._mixed_cls_mode = True
        self._height_path = height_path
        # Load grayscale
        gray_img = Image.open(gray_path).convert("RGB")
        self._gray_image = gray_img
        self.image = gray_img
        self._cache_qimage()
        self.current_image_path = gray_path
        # Load height map if available (keep native bit depth for rainbow render)
        import numpy as np
        if height_path and os.path.exists(height_path):
            self._height_image = Image.open(height_path)
        else:
            base = os.path.splitext(gray_path)[0]
            for ext in [".tif", ".tiff"]:
                candidate = base + ext
                if os.path.exists(candidate):
                    try:
                        self._height_image = Image.open(candidate)
                        self._height_path = candidate
                    except Exception:
                        pass
                    break
        self._cache_height_qimage()
        self._heatmap_qimage = None
        self._ocr_overlay = None
        w, h = self.image.size
        class_names = self.label_manager.classes if self.label_manager else ["background"]
        loaded_mask = load_mask_from_json(gray_path, self.project_dir, class_names)
        if loaded_mask is not None:
            self.mask = loaded_mask
        else:
            self.mask = np.zeros((h, w), dtype=np.uint8)
        self._overlay_dirty = True
        self._fit_to_window()
        self._load_prediction_overlay(gray_path)
        self._view_mode = 2  # force 1x2 for mixed classification
        self.update()

    def set_height_range(self, vmin=None, vmax=None, colormap=None):
        self._height_vmin = vmin
        self._height_vmax = vmax
        if colormap is not None:
            self._height_colormap = colormap
        self._cache_height_qimage()
        self.update()

    def _get_height_auto_range(self):
        if self._height_image is None:
            return 0, 1
        arr = np.array(self._height_image, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        valid = arr[(arr > -100) & (arr < 100) & (arr != 0)]
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, (0.5, 99.5))
        else:
            vmin, vmax = arr.min(), arr.max()
        if vmax <= vmin:
            vmin, vmax = arr.min(), arr.max()
        if vmax <= vmin:
            vmax = vmin + 1
        return float(vmin), float(vmax)

    def _cache_height_qimage(self):
        """Cache height map as rainbow-colored QImage."""
        if self._height_image is None:
            self._cached_height_qimage = None
            return
        arr = np.array(self._height_image, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        # Mask: exclude sentinel/background values (-999, 0) and extreme outliers (>|100|)
        # Real height data is typically -5 to 0 mm; sentinels are -999 or 0
        valid_mask = (arr > -100) & (arr < 100) & (arr != 0)
        # Use user-specified range if set, otherwise auto-detect from valid pixels only
        if self._height_vmin is not None and self._height_vmax is not None:
            vmin, vmax = self._height_vmin, self._height_vmax
        else:
            if valid_mask.any():
                vmin, vmax = np.percentile(arr[valid_mask], (0.5, 99.5))
            else:
                vmin, vmax = arr.min(), arr.max()
            if vmax <= vmin:
                vmin, vmax = arr.min(), arr.max()
            if vmax <= vmin:
                vmax = vmin + 1
        # Clip to range and normalize
        arr_clipped = np.clip(arr, vmin, vmax)
        arr_norm = ((arr_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        h, w = arr_norm.shape
        # Invert so high areas get warm colors
        arr_inv = 255 - arr_norm
        # Apply colormap
        color = cv2.applyColorMap(arr_inv, self._height_colormap)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        # Paint invalid (background) pixels dark gray instead of distorting colormap
        color[~valid_mask] = [40, 40, 40]
        color = np.ascontiguousarray(color)
        self._cached_height_pil = Image.fromarray(color, mode="RGB")
        h, w, _ = color.shape
        qimg = QImage(color.data, w, h, 3 * w, QImage.Format_RGB888)
        self._cached_height_qimage = qimg.copy() if not qimg.isNull() else None

    def _cache_qimage(self):
        if self.image is None:
            return
        arr = np.array(self.image)
        h, w, ch = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        self._cached_qimage = qimg.copy()
        self._cached_gray = np.mean(arr.astype(np.float32), axis=2)

    def _build_overlay(self):
        """Build annotation overlay (原图 + 掩膜 的合成图)。

        叠加层的画法统一来自 `core/mask_overlay.py`：**低透明度纯色底 + 高透明度斜纹
        + 轮廓线**。早期这里只画 25% 密度的细斜纹（12px 周期/3px 宽），缩到 fit
        （约 0.12 倍）时斜纹周期只剩 1.5px，屏幕上直接被采样掉；而参考窗用的又是
        `IGNORE_CLASS_COLOR`=(128,128,128) 灰，跟形状图1 的底色 130 几乎同色 ——
        结果就是"预览窗口2 不显示 ignore"。两处现在共用同一份实现，外观也一致了。
        """
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
        from core.mask_overlay import build_overlay_rgba, composite_over_image
        rgba = build_overlay_rgba(self.mask,
                                  show_classes=bool(getattr(self, "_show_annotation", True)),
                                  show_ignore=bool(getattr(self, "_show_ignore", True)))
        if rgba is None:
            self._cached_overlay_qimage = None
            return
        overlay = composite_over_image(img_arr, rgba)
        qimg = QImage(overlay.data, w, h, 3 * w, QImage.Format_RGB888)
        self._cached_overlay_qimage = qimg.copy()
        self._overlay_dirty = False

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.zoom_level < 1.0:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), getattr(self, "_bg_color", QColor(40, 40, 40)))
        if self.image is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return
        rect = self._image_to_widget_rect()
        # Fallback: if mixed mode but height QImage is missing, try to regenerate
        _mixed = getattr(self, "_mixed_cls_mode", False)
        if _mixed and self._cached_qimage is not None and self._height_image is not None and self._cached_height_qimage is None:
            try:
                self._cache_height_qimage()
            except Exception:
                pass
        side_by_side = (_mixed
                        and self._cached_height_qimage is not None and self._cached_qimage is not None)
        if side_by_side:
            # Side-by-side: gray (left) | height (right)
            iw, ih = self.image.size
            hiw, hih = self._height_image.size
            gap = 10
            scale = self.zoom_level
            # Gray image on left
            gray_rect = QRectF(rect.x(), rect.y() + (rect.height() - ih * scale) / 2.0,
                               iw * scale, ih * scale)
            painter.drawImage(gray_rect, self._cached_qimage)
            # Height image on right
            height_rect = QRectF(rect.x() + iw * scale + gap * scale,
                                 rect.y() + (rect.height() - hih * scale) / 2.0,
                                 hiw * scale, hih * scale)
            painter.drawImage(height_rect, self._cached_height_qimage)
            # Divider line
            div_x = rect.x() + iw * scale + gap * scale / 2.0
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            painter.drawLine(QPointF(div_x, rect.y()), QPointF(div_x, rect.y() + rect.height()))
            # Labels
            font = QFont("Consolas", max(8, int(10 * scale)))
            painter.setFont(font)
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(gray_rect, Qt.AlignBottom | Qt.AlignHCenter, "Grayscale")
            painter.setPen(QColor(200, 180, 100))
            painter.drawText(height_rect, Qt.AlignBottom | Qt.AlignHCenter, "Height Map")
        elif self._cached_qimage is not None:
            _vm2 = getattr(self, "_view_mode", 2)
            _hq = self._cached_height_qimage
            _mixed2 = getattr(self, "_mixed_cls_mode", False)
            if _mixed2 and _hq is not None and _vm2 == 3:
                # 2x1 stacked: gray (top) | height (bottom)
                iw2, ih2 = self.image.size
                hiw2, hih2 = self._height_image.size
                gap = 10
                scale = self.zoom_level
                cw = max(iw2, hiw2)
                gray_rect = QRectF(rect.x() + (rect.width() - cw * scale) / 2.0,
                                   rect.y(), cw * scale, ih2 * scale)
                painter.drawImage(gray_rect, self._cached_qimage)
                height_rect = QRectF(rect.x() + (rect.width() - cw * scale) / 2.0,
                                     rect.y() + ih2 * scale + gap * scale,
                                     cw * scale, hih2 * scale)
                painter.drawImage(height_rect, _hq)
                div_y = rect.y() + ih2 * scale + gap * scale / 2.0
                painter.setPen(QPen(QColor(100, 100, 100), 2))
                painter.drawLine(QPointF(rect.x(), div_y), QPointF(rect.x() + rect.width(), div_y))
                font = QFont("Consolas", max(8, int(10 * scale)))
                painter.setFont(font)
                painter.setPen(QColor(200, 200, 200))
                painter.drawText(gray_rect, Qt.AlignBottom | Qt.AlignHCenter, "Grayscale")
                painter.setPen(QColor(200, 180, 100))
                painter.drawText(height_rect, Qt.AlignBottom | Qt.AlignHCenter, "Height Map")
            elif _mixed2 and _hq is not None and _vm2 == 0:
                # 1x1 single page flip
                _sp = getattr(self, "_single_page", 0)
                if _sp == 1:
                    sp_img = self._cached_height_pil or self._height_image
                    sp_qimg = _hq
                    sp_label = "Height Map"
                else:
                    sp_img = self._gray_image or self.image
                    sp_qimg = self._cached_qimage
                    sp_label = "Grayscale"
                if sp_img:
                    siw, sih = sp_img.size
                    scale = self.zoom_level
                    sr = QRectF(rect.x() + (rect.width() - siw * scale) / 2.0,
                                rect.y() + (rect.height() - sih * scale) / 2.0,
                                siw * scale, sih * scale)
                    painter.drawImage(sr, sp_qimg)
                    font = QFont("Consolas", max(8, int(10 * scale)))
                    painter.setFont(font)
                    painter.setPen(QColor(200, 200, 200))
                    painter.drawText(sr, Qt.AlignBottom | Qt.AlignHCenter, sp_label)
                else:
                    painter.drawImage(rect, sp_qimg)
            elif self._view_mode == 1 and self._cached_height_qimage is not None:
                painter.drawImage(rect, self._cached_height_qimage)
            else:
                painter.drawImage(rect, self._cached_qimage)
        self._build_overlay()
        # Draw annotation overlay (hatch pattern)
        # 两个开关任一为开都要画（叠加图本身已经按开关过滤过类别/忽略区）
        if (getattr(self, "_show_annotation", True) or getattr(self, "_show_ignore", True)) \
                and self._cached_overlay_qimage is not None:
            painter.drawImage(rect, self._cached_overlay_qimage)
        # Draw prediction overlay on top
        if self._show_prediction:
            # 标注层（类别或忽略区）任一看得见时，预测层就不再铺底，免得盖住标注
            with_bg = not (getattr(self, "_show_annotation", True)
                           or getattr(self, "_show_ignore", True))
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
        if self._mode in (self.MODE_RECT, self.MODE_CIRCLE, self.MODE_ELLIPSE) and self._shape_start is not None:
            painter.setClipping(False)
            self._draw_shape_preview(painter)
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

        self._draw_class_labels(painter)
        if getattr(self, "_status_overlay", True):
            self._draw_status_overlay(painter)

    def status_info_text(self):
        """状态信息文本（`1175x3739 | Pan | Class: Zw | Annot:ON | Ignore:ON | Pred:OFF | Zoom: 100% | Brush: 15`）。

        只有**画布自己的叠加层**在用（`_draw_status_overlay`），多视图模式下由
        `set_status_overlay(False)` 关掉 —— 那种模式下这些状态改由**标注工具栏右侧**
        承载（三个可点开关 + 一个类别格，都是控件，不是这个函数拼的字符串）。

        2026-10-14：原先这里还有个 `full=False` 分支（去掉尺寸与缩放）专门给多视图顶栏用。
        顶栏整行拆掉之后它**一个调用者都没有了**，索性删掉 —— 留一条没人走的路，
        下一个人想接就会直接接上，于是又多出一处"同一个状态显示两遍"，
        而那正是这次重构要消掉的毛病。
        """
        zoom_pct = 100
        try:
            if self._fit_zoom:
                zoom_pct = int(self.zoom_level / self._fit_zoom * 100)
        except Exception:
            zoom_pct = 100
        mode_names = {self.MODE_BRUSH: 'Brush', self.MODE_ERASER: 'Eraser', self.MODE_IGNORE: 'Ignore',
                      self.MODE_POLYGON: "Polygon", self.MODE_LINE: "Line",
                      self.MODE_RECT: "Rect", self.MODE_CIRCLE: "Circle", self.MODE_ELLIPSE: "Ellipse",
                      self.MODE_PAN: "Pan", self.MODE_SAM: "SAM"}
        ann_status = "Annot:ON" if self._show_annotation else "Annot:OFF"
        ign_status = "Ignore:ON" if getattr(self, "_show_ignore", True) else "Ignore:OFF"
        if self._det_overlay is not None:
            pred_status = "Pred:ON" if self._show_pred_boxes else "Pred:OFF"
        else:
            pred_status = "Pred:ON" if self._show_prediction else "Pred:OFF"
        cls_name = "?"
        if self.label_manager is not None and 0 <= self.current_class_id < len(self.label_manager.classes):
            cls_name = self.label_manager.classes[self.current_class_id]
        parts = ["%dx%d" % (self.image.size[0], self.image.size[1]) if self.image else "N/A",
                 mode_names.get(self._mode, self._mode),
                 "Class: %s" % cls_name, ann_status, ign_status, pred_status,
                 "Zoom: %d%%" % zoom_pct]
        # 笔宽：这里以前写 "Width"，而 mask_editor_part1 里同一件事写 "Brush" ——
        # 同一个产品两个名字，用户看到会以为是两个参数。统一成 Brush。
        parts.append("Brush: %d" % self.brush_size)
        return " | ".join(parts)

    def set_status_overlay(self, enabled):
        """是否把状态信息叠加画在图像左下角（多视图模式下改由顶部信息栏显示）。"""
        enabled = bool(enabled)
        if enabled != getattr(self, "_status_overlay", True):
            self._status_overlay = enabled
            self.update()

    def set_bg_color(self, color):
        """设置图像区底色（多视图模式下与参考窗同色）。"""
        try:
            c = QColor(color)
        except Exception:
            return
        if not c.isValid() or c == getattr(self, "_bg_color", None):
            return
        self._bg_color = c
        self.update()

    def _draw_status_overlay(self, painter):
        info = self.status_info_text()
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
        """在每个标注区域左上角画类别名（忽略区画 IGNORE）。

        实现已抽到 `core/label_overlay.paint_shape_labels` —— **多视图参考窗用的是同一份**。
        两边各写一套必然漂移（同 `core/mask_overlay.py` 的理由），症状是"同一个标注在
        窗口 1 和窗口 2 字号/配色不一样"。

        两个开关的归属与掩膜层保持一致：
          Ctrl+Space(_show_annotation) -> 缺陷类别标签
          Ctrl+Alt  (_show_ignore)     -> 忽略区 IGNORE 标签
        历史实现是"一刀切"（`_show_annotation` 一关全不画），会出现"忽略区掩膜都隐藏了、
        IGNORE 字样还挂在图上"。
        """
        if not (self._show_annotation or getattr(self, "_show_ignore", True)):
            return
        if not self.current_image_path or not self.label_manager:
            return
        try:
            json_path = get_json_path(self.current_image_path)
            shapes, _ = load_labelme_json(json_path)
            if not shapes:
                return
            from core.label_overlay import paint_shape_labels
            paint_shape_labels(painter, shapes, self._image_to_widget, self.zoom_level,
                               show_classes=bool(self._show_annotation),
                               show_ignore=bool(getattr(self, "_show_ignore", True)))
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
                self.det_selection_changed.emit()
                return

            return
        if not self._mask_edit_enabled() and self._mode != self.MODE_PAN:
            self.status_message.emit("Mask editing is disabled in this task mode")
            return
        pt = self._window_to_image(event.pos())
        if pt is None and self._mode not in (self.MODE_POLYGON, self.MODE_IGNORE, self.MODE_LINE, self.MODE_SAM,
                                             self.MODE_RECT, self.MODE_CIRCLE, self.MODE_ELLIPSE):
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
        elif self._mode in (self.MODE_RECT, self.MODE_CIRCLE, self.MODE_ELLIPSE):
            self._push_undo(self._mode)
            self._shape_start = self._window_to_image_unclamped(event.pos())
            self._shape_cur = self._shape_start
            self._drawing = True
            self.update()
        elif self._mode == self.MODE_SAM:
            self._handle_sam_click(event)

    def _emit_cursor_pos(self, pos):
        """把光标像素坐标推给底部信息栏（坐标没变就不发）。

        放在 mouseMoveEvent 的**最前面**：下面几条分支（平移 / 检测层 / 未使能）都会
        early-return，放在后面就会出现"平移时坐标不刷新"。
        """
        try:
            pt = self._window_to_image(pos)
        except Exception:
            pt = None
        if pt == self._cursor_pt:
            return
        self._cursor_pt = pt
        if pt is None:
            self.cursor_moved.emit(-1, -1)
        else:
            self.cursor_moved.emit(int(pt[0]), int(pt[1]))

    def leaveEvent(self, event):
        """光标离开控件：清掉坐标读数。留着"最后一次"的值会像是当前光标位置。"""
        if self._cursor_pt is not None:
            self._cursor_pt = None
            self.cursor_moved.emit(-1, -1)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        self._emit_cursor_pos(event.pos())
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
        if not self._mask_edit_enabled() and self._mode != self.MODE_PAN:
            return
        if self._drawing and self._mode == self.MODE_BRUSH:
            self._draw_brush_at(event.pos())
        elif self._drawing and self._mode == self.MODE_ERASER:
            self._draw_eraser_at(event.pos())
        if self._drawing and self._mode in (self.MODE_RECT, self.MODE_CIRCLE, self.MODE_ELLIPSE):
            self._shape_cur = self._window_to_image_unclamped(event.pos())
            self.update()
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
        if self._mode in (self.MODE_RECT, self.MODE_CIRCLE, self.MODE_ELLIPSE) and self._shape_start is not None:
            self._shape_cur = self._window_to_image_unclamped(event.pos())
            self._fill_shape()
            self._shape_start = None
            self._shape_cur = None
            self._drawing = False
            self._overlay_dirty = True
            self.update()
            self.mask_changed.emit()

    def mouseDoubleClickEvent(self, event):
        if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and len(self._polygon_points) >= 3:
            self._close_polygon()

    def _zoom_step(self, factor):
        """以视图中心为基准缩放。"""
        if self.image is None:
            return
        c = self.rect().center()
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom_level * factor))
        actual = new_zoom / self.zoom_level
        self.zoom_level = new_zoom
        self.pan_offset_x = c.x() - actual * (c.x() - self.pan_offset_x)
        self.pan_offset_y = c.y() - actual * (c.y() - self.pan_offset_y)
        self.update()

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
        self.view_changed.emit()      # 滚轮缩放：立即推给多视图参考窗（拖动平移仍靠 40ms 轮询，避免每个鼠标移动都重画）

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
        key = event.key()
        mods = event.modifiers()
        # 工具快捷键（不与 Ctrl 组合冲突）
        _tool_keys = {
            Qt.Key_B: self.MODE_BRUSH,
            Qt.Key_L: self.MODE_LINE,
            Qt.Key_R: self.MODE_RECT,
            Qt.Key_C: self.MODE_CIRCLE,
            Qt.Key_O: self.MODE_ELLIPSE,
            Qt.Key_P: self.MODE_POLYGON,
            Qt.Key_E: self.MODE_ERASER,
            Qt.Key_I: self.MODE_IGNORE,
            Qt.Key_A: self.MODE_SAM,
            Qt.Key_H: self.MODE_PAN,
        }
        if not self._mask_edit_enabled() and key in _tool_keys:
            return
        if key in _tool_keys and not (mods & Qt.ControlModifier):
            self.set_mode(_tool_keys[key])
            return
        if Qt.Key_1 <= key <= Qt.Key_9 and not (mods & Qt.ControlModifier):
            self.class_select_requested.emit(key - Qt.Key_0)
            return
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom_step(1.25)
            return
        if key == Qt.Key_Minus:
            self._zoom_step(1 / 1.25)
            return
        if key in (Qt.Key_0, Qt.Key_F):
            self._fit_to_window()
            self.update()
            return
        if key == Qt.Key_BracketRight:
            self.brush_size = min(100, self.brush_size + 2)
            self.width_changed.emit(self.brush_size)
            self.status_message.emit("Brush size: %d" % self.brush_size)
            self.update()
            return
        if key == Qt.Key_BracketLeft:
            self.brush_size = max(1, self.brush_size - 2)
            self.width_changed.emit(self.brush_size)
            self.status_message.emit("Brush size: %d" % self.brush_size)
            self.update()
            return

        # Left/Right arrow to switch between grayscale and rainbow height map
        if self._height_image is not None and self._view_mode != 2:
            if event.key() == Qt.Key_Left or event.key() == Qt.Key_Right:
                if self._height_image is not None:
                    if self._view_mode == 0:
                        # Switch to grayscale
                        self._view_mode = 1
                        self.image = self._gray_image
                        self._cache_qimage()
                        self.status_message.emit("View: Grayscale")
                    else:
                        # Switch to rainbow height map
                        self._view_mode = 0
                        self._cache_height_qimage()
                        if self._cached_height_pil is not None:
                            self.image = self._cached_height_pil
                        self._cache_qimage()
                        self.status_message.emit("View: Height Map (rainbow)")
                    self._fit_to_window()
                    self.update()
                return

        # Detection mode: Delete = remove selected box
        if self._det_overlay is not None and (event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace):
            self._det_overlay.delete_active_box()
            self.update()
            self.det_boxes_changed.emit()
            return

        # Detection mode: Ctrl+Z / Ctrl+Y / Esc operate on boxes (not masks)
        if self._det_overlay is not None:
            if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
                if self._det_overlay.box_manager.undo():
                    self.update()
                    self.det_boxes_changed.emit()
                return
            if event.key() == Qt.Key_Y and event.modifiers() == Qt.ControlModifier:
                if self._det_overlay.box_manager.redo():
                    self.update()
                    self.det_boxes_changed.emit()
                return
            if event.key() == Qt.Key_Escape:
                self._det_overlay.cancel_drawing()
                self.update()
                return

        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            # Ctrl+Z: undo last mask edit
            self.undo()
        elif event.key() == Qt.Key_Y and event.modifiers() == Qt.ControlModifier:
            # Ctrl+Y: redo
            self.redo()
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            # Ctrl+S: save annotation
            self.save_requested.emit()
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
            self.toggle_annotation_visibility()
            event.accept()
        elif self._is_ctrl_alt_toggle(event):
            # Ctrl+Alt（只按修饰键，不配字母）：单独开关忽略区
            self.toggle_ignore_visibility()
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

    def _is_ctrl_alt_toggle(self, event):
        """是否"Ctrl+Alt"这个只看修饰键的开关组合。

        ⚠️ 只按修饰键时 Qt 仍会派发 KeyPress，`key()` 就是修饰键本身的码：
        先按住 Ctrl 再按 Alt，事件是 key=Key_Alt、modifiers=ControlModifier。
        再按 Alt 也可以（key=Key_Control、modifiers=AltModifier），两种顺序都认。
        必须排除 AltGr（Windows 上有些键盘布局用它输入字符，此时 modifiers 会带
        GroupSwitchModifier），否则会误触发。
        """
        if event.isAutoRepeat():
            return False
        mods = event.modifiers()
        if mods & Qt.GroupSwitchModifier:
            return False
        # ⚠️ 不使用 == 精确比较：修饰键自身的按下事件里，modifiers() 到底含不含它自己
        #    在 Qt 各版本/平台上是"按下前"还是"按下后"并不统一。改成"看另一个键在不在"，
        #    两种顺序（先 Ctrl 后 Alt / 先 Alt 后 Ctrl）与两种语义都能命中。
        ctrl = bool(mods & Qt.ControlModifier)
        alt = bool(mods & Qt.AltModifier)
        if event.key() == Qt.Key_Alt and ctrl:
            return True
        if event.key() == Qt.Key_Control and alt:
            return True
        # 备用绑定：万一手感/键盘布局导致"只按修饰键"不派发按键事件，
        # Ctrl+Alt+I 同样能切忽略区（I = Ignore）
        if event.key() == Qt.Key_I and ctrl and alt:
            return True
        return False

    def toggle_ignore_visibility(self):
        """Ctrl+Alt 的忽略区显示开关（只影响 __ignore__ / 掩膜值 255 的区域）。"""
        self._show_ignore = not getattr(self, "_show_ignore", True)
        self._overlay_dirty = True
        self.status_message.emit("Ignore: %s" % ("ON" if self._show_ignore else "OFF"))
        self.update()
        self.view_changed.emit()

    def toggle_annotation_visibility(self):
        """Ctrl+Space 的标注显示开关（检测模式切框，分割模式切掩膜）。

        单独抽成一个方法，是为了让多视图预览里的参考窗也能触发同一个动作：
        焦点在参考窗上时按键根本传不到画布（QFrame 默认忽略后往父级冒泡），
        以前只能等轮询才发现状态变了 —— 就是"窗口 2/3 比窗口 1 慢一点"的原因。
        """
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
        self.view_changed.emit()

    def toggle_prediction_visibility(self):
        """预测结果显示开关（与上面那个标注开关对称）。

        本来是 Space 的活儿（`keyPressEvent` 里就地改字段）。抽成方法是因为
        2026-10-14 顶栏并进底栏之后，工具栏上多了一个「预测」按钮要调它 ——
        同一件事有两处入口时，**必须都走同一个方法**，否则"按钮亮了但状态没变"
        或者反过来（按钮切了框、快捷键切了掩膜）迟早出现。
        """
        if self._det_overlay is not None:
            self._show_pred_boxes = not self._show_pred_boxes
            self.status_message.emit("Prediction boxes: %s"
                                     % ("ON" if self._show_pred_boxes else "OFF"))
        elif getattr(self, "_cls_mode", False):
            self._show_heatmap = not getattr(self, "_show_heatmap", True)
            self.status_message.emit("Heatmap: %s" % ("ON" if self._show_heatmap else "OFF"))
        else:
            self._show_prediction = not self._show_prediction
            self.status_message.emit("Prediction: %s"
                                     % ("ON" if self._show_prediction else "OFF"))
        self.update()
        self.view_changed.emit()

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

    def _fill_shape(self):
        """把矩形/圆/椭圆选区填充为当前类别（图像坐标）。"""
        if self._shape_start is None or self._shape_cur is None or self.mask is None:
            return
        x0 = min(self._shape_start[0], self._shape_cur[0])
        y0 = min(self._shape_start[1], self._shape_cur[1])
        x1 = max(self._shape_start[0], self._shape_cur[0])
        y1 = max(self._shape_start[1], self._shape_cur[1])
        h, w = self.mask.shape
        x0 = max(0, int(round(x0))); y0 = max(0, int(round(y0)))
        x1 = min(w - 1, int(round(x1))); y1 = min(h - 1, int(round(y1)))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        region = np.ones_like(xx, dtype=bool)
        if self._mode in (self.MODE_CIRCLE, self.MODE_ELLIPSE):
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            if self._mode == self.MODE_CIRCLE:
                r = max(min(x1 - x0, y1 - y0) / 2.0, 0.5)
                rx = ry = r
            else:
                rx = max((x1 - x0) / 2.0, 0.5)
                ry = max((y1 - y0) / 2.0, 0.5)
            region = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        self.mask[y0:y1 + 1, x0:x1 + 1][region] = self.current_class_id
        self.status_message.emit("%s filled (class %d)" % (self._mode, self.current_class_id))

    def _draw_shape_preview(self, painter):
        """拖拽时的虚线预览。"""
        if self._shape_start is None or self._shape_cur is None:
            return
        sp = self._image_to_widget(self._shape_start[0], self._shape_start[1])
        ep = self._image_to_widget(self._shape_cur[0], self._shape_cur[1])
        r = QRectF(sp, ep).normalized()
        color = CLASS_COLORS[self.current_class_id % len(CLASS_COLORS)]
        painter.setPen(QPen(QColor(*color), 2, Qt.DashLine))
        painter.setBrush(QBrush(QColor(color[0], color[1], color[2], 40)))
        if self._mode == self.MODE_RECT:
            painter.drawRect(r)
        elif self._mode in (self.MODE_CIRCLE, self.MODE_ELLIPSE):
            if self._mode == self.MODE_CIRCLE:
                d = min(r.width(), r.height())
                r = QRectF(r.center().x() - d / 2.0, r.center().y() - d / 2.0, d, d)
            painter.drawEllipse(r)

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

    def _mask_edit_enabled(self):
        """Whether mask drawing is allowed (segmentation annotation only)."""
        return getattr(self, "_mask_editing", True)

    def set_mode(self, mode):
        if self._mode == self.MODE_SAM and mode != self.MODE_SAM:
            self._sam_points = []
            self._sam_labels = []
            self._sam_mask = None
        if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and self._polygon_points:
            self._polygon_points = []
        self._line_start = None
        self._line_end = None
        self._shape_start = None
        self._shape_cur = None
        self._drawing = False
        self._panning = False
        self._mode = mode
        self._update_mode_cursor()
        self.update()
        self.mode_changed.emit(mode)

    def _update_mode_cursor(self):
        cursors = {
            self.MODE_BRUSH: Qt.CrossCursor,
            self.MODE_ERASER: Qt.CrossCursor,
            self.MODE_IGNORE: Qt.CrossCursor,
            self.MODE_POLYGON: Qt.CrossCursor,
            self.MODE_LINE: Qt.CrossCursor,
            self.MODE_RECT: Qt.CrossCursor,
            self.MODE_CIRCLE: Qt.CrossCursor,
            self.MODE_ELLIPSE: Qt.CrossCursor,
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

    def cancel_current(self):
        """取消当前操作（等价于 Esc）：多边形/SAM/形状/直线。"""
        if self._mode in (self.MODE_POLYGON, self.MODE_IGNORE) and self._polygon_points:
            self._polygon_points = []
            self.status_message.emit("Polygon cancelled")
        if self._mode == self.MODE_SAM:
            self._cancel_sam()
        self._line_start = None
        self._line_end = None
        self._shape_start = None
        self._shape_cur = None
        self._drawing = False
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
            self._redo_stack.clear()

    def undo(self):
        """撤销上一次标注操作。"""
        if not self._undo_stack:
            self.status_message.emit("Nothing to undo")
            return
        if self.mask is not None:
            self._redo_stack.append((self.mask.copy(), self._undo_stack[-1][1]))
        self.mask, desc = self._undo_stack.pop()
        self._overlay_dirty = True
        self.update()
        self.mask_changed.emit()
        self.status_message.emit("Undo: %s (%d left)" % (desc, len(self._undo_stack)))

    def redo(self):
        """重做被撤销的标注操作。"""
        if not self._redo_stack:
            self.status_message.emit("Nothing to redo")
            return
        if self.mask is not None:
            self._undo_stack.append((self.mask.copy(), self._redo_stack[-1][1]))
        self.mask, desc = self._redo_stack.pop()
        self._overlay_dirty = True
        self.update()
        self.mask_changed.emit()
        self.status_message.emit("Redo: %s (%d left)" % (desc, len(self._redo_stack)))

    def _auto_save_json(self):
        """Auto-save current mask as LabelMe JSON after each edit."""
        if not self._mask_edit_enabled():
            return
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
        if not self._mask_edit_enabled():
            return
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
