"""Detection box overlay - draws bounding boxes on the AnnotationCanvas.

Low coupling: attaches to existing canvas via event hooks,
does not modify mask_editor.py internals.
"""
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPolygonF


# Bright, distinct colors for detection boxes (different from segmentation palette)
DET_COLORS = [
    QColor(255, 50, 50),     # Red
    QColor(50, 255, 50),     # Green
    QColor(50, 50, 255),     # Blue
    QColor(255, 255, 50),    # Yellow
    QColor(255, 50, 255),    # Magenta
    QColor(50, 255, 255),    # Cyan
    QColor(255, 150, 50),    # Orange
    QColor(150, 50, 255),    # Purple
    QColor(50, 255, 150),    # Mint
    QColor(255, 100, 200),   # Pink
    QColor(100, 200, 255),   # Sky
    QColor(200, 255, 100),   # Lime
    QColor(255, 200, 100),   # Gold
    QColor(150, 150, 255),   # Lavender
    QColor(100, 255, 200),   # Teal
    QColor(255, 150, 150),   # Salmon
]


def get_color_for_class(class_name, class_list):
    """Get consistent color for a class name."""
    if class_name in class_list:
        idx = class_list.index(class_name)
    else:
        idx = hash(class_name) % len(DET_COLORS)
    return DET_COLORS[idx % len(DET_COLORS)]


class DetectionOverlay:
    """Draws bounding boxes and handles interaction on the existing AnnotationCanvas.
    
    This class is designed to be used alongside the canvas. It hooks into the
    canvas's paintEvent and mouse events to draw and edit detection boxes.
    
    Usage:
        overlay = DetectionOverlay(canvas, box_manager)
        canvas._det_overlay = overlay  # canvas checks this in paintEvent
    """

    # Interaction states
    STATE_IDLE = 0
    STATE_DRAWING = 1       # Drawing a new box
    STATE_DRAGGING = 2      # Moving an existing box
    STATE_RESIZING = 3      # Resizing via corner handle
    STATE_ROTATING = 4      # Rotating (OBB only)

    def __init__(self, canvas, box_manager):
        self.canvas = canvas
        self.box_manager = box_manager
        self._state = self.STATE_IDLE
        self._draw_start = None       # QPointF in image coords
        self._draw_current = None
        self._active_box_id = None    # Which box is being edited
        self._resize_handle = None    # Which corner handle
        self._drag_offset = None
        self._rotation_start_angle = 0.0
        self._hovered_box_id = None

    # ---- Drawing ----

    def paint(self, painter, image_to_widget_scale, offset_x=0.0, offset_y=0.0,
              show_annotations=True, show_predictions=True):
        """Draw all bounding boxes. Call from canvas paintEvent.
        
        Args:
            show_annotations: if True, draw annotation boxes (dashed)
            show_predictions: if True, draw prediction boxes (solid)
        """
        scale = image_to_widget_scale
        bbox_type = self.box_manager.bbox_type
        classes = self.box_manager.categories

        # 1. Annotation boxes (dashed)
        if show_annotations and self.box_manager.boxes:
            for box in self.box_manager.boxes:
                self._draw_single_box(painter, box, scale, bbox_type, classes,
                                      offset_x, offset_y, is_prediction=False)

        # 2. Prediction boxes (solid, with confidence scores)
        if show_predictions and self.box_manager.pred_boxes:
            for box in self.box_manager.pred_boxes:
                self._draw_single_box(painter, box, scale, bbox_type, classes,
                                      offset_x, offset_y, is_prediction=True)

    def _draw_single_box(self, painter, box, scale, bbox_type, classes,
                         ox, oy, is_prediction=False):
        """Draw a single bounding box."""
        bbox = box["bbox"]
        cat = box.get("category", "object")
        score = box.get("score")
        box_id = box.get("id")
        color = get_color_for_class(cat, classes)

        is_active = (box_id == self._active_box_id)
        is_hovered = (box_id == self._hovered_box_id)

        pen_width = 3 if is_active else (2 if is_hovered else 1.5)
        pen = QPen(color, pen_width)
        if is_prediction:
            pen.setStyle(Qt.SolidLine)  # Prediction = solid
        else:
            pen.setStyle(Qt.DashLine)   # Annotation = dashed
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if bbox_type == "obb" and len(bbox) >= 5:
            self._draw_obb(painter, bbox, scale, ox, oy)
        else:
            self._draw_hbb(painter, bbox, scale, ox, oy)

        # Label with score
        label = cat
        if score is not None:
            label += f" {score:.2f}"
        self._draw_label(painter, bbox, scale, label, color, is_prediction, ox, oy)

        # Resize handles only for active annotation box
        if is_active and not is_prediction:
            self._draw_handles(painter, bbox, scale, color, bbox_type, ox, oy)

        # Rotation handle only for active OBB annotation box
        if is_active and not is_prediction and bbox_type == "obb":
            self._draw_rotation_handle(painter, bbox, scale, color, ox, oy)

    def _draw_hbb(self, painter, bbox, scale, ox=0.0, oy=0.0):
        x, y, w, h = [v * scale for v in bbox[:4]]
        painter.drawRect(QRectF(x + ox, y + oy, w, h))

    def _draw_obb(self, painter, bbox, scale, ox=0.0, oy=0.0):
        corners = self._get_obb_corners_scaled(bbox, scale, ox, oy)
        polygon = QPolygonF([QPointF(cx, cy) for cx, cy in corners])
        painter.drawPolygon(polygon)

    def _draw_label(self, painter, bbox, scale, text, color, is_prediction, ox=0.0, oy=0.0):
        """Draw label at top-left of box."""
        if len(bbox) >= 5:
            # OBB: use top-left corner
            corners = self._get_obb_corners_scaled(bbox, scale, ox, oy)
            tx, ty = corners[0]
        else:
            tx = bbox[0] * scale + ox
            ty = bbox[1] * scale + oy - 2

        font = QFont("Segoe UI", max(8, int(10 * min(scale, 1.5))))
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(text)
        text_height = fm.height()
        ty = max(text_height, ty)  # Don't go above image

        # Background rectangle
        bg_rect = QRectF(tx, ty - text_height, text_width + 4, text_height + 2)
        if is_prediction:
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 180)))
        else:
            painter.setBrush(QBrush(QColor(40, 40, 40, 200)))
        painter.setPen(Qt.NoPen)
        painter.drawRect(bg_rect)

        # Text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QPointF(tx + 2, ty - 2), text)

    def _draw_handles(self, painter, bbox, scale, color, bbox_type, ox=0.0, oy=0.0):
        """Draw 4 corner resize handles."""
        handle_size = 6
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(Qt.white, 1))

        if bbox_type == "obb" and len(bbox) >= 5:
            corners = self._get_obb_corners_scaled(bbox, scale, ox, oy)
        else:
            x, y, w, h = [v * scale for v in bbox[:4]]
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        for cx, cy in corners:
            painter.drawRect(QRectF(cx - handle_size, cy - handle_size,
                                    handle_size * 2, handle_size * 2))

    def _draw_rotation_handle(self, painter, bbox, scale, color, ox=0.0, oy=0.0):
        """Draw rotation handle above the OBB."""
        corners = self._get_obb_corners_scaled(bbox, scale)
        top_center_x = (corners[0][0] + corners[1][0]) / 2
        top_center_y = (corners[0][1] + corners[1][1]) / 2
        handle_y = top_center_y - 20 * scale

        painter.setPen(QPen(color, 1))
        painter.drawLine(QPointF(top_center_x, top_center_y),
                         QPointF(top_center_x, handle_y))
        painter.setBrush(QBrush(Qt.white))
        painter.drawEllipse(QPointF(top_center_x, handle_y), 5, 5)

    # ---- Mouse interaction ----

    def mouse_press(self, pos_image, modifiers):
        if isinstance(pos_image, tuple):
            pos_image = QPointF(pos_image[0], pos_image[1])
        """Handle mouse press at image coordinates. Returns True if handled."""
        # Check if clicking on a handle of active box
        if self._active_box_id is not None:
            handle = self.box_manager.find_handle_at(
                pos_image.x(), pos_image.y(), self._active_box_id)
            if handle:
                self._state = self.STATE_RESIZING
                self._resize_handle = handle
                self._draw_start = pos_image
                return True

        # Check if clicking on a box
        box = self.box_manager.find_box_at(pos_image.x(), pos_image.y())
        if box:
            self._active_box_id = box["id"]
            self._state = self.STATE_DRAGGING
            self._draw_start = pos_image
            # Store offset from box origin
            bbox = box["bbox"]
            if len(bbox) >= 5:
                self._drag_offset = (pos_image.x() - bbox[0], pos_image.y() - bbox[1])
            else:
                self._drag_offset = (pos_image.x() - bbox[0], pos_image.y() - bbox[1])
            return True

        # Start drawing new box
        self._state = self.STATE_DRAWING
        self._draw_start = pos_image
        self._draw_current = pos_image
        self._active_box_id = None
        return True

    def mouse_move(self, pos_image):
        if isinstance(pos_image, tuple):
            pos_image = QPointF(pos_image[0], pos_image[1])
        """Handle mouse move. Updates draw/transform preview."""
        if self._state == self.STATE_DRAWING:
            self._draw_current = pos_image
            return True

        elif self._state == self.STATE_DRAGGING and self._active_box_id is not None:
            dx = pos_image.x() - self._draw_start.x()
            dy = pos_image.y() - self._draw_start.y()
            box = self.box_manager._get_box(self._active_box_id)
            if box:
                bbox = box["bbox"]
                new_bbox = list(bbox)
                if len(bbox) >= 5:
                    new_bbox[0] = bbox[0] + dx
                    new_bbox[1] = bbox[1] + dy
                else:
                    new_bbox[0] = bbox[0] + dx
                    new_bbox[1] = bbox[1] + dy
                self.box_manager.update_box(self._active_box_id, bbox=new_bbox)
                self._draw_start = pos_image
            return True

        elif self._state == self.STATE_RESIZING and self._active_box_id is not None:
            box = self.box_manager._get_box(self._active_box_id)
            if box:
                bbox = list(box["bbox"])
                px, py = pos_image.x(), pos_image.y()
                if len(bbox) >= 5:
                    self._resize_obb(bbox, px, py)
                else:
                    self._resize_hbb(bbox, px, py)
                self.box_manager.update_box(self._active_box_id, bbox=bbox)
            return True

        else:
            # Hover detection
            box = self.box_manager.find_box_at(pos_image.x(), pos_image.y())
            self._hovered_box_id = box["id"] if box else None

        return False

    def mouse_release(self, pos_image):
        if isinstance(pos_image, tuple):
            pos_image = QPointF(pos_image[0], pos_image[1])
        """Handle mouse release. Finalize draw/transform."""
        if self._state == self.STATE_DRAWING:
            # Create box from start -> current
            if self._draw_start and self._draw_current:
                x1, y1 = self._draw_start.x(), self._draw_start.y()
                x2, y2 = self._draw_current.x(), self._draw_current.y()
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:  # Minimum size
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    if self.box_manager.bbox_type == "obb":
                        bbox = [x + w / 2, y + h / 2, w, h, 0.0]
                    else:
                        bbox = [x, y, w, h]
                    # Use currently selected category from canvas
                    cat = self._get_current_category()
                    self.box_manager.add_box(bbox, category=cat)

        self._state = self.STATE_IDLE
        self._draw_start = None
        self._draw_current = None
        self._drag_offset = None

    def cancel_drawing(self):
        """Cancel current draw/transform operation."""
        if self._state == self.STATE_DRAWING:
            self.box_manager.undo()
        self._state = self.STATE_IDLE
        self._draw_start = None
        self._draw_current = None

    def delete_active_box(self):
        """Delete the currently active/selected box."""
        if self._active_box_id is not None:
            self.box_manager.delete_box(self._active_box_id)
            self._active_box_id = None
            self._state = self.STATE_IDLE

    def clear_selection(self):
        self._active_box_id = None
        self._state = self.STATE_IDLE

    def get_drawing_preview_rect(self):
        """Get current drawing preview rect in image coords. 
        Returns (x, y, w, h) or None."""
        if self._state == self.STATE_DRAWING and self._draw_start and self._draw_current:
            x1, y1 = self._draw_start.x(), self._draw_start.y()
            x2, y2 = self._draw_current.x(), self._draw_current.y()
            return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        return None

    def switch_bbox_type(self, new_type):
        """Switch between HBB and OBB."""
        self.box_manager.bbox_type = new_type
        self.clear_selection()

    # ---- Helpers ----

    def _get_current_category(self):
        """Get currently selected category from canvas or default."""
        # Try to get from canvas's label manager
        lm = getattr(self.canvas, "label_manager", None)
        if lm and hasattr(lm, "current_class"):
            idx = lm.current_class
            if idx < len(self.box_manager.categories):
                return self.box_manager.categories[idx]
        return self.box_manager.categories[1] if len(self.box_manager.categories) > 1 else "object"

    def _get_obb_corners_scaled(self, bbox, scale, ox=0.0, oy=0.0):
        cx, cy, w, h, angle = bbox[:5]
        import math
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        hw, hh = w / 2, h / 2
        corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        corners = []
        for lx, ly in corners_local:
            gx = (cx + lx * cos_a - ly * sin_a) * scale + ox
            gy = (cy + lx * sin_a + ly * cos_a) * scale + oy
            corners.append((gx, gy))
        return corners

    def _resize_hbb(self, bbox, px, py):
        """Resize HBB based on which handle is being dragged."""
        x, y, w, h = bbox[:4]
        if self._resize_handle == "tl":
            new_w = (x + w) - px
            new_h = (y + h) - py
            if new_w > 5: bbox[0] = px; bbox[2] = new_w
            if new_h > 5: bbox[1] = py; bbox[3] = new_h
        elif self._resize_handle == "tr":
            new_w = px - x
            new_h = (y + h) - py
            if new_w > 5: bbox[2] = new_w
            if new_h > 5: bbox[1] = py; bbox[3] = new_h
        elif self._resize_handle == "bl":
            new_w = (x + w) - px
            new_h = py - y
            if new_w > 5: bbox[0] = px; bbox[2] = new_w
            if new_h > 5: bbox[3] = new_h
        elif self._resize_handle == "br":
            new_w = px - x
            new_h = py - y
            if new_w > 5: bbox[2] = new_w
            if new_h > 5: bbox[3] = new_h

    def _resize_obb(self, bbox, px, py):
        """Resize OBB - simpler: adjust w/h based on corner distance from center."""
        cx, cy, w, h, angle = bbox[:5]
        import math
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(-rad), math.sin(-rad)
        dx = px - cx
        dy = py - cy
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        bbox[2] = max(10, abs(local_x) * 2)
        bbox[3] = max(10, abs(local_y) * 2)
