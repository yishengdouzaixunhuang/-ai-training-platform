"""Box Manager - CRUD operations for detection bounding boxes.

Stores boxes in memory, supports undo, and auto-saves to COCO JSON.
Designed to be used alongside (not inside) the existing segmentation annotation system.
"""
import os, json, copy
import numpy as np
from detection.coco_io import load_coco_json, save_coco_json, get_det_json_path


class BoxManager:
    """Manages a list of bounding boxes for one image."""

    def __init__(self, categories=None):
        self._boxes = []          # annotation boxes [{id, category, bbox, area, score=None}, ...]
        self._pred_boxes = []     # prediction boxes [{category, bbox, score}, ...] (read-only)
        self._undo_stack = []     # List of box snapshots for undo
        self._next_id = 1
        self._categories = categories or ["background"]
        self._image_path = None
        self._image_width = 0
        self._image_height = 0
        self._bbox_type = "hbb"   # "hbb" or "obb"
        self._dirty = False

    # ---- Properties ----

    @property
    def boxes(self):
        return self._boxes

    @property
    def pred_boxes(self):
        return self._pred_boxes

    @property
    def count(self):
        return len(self._boxes)

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, value):
        self._categories = value

    @property
    def bbox_type(self):
        return self._bbox_type

    @bbox_type.setter
    def bbox_type(self, value):
        self._bbox_type = value

    # ---- Load / Save ----

    def load(self, image_path):
        """Load boxes from detection JSON file (if exists)."""
        self._image_path = image_path
        json_path = get_det_json_path(image_path)
        bboxes, info = load_coco_json(json_path)
        if bboxes is not None:
            self._boxes = bboxes
            self._next_id = max((b.get("id", 0) for b in bboxes), default=0) + 1
            self._image_width = info.get("imageWidth", 0)
            self._image_height = info.get("imageHeight", 0)
            self._bbox_type = info.get("bbox_type", "hbb")
        else:
            self._boxes = []
            self._next_id = 1
            # Image dimensions will be set by caller
        self._undo_stack.clear()
        self._dirty = False

    def save(self):
        """Save boxes to detection JSON file."""
        if not self._image_path:
            return
        json_path = get_det_json_path(self._image_path)
        save_coco_json(
            json_path, self._image_path,
            self._image_width, self._image_height,
            self._boxes, self._categories, self._bbox_type
        )
        self._dirty = False

    def load_predictions(self, outputs_dir):
        """Load prediction boxes from outputs/xxx_det.json.
        
        Args:
            outputs_dir: path to the outputs/ directory containing _det.json files
        """
        if not self._image_path:
            return
        base = os.path.splitext(os.path.basename(self._image_path))[0]
        pred_json = os.path.join(outputs_dir, base + "_det.json")
        if not os.path.exists(pred_json):
            self._pred_boxes = []
            return
        bboxes, _ = load_coco_json(pred_json)
        if bboxes is not None:
            self._pred_boxes = bboxes
        else:
            self._pred_boxes = []

    def set_image_size(self, width, height):
        self._image_width = width
        self._image_height = height

    # ---- CRUD ----

    def _snapshot(self):
        """Save undo snapshot."""
        self._undo_stack.append(copy.deepcopy(self._boxes))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def add_box(self, bbox, category="object", score=None):
        """Add a new bounding box.
        
        Args:
            bbox: [x, y, w, h] for HBB or [cx, cy, w, h, angle] for OBB
            category: class name string
            score: optional confidence score (for predictions)
        Returns:
            box dict with id
        """
        self._snapshot()
        box = {
            "id": self._next_id,
            "category": category,
            "category_id": self._get_category_id(category),
            "bbox": list(bbox),
            "area": self._compute_area(bbox),
            "score": score,
        }
        self._boxes.append(box)
        self._next_id += 1
        self._dirty = True
        self.save()
        return box

    def update_box(self, box_id, bbox=None, category=None):
        """Update an existing box's geometry or category."""
        for b in self._boxes:
            if b["id"] == box_id:
                self._snapshot()
                if bbox is not None:
                    b["bbox"] = list(bbox)
                    b["area"] = self._compute_area(bbox)
                if category is not None:
                    b["category"] = category
                    b["category_id"] = self._get_category_id(category)
                self._dirty = True
                self.save()
                return b
        return None

    def delete_box(self, box_id):
        """Delete a box by id."""
        self._snapshot()
        self._boxes = [b for b in self._boxes if b["id"] != box_id]
        self._dirty = True
        self.save()

    def delete_all(self):
        """Delete all boxes."""
        if not self._boxes:
            return
        self._snapshot()
        self._boxes.clear()
        self._dirty = True
        self.save()

    def undo(self):
        """Undo last operation."""
        if not self._undo_stack:
            return False
        self._boxes = self._undo_stack.pop()
        self._dirty = True
        self.save()
        return True

    def find_box_at(self, px, py, margin=8):
        """Find box id at pixel position (px, py). 
        Returns box dict or None. Margin adds tolerance in pixels.
        """
        for b in reversed(self._boxes):
            if self._bbox_type == "obb":
                if self._point_in_obb(px, py, b["bbox"], margin):
                    return b
            else:
                x, y, w, h = b["bbox"][:4]
                if x - margin <= px <= x + w + margin and y - margin <= py <= y + h + margin:
                    return b
        return None

    def find_handle_at(self, px, py, box_id, handle_size=8):
        """Check if (px,py) is on a resize handle of the given box.
        Returns handle name or None.
        HBB handles: tl, tr, bl, br (corners)
        OBB handles: tl, tr, bl, br (4 rotated corners)
        """
        b = self._get_box(box_id)
        if not b:
            return None
        
        if self._bbox_type == "obb":
            corners = self._get_obb_corners(b["bbox"])
            names = ["tl", "tr", "br", "bl"]
            for name, (cx, cy) in zip(names, corners):
                if abs(px - cx) <= handle_size and abs(py - cy) <= handle_size:
                    return name
        else:
            x, y, w, h = b["bbox"][:4]
            handles = {
                "tl": (x, y),
                "tr": (x + w, y),
                "bl": (x, y + h),
                "br": (x + w, y + h),
            }
            for name, (hx, hy) in handles.items():
                if abs(px - hx) <= handle_size and abs(py - hy) <= handle_size:
                    return name
        return None

    def _get_box(self, box_id):
        for b in self._boxes:
            if b["id"] == box_id:
                return b
        return None

    def _get_category_id(self, category):
        try:
            return self._categories.index(category)
        except ValueError:
            return 0

    def _compute_area(self, bbox):
        if self._bbox_type == "obb" and len(bbox) >= 5:
            return bbox[2] * bbox[3]
        elif len(bbox) >= 4:
            return bbox[2] * bbox[3]
        return 0

    def _point_in_obb(self, px, py, obb, margin=0):
        """Point-in-rotated-rectangle test."""
        cx, cy, w, h, angle = obb[:5]
        import math
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(-rad), math.sin(-rad)
        dx = px - cx
        dy = py - cy
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        half_w, half_h = w / 2 + margin, h / 2 + margin
        return abs(local_x) <= half_w and abs(local_y) <= half_h

    def _get_obb_corners(self, obb):
        """Get 4 corners of rotated bbox."""
        cx, cy, w, h, angle = obb[:5]
        import math
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        hw, hh = w / 2, h / 2
        corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        corners = []
        for lx, ly in corners_local:
            gx = cx + lx * cos_a - ly * sin_a
            gy = cy + lx * sin_a + ly * cos_a
            corners.append((gx, gy))
        return corners
