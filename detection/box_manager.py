"""Box Manager - CRUD operations for detection bounding boxes.

Stores boxes in memory, supports undo, and auto-saves to COCO JSON.
Designed to be used alongside (not inside) the existing segmentation annotation system.
"""
import os, json, copy, time
import numpy as np
from detection.coco_io import load_coco_json, save_coco_json, get_det_json_path


class BoxManager:
    """Manages a list of bounding boxes for one image."""

    def __init__(self, categories=None):
        self._boxes = []          # annotation boxes [{id, category, bbox, area, score=None}, ...]
        self._pred_boxes = []     # prediction boxes [{category, bbox, score}, ...] (read-only)
        self._undo_stack = []     # List of box snapshots for undo
        self._redo_stack = []     # List of box snapshots for redo
        self._next_id = 1
        self._categories = categories or ["background"]
        self._image_path = None
        self._image_width = 0
        self._image_height = 0
        self._bbox_type = "hbb"   # "hbb" or "obb"
        self._dirty = False
        #: 上次 load / load_predictions 的失败原因（有值 = 那个文件坏了/读不动）。
        #: ★ 2026-11-16：以前 `_det.json` 损坏时 JSONDecodeError 会一路冒到
        #:   `_open_project_by_name` 的兜底 except，用户看到的是
        #:   "Expecting property name enclosed in double quotes: line 1 column 3"，
        #:   而且工程处于**半开**状态（有预览图、列表空）。现在吞在加载点，
        #:   按 0 个框显示并留下原因，由界面打一行日志。
        self.load_errors = {}

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
        """Load boxes from detection JSON file (if exists).

        ⚠️ 文件坏了**不抛**：按 0 个框显示，把原因记进 `self.load_errors["annot"]`。
           抛出去会一路冒到打开工程的兜底 except（用户看到原始 JSON 异常，
           且工程半开）。空/坏文件的语义是"这张图没有框"。
        """
        self._image_path = image_path
        json_path = get_det_json_path(image_path)
        self.load_errors.pop("annot", None)
        try:
            bboxes, info = load_coco_json(json_path)
        except Exception as e:                       # noqa: BLE001
            self.load_errors["annot"] = "%s: %s" % (type(e).__name__, e)
            bboxes, info = None, None
            # ★ 2026-11-16：坏文件**先改名留档**再允许后续写。
            #   容错让"打不开工程"变成"打得开、显示 0 个框"，而 save() 写的就是
            #   self._boxes —— 用户看到 0 个框、随手加一个框，原文件就被 1 个框整体替换。
            #   哪怕原文只是被截断 2 个字节、还能手工救回 9 个框，也没了。
            archived = self._quarantine(json_path)
            if archived:
                self.load_errors["annot"] += "；原文件已留档: %s" % os.path.basename(archived)
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
        self._redo_stack.clear()
        self._dirty = False

    def _quarantine(self, json_path):
        """把读不动的标注文件改名留档，返回新路径（失败返回 None，绝不抛）。

        `x_det.json` → `x_det.json.bad-20261116T170500`（重名再加 `-1`、`-2`）。

        改名失败（文件被别的进程占着）也**不挡**"按 0 个框显示"，但要把原因记进
        `load_errors` —— 否则用户会以为原文件还在、内容也还在，其实一编辑就被覆盖。
        """
        if not os.path.exists(json_path):
            return None
        stamp = time.strftime("%Y%m%dT%H%M%S")
        dst = "%s.bad-%s" % (json_path, stamp)
        n = 1
        while os.path.exists(dst):
            dst = "%s.bad-%s-%d" % (json_path, stamp, n)
            n += 1
        try:
            os.replace(json_path, dst)
        except OSError as e:
            self.load_errors["annot"] += "；留档失败: %s: %s" % (type(e).__name__, e)
            return None
        return dst

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
        self.load_errors.pop("pred", None)
        if not os.path.exists(pred_json):
            self._pred_boxes = []
            return
        try:                                   # 坏文件同理：按"没有预测"处理，别抛
            bboxes, _ = load_coco_json(pred_json)
        except Exception as e:                 # noqa: BLE001
            self.load_errors["pred"] = "%s: %s" % (type(e).__name__, e)
            bboxes = None
        if bboxes is not None:
            self._pred_boxes = bboxes
        else:
            self._pred_boxes = []

    def set_pred_boxes(self, boxes):
        """直接替换**预测层**（单图推理用）。只影响显示，不落盘、不碰标注。

        ★ 2026-11-16：单图推理以前走 `delete_all()` + `add_box()`，而 `add_box`
          里直接 `save()` —— 写的是 `images/<base>_det.json`（人的标注）。
          "标完了跑一下模型看看效果"会把标注整体换成预测。现在预测只进这一层。
        """
        self._pred_boxes = list(boxes or [])

    def set_image_size(self, width, height):
        self._image_width = width
        self._image_height = height

    # ---- CRUD ----

    def _snapshot(self):
        """Save undo snapshot."""
        self._undo_stack.append(copy.deepcopy(self._boxes))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def add_boxes(self, items):
        """批量加框：**一次快照、一次写盘**（撤销一次回到批量之前）。

        items: `[{"bbox": [...], "category": "...", "score": ...}, ...]`

        ★ 2026-11-16：`add_box` 原来每个框一次 `save()`（6 个框 = 6 次写盘），
          而"把预测整体转成标注"动辄几十个框。批量入口保证：
          · 只写一次盘（写到一半中断不会留下半个标注）；
          · 撤销栈**只压一次** —— 一次 Ctrl+Z 整体回退，而不是按框回退 30 次。
        """
        items = [it for it in (items or []) if it and it.get("bbox")]
        if not items:
            return []
        self._snapshot()
        added = []
        for it in items:
            bbox = it["bbox"]
            category = it.get("category") or "object"
            box = {
                "id": self._next_id,
                "category": category,
                "category_id": self._get_category_id(category),
                "bbox": list(bbox),
                "area": self._compute_area(bbox),
                "score": it.get("score"),
            }
            self._boxes.append(box)
            self._next_id += 1
            added.append(box)
        self._dirty = True
        self.save()
        return added

    def add_box(self, bbox, category="object", score=None):
        """Add a new bounding box.

        Args:
            bbox: [x, y, w, h] for HBB or [cx, cy, w, h, angle] for OBB
            category: class name string
            score: optional confidence score (for predictions)
        Returns:
            box dict with id
        """
        added = self.add_boxes([{"bbox": bbox, "category": category, "score": score}])
        return added[0] if added else None

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
        self._redo_stack.append(copy.deepcopy(self._boxes))
        self._boxes = self._undo_stack.pop()
        self._dirty = True
        self.save()
        return True

    def redo(self):
        """Redo last undone operation."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(copy.deepcopy(self._boxes))
        self._boxes = self._redo_stack.pop()
        self._dirty = True
        self.save()
        return True

    def can_undo(self):
        """Whether an undo is available."""
        return bool(self._undo_stack)

    def can_redo(self):
        """Whether a redo is available."""
        return bool(self._redo_stack)

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
