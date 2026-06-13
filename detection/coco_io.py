"""COCO-format JSON I/O for object detection annotations.

Compatible with standard COCO JSON for cross-software import/export.
Supports both horizontal bounding boxes (HBB) and rotated bounding boxes (OBB).
"""
import os, json, numpy as np

DETECTION_JSON_VERSION = "1.0.0"
COCO_CATEGORIES_KEY = "detection_categories"

# ============================================================
#  COCO JSON helpers
# ============================================================

def _get_det_json_path(image_path):
    """Get detection JSON path: image.bmp -> image_det.json"""
    base, _ = os.path.splitext(image_path)
    return base + "_det.json"


def load_coco_json(json_path):
    """Load a detection COCO JSON file.
    
    Returns:
        (bboxes, image_info) where bboxes is a list of dicts:
        [{id, category, bbox: [x,y,w,h], score, area, ...}, ...]
        For OBB: bbox is [cx,cy,w,h,angle_deg]
        Returns (None, None) if file not found.
    """
    if not os.path.exists(json_path):
        return None, None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Compatibility: old batch inference saved a plain list
    if isinstance(data, list):
        converted = []
        for idx, item in enumerate(data):
            converted.append({
                "id": idx + 1,
                "category": item.get("category", "unknown"),
                "bbox": item.get("bbox", [0, 0, 0, 0]),
                "score": item.get("confidence", item.get("score", 0.0)),
            })
        data = {"version": DETECTION_JSON_VERSION, "annotations": converted, "image": {}, "categories": []}
    
    bboxes = data.get("annotations", [])
    image_info = {
        "imageWidth": data.get("image", {}).get("width", 0),
        "imageHeight": data.get("image", {}).get("height", 0),
        "imagePath": data.get("image", {}).get("file_name", ""),
        "version": data.get("version", DETECTION_JSON_VERSION),
        "bbox_type": data.get("bbox_type", "hbb"),  # "hbb" or "obb"
    }
    return bboxes, image_info


def save_coco_json(json_path, image_path, image_width, image_height, 
                   bboxes, categories, bbox_type="hbb"):
    """Save detection annotations to COCO JSON format.
    
    Args:
        json_path: output JSON path
        image_path: source image path
        image_width, image_height: image dimensions
        bboxes: list of dicts [{category, bbox: [x,y,w,h], score, area}, ...]
                For OBB: bbox: [cx, cy, w, h, angle_deg]
        categories: list of category name strings
        bbox_type: "hbb" (horizontal) or "obb" (rotated)
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    annotations = []
    for i, b in enumerate(bboxes):
        ann = {
            "id": b.get("id", i + 1),
            "category_id": b.get("category_id", categories.index(b["category"]) if b.get("category") in categories else 0),
            "category": b.get("category", "unknown"),
            "bbox": b["bbox"],
            "area": b.get("area", _compute_area(b["bbox"], bbox_type)),
            "score": b.get("score", b.get("confidence", None)),
        }
        annotations.append(ann)
    
    data = {
        "version": DETECTION_JSON_VERSION,
        "bbox_type": bbox_type,
        "image": {
            "file_name": os.path.basename(image_path),
            "width": image_width,
            "height": image_height,
        },
        "categories": [{"id": i, "name": name} for i, name in enumerate(categories)],
        "annotations": annotations,
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def bboxes_to_coco(bboxes, categories, image_path, image_width, image_height, bbox_type="hbb"):
    """Convert internal bbox list to COCO dict (for in-memory use)."""
    return {
        "version": DETECTION_JSON_VERSION,
        "bbox_type": bbox_type,
        "image": {
            "file_name": os.path.basename(image_path),
            "width": image_width,
            "height": image_height,
        },
        "categories": [{"id": i, "name": name} for i, name in enumerate(categories)],
        "annotations": [
            {
                "id": b.get("id", i + 1),
                "category_id": b.get("category_id", categories.index(b["category"]) if b.get("category") in categories else 0),
                "category": b.get("category", "unknown"),
                "bbox": b["bbox"],
                "area": b.get("area", _compute_area(b["bbox"], bbox_type)),
                "score": b.get("score", None),
            }
            for i, b in enumerate(bboxes)
        ],
    }


def coco_to_bboxes(coco_data):
    """Convert COCO dict back to internal bbox list."""
    categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    bboxes = []
    for ann in coco_data.get("annotations", []):
        cat_id = ann.get("category_id", 0)
        bboxes.append({
            "id": ann.get("id", len(bboxes) + 1),
            "category": ann.get("category", categories.get(cat_id, "unknown")),
            "category_id": cat_id,
            "bbox": ann["bbox"],
            "area": ann.get("area", 0),
            "score": ann.get("score", None),
        })
    return bboxes


def _compute_area(bbox, bbox_type="hbb"):
    """Compute area of a bounding box."""
    if bbox_type == "hbb" and len(bbox) >= 4:
        x, y, w, h = bbox[:4]
        return w * h
    elif bbox_type == "obb" and len(bbox) >= 5:
        cx, cy, w, h, angle = bbox[:5]
        return w * h
    return 0


def get_det_json_path(image_path):
    """Get detection JSON path for an image."""
    return _get_det_json_path(image_path)
