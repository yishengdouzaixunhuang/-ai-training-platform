"""LabelMe JSON format reader/writer for annotation import/export."""
import os, json, numpy as np, cv2

LABELME_VERSION = "5.3.1"


def get_json_path(image_path):
    """Get the LabelMe JSON path for an image: image.bmp -> image.json"""
    base, _ = os.path.splitext(image_path)
    return base + ".json"


def load_labelme_json(json_path):
    """Load a LabelMe JSON file, return (shapes, image_info) or (None, None)."""
    if not os.path.exists(json_path):
        return None, None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("shapes", []), {
        "imageWidth": data.get("imageWidth", 0),
        "imageHeight": data.get("imageHeight", 0),
        "imagePath": data.get("imagePath", ""),
        "version": data.get("version", LABELME_VERSION),
    }


def save_labelme_json(json_path, image_path, image_width, image_height, shapes, image_set="Train"):
    """Save shapes to a LabelMe JSON file."""
    # Auto-version: save current JSON as a version snapshot before overwriting
    if os.path.exists(json_path):
        try:
            from annotation.version_manager import save_version
            project_dir = os.path.dirname(os.path.dirname(json_path))
            save_version(project_dir, image_path)
        except Exception:
            pass  # Never block save for versioning

    data = {
        "version": LABELME_VERSION,
        "flags": {},
        "imageData": None,
        "imagePath": os.path.basename(image_path),
        "imageWidth": image_width,
        "imageHeight": image_height,
        "imageSet": image_set,
        "baseData": [
            {
                "Roi": {
                    "h": image_height,
                    "phi": 0,
                    "w": image_width,
                    "x": (image_width - 1) / 2.0,
                    "y": (image_height - 1) / 2.0
                },
                "category": "",
                "dataSet": image_set
            }
        ],
        "shapes": shapes,
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def shapes_to_mask(shapes, class_names, image_size):
    """Convert shapes to a numpy mask. class_names[0] = background.
    Returns mask (H, W) uint8."""
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    name_to_id = {name: i for i, name in enumerate(class_names)}

    for shape in shapes:
        label = shape.get("label", "")
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])

        # Handle ignore label -> class 255 (any label containing "ignore" or "gnore")
        if "ignore" in label.lower() or "gnore" in label.lower():
            class_id = 255
        elif label in name_to_id:
            class_id = name_to_id[label]
        else:
            class_id = len(name_to_id)
            name_to_id[label] = class_id

        if shape_type == "polygon" and len(points) >= 3:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color=class_id)

        elif shape_type == "rectangle" and len(points) == 2:
            (x1, y1), (x2, y2) = points
            x1, x2 = sorted([int(x1), int(x2)])
            y1, y2 = sorted([int(y1), int(y2)])
            mask[y1:y2+1, x1:x2+1] = class_id

        elif shape_type == "line" and len(points) >= 2:
            for i in range(len(points) - 1):
                (x1, y1), (x2, y2) = points[i], points[i+1]
                cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)),
                         color=class_id, thickness=3)

    return mask


def mask_to_shapes(mask, class_names, simplify_epsilon=2.0):
    """Convert a numpy mask to LabelMe shapes.
    simplify_epsilon: Douglas-Peucker simplification (pixels). 0 = no simplification."""
    shapes = []
    h, w = mask.shape

    # Use all class IDs present in the mask
    unique_ids = np.unique(mask)
    for class_id in unique_ids:
        if class_id == 0:  # background
            continue
        binary = (mask == class_id).astype(np.uint8)
        if not binary.any():
            continue

        # Handle ignore class
        if class_id == 255:
            label = "ignore"
        elif class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = "class_%d" % class_id

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:
                continue

            # Flatten and convert to list of [x, y]
            pts = contour.reshape(-1, 2).astype(np.float32)

            # Simplify polygon
            if simplify_epsilon > 0 and len(pts) > 3:
                pts = cv2.approxPolyDP(pts, simplify_epsilon, True)
                pts = pts.reshape(-1, 2)

            points = [[int(p[0]), int(p[1])] for p in pts]
            # Get color for this class
            from core.config import CLASS_COLORS
            color_rgb = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            hex_color = "#%02x%02x%02x" % tuple(color_rgb)
            shapes.append({
                "label": label,
                "points": points,
                "shape_type": "polygon",
                "color": hex_color,
                "description": "Polygon",
                "eraser points": [],
                "flags": {},
                "group_id": None,
            })

    return shapes


def load_mask_from_json(image_path, project_dir, class_names):
    """Load annotations from a LabelMe JSON for the given image.
    Falls back to PNG mask if JSON doesn't exist."""
    json_path = get_json_path(image_path)
    shapes, info = load_labelme_json(json_path)

    if shapes is not None and len(shapes) > 0:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
        elif info:
            h, w = info["imageHeight"], info["imageWidth"]
        else:
            return None
        mask = shapes_to_mask(shapes, class_names, (h, w))
        # Also save PNG for backward compatibility
        base = os.path.splitext(os.path.basename(image_path))[0]
        mp = os.path.join(project_dir, "annotations", base + "_mask.png")
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        from PIL import Image
        Image.fromarray(mask.astype(np.uint8)).save(mp)
        return mask

    # Fallback to PNG mask
    base = os.path.splitext(os.path.basename(image_path))[0]
    mp = os.path.join(project_dir, "annotations", base + "_mask.png")
    if os.path.exists(mp):
        from PIL import Image
        return np.array(Image.open(mp)).astype(np.uint8)

    return None


def save_mask_to_json(mask, image_path, project_dir, class_names, simplify_epsilon=1.0):
    """Save current mask as LabelMe JSON. Also saves PNG for backward compat."""
    if mask is None:
        return

    shapes = mask_to_shapes(mask, class_names, simplify_epsilon)

    h, w = mask.shape
    json_path = get_json_path(image_path)
    save_labelme_json(json_path, image_path, w, h, shapes, image_set="Train")

    # Also save PNG mask for backward compatibility
    base = os.path.splitext(os.path.basename(image_path))[0]
    mp = os.path.join(project_dir, "annotations", base + "_mask.png")
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    from PIL import Image
    Image.fromarray(mask.astype(np.uint8)).save(mp)

    return json_path
