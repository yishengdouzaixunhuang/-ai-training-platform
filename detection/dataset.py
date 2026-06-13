# -*- coding: utf-8 -*-
"""Detection Dataset — COCO JSON annotations to YOLO format.

Converts detection annotations stored as COCO JSON (via coco_io.py) into
YOLO-format label files for training with ultralytics YOLO.
"""
import os, json, shutil, random
from pathlib import Path

def build_yolo_dataset(project_dir, output_dir=None, val_split=0.2, seed=42):
    """Build YOLO-format dataset from project detection annotations.
    
    Scans images/ for *_det.json files, converts boxes to YOLO normalized
    format, and creates train/val split with data.yaml.
    
    Args:
        project_dir: Path to project root (contains images/, annotations/)
        output_dir: Output directory for YOLO dataset. Default: project_dir/yolo_dataset/
        val_split: Fraction of data for validation (0.0-1.0)
        seed: Random seed for reproducible splits
    
    Returns:
        dict with keys: data_yaml, train_images, val_images, num_classes, class_names
    """
    project_dir = Path(project_dir)
    img_dir = project_dir / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    
    if output_dir is None:
        output_dir = project_dir / "yolo_dataset"
    output_dir = Path(output_dir)
    
    # Collect class names from all det JSONs
    all_categories = set()
    annotated = []
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in (".bmp", ".png", ".jpg", ".jpeg"):
            continue
        det_json = f.parent / (f.stem + "_det.json")
        if det_json.exists():
            annotated.append((f, det_json))
            try:
                with open(det_json, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                for cat in data.get("categories", []):
                    name = cat.get("name", "")
                    if name and name.lower() != "background":
                        all_categories.add(name)
            except Exception:
                pass
    
    if not annotated:
        raise RuntimeError("No annotated images with _det.json found")
    
    # Sort categories for consistent class IDs
    class_names = sorted(all_categories)
    class_to_id = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    
    # Shuffle and split
    random.seed(seed)
    annotated_sorted = sorted(annotated, key=lambda x: x[0].name)
    random.shuffle(annotated_sorted)
    split_idx = int(len(annotated_sorted) * (1 - val_split))
    train_pairs = annotated_sorted[:split_idx]
    val_pairs = annotated_sorted[split_idx:]
    
    # Create directory structure
    for subset, pairs in [("train", train_pairs), ("val", val_pairs)]:
        (output_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)
    
    # Convert each image
    def convert_pair(img_path, det_path, subset):
        # Copy/symlink image
        dst_img = output_dir / "images" / subset / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        
        # Read det JSON, convert boxes to YOLO format
        with open(det_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Image dimensions
        img_w = data.get("imageWidth", 0)
        img_h = data.get("imageHeight", 0)
        if img_w <= 0 or img_h <= 0:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as im:
                img_w, img_h = im.size
        
        # Get bbox type
        bbox_type = data.get("bbox_type", "hbb")
        
        labels = []
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox", [])
            category = ann.get("category", "")
            if not bbox or not category:
                continue
            cid = class_to_id.get(category)
            if cid is None:
                continue
            
            if bbox_type == "obb" and len(bbox) >= 5:
                # OBB: convert to poly (cx,cy,w,h,angle) -> 4 corner points normalized
                import math
                cx, cy, w, h, angle = bbox[:5]
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                hw, hh = w / 2, h / 2
                corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                corners_norm = []
                for lx, ly in corners_local:
                    gx = (cx + lx * cos_a - ly * sin_a) / img_w
                    gy = (cy + lx * sin_a + ly * cos_a) / img_h
                    corners_norm.extend([gx, gy])
                labels.append(f"{cid} " + " ".join(f"{c:.6f}" for c in corners_norm))
            else:
                # HBB: [x, y, w, h] -> normalized [cx, cy, w, h]
                x, y, bw, bh = bbox[:4]
                cx = (x + bw / 2) / img_w
                cy = (y + bh / 2) / img_h
                nw = bw / img_w
                nh = bh / img_h
                labels.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        # Write labels file
        label_path = output_dir / "labels" / subset / (img_path.stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))
    
    for img_path, det_path in train_pairs:
        convert_pair(img_path, det_path, "train")
    for img_path, det_path in val_pairs:
        convert_pair(img_path, det_path, "val")
    
    # Write data.yaml
    data_yaml = output_dir / "data.yaml"
    train_rel = "images/train"
    val_rel = "images/val"
    yaml_content = f"""# YOLO Detection Dataset
# Generated by AI Training Platform
path: {output_dir.as_posix()}
train: {train_rel}
val: {val_rel}

nc: {num_classes}
names: {json.dumps(class_names)}
"""
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    return {
        "data_yaml": str(data_yaml),
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "num_classes": num_classes,
        "class_names": class_names,
        "output_dir": str(output_dir),
    }


def get_annotated_count(project_dir):
    """Return count of images with detection annotations (_det.json)."""
    img_dir = Path(project_dir) / "images"
    if not img_dir.exists():
        return 0
    count = 0
    for f in img_dir.iterdir():
        if f.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"):
            if (f.parent / (f.stem + "_det.json")).exists():
                count += 1
    return count
