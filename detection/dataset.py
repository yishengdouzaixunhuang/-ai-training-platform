# -*- coding: utf-8 -*-
"""Detection Dataset — COCO JSON annotations to YOLO format.

Converts detection annotations stored as COCO JSON (via coco_io.py) into
YOLO-format label files for training with ultralytics YOLO.
"""
import os, json, shutil, random
from pathlib import Path

#: 构建出来的数据集长这样：`data.yaml` + `images/` + `labels/`。
#: 用它判断"这个目录像不像我们自己生成的"，清理前挡一下。
_DATASET_FILE = "data.yaml"
_DATASET_DIRS = ("images", "labels")


def _looks_like_dataset(path):
    """`data.yaml` 在 → 是数据集；否则要 `images/` 与 `labels/` **同时**在。

    ⚠️ 别退回 `any(...)`（三个有一个就算）：那样**任何带 `images/` 子目录的非空目录**
       都会被认为是数据集 —— 而"另一个工程根"正好长这样，会被整个删掉。
       放宽到"images + labels 同时有"是为了容忍"建到一半、还没写 data.yaml"的旧目录。
    """
    if (path / _DATASET_FILE).exists():
        return True
    return all((path / d).is_dir() for d in _DATASET_DIRS)


def _wipe_previous_dataset(output_dir, project_dir):
    """把上一轮生成的 `yolo_dataset/` **整个**删掉，返回删掉的文件数。

    ## 为什么必须清

    图片按 base 名落在 `images/train|val/` 下，而拷贝是
    `if not dst_img.exists(): copy2`。于是：

    - 补图/删图后列表长度变了 → `random.seed(42)` 的 shuffle 结果**整体改变**
      （列表长度固定时才恰好每次一样），某张图这次被分到 val，
      上一轮留在 `images/train/` 的那一份永远不会被删；
    - 结果：同一张图**同时出现在 train 和 val**。模型训练时见过它、验证时又考它，
      mAP 虚高，而界面上完全看不出来 —— 比"划分不生效"危险得多。

    删图方向同样会残留：工程里删掉的图，旧副本还在训练集里继续被训练。
    `data.yaml` 之类生成物也一起清，避免新旧口径混着用。

    ## 为什么失败必须抛

    上一次训练的 DataLoader 可能还占着文件（Windows 上是 `PermissionError`）。
    此时若 `ignore_errors=True` 继续重建，结果是"看着重建了、其实还是旧的"，
    比不清理更难查。所以这里转换成带说明的 RuntimeError，让界面上真的报错。

    ## 不删什么（两道防护，第一条是**不变式**）

    1. **数据集目录必须在工程目录里面** —— 判定条件 `proj in out.parents`。
       一条规则同时盖住三种误传，不用逐个列举：
         · `out == proj`（工程根）
         · `out` 是工程的上级目录
         · `out` 是**另一个工程**（★ 原来漏的就是这条：它同样有 `images/`，
           会被"像不像数据集"的弱标记判成数据集，然后整个删掉）
       反过来，`proj/yolo_dataset`、`proj/yolo_dataset_fold0` 都天然通过。
    2. 目录非空、但**不像数据集** → 拒绝。挡的是工程内的误传，最典型是
       `output_dir=proj/images`（有图片，但没有 `images/labels` 子目录）。
    3. 目录不存在 → 直接返回，没什么可清。
    """
    out = Path(output_dir).resolve()
    proj = Path(project_dir).resolve()

    # ① 不变式：必须在工程内。⚠️ 别写成 `out in proj.parents`（那是"不能是上级"），
    #    方向反了 —— 要的是"out 的祖先里有 proj"。
    if proj not in out.parents:
        raise ValueError(
            "数据集输出目录必须在工程目录内：%s（工程：%s）" % (out, proj))

    if not out.exists():
        return 0

    # ② 工程内也要看长相：`proj/images` 这种误传不能删
    if any(out.iterdir()) and not _looks_like_dataset(out):
        raise ValueError(
            "拒绝清空 %s：里面没有 data.yaml，也没有 images/ + labels/，"
            "不像构建出来的数据集。确认 output_dir 是否传错。" % out)

    n = sum(1 for _ in out.rglob("*"))
    try:
        shutil.rmtree(str(out))
    except OSError as e:
        raise RuntimeError(
            "清理旧数据集失败：%s（%s）。多半是上一次训练还在跑、文件被占用 —— "
            "等它结束再重试，别用忽略错误的方式硬上。" % (out, e)) from e
    return n

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

    # ★ 重建前把上一轮的数据集整个删掉（见 _wipe_previous_dataset 的说明）
    removed = _wipe_previous_dataset(output_dir, project_dir)

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
        "removed_files": removed,      # 重建前清掉的旧副本数（界面上顺手报一句）
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
