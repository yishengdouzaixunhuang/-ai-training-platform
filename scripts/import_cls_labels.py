# -*- coding: utf-8 -*-
"""Batch import classification labels from company-exported JSON files."""
import json, os, sys
from pathlib import Path
from collections import Counter

SRC_DIR = Path(r"D:\01_D盘\01_Arthur\15_标注\铜极分类的标注")
PROJ_DIR = Path(r"C:\Users\Administrator\Documents\AI_Training_Platform\Classify")

def main():
    img_dir = PROJ_DIR / "images"
    mapping = {}
    split_map = {}
    all_categories = set()
    skipped = []

    for jf in sorted(SRC_DIR.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        cat = data["baseData"][0]["category"]
        ds = data.get("imageSet", data["baseData"][0].get("dataSet", "Train")).lower()
        img_name = data.get("imagePath", jf.stem + ".bmp")
        all_categories.add(cat)

        if ds in ("train", "training"):
            ds = "train"
        elif ds in ("val", "validation", "valid", "test"):
            ds = "val"
        else:
            ds = "train"

        img_path = img_dir / img_name
        if img_path.exists():
            rel = f"images/{img_name}"
            base = os.path.splitext(img_name)[0]
            mapping[rel] = cat
            split_map[base] = ds
        else:
            skipped.append(img_name)

    class_names = sorted(all_categories)
    print(f"Classes: {class_names}")

    final_mapping = {}
    for rel, cat_name in mapping.items():
        final_mapping[rel] = class_names.index(cat_name)

    (PROJ_DIR / "annotations").mkdir(parents=True, exist_ok=True)

    label_data = {"labels": class_names, "mapping": final_mapping}
    with open(PROJ_DIR / "annotations" / "class_labels.json", "w", encoding="utf-8") as f:
        json.dump(label_data, f, indent=2, ensure_ascii=False)

    with open(PROJ_DIR / "train_test_split.json", "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2, ensure_ascii=False)

    train_cnt = sum(1 for v in split_map.values() if v == "train")
    val_cnt = sum(1 for v in split_map.values() if v == "val")
    per_class = Counter(mapping.values())

    print(f"\nTotal imported: {len(final_mapping)}")
    print(f"Train: {train_cnt}, Val: {val_cnt}")
    for cls, cnt in per_class.most_common():
        print(f"  {cls}: {cnt}")
    if skipped:
        print(f"\nSkipped (image not found): {len(skipped)}")
        for s in skipped[:5]:
            print(f"  - {s}")
    print(f"\nSaved:")
    print(f"  {PROJ_DIR / 'annotations' / 'class_labels.json'}")
    print(f"  {PROJ_DIR / 'train_test_split.json'}")

if __name__ == "__main__":
    main()
