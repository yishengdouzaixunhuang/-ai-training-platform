with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add cls_text after pred_cnt
old_pred = '            # Prediction count\n            pred_cnt = getattr(self, "_pred_count", {}).get(base, 0)\n\n            # Inference time'
new_pred = '            # Prediction count / classification result\n            pred_cnt = getattr(self, "_pred_count", {}).get(base, 0)\n            cls_text = ""\n            if self._cls_mode:\n                cls_data = getattr(self, "_cls_predictions", {}).get(path, None) or getattr(self, "_cls_predictions", {}).get(fname, None)\n                if cls_data is not None:\n                    if isinstance(cls_data, dict):\n                        cls_text = cls_data.get("class", "")\n                        conf = cls_data.get("confidence", None)\n                        if conf is not None:\n                            score_str_local = f"{conf:.2f}"\n            \n            # Inference time'

if old_pred in content:
    content = content.replace(old_pred, new_pred)
    print("1. Added cls_text")

# 2. Update the score logic to use cls confidence
old_score = '            raw_score = getattr(self, "_img_scores", {}).get(base, None)\n            score_str = f"{raw_score:.2f}" if raw_score is not None else ""'
new_score = '            raw_score = getattr(self, "_img_scores", {}).get(base, None)\n            score_str = f"{raw_score:.2f}" if raw_score is not None else ""\n            if self._cls_mode and cls_data is not None:\n                if isinstance(cls_data, dict) and "confidence" in cls_data:\n                    score_str = f\"{cls_data[\"confidence\"]:.2f}\"'

if old_score in content:
    content = content.replace(old_score, new_score)
    print("2. Fixed score for classification")

# 3. Add 耗时 column to table header
old_header = 'self.image_list_widget.setHorizontalHeaderLabels(["序号", "图片名", "大小", "数据集", "标注", "结果", "分数"])'
new_header = 'self.image_list_widget.setHorizontalHeaderLabels(["序号", "图片名", "大小", "数据集", "标注", "结果", "分数", "耗时"])'

if old_header in content:
    content = content.replace(old_header, new_header)
    print("3. Added 耗时 column")

# 4. Update column count
old_col_count = 'self.image_list_widget.setColumnCount(7)'
new_col_count = 'self.image_list_widget.setColumnCount(8)'
if old_col_count in content:
    content = content.replace(old_col_count, new_col_count)
    print("4. Updated column count to 8")

# 5. Add infer_ms to items_data
old_items = '                (cls_text if self._cls_mode else (str(pred_cnt) if pred_cnt > 0 else ""), True),\n                (score_str, True),\n            ]'
new_items = '                (cls_text if self._cls_mode else (str(pred_cnt) if pred_cnt > 0 else ""), True),\n                (score_str, True),\n                (str(infer_ms) if infer_ms else "", True),\n            ]'
if old_items in content:
    content = content.replace(old_items, new_items)
    print("5. Added infer_ms to items")
else:
    # Try alternative
    alt_old = '                (cls_text if self._cls_mode else (str(pred_cnt) if pred_cnt > 0 else ""), True),\n                (score_str, True),\n            ]\n            for col, (val, center) in enumerate(items_data):'
    alt_new = '                (cls_text if self._cls_mode else (str(pred_cnt) if pred_cnt > 0 else ""), True),\n                (score_str, True),\n                (str(infer_ms) if infer_ms else "", True),\n            ]\n            for col, (val, center) in enumerate(items_data):'
    if alt_old in content:
        content = content.replace(alt_old, alt_new)
        print("5b. Added infer_ms (alt match)")
    else:
        print("5. ERROR: items_data not matched")

# 6. Update color loop from 7 to 8 columns
old_color = '            for col in range(7):'
new_color = '            for col in range(8):'
if old_color in content:
    content = content.replace(old_color, new_color)
    print("6. Updated color loop to 8 columns")

# 7. Update score_str reference - need to handle the local score_str issue
# The cls_data block uses score_str_local but should update score_str
old_local = '                            score_str_local = f"{conf:.2f}"'
new_local = '                            score_str = f"{conf:.2f}"'
if old_local in content:
    content = content.replace(old_local, new_local)
    print("7. Fixed score_str reference")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
