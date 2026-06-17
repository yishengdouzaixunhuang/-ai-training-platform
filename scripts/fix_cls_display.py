with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add cls_text computation in _filter_images before items_data
old_pred_cnt = '''            # Prediction count / classification result
            pred_cnt = getattr(self, "_pred_count", {}).get(base, 0)
            cls_text = ""
            # Score
            raw_score = getattr(self, "_img_scores", {}).get(base, None)
            score_str = f"{raw_score:.2f}" if raw_score is not None else ""'''

# Check if this block exists
if 'cls_text = ""' not in content:
    # Need to add the whole classification result block
    old_score_block = '            # Prediction count / classification result\n            pred_cnt = getattr(self, "_pred_count", {}).get(base, 0)\n            # Score\n            raw_score = getattr(self, "_img_scores", {}).get(base, None)\n            score_str = f"{raw_score:.2f}" if raw_score is not None else ""'
    
    new_score_block = '''            # Prediction count / classification result
            pred_cnt = getattr(self, "_pred_count", {}).get(base, 0)
            cls_text = ""
            # Score
            raw_score = getattr(self, "_img_scores", {}).get(base, None)
            score_str = f"{raw_score:.2f}" if raw_score is not None else ""

            if self._cls_mode:
                cls_data = getattr(self, "_cls_predictions", {}).get(path, None) or getattr(self, "_cls_predictions", {}).get(fname, None)
                if cls_data is not None:
                    if isinstance(cls_data, dict):
                        cls_text = cls_data.get("class", str(cls_data.get("class_id", "")))
                        conf = cls_data.get("confidence", None)
                        if conf is not None:
                            score_str = f"{conf:.2f}"
                    elif isinstance(cls_data, (int, float)):
                        cls_text = str(int(cls_data))
                    else:
                        cls_text = str(cls_data)'''

    if old_score_block in content:
        content = content.replace(old_score_block, new_score_block)
        print("1. Added cls_text computation in _filter_images")
    else:
        print("ERROR: old_score_block not found")
else:
    print("1. cls_text already exists")

# 2. After batch classification, store predictions and refresh list
old_batch_done = '''                self.log_signal.emit(f"Batch classification complete! {len(results)} results saved to {out_path}")
                self.cls_infer_status.setText("Complete")'''

new_batch_done = '''                self.log_signal.emit(f"Batch classification complete! {len(results)} results saved to {out_path}")
                self._cls_predictions = {}
                for base, info in results.items():
                    for ext in (".bmp", ".png", ".jpg", ".jpeg"):
                        img_path = os.path.join(img_dir, base + ext)
                        if os.path.exists(img_path):
                            self._cls_predictions[img_path] = info
                            break
                    if base not in [os.path.splitext(k)[0] for k in self._cls_predictions]:
                        self._cls_predictions[base] = info
                self.refresh_list_signal.emit()
                self.cls_infer_status.setText("Complete")'''

if old_batch_done in content:
    content = content.replace(old_batch_done, new_batch_done)
    print("2. Batch inference now refreshes prediction display")
else:
    print("ERROR: old_batch_done not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
