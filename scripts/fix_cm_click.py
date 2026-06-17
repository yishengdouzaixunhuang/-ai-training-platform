with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. After computing cm, store per-image (gt, pred) mapping and connect click signal
old_after_cm = '                    self.cls_eval_status.setText("Acc: {:.2%}  ({}/{} matched)".format(acc, matched, total_matched))'

new_after_cm = '''                    self.cls_eval_status.setText("Acc: {:.2%}  ({}/{} matched)".format(acc, matched, total_matched))
                    # Store per-image mapping for click-to-filter
                    self._cls_cm_data = {"cm": cm_display, "names": display_names, "gt_map": gt_map, "pred_map": pred_map}
                    self.cls_cm_table.cellClicked.disconnect() if self.cls_cm_table.receivers(self.cls_cm_table.cellClicked) > 0 else None
                    self.cls_cm_table.cellClicked.connect(self._on_cm_cell_clicked)'''

if old_after_cm in content:
    content = content.replace(old_after_cm, new_after_cm)
    print("1. Added CM click handler + data storage")

# 2. Build gt_map and pred_map during the CM computation loop
old_cm_loop = '''                    matched = 0
                    for base, info in results.items():
                        pred_cls = info.get("class", "")
                        # Find ground truth
                        gt_cls = None
                        for rel_path, cls_id in gt_mapping.items():
                            if base in rel_path or rel_path.endswith("/" + base):
                                gt_cls = class_names[cls_id] if cls_id < n else None
                                break
                        if gt_cls is None:
                            continue
                        pred_idx = class_names.index(pred_cls) if pred_cls in class_names else -1
                        gt_idx = class_names.index(gt_cls)
                        if pred_idx >= 0:
                            cm[gt_idx][pred_idx] += 1
                            if pred_idx == gt_idx:
                                matched += 1'''

new_cm_loop = '''                    matched = 0
                    gt_map = {}  # base_name -> gt class name
                    pred_map = {}  # base_name -> pred class name
                    for base, info in results.items():
                        pred_cls = info.get("class", "")
                        # Find ground truth
                        gt_cls = None
                        for rel_path, cls_id in gt_mapping.items():
                            if base in rel_path or rel_path.endswith("/" + base):
                                gt_cls = class_names[cls_id] if cls_id < n else None
                                break
                        if gt_cls is None:
                            continue
                        pred_idx = class_names.index(pred_cls) if pred_cls in class_names else -1
                        gt_idx = class_names.index(gt_cls)
                        if pred_idx >= 0:
                            cm[gt_idx][pred_idx] += 1
                            if pred_idx == gt_idx:
                                matched += 1
                        gt_map[base] = gt_cls
                        pred_map[base] = pred_cls'''

if old_cm_loop in content:
    content = content.replace(old_cm_loop, new_cm_loop)
    print("2. Added gt_map/pred_map tracking")
else:
    print("ERROR: cm_loop not found")

# 3. Add _on_cm_cell_clicked method before _toggle_panel
click_method = '''
    def _on_cm_cell_clicked(self, row, col):
        """Filter image list when clicking a CM cell."""
        data = getattr(self, "_cls_cm_data", None)
        if not data or row >= len(data["names"]) or col >= len(data["names"]):
            return
        gt_cls = data["names"][row]
        pred_cls = data["names"][col]
        # Find images matching this cell
        matched = []
        for base, gt in data["gt_map"].items():
            pd = data["pred_map"].get(base, "")
            if gt == gt_cls and pd == pred_cls:
                # Find full image path
                for ext in (".bmp", ".png", ".jpg", ".jpeg"):
                    p = os.path.join(self._project_dir, "images", base + ext)
                    if os.path.exists(p):
                        matched.append(base)
                        break
        # Filter the image list
        self.search_input.setText(" ".join(matched[:5]))
        self.log(f"CM click: {gt_cls}->{pred_cls} = {len(matched)} images")
'''

toggle_marker = '    def _toggle_panel(self, side):'
if toggle_marker in content and '_on_cm_cell_clicked' not in content:
    content = content.replace(toggle_marker, click_method + '\n' + toggle_marker)
    print("3. Added _on_cm_cell_clicked method")
else:
    print("3. Already exists or marker not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
