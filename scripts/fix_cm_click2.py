with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

method = '''
    def _on_cm_cell_clicked(self, row, col):
        """Filter image list when clicking a confusion matrix cell."""
        import os
        data = getattr(self, "_cls_cm_data", None)
        if not data or row >= len(data["names"]) or col >= len(data["names"]):
            return
        gt_cls = data["names"][row]
        pred_cls = data["names"][col]
        matched = []
        for base, gt in data["gt_map"].items():
            pd = data["pred_map"].get(base, "")
            if gt == gt_cls and pd == pred_cls:
                for ext in (".bmp", ".png", ".jpg", ".jpeg"):
                    p = os.path.join(getattr(self, "_project_dir", ""), "images", base + ext)
                    if os.path.exists(p):
                        matched.append(base)
                        break
        if matched:
            self.search_input.setText("")
            self._filter_images(" ".join(matched[:10]))
        self.log(f"CM click: {gt_cls}->{pred_cls} = {len(matched)} images")
'''

marker = '    def _toggle_panel(self, side):'
if '_on_cm_cell_clicked' not in content:
    content = content.replace(marker, method + '\n' + marker)
    print("Added _on_cm_cell_clicked")
else:
    print("Already exists")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
