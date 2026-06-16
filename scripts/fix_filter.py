with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add cls_mapping loading before the for loop in _filter_images
old_loop = "        for path in self._all_images:"
cls_block = """        # Classification mode: load class_labels.json mapping once
        cls_mapping = {}
        if self._cls_mode and self.current_project:
            try:
                from classification.label_manager import LabelManager as ClsLabelManager
                pd = str(self.pm.get_project_dir(self.current_project["name"]))
                cls_mapping = ClsLabelManager(pd).get_mapping()
            except Exception:
                pass
        for path in self._all_images:"""

if old_loop in content and "cls_mapping = {}" not in content:
    content = content.replace(old_loop, cls_block, 1)
    print("Added cls_mapping loading")

# Add classification annotation check after shape_labels = []
old_shape_check = """            shape_labels = []
            if os.path.exists(det_json_path):"""
new_shape_check = """            shape_labels = []
            # Classification mode: check class_labels.json mapping
            cls_id = None
            if self._cls_mode and cls_mapping:
                rel_path = f"images/{fname}"
                cls_id = cls_mapping.get(rel_path, cls_mapping.get(fname, None))
                if cls_id is not None:
                    has_ann = True
                    shape_count = 1
                    labels = getattr(self, "_cached_classes", []) or []
                    if int(cls_id) < len(labels):
                        shape_labels.append(labels[int(cls_id)])
            if os.path.exists(det_json_path):"""

if old_shape_check in content and "Classification mode: check class_labels.json" not in content:
    content = content.replace(old_shape_check, new_shape_check, 1)
    print("Added classification annotation check")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
