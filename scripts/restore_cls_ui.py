# -*- coding: utf-8 -*-
"""Restore classification UI integration into main_window.py"""
import re

path = r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# === 1. Add _cls_mode init in __init__ after _detection_mode ===
old = "self._detection_mode = False"
new = "self._detection_mode = False\n        self._cls_mode = False"
if "self._cls_mode = False" not in content:
    content = content.replace(old, new, 1)
    print("1. Added _cls_mode init")

# === 2. Add classification action to Tools menu ===
old = 'tm.addAction("Object Detection", lambda: self._set_task("detection"))'
new = 'tm.addAction("Object Detection", lambda: self._set_task("detection"))\n        tm.addAction("Image Classification", lambda: self._set_task("classification"))'
if 'Image Classification' not in content:
    content = content.replace(old, new)
    print("2. Added classification menu item")

# === 3. Add classification branch to _set_task ===
old_set_task_else = '''        else:
            self._detection_mode = False
            self.log("Task: Semantic Segmentation")
            self.canvas._det_overlay = None
            self.canvas.update()
            self._right_stack.setCurrentIndex(0)
            self._right_tabs.button(0).setChecked(True) if self._right_tabs.button(0) else None'''
new_set_task = '''        elif task == "classification":
            self._detection_mode = False
            self._cls_mode = True
            self.log("Task: Image Classification")
            self.canvas._det_overlay = None
            self.canvas.update()
            self._load_classification_labels()
            self._filter_images(self.search_input.text())
        else:
            self._detection_mode = False
            self._cls_mode = False
            self.log("Task: Semantic Segmentation")
            self.canvas._det_overlay = None
            self.canvas.update()
            self._right_stack.setCurrentIndex(0)'''
if '"_set_task("classification")' in content and '"_cls_mode = True"' not in content:
    content = content.replace(old_set_task_else, new_set_task)
    print("3. Added classification branch to _set_task")

# === 4. Add _load_classification_labels method ===
if '_load_classification_labels' not in content:
    cls_label_method = '''
    def _load_classification_labels(self):
        """Load classification labels from class_labels.json."""
        if not self.current_project:
            return
        try:
            project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
            from classification.label_manager import LabelManager as ClsLabelManager
            lm = ClsLabelManager(project_dir)
            data = lm.load()
            labels = data.get("labels", [])
            if not labels:
                data = lm.auto_label_from_folders()
                labels = data.get("labels", [])
            if labels:
                self._cached_classes = labels
                self.class_list.clear()
                for c in labels:
                    self.class_list.addItem(c)
                self.log(f"Classification labels: {len(labels)} classes")
            else:
                self.log("No classification labels found")
        except Exception as e:
            self.log(f"Failed to load classification labels: {e}")
'''
    # Insert before _set_workspace
    insert_pos = content.find('\n    def _set_workspace(self):')
    if insert_pos > 0:
        content = content[:insert_pos] + cls_label_method + '\n' + content[insert_pos:]
        print("4. Added _load_classification_labels method")

# === 5. Fix _filter_images to support classification ===
# The cls_mapping variable and classification checks were lost.
# Find the _filter_images method and add cls_mapping loading + classification checks
old_filter_start = '        # Count shapes and collect labels for search\n            shape_count = 0\n            shape_labels = []\n            # Classification mode: check class_labels.json mapping'
if old_filter_start in content:
    # Already has classification placeholder - need to make it work
    # Add cls_mapping loading before the loop
    old_loop_start = '        for path in self._all_images:'
    cls_mapping_block = '''        # Classification mode: load class_labels.json mapping once
        cls_mapping = {}
        if self._cls_mode and self.current_project:
            try:
                from classification.label_manager import LabelManager as ClsLabelManager
                pd = str(self.pm.get_project_dir(self.current_project["name"]))
                cls_mapping = ClsLabelManager(pd).get_mapping()
            except Exception:
                pass
        for path in self._all_images:'''
    if 'cls_mapping = {}' not in content:
        content = content.replace(old_loop_start, cls_mapping_block)
        print("5a. Added cls_mapping loading")

    # Fix classification annotation check
    old_cls_check = "            if self._cls_mode and cls_mapping:"
    new_cls_check = '''            # Classification mode: check class_labels.json mapping
            cls_id = None
            if self._cls_mode and cls_mapping:
                rel_path = f"images/{fname}"
                cls_id = cls_mapping.get(rel_path, cls_mapping.get(fname, None))
                if cls_id is not None:
                    has_ann = True
                    shape_count = 1
                    labels = getattr(self, "_cached_classes", []) or []
                    if int(cls_id) < len(labels):
                        shape_labels.append(labels[int(cls_id)])'''
    # Remove old broken classification block and add new one
    old_broken_cls = '''            # Classification mode: check class_labels.json mapping
            if self._cls_mode and cls_mapping:
                # Build relative path patterns to match class_labels.json mapping
                rel_path = f"images/{fname}"
                cls_id = cls_mapping.get(rel_path, cls_mapping.get(fname, None))'''
    if old_broken_cls in content:
        # Replace the broken block with working code
        content = content.replace(
            old_broken_cls + '\n                if cls_id is not None:\n                    has_ann = True\n                    shape_count = 1\n                    labels = getattr(self, "_cached_classes", []) or []\n                    if int(cls_id) < len(labels):\n                        shape_labels.append(labels[int(cls_id)])',
            new_cls_check
        )
        print("5b. Fixed classification annotation check")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("\nAll patches applied!")
