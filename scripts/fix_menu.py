with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add "Start Cls Training..." to Model Mgmt menu
old_mm = '        mm.addAction("Start Seg Training...", self._start_training)\n        mm.addAction("Start Det Training...", self._start_det_training)'
if old_mm in content:
    new_mm = '        mm.addAction("Start Seg Training...", self._start_training)\n        mm.addAction("Start Det Training...", self._start_det_training)\n        mm.addAction("Start Cls Training...", self._start_cls_training)'
    content = content.replace(old_mm, new_mm)
    print("1. Added Start Cls Training to Model Mgmt menu")
else:
    print("ERROR: old_mm not found")

# 2. Fix _set_task classification to switch to CL panel (panel 7)
old_cls_task = """            self._load_classification_labels()
            self._filter_images(self.search_input.text())
        else:
            self._detection_mode = False
            self._cls_mode = False"""

new_cls_task = """            self._load_classification_labels()
            self._filter_images(self.search_input.text())
            self._right_stack.setCurrentIndex(7)
        else:
            self._detection_mode = False
            self._cls_mode = False"""

if old_cls_task in content:
    content = content.replace(old_cls_task, new_cls_task)
    print("2. Classification mode now switches to panel 7 (CL)")
else:
    print("ERROR: old_cls_task not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
