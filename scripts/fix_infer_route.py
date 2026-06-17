with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add _refresh_cls_model_list call when switching to classification mode
old_cls_task = '            self._load_classification_labels()\n            self._filter_images(self.search_input.text())\n            self._right_stack.setCurrentIndex(7)'
new_cls_task = '            self._load_classification_labels()\n            self._refresh_cls_model_list()\n            self._filter_images(self.search_input.text())\n            self._right_stack.setCurrentIndex(7)'
if old_cls_task in content:
    content = content.replace(old_cls_task, new_cls_task)
    print("1. Added _refresh_cls_model_list on mode switch")

# 2. Make _start_inference (single) aware of classification mode
old_single_infer = '    def _start_inference(self):\n        if not self.current_project:\n            return QMessageBox.warning(self, "Warning", "Please open a project first")\n        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))'

new_single_infer = '''    def _start_inference(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        if self._cls_mode:
            return self._start_cls_inference()
        if self._detection_mode:
            return self._start_det_inference()
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))'''

if old_single_infer in content:
    content = content.replace(old_single_infer, new_single_infer)
    print("2. _start_inference now routes to cls/det/seg")
else:
    print("ERROR: old_single_infer not found")

# 3. Make _batch_inference aware of classification mode
old_batch_infer = '    def _batch_inference(self):\n        if not self.current_project:\n            return QMessageBox.warning(self, "Warning", "Please open a project first")\n        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))'

new_batch_infer = '''    def _batch_inference(self):
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        if self._cls_mode:
            return self._start_cls_batch_inference()
        if self._detection_mode:
            return self._start_det_batch_inference()
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))'''

# Only replace the first occurrence (the batch_inference method, not the run() inner)
if content.count(old_batch_infer) >= 1:
    content = content.replace(old_batch_infer, new_batch_infer, 1)
    print("3. _batch_inference now routes to cls/det/seg")
else:
    print("ERROR: old_batch_infer not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
