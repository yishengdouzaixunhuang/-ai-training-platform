with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

old = ('    def _start_inference(self):\n'
       '        if not self.current_project:\n'
       '            return QMessageBox.warning(self, "Warning", "Please open a project first")\n'
       '        if self.canvas.image is None:\n'
       '            return QMessageBox.warning(self, "Warning", "Please load an image first")\n'
       '        project_dir')

new = ('    def _start_inference(self):\n'
       '        if not self.current_project:\n'
       '            return QMessageBox.warning(self, "Warning", "Please open a project first")\n'
       '        if self._cls_mode:\n'
       '            return self._start_cls_inference()\n'
       '        if self.canvas.image is None:\n'
       '            return QMessageBox.warning(self, "Warning", "Please load an image first")\n'
       '        project_dir')

if old in content:
    content = content.replace(old, new, 1)
    print("Fixed _start_inference with cls routing")
else:
    print("ERROR: not matched")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
