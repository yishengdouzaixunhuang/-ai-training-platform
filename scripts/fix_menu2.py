with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

old_mm = '        mm.addAction("Start Det Training", self._start_det_training)'
new_mm = '        mm.addAction("Start Det Training", self._start_det_training)\n        mm.addAction("Start Cls Training", self._start_cls_training)'
if old_mm in content:
    content = content.replace(old_mm, new_mm)
    print("Added Start Cls Training to Model Mgmt menu")
else:
    print("ERROR: not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
