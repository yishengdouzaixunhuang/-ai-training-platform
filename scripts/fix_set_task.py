import sys
path = r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix _set_task: change "else:" to classification + segmentation branches
old_else = (
    '        else:\n'
    '            self._detection_mode = False\n'
    '            self.log("Task: Semantic Segmentation")\n'
    '            self.canvas._det_overlay = None\n'
    '            self.canvas.update()\n'
    '            self._right_stack.setCurrentIndex(0)\n'
    '            self._right_tabs.button(0).setChecked(True) if self._right_tabs.button(0) else None'
)

new_branches = (
    '        elif task == "classification":\n'
    '            self._detection_mode = False\n'
    '            self._cls_mode = True\n'
    '            self.log("Task: Image Classification")\n'
    '            self.canvas._det_overlay = None\n'
    '            self.canvas.update()\n'
    '            self._load_classification_labels()\n'
    '            self._filter_images(self.search_input.text())\n'
    '        else:\n'
    '            self._detection_mode = False\n'
    '            self._cls_mode = False\n'
    '            self.log("Task: Semantic Segmentation")\n'
    '            self.canvas._det_overlay = None\n'
    '            self.canvas.update()\n'
    '            self._right_stack.setCurrentIndex(0)'
)

if old_else in content:
    content = content.replace(old_else, new_branches)
    print("OK: _set_task fixed")
else:
    print("ERROR: _set_task else block not found")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
