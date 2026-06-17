with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add "Full Image" checkbox after Image Size spinbox in CL panel
old_img_size = '        cfl.addRow("Image Size:", self.cls_img_size_spin)'
new_img_size = '''        cfl.addRow("Image Size:", self.cls_img_size_spin)
        self.cls_full_img_check = QCheckBox("Full Image (no resize)")
        cfl.addRow("", self.cls_full_img_check)'''

if old_img_size in content:
    content = content.replace(old_img_size, new_img_size)
    print("1. Added Full Image checkbox")

# 2. In _start_cls_training, use image_size=None when checked
old_get_size = '        image_size = self.cls_img_size_spin.value()'
new_get_size = '''        image_size = self.cls_img_size_spin.value()
        if self.cls_full_img_check.isChecked():
            image_size = None'''

if old_get_size in content:
    content = content.replace(old_get_size, new_get_size)
    print("2. Full image logic in training")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
