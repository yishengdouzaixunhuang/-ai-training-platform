with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add toggle connection after checkbox creation
old_checkbox = '        self.cls_full_img_check = QCheckBox("Full Image (no resize)")\n        cfl.addRow("", self.cls_full_img_check)'
new_checkbox = '''        self.cls_full_img_check = QCheckBox("Full Image (no resize)")
        self.cls_full_img_check.toggled.connect(lambda checked: self.cls_img_size_spin.setDisabled(checked))
        cfl.addRow("", self.cls_full_img_check)'''

if old_checkbox in content:
    content = content.replace(old_checkbox, new_checkbox)
    print("Added toggle: Full Image disables Image Size spinbox")
else:
    print("ERROR: not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
