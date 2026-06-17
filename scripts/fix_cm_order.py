with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Reverse class_names order so OK comes before NG (more intuitive)
old_class_order = '                    class_names = label_data.get("labels", [])'
new_class_order = '                    class_names = list(reversed(label_data.get("labels", [])))'

if old_class_order in content:
    content = content.replace(old_class_order, new_class_order)
    print("OK-first display order")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
