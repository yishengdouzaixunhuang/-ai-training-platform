with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix: don't reverse labels, keep original order for ID mapping
old = '                    class_names = list(reversed(label_data.get("labels", [])))'
new = '                    class_names = label_data.get("labels", [])'

if old in content:
    content = content.replace(old, new)
    print("Fixed: use original label order (IDs match)")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
