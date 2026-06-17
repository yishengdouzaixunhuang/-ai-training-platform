with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\dataset.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix: also check with "images/" prefix
old_check = """                if rel in mapping_dict:
                    label_id = mapping_dict[rel]"""

new_check = """                if rel in mapping_dict:
                    label_id = mapping_dict[rel]
                elif f"images/{rel}" in mapping_dict:
                    label_id = mapping_dict[f"images/{rel}"]"""

if old_check in content:
    content = content.replace(old_check, new_check)
    print("Fixed: added images/ prefix fallback")
else:
    print("ERROR: pattern not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\classification\dataset.py", "w", encoding="utf-8") as f:
    f.write(content)
