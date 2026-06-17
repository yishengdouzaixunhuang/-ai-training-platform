with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()
old = '                (str(pred_cnt) if pred_cnt > 0 else "", True),'
new = '                (cls_text if self._cls_mode else (str(pred_cnt) if pred_cnt > 0 else ""), True),'
if old in content:
    content = content.replace(old, new)
    print("Fixed cls_text in items_data")
else:
    print("ERROR: not found")
with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
