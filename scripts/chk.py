with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    c = f.read()

# Check if _refresh_cls_model_list is called when switching to classification
idx = c.find('elif task == "classification"')
if idx > 0:
    block = c[idx:idx+300]
    print("_set_task classification branch:")
    print(block)
    print()
    print("_refresh_cls_model_list called:", "_refresh_cls_model_list" in block)
