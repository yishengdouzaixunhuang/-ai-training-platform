with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    c = f.read()
print("cls_mode init:", "self._cls_mode = False" in c)
print("menu item:", "Image Classification" in c)
print("set_task cls branch:", 'elif task == "classification"' in c)
print("load_cls_labels method:", "_load_classification_labels" in c)
print("cls_mapping loading:", "cls_mapping = {}" in c)
print("cls annotation check:", "if self._cls_mode and cls_mapping:" in c)
print("shape_labels in column:", "shape_labels) if self._cls_mode" in c)
print("status bar Classes:", "Classes:" in c)
print("---")
print("Syntax check:", end=" ")
import py_compile
try:
    py_compile.compile(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", doraise=True)
    print("OK")
except py_compile.PyCompileError as e:
    print(f"FAIL: {e}")
