import sys
path = r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Change 1: annotation column shows class name in classification mode
old1 = '(str(shape_count) if shape_count > 0 else "", True),'
new1 = '(", ".join(shape_labels) if self._cls_mode else (str(shape_count) if shape_count > 0 else ""), True),'
content = content.replace(old1, new1)

# Change 2: status bar shows Classes instead of Blocks in classification mode
old2 = '        parts.append(f"Blocks: {total_shapes}")'
new2 = '        if not self._cls_mode:\n            parts.append(f"Blocks: {total_shapes}")\n        else:\n            parts.append(f"Classes: {len(getattr(self, \"_cached_classes\", []))}")'
content = content.replace(old2, new2)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied OK")
print("Change 1 found:", old1 in content)
print("Change 2 found:", "Classes:" in content)
