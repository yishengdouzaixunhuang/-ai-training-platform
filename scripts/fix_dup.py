with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the duplicated blocks from double-patching
broken = '        if not self._cls_mode:\n            if not self._cls_mode:\n            parts.append(f"Blocks: {total_shapes}")\n        else:\n            parts.append(f"Classes: {len(getattr(self, \"_cached_classes\", []))}")\n        else:\n            parts.append(f"Classes: {len(getattr(self, \"_cached_classes\", []))}")'

fixed = '        if not self._cls_mode:\n            parts.append(f"Blocks: {total_shapes}")\n        else:\n            parts.append(f"Classes: {len(getattr(self, \"_cached_classes\", []))}")'

count = content.count(broken)
print(f"Found {count} occurrences of broken block")

if count > 0:
    content = content.replace(broken, fixed)
    with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed!")
else:
    print("Broken block not found. Checking lines 1130-1140:")
    lines = content.split("\n")
    for i in range(1129, 1140):
        if i < len(lines):
            print(f"  L{i+1}: {lines[i]}")
