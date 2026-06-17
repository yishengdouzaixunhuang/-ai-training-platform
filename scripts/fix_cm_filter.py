with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the click handler to use _cm_filter_bases
old_click = '''        if matched:
            self.search_input.setText("")
            self._filter_images(" ".join(matched[:10]))
        self.log(f"CM click: {gt_cls}->{pred_cls} = {len(matched)} images")'''

new_click = '''        if matched:
            self._cm_filter_bases = set(matched)
            self.search_input.setText("")
            self._filter_images("")
            self._cm_filter_bases = None
        self.log(f"CM click: {gt_cls}->{pred_cls} = {len(matched)} images")'''

if old_click in content:
    content = content.replace(old_click, new_click)
    print("1. Fixed click handler")

# Add _cm_filter_bases check in _filter_images - before search check
old_search = '''            # Search: match filename OR class label
            if text_lower and text_lower not in fname.lower():'''

new_search = '''            # CM cell click filter
            cm_filter = getattr(self, "_cm_filter_bases", None)
            if cm_filter is not None and base not in cm_filter:
                continue
            # Search: match filename OR class label
            if text_lower and text_lower not in fname.lower():'''

if old_search in content:
    content = content.replace(old_search, new_search)
    print("2. Added CM filter in _filter_images")
else:
    print("ERROR: search block not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
