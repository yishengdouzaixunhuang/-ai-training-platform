with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix: ColumnCount = n+1 (not n+2), headers = class_names + ["Total"] (no empty first column)
# Also revert j+1 back to j
old_col = 'self.cls_cm_table.setColumnCount(n + 2)'
new_col = 'self.cls_cm_table.setColumnCount(n + 1)'
content = content.replace(old_col, new_col)

# Fix headers: remove empty first column
old_headers = '                    headers = [""] + display_names + ["Total"]'
new_headers = '                    headers = display_names + ["Total"]'
content = content.replace(old_headers, new_headers)

# Fix setItem: revert j+1 -> j, n+1 -> n
content = content.replace('self.cls_cm_table.setItem(i, j + 1, item)', 'self.cls_cm_table.setItem(i, j, item)')
content = content.replace('self.cls_cm_table.setItem(i, n + 1, item)', 'self.cls_cm_table.setItem(i, n, item)')
content = content.replace('self.cls_cm_table.setItem(n, j + 1, item)', 'self.cls_cm_table.setItem(n, j, item)')
content = content.replace('self.cls_cm_table.setItem(n, n + 1, item)', 'self.cls_cm_table.setItem(n, n, item)')

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Removed empty first column, use vertical header for row labels")
