with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# After computing cm with original label order, reorder for display (OK first, NG second)
# Find the "if cm and len(class_names)" or equivalent display code
old_display = '                    headers = [""] + class_names + ["Total"]'
new_display = '''                    # Reorder: OK first, NG second for display
                    display_order = list(range(n))
                    if "OK" in class_names and "NG" in class_names:
                        ok_idx = class_names.index("OK")
                        ng_idx = class_names.index("NG")
                        if ok_idx > ng_idx:
                            display_order = [ok_idx, ng_idx] if n == 2 else sorted(range(n), key=lambda i: (class_names[i] != "OK", class_names[i] != "NG", class_names[i]))
                    display_names = [class_names[i] for i in display_order]
                    # Reorder cm matrix
                    cm_display = [[cm[display_order[i]][display_order[j]] for j in range(n)] for i in range(n)]
                    headers = [""] + display_names + ["Total"]'''

if old_display in content:
    content = content.replace(old_display, new_display)
    print("Added OK-first reorder logic")

# Update the table build to use cm_display and display_names
# Change all references from class_names and cm in the table building loop
old_build = '''                    self.cls_cm_table.setVerticalHeaderLabels(class_names + ["Total"])
                    row_sums = [sum(row) for row in cm]
                    col_sums = [sum(cm[r][c] for r in range(n)) for c in range(n)]
                    cm_total = sum(row_sums)
                    for i in range(n):
                        for j in range(n):
                            item = QTableWidgetItem(str(cm[i][j]))'''
new_build = '''                    self.cls_cm_table.setVerticalHeaderLabels(display_names + ["Total"])
                    row_sums = [sum(row) for row in cm_display]
                    col_sums = [sum(cm_display[r][c] for r in range(n)) for c in range(n)]
                    cm_total = sum(row_sums)
                    for i in range(n):
                        for j in range(n):
                            item = QTableWidgetItem(str(cm_display[i][j]))'''

if old_build in content:
    content = content.replace(old_build, new_build)
    print("CM display uses cm_display matrix")
else:
    # Try alternate match
    old_build2 = '''                    self.cls_cm_table.setVerticalHeaderLabels(class_names + ["Total"])
                    row_sums = [sum(row) for row in cm]
                    col_sums = [sum(cm[r][c] for r in range(n)) for c in range(n)]'''
    if old_build2 in content:
        content = content.replace(old_build2, new_build.split('\n')[0] + '\n' + '\n'.join(new_build.split('\n')[1:3]))
        print("CM display uses cm_display (alt match)")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
