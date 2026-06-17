with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix both occurrences: total -> cm_total in confusion matrix code
content = content.replace('                        total = sum(row_sums)', '                        cm_total = sum(row_sums)')
# Fix references to the old 'total'
content = content.replace("                        item = QTableWidgetItem(str(total))", "                        item = QTableWidgetItem(str(cm_total))")
content = content.replace("                        self.cls_cm_table.setItem(n, n, item)", "                        self.cls_cm_table.setItem(n, n, cm_item)")
# Fix the second cm_total reference  
content = content.replace('                    total = sum(row_sums)', '                    cm_total = sum(row_sums)')
content = content.replace("                    item = QTableWidgetItem(str(total))", "                    cm_item = QTableWidgetItem(str(cm_total))")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed total -> cm_total")
