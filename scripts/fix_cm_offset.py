with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the CM table building: all column indices need +1 offset (column 0 = row label)
old_loop = '''                    for i in range(n):
                        for j in range(n):
                            item = QTableWidgetItem(str(cm_display[i][j]))
                            item.setTextAlignment(Qt.AlignCenter)
                            if i == j:
                                item.setBackground(QColor(180, 255, 180))
                            self.cls_cm_table.setItem(i, j, item)
                        item = QTableWidgetItem(str(row_sums[i]))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.cls_cm_table.setItem(i, n, item)
                    for j in range(n):
                        item = QTableWidgetItem(str(col_sums[j]))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.cls_cm_table.setItem(n, j, item)
                    item = QTableWidgetItem(str(cm_total))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.cls_cm_table.setItem(n, n, item)'''

new_loop = '''                    for i in range(n):
                        for j in range(n):
                            item = QTableWidgetItem(str(cm_display[i][j]))
                            item.setTextAlignment(Qt.AlignCenter)
                            if i == j:
                                item.setBackground(QColor(180, 255, 180))
                            self.cls_cm_table.setItem(i, j + 1, item)
                        item = QTableWidgetItem(str(row_sums[i]))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.cls_cm_table.setItem(i, n + 1, item)
                    for j in range(n):
                        item = QTableWidgetItem(str(col_sums[j]))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.cls_cm_table.setItem(n, j + 1, item)
                    item = QTableWidgetItem(str(cm_total))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.cls_cm_table.setItem(n, n + 1, item)'''

if old_loop in content:
    content = content.replace(old_loop, new_loop)
    print("Fixed column offsets (+1)")
else:
    print("ERROR: not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
