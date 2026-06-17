with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the silent except with error log
old_except = '                    self.cls_eval_status.setText("Top-1: {:.2%}  Top-5: {:.2%}".format(top1, top5))\n                except Exception:\n                    pass'

new_except = '                    self.cls_eval_status.setText("Top-1: {:.2%}  Top-5: {:.2%}".format(top1, top5))\n                except Exception as ex:\n                    self.log_signal.emit("CM auto-eval: " + str(ex))'

if old_except in content:
    content = content.replace(old_except, new_except)
    print("Fixed silent except -> log error")
else:
    print("ERROR: not found")

# Also ensure cls_cm_table is visible when showing data
old_visible = '                        self.cls_cm_table.resizeColumnsToContents()'  
if old_visible not in content:
    # Add resize after the last setItem
    old_last_item = "                        self.cls_cm_table.setItem(n, n, cm_item)"
    new_last_item = "                        self.cls_cm_table.setItem(n, n, cm_item)\n                        self.cls_cm_table.resizeColumnsToContents()"
    if old_last_item in content:
        content = content.replace(old_last_item, new_last_item)
        print("Added resizeColumnsToContents")
else:
    print("resizeColumnsToContents already present")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
