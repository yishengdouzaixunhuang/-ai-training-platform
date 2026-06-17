with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

old_epoch_cb = """                def epoch_cb(epoch, total, train_loss, val_acc):
                    if self._stop_flag:
                        self._cls_trainer._stop_flag = True
                    self.train_progress_signal.emit(epoch, total)
                    self.log_signal.emit(
                        f"Epoch {epoch}/{total} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
                    self.cls_train_status.setText(f"Epoch {epoch}/{total}")"""

new_epoch_cb = """                def epoch_cb(epoch, total):
                    if self._stop_flag:
                        self._cls_trainer._stop_flag = True
                    self.train_progress_signal.emit(epoch, total)
                    self.cls_train_status.setText(f"Epoch {epoch}/{total}")"""

if old_epoch_cb in content:
    content = content.replace(old_epoch_cb, new_epoch_cb)
    print("Fixed epoch_cb signature")
else:
    print("ERROR: old_epoch_cb not found - checking...")
    # Try to find it
    idx = content.find("epoch_cb")
    if idx > 0:
        print(content[idx:idx+300])

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
