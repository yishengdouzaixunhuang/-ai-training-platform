with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the _start_cls_training call to match actual trainer API
old_call = """                self._cls_trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    optim_name=optim_name, loss_name=loss_name,
                    image_size=image_size, use_amp=use_amp,
                    augment=augment, k_folds=k_folds, resume=resume,
                    epoch_callback=epoch_cb, stop_check=stop_check,
                )"""

new_call = """                self._cls_trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    optimizer=optim_name, loss_func=loss_name,
                    image_size=image_size, use_amp=use_amp,
                    augment=augment, k_folds=k_folds, resume=resume,
                    progress_callback=epoch_cb, stop_check=stop_check,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                )"""

if old_call in content:
    content = content.replace(old_call, new_call)
    print("Fixed train() call params")
else:
    print("ERROR: old train call not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
