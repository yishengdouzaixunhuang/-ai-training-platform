with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix: cap top_k to num_classes in both evaluation locations
# Find the evaluate_classification calls and add top_k param
old_eval1 = '                    eval_result = evaluate_classification(\n                        project_dir, split="val",\n                        model_path="classification/" + model_file,\n                        batch_size=32, image_size=ct2._image_size,\n                        compute_confusion=True,\n                    )'
new_eval1 = '                    eval_result = evaluate_classification(\n                        project_dir, split="val",\n                        model_path="classification/" + model_file,\n                        batch_size=32, image_size=ct2._image_size,\n                        compute_confusion=True, top_k=(1, min(5, ct2.num_classes)),\n                    )'

if old_eval1 in content:
    content = content.replace(old_eval1, new_eval1)
    print("Fixed eval1 top_k")

old_eval2 = '                result = evaluate_classification(\n                    project_dir, split="val",\n                    model_path="classification/" + model_file,\n                    batch_size=32, image_size=ct._image_size,\n                    compute_confusion=True,\n                    log_callback=lambda msg: self.log_signal.emit(msg),\n                )'
new_eval2 = '                result = evaluate_classification(\n                    project_dir, split="val",\n                    model_path="classification/" + model_file,\n                    batch_size=32, image_size=ct._image_size,\n                    compute_confusion=True, top_k=(1, min(5, ct.num_classes)),\n                    log_callback=lambda msg: self.log_signal.emit(msg),\n                )'

if old_eval2 in content:
    content = content.replace(old_eval2, new_eval2)
    print("Fixed eval2 top_k")
else:
    print("ERROR: eval2 not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
