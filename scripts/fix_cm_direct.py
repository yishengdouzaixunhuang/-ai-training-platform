with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the auto-eval block with direct CM from batch results
old_auto_eval = '''                # Auto-evaluate confusion matrix
                try:
                    from classification.trainer import ClassificationTrainer
                    ct2 = ClassificationTrainer(project_dir)
                    ct2.load_model(model_file)
                    from classification.eval import evaluate_classification
                    eval_result = evaluate_classification(
                        project_dir, split="val",
                        model_path="classification/" + model_file,
                        batch_size=32, image_size=ct2._image_size,
                        compute_confusion=True, top_k=(1, min(5, ct2.num_classes)),
                    )
                    cm = eval_result.get("confusion_matrix")
                    class_names = eval_result.get("class_names", [])
                    top1 = eval_result.get("top_k_results", {}).get("top1_acc", 0)
                    top5 = eval_result.get("top_k_results", {}).get("top5_acc", 0)
                    if cm and len(class_names) > 0:
                        n = len(class_names)
                        self.cls_cm_table.setRowCount(n + 1)
                        self.cls_cm_table.setColumnCount(n + 1)
                        headers = [""] + class_names + ["Total"]
                        self.cls_cm_table.setHorizontalHeaderLabels(headers)
                        self.cls_cm_table.setVerticalHeaderLabels(class_names + ["Total"])
                        row_sums = [sum(row) for row in cm]
                        col_sums = [sum(cm[r][c] for r in range(n)) for c in range(n)]
                        cm_total = sum(row_sums)
                        for i in range(n):
                            for j in range(n):
                                item = QTableWidgetItem(str(cm[i][j]))
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
                        self.cls_cm_table.setItem(n, n, item)
                        self.cls_cm_table.resizeColumnsToContents()
                    self.cls_eval_status.setText("Top-1: {:.2%}  Top-5: {:.2%}".format(top1, top5))
                except Exception as ex:
                    self.log_signal.emit("CM auto-eval: " + str(ex))'''

new_auto_eval = '''                # Compute confusion matrix from batch results
                try:
                    from classification.label_manager import LabelManager as ClsLabelManager
                    lm = ClsLabelManager(project_dir)
                    label_data = lm.load()
                    gt_mapping = label_data.get("mapping", {})
                    class_names = label_data.get("labels", [])
                    if len(class_names) < 2:
                        class_names = ["NG", "OK"]
                    n = len(class_names)
                    cm = [[0]*n for _ in range(n)]
                    matched = 0
                    for base, info in results.items():
                        pred_cls = info.get("class", "")
                        # Find ground truth
                        gt_cls = None
                        for rel_path, cls_id in gt_mapping.items():
                            if base in rel_path or rel_path.endswith("/" + base):
                                gt_cls = class_names[cls_id] if cls_id < n else None
                                break
                        if gt_cls is None:
                            continue
                        pred_idx = class_names.index(pred_cls) if pred_cls in class_names else -1
                        gt_idx = class_names.index(gt_cls)
                        if pred_idx >= 0:
                            cm[gt_idx][pred_idx] += 1
                            if pred_idx == gt_idx:
                                matched += 1
                    total_matched = sum(sum(row) for row in cm)
                    acc = matched / total_matched if total_matched > 0 else 0
                    # Update table
                    self.cls_cm_table.setRowCount(n + 1)
                    self.cls_cm_table.setColumnCount(n + 1)
                    headers = [""] + class_names + ["Total"]
                    self.cls_cm_table.setHorizontalHeaderLabels(headers)
                    self.cls_cm_table.setVerticalHeaderLabels(class_names + ["Total"])
                    row_sums = [sum(row) for row in cm]
                    col_sums = [sum(cm[r][c] for r in range(n)) for c in range(n)]
                    cm_total = sum(row_sums)
                    for i in range(n):
                        for j in range(n):
                            item = QTableWidgetItem(str(cm[i][j]))
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
                    self.cls_cm_table.setItem(n, n, item)
                    self.cls_cm_table.resizeColumnsToContents()
                    self.cls_eval_status.setText("Acc: {:.2%}  ({}/{} matched)".format(acc, matched, total_matched))
                except Exception as ex:
                    self.log_signal.emit("CM error: " + str(ex))'''

if old_auto_eval in content:
    content = content.replace(old_auto_eval, new_auto_eval)
    print("Fixed: CM now computed from batch results directly")
else:
    print("ERROR: not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
