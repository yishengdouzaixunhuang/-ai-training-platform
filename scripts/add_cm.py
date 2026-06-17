with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find where CI panel is built and add confusion matrix section
# Add after the buttons, before cil.addStretch()

old_ci_buttons = '        ih.addWidget(QPushButton("Batch Inference", clicked=self._start_cls_batch_inference))\n        cil.addLayout(ih)\n        cil.addStretch()\n        self._right_stack.addWidget(panel_cls_infer)  # 8'

new_ci_section = '''        ih.addWidget(QPushButton("Batch Inference", clicked=self._start_cls_batch_inference))
        cil.addLayout(ih)
        # Confusion Matrix section
        cil.addWidget(QLabel("<b>Confusion Matrix</b>"))
        eh = QHBoxLayout()
        eh.addWidget(QPushButton("Evaluate", clicked=self._eval_cls_model))
        self.cls_eval_status = QLabel("")
        eh.addWidget(self.cls_eval_status)
        cil.addLayout(eh)
        self.cls_cm_table = QTableWidget()
        self.cls_cm_table.setEditTriggers(QTableWidget.NoEditTriggers)
        cil.addWidget(self.cls_cm_table)
        cil.addStretch()
        self._right_stack.addWidget(panel_cls_infer)  # 8'''

if old_ci_buttons in content:
    content = content.replace(old_ci_buttons, new_ci_section)
    print("1. Added confusion matrix section to CI panel")
else:
    print("ERROR: CI panel not found")

# Add _eval_cls_model method before _toggle_panel
eval_method = '''
    def _eval_cls_model(self):
        """Evaluate classification model and show confusion matrix."""
        if not self.current_project:
            return
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        self._refresh_cls_model_list()
        model_file = self.cls_infer_model_combo.currentText()
        if not model_file or "(no models" in model_file:
            return QMessageBox.warning(self, "Error", "No classification model found.")
        self.cls_eval_status.setText("Evaluating...")
        def run():
            try:
                from classification.trainer import ClassificationTrainer
                ct = ClassificationTrainer(project_dir)
                ct.load_model(model_file)
                from classification.eval import evaluate_classification
                result = evaluate_classification(
                    project_dir, split="val",
                    model_path=f"classification/{model_file}",
                    batch_size=32, image_size=ct._image_size,
                    compute_confusion=True,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                )
                cm = result.get("confusion_matrix")
                class_names = result.get("class_names", [])
                top1 = result.get("top_k_results", {}).get("top1_acc", 0)
                top5 = result.get("top_k_results", {}).get("top5_acc", 0)
                self.log_signal.emit(f"Evaluation: Top-1={top1:.4f}, Top-5={top5:.4f}")
                # Update table
                if cm and len(class_names) > 0:
                    n = len(class_names)
                    self.cls_cm_table.setRowCount(n + 1)
                    self.cls_cm_table.setColumnCount(n + 1)
                    headers = [""] + class_names + ["Total"]
                    self.cls_cm_table.setHorizontalHeaderLabels(headers)
                    self.cls_cm_table.setVerticalHeaderLabels(class_names + ["Total"])
                    row_sums = [sum(row) for row in cm]
                    col_sums = [sum(cm[r][c] for r in range(n)) for c in range(n)]
                    total = sum(row_sums)
                    for i in range(n):
                        for j in range(n):
                            item = QTableWidgetItem(str(cm[i][j]))
                            item.setTextAlignment(Qt.AlignCenter)
                            if i == j:
                                item.setBackground(QColor(180, 255, 180))
                            self.cls_cm_table.setItem(i, j, item)
                        # Row total
                        item = QTableWidgetItem(str(row_sums[i]))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.cls_cm_table.setItem(i, n, item)
                    # Column totals
                    for j in range(n):
                        item = QTableWidgetItem(str(col_sums[j]))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.cls_cm_table.setItem(n, j, item)
                    # Grand total
                    item = QTableWidgetItem(str(total))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.cls_cm_table.setItem(n, n, item)
                    self.cls_cm_table.resizeColumnsToContents()
                self.cls_eval_status.setText(f"Top-1: {top1:.2%}  Top-5: {top5:.2%}")
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Evaluation error: {e}")
                self.cls_eval_status.setText(f"Error: {e}")
        import threading
        threading.Thread(target=run, daemon=True).start()

'''

toggle_marker = '    def _toggle_panel(self, side):'
if toggle_marker in content and '_eval_cls_model' not in content:
    content = content.replace(toggle_marker, eval_method + '\n' + toggle_marker)
    print("2. Added _eval_cls_model method")
else:
    print("2. _eval_cls_model already exists or marker not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
