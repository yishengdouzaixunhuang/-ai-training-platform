with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add plot_callback to classification training call
old_train_call = """                self._cls_trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    optimizer=optim_name, loss_func=loss_name,
                    image_size=image_size, use_amp=use_amp,
                    augment=augment, k_folds=k_folds, resume=resume,
                    progress_callback=epoch_cb, stop_check=stop_check,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                )"""

new_train_call = """                self._cls_trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    optimizer=optim_name, loss_func=loss_name,
                    image_size=image_size, use_amp=use_amp,
                    augment=augment, k_folds=k_folds, resume=resume,
                    progress_callback=epoch_cb, stop_check=stop_check,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                    plot_callback=lambda h: self.chart_signal.emit(h),
                )"""

if old_train_call in content:
    content = content.replace(old_train_call, new_train_call)
    print("1. Added real-time plot_callback to classification training")
else:
    print("ERROR: train call not found")

# 2. Modify _update_loss_chart to handle classification mode (val_acc)
old_loss_chart = """    def _update_loss_chart(self, history):
        \"\"\"Update loss curve from training thread.\"\"\"
        try:
            self.loss_ax.clear()
            train_loss = history.get("train_loss", [])
            val_loss = history.get("val_loss", [])
            epochs = range(1, len(train_loss) + 1)
            if len(epochs) > 0:
                self.loss_ax.plot(epochs, train_loss, "b-", label="Train", linewidth=0.8, alpha=0.7)
                self.loss_ax.plot(epochs, val_loss, "r-", label="Val", linewidth=0.8, alpha=0.7)
                self.loss_ax.legend(fontsize=7)
                self.loss_ax.set_xlabel("Epoch", fontsize=7)
                self.loss_ax.set_ylabel("Loss", fontsize=7)
                self.loss_ax.tick_params(labelsize=6)
            self.loss_canvas.draw()
        except Exception:
            pass"""

new_loss_chart = """    def _update_loss_chart(self, history):
        \"\"\"Update loss curve from training thread (seg or cls).\"\"\"
        try:
            if self._cls_mode and hasattr(self, "cls_loss_ax"):
                # Classification mode: show on CL panel
                self.cls_loss_ax.clear()
                train_loss = history.get("train_loss", [])
                val_acc = history.get("val_acc", [])
                epochs = range(1, len(train_loss) + 1)
                if len(epochs) > 0:
                    self.cls_loss_ax.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1)
                    if val_acc and len(val_acc) == len(train_loss):
                        ax2 = self.cls_loss_ax.twinx()
                        ax2.plot(epochs, val_acc, "r-", label="Val Acc", linewidth=1)
                        ax2.set_ylabel("Accuracy", color="r")
                        ax2.tick_params(axis="y", labelcolor="r")
                        ax2.set_ylim(0, 1.05)
                        lines2, labels2 = ax2.get_legend_handles_labels()
                    else:
                        lines2, labels2 = [], []
                    self.cls_loss_ax.set_xlabel("Epoch")
                    self.cls_loss_ax.set_ylabel("Loss", color="b")
                    self.cls_loss_ax.tick_params(axis="y", labelcolor="b")
                    self.cls_loss_ax.grid(True, alpha=0.3)
                    lines1, labels1 = self.cls_loss_ax.get_legend_handles_labels()
                    self.cls_loss_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
                    self.cls_loss_ax.figure.tight_layout(pad=0.5)
                self.cls_loss_canvas.setVisible(True)
                self.cls_loss_canvas.draw()
            else:
                # Segmentation mode: show on Loss Curve panel
                self.loss_ax.clear()
                train_loss = history.get("train_loss", [])
                val_loss = history.get("val_loss", [])
                epochs = range(1, len(train_loss) + 1)
                if len(epochs) > 0:
                    self.loss_ax.plot(epochs, train_loss, "b-", label="Train", linewidth=0.8, alpha=0.7)
                    self.loss_ax.plot(epochs, val_loss, "r-", label="Val", linewidth=0.8, alpha=0.7)
                    self.loss_ax.legend(fontsize=7)
                    self.loss_ax.set_xlabel("Epoch", fontsize=7)
                    self.loss_ax.set_ylabel("Loss", fontsize=7)
                    self.loss_ax.tick_params(labelsize=6)
                self.loss_canvas.draw()
        except Exception:
            pass"""

if old_loss_chart in content:
    content = content.replace(old_loss_chart, new_loss_chart)
    print("2. Modified _update_loss_chart for cls/seg dual mode")
else:
    print("ERROR: loss_chart not found")

# 3. After batch inference, auto-evaluate and show confusion matrix
old_batch_end = '                self.refresh_list_signal.emit()\n                self.cls_infer_status.setText("Complete")'

new_batch_end = '''                self.refresh_list_signal.emit()
                self.cls_infer_status.setText("Complete")
                # Auto-evaluate confusion matrix
                try:
                    from classification.trainer import ClassificationTrainer
                    ct2 = ClassificationTrainer(project_dir)
                    ct2.load_model(model_file)
                    from classification.eval import evaluate_classification
                    eval_result = evaluate_classification(
                        project_dir, split="val",
                        model_path="classification/" + model_file,
                        batch_size=32, image_size=ct2._image_size,
                        compute_confusion=True,
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
                        total = sum(row_sums)
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
                        item = QTableWidgetItem(str(total))
                        item.setTextAlignment(Qt.AlignCenter)
                        self.cls_cm_table.setItem(n, n, item)
                        self.cls_cm_table.resizeColumnsToContents()
                    self.cls_eval_status.setText("Top-1: {:.2%}  Top-5: {:.2%}".format(top1, top5))
                except Exception:
                    pass'''

if old_batch_end in content:
    content = content.replace(old_batch_end, new_batch_end)
    print("3. Confusion matrix auto-refresh after batch inference")
else:
    print("ERROR: batch_end not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
