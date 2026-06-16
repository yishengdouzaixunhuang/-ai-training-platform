import os, sys

path = r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# ===== 1. Add panels 7 & 8 after panel 6 =====
old_marker = '        self._right_stack.addWidget(panel_det_train)  # 6\n\n        content_layout.addWidget(self._right_stack)'

panel_7_cls_train = '''
        # Panel 7: Classification Training
        panel_cls_train = QWidget(); ctl = QVBoxLayout(panel_cls_train); ctl.setContentsMargins(4, 2, 2, 2)
        ctl.addWidget(QLabel("<b>Classification Training</b>"))
        self.cls_train_progress = QProgressBar(); ctl.addWidget(self.cls_train_progress)
        self.cls_train_status = QLabel("Ready"); ctl.addWidget(self.cls_train_status)
        btn_cls_stop = QPushButton("Stop"); btn_cls_stop.clicked.connect(self._stop_training)
        btn_cls_stop.setEnabled(False); ctl.addWidget(btn_cls_stop)
        self._cls_stop_btn = btn_cls_stop
        cfl = QFormLayout()
        self.cls_model_combo = QComboBox()
        self.cls_model_combo.addItems(["resnet18", "resnet34", "resnet50", "resnet101", "efficientnet_b0",
                                        "efficientnet_b3", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
                                        "vit_b_16", "vit_b_32"])
        cfl.addRow("Model:", self.cls_model_combo)
        self.cls_epochs_spin = QSpinBox(); self.cls_epochs_spin.setRange(1, 2000); self.cls_epochs_spin.setValue(200)
        cfl.addRow("Epochs:", self.cls_epochs_spin)
        self.cls_batch_spin = QSpinBox(); self.cls_batch_spin.setRange(1, 256); self.cls_batch_spin.setValue(32)
        cfl.addRow("Batch:", self.cls_batch_spin)
        self.cls_img_size_spin = QSpinBox(); self.cls_img_size_spin.setRange(32, 512); self.cls_img_size_spin.setValue(224)
        cfl.addRow("Image Size:", self.cls_img_size_spin)
        self.cls_optim_combo = QComboBox()
        self.cls_optim_combo.addItems(["AdamW", "Adam", "SGD"])
        cfl.addRow("Optimizer:", self.cls_optim_combo)
        self.cls_lr_spin = QDoubleSpinBox(); self.cls_lr_spin.setRange(0.00001, 0.1); self.cls_lr_spin.setSingleStep(0.0001)
        self.cls_lr_spin.setDecimals(5); self.cls_lr_spin.setValue(0.001)
        cfl.addRow("LR:", self.cls_lr_spin)
        self.cls_loss_combo = QComboBox()
        self.cls_loss_combo.addItems(["cross_entropy", "label_smoothing", "focal"])
        cfl.addRow("Loss:", self.cls_loss_combo)
        self.cls_augment_combo = QComboBox()
        self.cls_augment_combo.addItems(["none", "light", "medium", "heavy"])
        self.cls_augment_combo.setCurrentText("medium")
        cfl.addRow("Augment:", self.cls_augment_combo)
        self.cls_amp_check = QCheckBox("AMP (mixed precision)"); self.cls_amp_check.setChecked(True)
        cfl.addRow("", self.cls_amp_check)
        self.cls_kfold_spin = QSpinBox(); self.cls_kfold_spin.setRange(1, 10); self.cls_kfold_spin.setValue(1)
        self.cls_kfold_spin.setToolTip("1 = no cross-validation")
        cfl.addRow("K-Fold:", self.cls_kfold_spin)
        self.cls_resume_check = QCheckBox("Resume from last")
        cfl.addRow("", self.cls_resume_check)
        ctl.addLayout(cfl)
        ctl.addWidget(QPushButton("Start Training", clicked=self._start_cls_training))
        ctl.addStretch()
        self._right_stack.addWidget(panel_cls_train)  # 7

        # Panel 8: Classification Inference
        panel_cls_infer = QWidget(); cil = QVBoxLayout(panel_cls_infer); cil.setContentsMargins(4, 2, 2, 2)
        cil.addWidget(QLabel("<b>Classification Inference</b>"))
        self.cls_infer_model_combo = QComboBox()
        self.cls_infer_model_combo.setToolTip("Select model checkpoint")
        cil.addWidget(self.cls_infer_model_combo)
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Top-K:"))
        self.cls_topk_spin = QSpinBox(); self.cls_topk_spin.setRange(1, 20); self.cls_topk_spin.setValue(5)
        hl2.addWidget(self.cls_topk_spin); cil.addLayout(hl2)
        self.cls_infer_progress = QProgressBar(); cil.addWidget(self.cls_infer_progress)
        self.cls_infer_status = QLabel("Ready"); cil.addWidget(self.cls_infer_status)
        self.cls_result_label = QLabel(""); self.cls_result_label.setWordWrap(True)
        self.cls_result_label.setStyleSheet("QLabel { font-family: Consolas; font-size: 11px; padding: 4px; }")
        cil.addWidget(self.cls_result_label)
        ih = QHBoxLayout()
        ih.addWidget(QPushButton("Single Inference", clicked=self._start_cls_inference))
        ih.addWidget(QPushButton("Batch Inference", clicked=self._start_cls_batch_inference))
        cil.addLayout(ih)
        cil.addStretch()
        self._right_stack.addWidget(panel_cls_infer)  # 8

        content_layout.addWidget(self._right_stack)'''

if old_marker in content:
    content = content.replace(old_marker, panel_7_cls_train)
    print("1. Added panels 7 & 8")
else:
    print("ERROR: old_marker not found")

# ===== 2. Add tab buttons for CL and CI =====
old_tab_defs = '            ("DT", "Det Training", 6),\n        ]'
new_tab_defs = '            ("DT", "Det Training", 6),\n            ("CL", "Cls Training", 7),\n            ("CI", "Cls Inference", 8),\n        ]'
if old_tab_defs in content:
    content = content.replace(old_tab_defs, new_tab_defs)
    print("2. Added tab buttons")
else:
    print("ERROR: old_tab_defs not found")

# ===== 3. Add classification methods before _toggle_panel =====
cls_methods = '''
    # ============ Classification ============

    def _start_cls_training(self):
        """Start classification model training."""
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        model_name = self.cls_model_combo.currentText()
        epochs = self.cls_epochs_spin.value()
        batch_size = self.cls_batch_spin.value()
        image_size = self.cls_img_size_spin.value()
        lr = self.cls_lr_spin.value()
        optim_name = self.cls_optim_combo.currentText()
        loss_name = self.cls_loss_combo.currentText()
        augment = self.cls_augment_combo.currentText()
        use_amp = self.cls_amp_check.isChecked()
        k_folds = self.cls_kfold_spin.value()
        resume = self.cls_resume_check.isChecked()

        self.log(f"Classification: {model_name}, {epochs} epochs, batch={batch_size}, lr={lr}")
        self.cls_train_progress.setVisible(True)
        self.cls_train_progress.setValue(0)
        self.cls_train_status.setText("Preparing...")
        self._cls_stop_btn.setEnabled(True)
        self._stop_flag = False

        def run():
            try:
                from classification.trainer import ClassificationTrainer
                self._cls_trainer = ClassificationTrainer(project_dir, model_name=model_name)
                def epoch_cb(epoch, total, train_loss, val_acc):
                    if self._stop_flag:
                        self._cls_trainer._stop_flag = True
                    self.train_progress_signal.emit(epoch, total)
                    self.log_signal.emit(
                        f"Epoch {epoch}/{total} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
                    self.cls_train_status.setText(f"Epoch {epoch}/{total}")
                def stop_check():
                    return self._stop_flag
                self._cls_trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    optim_name=optim_name, loss_name=loss_name,
                    image_size=image_size, use_amp=use_amp,
                    augment=augment, k_folds=k_folds, resume=resume,
                    epoch_callback=epoch_cb, stop_check=stop_check,
                )
                self._refresh_cls_model_list()
                self.log_signal.emit("Classification training complete!")
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Classification training error: {e}")
                self.log_signal.emit(traceback.format_exc())
            finally:
                self.cls_train_progress.setVisible(False)
                self.cls_train_status.setText("Ready")
                self._cls_stop_btn.setEnabled(False)
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _refresh_cls_model_list(self):
        """Refresh classification model list for inference."""
        if not self.current_project:
            return
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        models_dir = os.path.join(project_dir, "models", "classification")
        self.cls_infer_model_combo.clear()
        if os.path.exists(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.endswith(".pth"):
                    self.cls_infer_model_combo.addItem(f)
        if self.cls_infer_model_combo.count() == 0:
            self.cls_infer_model_combo.addItem("(no models found)")

    def _start_cls_inference(self):
        """Run classification inference on current image."""
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        self._refresh_cls_model_list()
        model_file = self.cls_infer_model_combo.currentText()
        if not model_file or "(no models" in model_file:
            return QMessageBox.warning(self, "Error", "No classification model found.")
        current_path = getattr(self.canvas, "current_image_path", None)
        if not current_path or not os.path.exists(current_path):
            return QMessageBox.warning(self, "Warning", "No image selected")
        top_k = self.cls_topk_spin.value()
        self.cls_infer_status.setText("Running...")
        def run():
            import time
            try:
                from classification.trainer import ClassificationTrainer
                ct = ClassificationTrainer(project_dir)
                t0 = time.time()
                result = ct.predict_single(current_path, model_file, top_k=top_k)
                elapsed = time.time() - t0
                if result and result.get("predictions"):
                    lines = [f"{p['class']}: {p['confidence']:.4f}" for p in result["predictions"]]
                    self.log_signal.emit(
                        f"Classification: {result['predictions'][0]['class']} "
                        f"({result['predictions'][0]['confidence']:.2%}) in {elapsed:.2f}s")
                    self.cls_result_label.setText("\\n".join(lines))
                else:
                    self.cls_result_label.setText("No result")
                    self.log_signal.emit("Classification: no result")
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Classification inference error: {e}")
                self.cls_result_label.setText(f"Error: {e}")
            finally:
                self.cls_infer_status.setText("Ready")
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _start_cls_batch_inference(self):
        """Run batch classification inference on all images."""
        if not self.current_project:
            return QMessageBox.warning(self, "Warning", "Please open a project first")
        project_dir = str(self.pm.get_project_dir(self.current_project["name"]))
        self._refresh_cls_model_list()
        model_file = self.cls_infer_model_combo.currentText()
        if not model_file or "(no models" in model_file:
            return QMessageBox.warning(self, "Error", "No classification model found.")
        img_dir = os.path.join(project_dir, "images")
        out_dir = os.path.join(project_dir, "outputs", "classification")
        os.makedirs(out_dir, exist_ok=True)
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                  if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))]
        if not images:
            return QMessageBox.warning(self, "Warning", "No images found")
        total = len(images)
        top_k = self.cls_topk_spin.value()
        self._stop_flag = False
        self.cls_infer_progress.setVisible(True)
        self.cls_infer_progress.setMaximum(total)
        self.cls_infer_progress.setValue(0)
        self.cls_infer_status.setText("Running...")
        self.log(f"Batch classification on {total} images...")
        def run():
            import time, json as _json
            results = {}
            try:
                from classification.trainer import ClassificationTrainer
                ct = ClassificationTrainer(project_dir)
                ct.load_model(model_file)
                for i, img_path in enumerate(images):
                    if self._stop_flag:
                        self.log_signal.emit(f"Batch classification stopped at {i}/{total}")
                        break
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    t0 = time.time()
                    try:
                        r = ct.predict_single(img_path, model_file, top_k=top_k)
                        elapsed = time.time() - t0
                        if r and r.get("predictions"):
                            top1 = r["predictions"][0]
                            results[base] = {"class": top1["class"], "confidence": top1["confidence"]}
                            self.log_signal.emit(
                                f"[{i+1}/{total}] {base}: {top1['class']} ({top1['confidence']:.2%}, {elapsed:.2f}s)")
                        else:
                            self.log_signal.emit(f"[{i+1}/{total}] {base}: no result")
                    except Exception as ex:
                        self.log_signal.emit(f"[{i+1}/{total}] {base}: FAILED - {ex}")
                    self.infer_progress_signal.emit(i + 1, total)
                out_path = os.path.join(out_dir, "classification_results.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    _json.dump(results, f, indent=2, ensure_ascii=False)
                self.log_signal.emit(f"Batch classification complete! {len(results)} results saved to {out_path}")
                self.cls_infer_status.setText("Complete")
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Batch classification error: {e}")
                self.log_signal.emit(traceback.format_exc())
                self.cls_infer_status.setText("Failed")
            finally:
                self.cls_infer_progress.setVisible(False)
        import threading
        threading.Thread(target=run, daemon=True).start()

'''

toggle_marker = '    def _toggle_panel(self, side):'
if toggle_marker in content:
    content = content.replace(toggle_marker, cls_methods + '\n' + toggle_marker)
    print("3. Added classification methods")
else:
    print("ERROR: toggle_marker not found")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
