with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add cls_loss_canvas to CL panel (panel 7) - after the Start Training button
old_cls_train_end = '        ctl.addWidget(QPushButton("Start Training", clicked=self._start_cls_training))\n        ctl.addStretch()\n        self._right_stack.addWidget(panel_cls_train)  # 7'

new_cls_train_end = '''        ctl.addWidget(QPushButton("Start Training", clicked=self._start_cls_training))
        ctl.addWidget(QLabel("<b>Loss / Accuracy Curve</b>"))
        self.cls_loss_canvas = FigureCanvas(Figure(figsize=(4, 2.5), dpi=100))
        self.cls_loss_ax = self.cls_loss_canvas.figure.subplots()
        self.cls_loss_ax.set_xlabel("Epoch"); self.cls_loss_ax.set_ylabel("Loss / Acc")
        self.cls_loss_ax.grid(True, alpha=0.3)
        self.cls_loss_canvas.figure.tight_layout(pad=0.5)
        ctl.addWidget(self.cls_loss_canvas)
        self.cls_loss_canvas.setVisible(False)
        ctl.addStretch()
        self._right_stack.addWidget(panel_cls_train)  # 7'''

if old_cls_train_end in content:
    content = content.replace(old_cls_train_end, new_cls_train_end)
    print("1. Added loss curve canvas to CL panel")
else:
    print("ERROR: CL panel end not found")

# 2. After training completes, update the loss curve
old_train_done = '''                self._refresh_cls_model_list()
                self.log_signal.emit("Classification training complete!")'''

new_train_done = '''                self._refresh_cls_model_list()
                self._update_cls_loss_curve()
                self.log_signal.emit("Classification training complete!")'''

if old_train_done in content:
    content = content.replace(old_train_done, new_train_done)
    print("2. Loss curve auto-update after training")
else:
    print("ERROR: train_done not found")

# 3. Add _update_cls_loss_curve method before _toggle_panel
loss_curve_method = '''
    def _update_cls_loss_curve(self):
        """Update classification loss/accuracy curve from training history."""
        if not hasattr(self, "_cls_trainer") or self._cls_trainer is None:
            return
        history = getattr(self._cls_trainer, "history", {})
        train_loss = history.get("train_loss", [])
        val_acc = history.get("val_acc", [])
        if not train_loss:
            return
        self.cls_loss_ax.clear()
        epochs = list(range(1, len(train_loss) + 1))
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
'''

toggle_marker = '    def _toggle_panel(self, side):'
if toggle_marker in content and '_update_cls_loss_curve' not in content:
    content = content.replace(toggle_marker, loss_curve_method + '\n' + toggle_marker)
    print("3. Added _update_cls_loss_curve method")
else:
    print("3. _update_cls_loss_curve exists or marker not found")

with open(r"C:\Users\Administrator\Documents\Codex\2026-06-04\c-users-administrator-documents-codex-2026\work\ai_training_platform\ui\main_window.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done!")
