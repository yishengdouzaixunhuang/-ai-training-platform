"""Image resize tool - fixed size / scale / max dimension, batch support."""
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QLineEdit,
    QFileDialog, QMessageBox, QListWidget, QProgressBar,
    QGroupBox, QCheckBox, QWidget
)
from PyQt5.QtCore import Qt


class ResizeToolDialog(QDialog):
    """Multi-mode image resize dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Resize Tool")
        self.setMinimumSize(700, 500)
        self._files = []
        self._output_dir = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ---- File selection ----
        fl = QHBoxLayout()
        fl.addWidget(QLabel("Images:"))
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(80)
        fl.addWidget(self._file_list)
        btn_add = QPushButton("+"); btn_add.setFixedWidth(28)
        btn_add.clicked.connect(self._add_files)
        fl.addWidget(btn_add)
        btn_clr = QPushButton("\u00d7"); btn_clr.setFixedWidth(28)
        btn_clr.setStyleSheet("color: #e74c3c; font-weight: bold;")
        btn_clr.clicked.connect(self._clear_files)
        fl.addWidget(btn_clr)
        layout.addLayout(fl)

        # ---- Output directory ----
        ol = QHBoxLayout()
        ol.addWidget(QLabel("Output:"))
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Select output folder...")
        self._out_edit.setReadOnly(True)
        ol.addWidget(self._out_edit)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._choose_output)
        ol.addWidget(btn_out)
        layout.addLayout(ol)

        # ---- Mode tabs ----
        self._tabs = QTabWidget()

        # Tab 1: Fixed Size
        tab1 = QWidget()
        f1 = QFormLayout(tab1)
        self._fw_w = QSpinBox(); self._fw_w.setRange(1, 99999); self._fw_w.setValue(256)
        f1.addRow("Width:", self._fw_w)
        self._fw_h = QSpinBox(); self._fw_h.setRange(1, 99999); self._fw_h.setValue(256)
        f1.addRow("Height:", self._fw_h)
        self._fw_keep_ar = QCheckBox("Keep aspect ratio (fit within WxH)")
        self._fw_keep_ar.setChecked(True)
        f1.addRow(self._fw_keep_ar)
        self._tabs.addTab(tab1, "Fixed Size")

        # Tab 2: Scale Factor
        tab2 = QWidget()
        f2 = QFormLayout(tab2)
        self._sc_pct = QSpinBox(); self._sc_pct.setRange(10, 500); self._sc_pct.setValue(50)
        self._sc_pct.setSuffix(" %")
        f2.addRow("Scale:", self._sc_pct)
        self._sc_label = QLabel("Original: ? x ?")
        f2.addRow(self._sc_label)
        self._tabs.addTab(tab2, "Scale")

        # Tab 3: Max Dimension
        tab3 = QWidget()
        f3 = QFormLayout(tab3)
        self._md_max = QSpinBox(); self._md_max.setRange(8, 99999); self._md_max.setValue(512)
        self._md_max.setSuffix(" px")
        f3.addRow("Max side:", self._md_max)
        self._md_label = QLabel("Longest side \u2264 N px, keep aspect ratio")
        f3.addRow(self._md_label)
        self._tabs.addTab(tab3, "Max Dim")

        layout.addWidget(self._tabs)

        # ---- Progress + Process button ----
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        bl = QHBoxLayout()
        bl.addStretch()
        self._preview_label = QLabel("")
        bl.addWidget(self._preview_label)
        btn_process = QPushButton("Process")
        btn_process.setFixedWidth(100)
        btn_process.setStyleSheet(
            "QPushButton { background-color: #2C5F8A; color: white; font-weight: bold; padding: 6px; }"
        )
        btn_process.clicked.connect(self._process)
        bl.addWidget(btn_process)
        layout.addLayout(bl)

    # ---- File management ----
    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All (*.*)"
        )
        for f in files:
            if f not in self._files:
                self._files.append(f)
                self._file_list.addItem(os.path.basename(f))
        self._update_preview()

    def _clear_files(self):
        self._files.clear()
        self._file_list.clear()
        self._update_preview()

    def _choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_dir = d
            self._out_edit.setText(d)

    def _update_preview(self):
        if not self._files:
            self._preview_label.setText("")
            return
        # Show first image's size as reference
        try:
            img = cv2.imdecode(np.fromfile(self._files[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                tab = self._tabs.currentIndex()
                if tab == 0:
                    nw, nh = self._fw_w.value(), self._fw_h.value()
                    if self._fw_keep_ar.isChecked():
                        scale = min(nw / w, nh / h)
                        nw, nh = int(w * scale), int(h * scale)
                    self._preview_label.setText(f"{w}x{h} \u2192 {nw}x{nh}")
                elif tab == 1:
                    pct = self._sc_pct.value() / 100.0
                    nw, nh = int(w * pct), int(h * pct)
                    self._preview_label.setText(f"{w}x{h} \u2192 {nw}x{nh} ({self._sc_pct.value()}%)")
                elif tab == 2:
                    max_dim = self._md_max.value()
                    scale = max_dim / max(w, h)
                    nw, nh = int(w * scale), int(h * scale)
                    self._preview_label.setText(f"{w}x{h} \u2192 {nw}x{nh}")
        except Exception:
            self._preview_label.setText("(preview error)")

    # ---- Processing ----
    def _process(self):
        if not self._files:
            QMessageBox.warning(self, "Warning", "Please add images first")
            return
        if not self._output_dir:
            QMessageBox.warning(self, "Warning", "Please select output directory")
            return

        tab = self._tabs.currentIndex()
        self._progress.setVisible(True)
        self._progress.setMaximum(len(self._files))
        self._progress.setValue(0)

        total = 0
        errors = []
        for idx, path in enumerate(self._files):
            try:
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None:
                    errors.append(os.path.basename(path))
                    continue
                h, w = img.shape[:2]

                if tab == 0:  # Fixed Size
                    nw, nh = self._fw_w.value(), self._fw_h.value()
                    if self._fw_keep_ar.isChecked():
                        scale = min(nw / w, nh / h)
                        nw, nh = int(w * scale), int(h * scale)
                elif tab == 1:  # Scale
                    pct = self._sc_pct.value() / 100.0
                    nw, nh = int(w * pct), int(h * pct)
                else:  # Max Dimension
                    max_dim = self._md_max.value()
                    scale = max_dim / max(w, h)
                    nw, nh = int(w * scale), int(h * scale)

                # Ensure at least 1 pixel
                nw = max(1, nw)
                nh = max(1, nh)

                # Choose interpolation
                if nw * nh > w * h:
                    interp = cv2.INTER_CUBIC
                else:
                    interp = cv2.INTER_AREA

                resized = cv2.resize(img, (nw, nh), interpolation=interp)

                base = os.path.splitext(os.path.basename(path))[0]
                # Keep original extension
                ext = os.path.splitext(path)[1].lower()
                out_name = f"{base}_{nw}x{nh}{ext}"
                out_path = os.path.join(self._output_dir, out_name)

                _, buf = cv2.imencode(ext, resized)
                buf.tofile(out_path)
                total += 1
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

            self._progress.setValue(idx + 1)

        self._progress.setVisible(False)
        msg = f"Resized {total} images saved to:\n{self._output_dir}"
        if errors:
            msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:10])
        QMessageBox.information(self, "Done", msg)
