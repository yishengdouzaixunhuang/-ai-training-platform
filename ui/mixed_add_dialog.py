# -*- coding: utf-8 -*-
"""Mixed Classification Add Images Dialog."""

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QListWidget, QListWidgetItem, QFileDialog, QFrame, QSizePolicy
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

STYLE = """
QDialog { background-color: #ffffff; }
QListWidget { border: 1px solid #d0d0d0; border-radius: 2px; background-color: #ffffff; font-size: 12px; color: #1a1a1a; }
QListWidget::item { padding: 3px 6px; border-bottom: 1px solid #f0f0f0; }
QListWidget::item:hover { background-color: #e8f0fe; }
QPushButton { border: 1px solid #c0c0c0; border-radius: 4px; padding: 6px 14px; background-color: #f5f5f5; font-size: 13px; color: #333333; }
QPushButton:hover { background-color: #e0e0e0; }
QPushButton#ok_btn { background-color: #1a73e8; color: white; border: none; font-weight: bold; }
QPushButton#ok_btn:hover { background-color: #1557b0; }
QPushButton#ok_btn:disabled { background-color: #a0c4f0; }
QPushButton#cancel_btn { background-color: transparent; border: 1px solid #c0c0c0; }
QCheckBox { font-size: 13px; color: #333333; }
QLabel#col_header { font-size: 14px; font-weight: bold; color: #333333; }
QLabel#error_label { font-size: 12px; color: #d93025; padding: 4px 0; }
QLabel#count_label { font-size: 13px; font-weight: bold; color: #666666; }
QFrame#placeholder { background-color: #f5f5f5; border: 1px dashed #cccccc; border-radius: 4px; }
"""

class MixedAddImagesDialog(QDialog):
    """Dialog for loading paired grayscale + height map images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Images - Mixed Classification")
        self.setMinimumSize(860, 600)
        self.resize(960, 640)
        self.setStyleSheet(STYLE)
        self._gray_files = []
        self._height_files = []
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(16, 12, 16, 16)

        top = QHBoxLayout()
        self._bind_check = QCheckBox("Bind View")
        self._bind_check.setChecked(True)
        self._bind_check.setToolTip("When checked, loading one column auto-pairs the other")
        top.addWidget(self._bind_check)
        top.addStretch()
        la = QPushButton("Load All Images")
        la.clicked.connect(self._load_all)
        top.addWidget(la)
        root.addLayout(top)

        hdr = QHBoxLayout(); hdr.setSpacing(0)
        self._gc_lbl = QLabel("0"); self._gc_lbl.setObjectName("count_label")
        self._hc_lbl = QLabel("0"); self._hc_lbl.setObjectName("count_label")
        hdr.addLayout(self._mk_hdr("Grayscale", self._on_gray_file, self._on_gray_dir, self._on_clr_gray, self._gc_lbl), 1)
        hdr.addSpacing(12)
        hdr.addLayout(self._mk_hdr("Height Map", self._on_hgt_file, self._on_hgt_dir, self._on_clr_hgt, self._hc_lbl), 1)
        root.addLayout(hdr)

        lst = QHBoxLayout(); lst.setSpacing(0)
        self._gl = QListWidget(); self._gl.setToolTip("Grayscale images")
        self._gl.itemEntered.connect(self._on_hover)
        self._hl = QListWidget(); self._hl.setToolTip("Height maps")
        self._hl.itemEntered.connect(self._on_hover)
        lst.addWidget(self._gl, 1); lst.addSpacing(12); lst.addWidget(self._hl, 1)
        root.addLayout(lst, 1)

        ph = QHBoxLayout(); ph.setSpacing(0)
        p1 = QFrame(); p1.setObjectName("placeholder"); p1.setMinimumHeight(32)
        p1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        p2 = QFrame(); p2.setObjectName("placeholder"); p2.setMinimumHeight(32)
        p2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        ph.addWidget(p1, 1); ph.addSpacing(12); ph.addWidget(p2, 1)
        root.addLayout(ph)

        self._err = QLabel(""); self._err.setObjectName("error_label"); self._err.setVisible(False)
        root.addWidget(self._err)

        btns = QHBoxLayout(); btns.addStretch()
        self._ok = QPushButton("OK"); self._ok.setObjectName("ok_btn"); self._ok.setMinimumWidth(90)
        self._ok.clicked.connect(self._on_ok)
        cancel = QPushButton("Cancel"); cancel.setObjectName("cancel_btn"); cancel.setMinimumWidth(90)
        cancel.clicked.connect(self.reject)
        btns.addWidget(self._ok); btns.addWidget(cancel)
        root.addLayout(btns)
        self._upd_ok()

    def _mk_hdr(self, title, fcb, dcb, clcb, cnt):
        r = QHBoxLayout(); r.setSpacing(4)
        lb = QLabel(title); lb.setObjectName("col_header")
        r.addWidget(lb); r.addStretch()
        icons = [("\U0001f4c2", fcb), ("\U0001f4c1", dcb), ("\U0001f5d1\ufe0f", clcb)]
        tips = ["Load single " + title, "Load " + title + " folder", "Clear " + title]
        for (ic, cb), tip in zip(icons, tips):
            b = QPushButton(ic); b.setFixedSize(32, 26); b.setToolTip(tip)
            if ic.startswith("\U0001f5d1"):
                b.setStyleSheet("QPushButton { color: #d93025; } QPushButton:hover { background-color: #fce8e6; }")
            b.clicked.connect(cb); r.addWidget(b)
        r.addWidget(cnt)
        return r

    def _on_gray_file(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Select Grayscale", "", "Images (*.bmp *.png *.jpg *.jpeg);;All (*)")
        if fs: self._add_g(fs)
    def _on_gray_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Grayscale Folder")
        if d: self._add_g(self._scan(d, {".bmp",".png",".jpg",".jpeg"}))
    def _on_hgt_file(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Select Height Maps", "", "Height Maps (*.tif *.tiff);;All (*)")
        if fs: self._add_h(fs)
    def _on_hgt_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Height Map Folder")
        if d: self._add_h(self._scan(d, {".tif",".tiff"}))
    def _load_all(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder with Both Types")
        if not d: return
        g, h = [], []
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if not os.path.isfile(fp): continue
            e = os.path.splitext(f)[1].lower()
            if e in (".bmp",".png",".jpg",".jpeg"): g.append(fp)
            elif e in (".tif",".tiff"): h.append(fp)
        if self._bind_check.isChecked(): g, h = self._pair(g, h)
        self._gray_files = g; self._height_files = h
        self._rf(); self._val()
    def _add_g(self, fs):
        g = self._gray_files + [f for f in fs if f not in self._gray_files]
        if self._bind_check.isChecked():
            g, h = self._pair(g, self._height_files)
            self._gray_files = g; self._height_files = h
        else: self._gray_files = g
        self._rf(); self._val()
    def _add_h(self, fs):
        h = self._height_files + [f for f in fs if f not in self._height_files]
        if self._bind_check.isChecked():
            g, h = self._pair(self._gray_files, h)
            self._gray_files = g; self._height_files = h
        else: self._height_files = h
        self._rf(); self._val()
    @staticmethod
    def _scan(d, exts):
        return sorted([os.path.join(d, f) for f in os.listdir(d)
                       if os.path.isfile(os.path.join(d, f)) and os.path.splitext(f)[1].lower() in exts])
    @staticmethod
    def _pair(g, h):
        hm = {os.path.splitext(os.path.basename(p))[0]: p for p in h}
        pg, ph = [], []
        for gp in g:
            b = os.path.splitext(os.path.basename(gp))[0]
            if b in hm: pg.append(gp); ph.append(hm[b])
        return pg, ph
    def _rf(self):
        self._gl.clear(); self._hl.clear()
        n = max(len(self._gray_files), len(self._height_files))
        for i in range(n):
            if i < len(self._gray_files):
                it = QListWidgetItem(str(i+1) + ". " + os.path.basename(self._gray_files[i]))
                it.setData(Qt.UserRole, self._gray_files[i]); self._gl.addItem(it)
            else:
                it = QListWidgetItem(str(i+1) + ". -"); it.setForeground(QColor("#cccccc")); self._gl.addItem(it)
            if i < len(self._height_files):
                it = QListWidgetItem(str(i+1) + ". " + os.path.basename(self._height_files[i]))
                it.setData(Qt.UserRole, self._height_files[i]); self._hl.addItem(it)
            else:
                it = QListWidgetItem(str(i+1) + ". -"); it.setForeground(QColor("#cccccc")); self._hl.addItem(it)
        self._gc_lbl.setText(str(len(self._gray_files)))
        self._hc_lbl.setText(str(len(self._height_files)))
    def _on_hover(self, item):
        p = item.data(Qt.UserRole)
        if p: item.setToolTip(p)
    def _on_clr_gray(self):
        self._gray_files = []; self._rf(); self._val()
    def _on_clr_hgt(self):
        self._height_files = []; self._rf(); self._val()
    def _val(self):
        gc, hc = len(self._gray_files), len(self._height_files)
        if gc == 0 and hc == 0: self._err.setVisible(False)
        elif gc != hc:
            self._err.setText("ERROR: Grayscale(%d) vs Height(%d) mismatch!" % (gc, hc))
            self._err.setVisible(True)
        else: self._err.setVisible(False)
        self._upd_ok()
    def _upd_ok(self):
        gc, hc = len(self._gray_files), len(self._height_files)
        self._ok.setEnabled(gc > 0 and gc == hc)
    def _on_ok(self):
        if len(self._gray_files) != len(self._height_files) or len(self._gray_files) == 0: return
        self.accept()
    def get_paired_paths(self):
        return list(self._gray_files), list(self._height_files)
