"""Color palette widget for multi-class segmentation annotation."""
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal


class ColorPalette(QWidget):
    """A grid of color swatches with class names for quick class selection."""

    class_selected = pyqtSignal(int)  # emits row index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(2)
        self._buttons = []
        self._colors = []
        self._classes = []
        self._current_idx = -1
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def set_classes(self, classes, colors):
        """Populate palette with classes and their colors.
        
        Args:
            classes: list of class name strings
            colors: list of (R,G,B) tuples
        """
        # Clear existing buttons
        for btn in self._buttons:
            btn.deleteLater()
        self._buttons.clear()
        self._colors = []
        self._classes = []
        self._current_idx = -1

        # Re-layout
        cols = 3
        for i, (name, color) in enumerate(zip(classes, colors)):
            if i == 0 and name.lower() == "background":
                # Background class - add but dimmed
                pass
            btn = QPushButton(name)
            btn.setFixedHeight(26)
            btn.setStyleSheet(self._btn_style(color, selected=False))
            btn.clicked.connect(lambda checked, idx=i: self._on_click(idx))
            btn.setCursor(Qt.PointingHandCursor)
            self._grid.addWidget(btn, i // cols, i % cols)
            self._buttons.append(btn)
            self._colors.append(color)
            self._classes.append(name)

    def _btn_style(self, color, selected=False):
        r, g, b = color
        border = "2px solid #fff" if selected else "1px solid #555"
        return (
            f"QPushButton {{"
            f"  background-color: rgb({r},{g},{b});"
            f"  color: {'#fff' if (r+g+b)/3 < 128 else '#000'};"
            f"  border: {border}; border-radius: 3px;"
            f"  font-size: 10px; font-weight: bold;"
            f"  padding: 0 4px;"
            f"}}"
        )

    def _on_click(self, idx):
        self.set_selected(idx)
        self.class_selected.emit(idx)

    def set_selected(self, idx):
        """Highlight the selected class button."""
        # Reset previous
        if 0 <= self._current_idx < len(self._buttons):
            btn = self._buttons[self._current_idx]
            color = self._colors[self._current_idx]
            btn.setStyleSheet(self._btn_style(color, selected=False))
        # Set new
        if 0 <= idx < len(self._buttons):
            self._current_idx = idx
            btn = self._buttons[idx]
            color = self._colors[idx]
            btn.setStyleSheet(self._btn_style(color, selected=True))

    def clear(self):
        """Remove all buttons."""
        for btn in self._buttons:
            btn.deleteLater()
        self._buttons.clear()
        self._colors.clear()
        self._classes.clear()
        self._current_idx = -1
