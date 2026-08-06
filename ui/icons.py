# -*- coding: utf-8 -*-
"""内嵌 SVG 图标工厂。

用 QSvgRenderer 把内嵌 SVG 渲染成 QIcon（2x 渲染保证高分屏锐利）。
- Normal/Off 态：teal 色（#2E7E8A）
- Normal/On 态（选中）：白色，配合工具栏 QSS 高亮背景
AI 徽标用 QPainter 绘制渐变圆角块。
"""
from PyQt5.QtCore import QByteArray, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer

COLOR_OFF = "#2E7E8A"  # 正常态
COLOR_ON = "#FFFFFF"   # 选中态

# 24x24 viewBox，stroke 用 currentColor 占位（渲染时替换）
_SVGS = {
    "line": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
             '<line x1="4" y1="20" x2="20" y2="4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'),
    "rect": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
             '<rect x="4.5" y="4.5" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"/></svg>'),
    "circle": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
               '<circle cx="12" cy="12" r="7.5" fill="none" stroke="currentColor" stroke-width="2"/></svg>'),
    "ellipse": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                '<ellipse cx="12" cy="12" rx="8.5" ry="5.5" fill="none" stroke="currentColor" stroke-width="2"/></svg>'),
    "polygon": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                '<polygon points="12,3.5 20.5,8 20.5,16 12,20.5 3.5,16 3.5,8" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>'),
    "brush": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
              '<path d="M3 16 C5 12.5 6.5 15.5 8.5 11.5 C10.5 7.5 11.5 12.5 13.5 8.5 C15.5 4.5 17.5 8.5 21 4" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>'),
    "eraser": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
               '<path d="M6.5 15.5 L13.5 5.5 L20 11.5 L13 20.5 L6.5 20.5 Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>'
               '<line x1="8" y1="13.5" x2="12" y2="17.5" stroke="currentColor" stroke-width="2"/></svg>'),
    "ignore": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
               '<circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/>'
               '<line x1="7.5" y1="7.5" x2="16.5" y2="16.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'),
    "pan": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            '<path d="M12 2.5 L15 6.5 L9 6.5 Z M12 21.5 L15 17.5 L9 17.5 Z M2.5 12 L6.5 9 L6.5 15 Z M21.5 12 L17.5 9 L17.5 15 Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>'),
    "undo": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
             '<path d="M9 6 L4 11 L9 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
             '<path d="M4 11 H13 A6 6 0 0 1 13 23 H9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'),
    "redo": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
             '<path d="M15 6 L20 11 L15 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
             '<path d="M20 11 H11 A6 6 0 0 0 11 23 H15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'),
    "cancel": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
               '<circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/>'
               '<line x1="9" y1="9" x2="15" y2="15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
               '<line x1="15" y1="9" x2="9" y2="15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'),
    "clear": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
              '<path d="M5 7 H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
              '<path d="M9.5 7 V4.5 H14.5 V7" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>'
              '<path d="M7.5 7 L8.5 20 H15.5 L16.5 7" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>'),
    "reload": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
               '<path d="M4.5 12 A7.5 7.5 0 0 1 18.5 6.5 M18.5 6.5 V3 M18.5 6.5 H15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
               '<path d="M19.5 12 A7.5 7.5 0 0 1 5.5 17.5 M5.5 17.5 V21 M5.5 17.5 H9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    "save": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
             '<rect x="4" y="4" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"/>'
             '<path d="M8 4 V10 H16 V4" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>'
             '<rect x="8.5" y="14" width="7" height="6" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>'),
    "prev": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
             '<path d="M14.5 5 L7.5 12 L14.5 19" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    "next": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
             '<path d="M9.5 5 L16.5 12 L9.5 19" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'),
}


def _render(svg, color, size):
    svg = svg.replace("currentColor", color)
    px = size * 2  # 2x 渲染，高分屏锐利
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    renderer.render(p, QRectF(0, 0, px, px))
    p.end()
    pm.setDevicePixelRatio(2.0)
    return pm


def make_icon(name, size=24):
    """双态图标：Off=teal，On=白色（配合工具栏选中高亮）。"""
    svg = _SVGS[name]
    icon = QIcon()
    icon.addPixmap(_render(svg, COLOR_OFF, size), QIcon.Normal, QIcon.Off)
    icon.addPixmap(_render(svg, COLOR_ON, size), QIcon.Normal, QIcon.On)
    return icon


def make_ai_icon(size=24):
    """AI 智能标注徽标（渐变蓝底白字）。"""
    px = size * 2
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    grad = QLinearGradient(0, 0, px, px)
    grad.setColorAt(0.0, QColor("#2563EB"))
    grad.setColorAt(1.0, QColor("#0EA5E9"))
    p.setBrush(grad)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(0, 0, px, px, px * 0.16, px * 0.16)
    f = QFont("Segoe UI", 0, QFont.Bold)
    f.setPixelSize(int(px * 0.52))
    p.setFont(f)
    p.setPen(QColor("#FFFFFF"))
    p.drawText(QRectF(0, 0, px, px), Qt.AlignCenter, "AI")
    p.end()
    pm.setDevicePixelRatio(2.0)
    icon = QIcon()
    icon.addPixmap(pm, QIcon.Normal, QIcon.Off)
    icon.addPixmap(pm, QIcon.Normal, QIcon.On)
    return icon