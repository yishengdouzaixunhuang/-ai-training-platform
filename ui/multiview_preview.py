# -*- coding: utf-8 -*-
"""多视图并排预览（对标友商 iSense 的 3*1 横排预览）

核心行为：
  * 第 0 格是**原有的 AnnotationCanvas**（标注照旧可用），其余格是参考视图
  * **三个窗口全部可缩放，任一窗口滚轮缩放，其余两个同步**
  * 同步的是"画布当前显示的图像区域"，所以三个视图**像素级对齐**同一块区域
    （而不是各自 fit 到窗口 —— 那样三个视图会各看各的，无法比对）
  * 单图模式只显示画布，布局栏隐藏，外观与改动前一致

同步原理：
  AnnotationCanvas 的绘制变换是
      dw = iw * zoom_level
      x  = (canvas.width() - dw)/2 + pan_offset_x
  反解出"画布中心对应的图像坐标"，再让每个参考面板把该图像坐标放到自己中心，
  即可保证三个窗口看到同一区域。每 120ms 轮询一次（画布缩放入口很多，
  轮询比逐个挂钩子更稳、且不用改动画布代码）。
"""
from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt, QRectF, QEvent, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QSizePolicy, QVBoxLayout, QWidget)

LAYOUT_MODES = ["自动（N*1）", "1*1（仅主视图）", "2*1", "3*1", "2*2", "4*1"]
_BG = "#0A0E17"
# 与 AnnotationCanvas.wheelEvent 完全一致的滚轮倍率。
# 两边倍率/锚点不一致时，在参考窗滚动会感觉"跳"（画布锚光标、参考窗锚中心）。
WHEEL_FACTOR = 1.12
# 单图模式下画布自己的底色（原样保留）
_CANVAS_BG = "#282828"
# 三个窗口的标题条高度（参考窗和主视图窗口必须一致，几何才能对齐）
_HEADER_H = 18


class _MainPanel(QFrame):
    """主视图窗口（window 1）的外框。

    为什么 window 1 也要套一层外框：参考窗有"标题条 + 边距"，主画布原来直接铺在格子
    里、什么都没有，于是两者的"可视图区"一个高一个矮——同样的缩放和中心点画出来，
    window 1 的图就比 window 2/3 高出半个标题条（实测 7px，用户一眼就看出来了）。
    套上**一模一样**的边距和标题高度后，三窗视图矩形完全重合，图像自然对齐。

    顺带：状态信息那行字改到顶部信息栏显示（画布叠加层关掉），不再压住图像左下角。
    """

    def __init__(self, title_text="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border:1px solid #2A3A5C; border-radius:4px; }")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(2, 2, 2, 2)
        self._lay.setSpacing(2)
        self.title = QLabel(title_text)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFixedHeight(_HEADER_H)
        self.title.setStyleSheet("color:#00E5FF; font-size:12px; border:none;")
        self._lay.addWidget(self.title)

    def set_body(self, w):
        self._lay.addWidget(w, 1)

    def detach_body(self, w):
        """把 body 摘出来（外框要销毁时，绝不能连带把画布一起删掉）。"""
        try:
            self._lay.removeWidget(w)
        except Exception:
            pass
        w.setParent(None)

# 解码缓存：切图时三个窗口会反复取同一批全分辨率图，
# 每次都 QPixmap(path) 重新解码 1450x3740 是"切图要等 1 秒"的主因之一。
_PIX_CACHE = {}
_PIX_CACHE_MAX = 24


def _load_pixmap(path):
    if not path:
        return None
    pm = _PIX_CACHE.get(path)
    if pm is not None:
        return pm
    pm = QPixmap(path)
    if len(_PIX_CACHE) >= _PIX_CACHE_MAX:
        _PIX_CACHE.clear()
    _PIX_CACHE[path] = pm
    return pm


LAST_OV_ERR = None      # 诊断用：记录 build_mask_overlay_rgba 最后一次失败原因


def build_mask_overlay_rgba(mask, show_classes=True, show_ignore=True):
    """从掩膜生成**仅掩膜**的 RGBA 叠加图（背景全透明）。

    ⚠️ 不能用主画布的 `_cached_overlay_qimage`！那张图是
    「**漫反射原图** + 掩膜」的合成，把它推给参考窗会让形状图1/形状图2
    也显示成漫反射（已踩过这个坑）。
    参考窗必须：各自画自己的视图 + 叠加这张透明掩膜。

    show_classes / show_ignore：分别对应 Ctrl+Space（缺陷类别）与 Ctrl+Alt（忽略区）
    两个独立开关 —— 参考窗必须跟主画布用同一组开关，否则又会出现"窗口之间不一致"。

    实现统一挪到 `core/mask_overlay.py`（主画布与参考窗共用一份）。
    """
    from core.mask_overlay import build_overlay_rgba
    rgba = build_overlay_rgba(mask, show_classes=show_classes, show_ignore=show_ignore)
    global LAST_OV_ERR
    if rgba is None:
        LAST_OV_ERR = "build_overlay_rgba 返回 None（mask 为 None 或形状异常）"
    else:
        LAST_OV_ERR = None
    return rgba


class _RefPanel(QFrame):
    """只读参考视图：与主画布同步显示同一图像区域，滚轮也能缩放。"""

    def __init__(self, on_wheel=None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border:1px solid #2A3A5C; border-radius:4px; }")
        self._on_wheel = on_wheel
        self._on_pan = None       # 由 MultiViewPreview 绑定：拖动平移画布
        self.setFocusPolicy(Qt.StrongFocus)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.title = QLabel("")
        self.title.setAlignment(Qt.AlignCenter)
        # 标题条高度写死：主视图窗口(_MainPanel)用同样的高度，
        # 三窗的"视图矩形"才能完全重合（否则参考窗的图会比主窗低半个标题条）。
        self.title.setFixedHeight(_HEADER_H)
        self.title.setStyleSheet("color:#00E5FF; font-size:12px; border:none;")
        lay.addWidget(self.title)
        self.view = QLabel("")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(40, 40)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setStyleSheet(f"border:none; color:#55607A; background:{_BG};")
        self.view.installEventFilter(self)      # 在图上滚动也要能缩放
        lay.addWidget(self.view, 1)
        self._src = None          # 本视图自己的原图（形状图就是形状图，绝不能换成别的）
        self._ov = None           # **仅掩膜**的 RGBA 叠加层（透明背景）
        self._ov_tag = None
        self._state = None        # 上次渲染用的 (zoom, cx, cy, w, h)
        self._buf = None
        self._dirty = True        # 需要重画（尺寸/图像变化时置位）
        self._drag = None
        self.setCursor(Qt.OpenHandCursor)   # 提示可拖动

    # ---- 数据 ----
    def set_image(self, path, name):
        self.title.setText(name)
        self._src = _load_pixmap(path)
        self._state = None
        self._dirty = True
        if self._src is None or self._src.isNull():
            self.view.setPixmap(QPixmap())
            self.view.setText("加载失败")
        else:
            self.view.setText("")

    def needs_render(self):
        """当前面板尺寸与上次渲染尺寸不一致 -> 需要重画。

        不依赖 resizeEvent 派发（隐藏控件/未 show 时 Qt 可能不派发，实测此时
        会一直停在"1 大 2 小"），直接比尺寸最可靠。
        """
        vw, vh = self.view.width(), self.view.height()
        return (self._state is None or self._state[3] != vw or self._state[4] != vh)

    def set_overlay(self, qimg, tag=None):
        """设置**仅掩膜**的透明叠加层。

        注意：这里只接收"掩膜层"（背景透明）；各面板仍画**各自的视图原图**。
        早期版本错误地把主画布的「漫反射+掩膜」合成图推给所有参考窗，
        导致形状图1/形状图2 被显示成漫反射。
        """
        if qimg is None:
            if self._ov is not None:
                self._ov = None
                self._ov_tag = None
                self._state = None
                self._dirty = True
            return
        if tag is not None and tag == self._ov_tag and self._ov is not None:
            return
        try:
            self._ov = QPixmap.fromImage(qimg)
        except Exception:
            return
        self._ov_tag = tag
        self._state = None
        self._dirty = True

    # ---- 同步渲染 ----
    def render_view(self, zoom, cx_img, cy_img):
        """把"图像坐标 (cx_img, cy_img)"这一点放到本面板中心，按 zoom 缩放。"""
        src = self._src
        if src is None or src.isNull() or zoom <= 0:
            return
        vw, vh = self.view.width(), self.view.height()
        if vw < 8 or vh < 8:
            return
        key = (round(zoom, 6), round(cx_img, 2), round(cy_img, 2), vw, vh,
               self._ov_tag if self._ov is not None else None)
        if key == self._state:
            return
        self._state = key
        iw, ih = src.width(), src.height()
        x = vw / 2.0 - cx_img * zoom
        y = vh / 2.0 - cy_img * zoom
        if self._buf is None or self._buf.size() != self.view.size():
            self._buf = QPixmap(self.view.size())
        self._buf.fill(QColor(_BG))
        # 只把"可见的那块图像"缩放绘制。
        # 早期版本每次都整张 4K 图缩放，滚轮放大时开销大、掉帧，手感发跳。
        sx = max(0.0, -x / zoom)
        sy = max(0.0, -y / zoom)
        sw = min(iw - sx, vw / zoom)
        sh = min(ih - sy, vh / zoom)
        if sw <= 0 or sh <= 0:
            self.view.setPixmap(self._buf)
            return
        p = QPainter(self._buf)
        p.setRenderHint(QPainter.SmoothPixmapTransform, zoom < 1.0)
        # ⚠️ 目标矩形取整：三个窗口的列宽可能是 146/147/147 这种"差 1px"的状态，
        #    不取整时每个窗口的采样相位都不同 —— 斜纹叠加层会被采样成完全不同的样子
        #    （实测同一忽略区，窗口2 只画出 1103 个像素、窗口3 画出 23291 个，
        #     用户看到的就是"窗口2 不显示标注"）。取整后三个窗口同相位。
        dx, dy = float(round(x + sx * zoom)), float(round(y + sy * zoom))
        dw, dh2 = float(max(1.0, round(sw * zoom))), float(max(1.0, round(sh * zoom)))
        dst = QRectF(dx, dy, dw, dh2)
        src_r = QRectF(sx, sy, sw, sh)
        # ① 先画本视图自己的原图
        p.drawPixmap(dst, src, src_r)
        # ② 再叠加"仅掩膜"的透明层（同一变换 -> 与主画布逐像素对齐）
        if self._ov is not None and not self._ov.isNull():
            p.drawPixmap(dst, self._ov, src_r)
        p.end()
        self.view.setPixmap(self._buf)
        self._dirty = False

    # ---- 交互：任一窗口都能滚轮缩放，且锚定光标（与主画布手感一致）----
    def _wheel_pos_in_view(self, src, e):
        """把滚轮事件坐标统一换算到 self.view 的坐标系。"""
        try:
            gp = src.mapToGlobal(e.pos())
            return self.view.mapFromGlobal(gp)
        except Exception:
            return e.pos()

    def wheelEvent(self, e):
        if self._on_wheel is not None:
            self._on_wheel(e.angleDelta().y(), self._wheel_pos_in_view(self, e))
            e.accept()
            return
        super().wheelEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._state = None        # 尺寸变了要重画
        self._dirty = True

    # ---- 拖动平移（三个窗口都能拖，且同步）----
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._on_pan is not None:
            self._drag = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag is not None and self._on_pan is not None:
            d = e.pos() - self._drag
            self._drag = e.pos()
            self._on_pan(d)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self._drag = None
            self.setCursor(Qt.OpenHandCursor)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # ---- 事件转发：在图上拖动也要能平移 ----
    def eventFilter(self, obj, ev):
        if obj is self.view and self._on_wheel is not None:
            t = ev.type()
            if t == QEvent.Wheel:
                self._on_wheel(ev.angleDelta().y(), self._wheel_pos_in_view(self.view, ev))
                ev.accept()
                return True
            if self._on_pan is not None and t in (QEvent.MouseButtonPress,
                                                  QEvent.MouseMove,
                                                  QEvent.MouseButtonRelease):
                self.event(ev)          # 交给本面板的鼠标处理（坐标已按 view 平移）
                ev.accept()
                return True
        return super().eventFilter(obj, ev)


class MultiViewPreview(QWidget):
    """N 视图并排预览容器（viewer stack 的 index 0）。"""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self._items = []
        self._panels = []
        self._last_sig = None
        self._build()
        # 轮询计时器在 _build() 里创建（装完画布事件过滤器之后），此处不要再建一个，
        # 否则会同时跑两个定时器做重复同步。

    # ────────── 构建 ──────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self.bar = QWidget()
        bar = QHBoxLayout(self.bar)
        bar.setContentsMargins(6, 2, 6, 0)
        bar.setSpacing(6)
        lb = QLabel("视图布局")
        lb.setStyleSheet("color:#8A93A6; font-size:12px;")
        bar.addWidget(lb)
        self.combo = QComboBox()
        self.combo.addItems(LAYOUT_MODES)
        self.combo.setFixedWidth(150)
        self.combo.setToolTip("多视图工程默认横排；三个窗口缩放自动同步")
        self.combo.currentIndexChanged.connect(lambda *_: self.apply_layout())
        bar.addWidget(self.combo)
        self.zoom_lbl = QLabel("")
        self.zoom_lbl.setStyleSheet("color:#00E5FF; font-size:12px;")
        bar.addWidget(self.zoom_lbl)
        self.hint = QLabel("")
        self.hint.setStyleSheet("color:#8A93A6; font-size:12px;")
        bar.addWidget(self.hint)
        # 状态信息（原来那行"1175x3739 | Pan | Class: ? | ..."）改到这条顶栏显示：
        # 顶栏本来就是多视图模式独占的一整行，横着放得下，且一点不占图像区 ——
        # 比原来压在 window 1 图像左下角强，三窗的图也就能完全对齐了。
        self.info_lbl = QLabel("")
        self.info_lbl.setStyleSheet("color:#C8D2E0; font-size:12px;")
        # 文本很长时不要撑宽顶栏（宁可裁掉尾巴），否则会把三个窗口挤窄
        self.info_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.info_lbl.setMinimumWidth(0)
        self.info_lbl.setToolTip("当前主视图状态（图像尺寸 / 工具 / 类别 / 标注开关 / 缩放 / 笔宽）\n"
                                 "快捷键：Ctrl+Space 显示/隐藏缺陷类别　Ctrl+Alt（或 Ctrl+Alt+I）显示/隐藏忽略区")
        bar.addWidget(self.info_lbl, 1)
        root.addWidget(self.bar)

        self.host = QWidget()
        self.grid = QGridLayout(self.host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        root.addWidget(self.host, 1)

        self.bar.setVisible(False)
        # 主视图外框（多视图模式下用来对齐 window 1 与 2/3 的视图矩形），单图模式下为 None
        self._main_panel = None
        self.grid.addWidget(self.canvas, 0, 0)
        # 即时同步（治本）：画布自己知道自己什么时候变了，直接发信号推过来，
        # 不再靠轮询"猜"——轮询最快也要等下一个周期，就是那一下肉眼可见的延迟。
        self.canvas.installEventFilter(self)
        try:
            self.canvas.view_changed.connect(self._on_canvas_view_changed)
        except Exception as e:
            # 老版本画布没有这个信号时退回纯轮询（不能让缺少信号导致崩）
            print(f"[多视图预览] 画布无 view_changed 信号，退回轮询同步: {e}")
        self._sync_timer = QTimer(self)
        # 轮询只兜底：拖动平移这类高频连续动作不逐个信号推送（每移动一次就重画两个窗
        # 反而更卡），交给 40ms 轮询跟进。数值比较开销可忽略。
        self._sync_timer.setInterval(40)
        self._sync_timer.timeout.connect(self._sync_from_canvas)
        self._sync_timer.start()

    def _on_canvas_view_changed(self):
        """画布 zoom/pan/标注开关变化 -> 立即同步（不等轮询）。"""
        self._sync_from_canvas(force=True)

    def _set_status_overlay(self, enabled):
        try:
            self.canvas.set_status_overlay(enabled)
        except Exception as e:
            print(f"[多视图预览] 画布不支持 set_status_overlay（{type(e).__name__}: {e}），状态行仍叠加在图像上")

    def _set_canvas_bg(self, color):
        try:
            self.canvas.set_bg_color(color)
        except Exception:
            pass

    def _refresh_info(self):
        """把画布的状态信息同步到顶部信息栏（文本没变就不动控件）。"""
        lb = getattr(self, "info_lbl", None)
        if lb is None:
            return
        try:
            txt = self.canvas.status_info_text()
        except Exception:
            return                      # 老版本画布没有这个方法：保持空白，不报错
        if txt != getattr(self, "_info_txt", None):
            self._info_txt = txt
            lb.setText(txt)

    def eventFilter(self, obj, ev):
        """把任意面板上的 Ctrl+Space / Ctrl+Alt 都变成同一次显示开关。

        画布自己会处理这两个按键（会发 view_changed），这里只补一次同步即可；
        参考窗是 QFrame，按键默认冒泡到父级、永远到不了画布，所以在参考窗上
        按下时由我们直接调用画布的开关方法。
        """
        try:
            if ev.type() == QEvent.KeyPress and not ev.isAutoRepeat():
                mods = ev.modifiers()
                is_space_ctrl = (ev.key() == Qt.Key_Space
                                 and mods == Qt.ControlModifier)
                is_ctrl_alt = (not (mods & Qt.GroupSwitchModifier)
                               and ((ev.key() == Qt.Key_Alt and (mods & Qt.ControlModifier))
                                    or (ev.key() == Qt.Key_Control and (mods & Qt.AltModifier))
                                    or (ev.key() == Qt.Key_I and (mods & Qt.ControlModifier)
                                        and (mods & Qt.AltModifier))))
                if is_space_ctrl or is_ctrl_alt:
                    if obj is self.canvas:
                        QTimer.singleShot(0, lambda: self._sync_from_canvas(force=True))
                    elif obj in self._panels:
                        if is_space_ctrl:
                            self.canvas.toggle_annotation_visibility()
                        else:
                            self.canvas.toggle_ignore_visibility()
                        return True
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    # ────────── 对外接口 ──────────
    def clear_views(self):
        self._items = []
        self.bar.setVisible(False)
        self.apply_layout()

    def set_views(self, items):
        """items = [(path, label), ...]；第 0 个已在主画布上。

        关键：**布局没变时只换图，不重建控件**。
        早期版本每次都走 apply_layout（销毁+重建面板），重建瞬间控件还没落到最终尺寸，
        就会先渲染成小图，要等下一次轮询才补成大图 —— 表现为切图时"1 大 2 小"停留一下。
        """
        items = list(items or [])
        prev = self._items
        self._items = items
        if len(items) <= 1:
            return self.clear_views()
        self.bar.setVisible(True)
        self.hint.setText(f"{len(items)} 视图：" + " / ".join(n for _, n in items))
        reuse = (len(prev) == len(items) and len(self._panels) == len(items) - 1
                 and all(p.view.width() > 8 for p in self._panels))
        if reuse:
            for p, (path, name) in zip(self._panels, items[1:]):
                p.set_image(path, name)
            self._sync_from_canvas(force=True)
        else:
            self.apply_layout()
            self._sync_from_canvas(force=True)

    def is_multiview(self):
        return len(self._items) > 1

    # ────────── 布局 ──────────
    def _grid_shape(self, n):
        m = self.combo.currentText()
        if m.startswith("1*1"):
            return 1, 1
        if m.startswith("2*1"):
            return min(2, n), 1
        if m.startswith("3*1"):
            return min(3, n), 1
        if m.startswith("2*2"):
            return min(2, n), 2
        if m.startswith("4*1"):
            return min(4, n), 1
        return n, 1

    def apply_layout(self):
        # 先把画布从主视图外框里摘出来：外框下面要销毁，画布绝不能跟着陪葬
        if self._main_panel is not None:
            self._main_panel.detach_body(self.canvas)
            self._main_panel.setParent(None)
            self._main_panel.deleteLater()
            self._main_panel = None
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
        for p in self._panels:
            p.setParent(None)
            p.deleteLater()
        self._panels = []

        n_ref = len(self._items) - 1
        if n_ref <= 0:
            # 单图模式：画布直接铺在格子里，状态信息仍叠加画在图像上（外观与改动前一致）
            self.grid.addWidget(self.canvas, 0, 0)
            self._set_status_overlay(True)
            self._set_canvas_bg(_CANVAS_BG)
            # 恢复画布原有的最小尺寸约束
            try:
                self.canvas.setMinimumSize(640, 480)
            except Exception:
                pass
            for c in range(1, 5):
                self.grid.setColumnStretch(c, 0)
            self.grid.setColumnStretch(0, 1)
            return
        # 多视图：给 window 1 套上和参考窗一模一样的外框（同样的边距 + 同样的标题高度），
        # 三窗"视图矩形"才会完全重合；否则主窗的图会高出半个标题条。
        _t = (self._items[0][1] if self._items and self._items[0][1] else "主视图")
        self._main_panel = _MainPanel(f"{_t}（主视图）")
        self._main_panel.set_body(self.canvas)
        self.grid.addWidget(self._main_panel, 0, 0)
        # 状态信息改由顶部信息栏显示，画布上不再叠那一行（免得压住图像左下角）
        self._set_status_overlay(False)
        # 底色也统一（画布原本是 #282828，参考窗是 #0A0E17，并排看像两个软件）
        self._set_canvas_bg(_BG)
        # 画布自带 640x480 的最小尺寸，会把第一列撑得比参考列宽。
        # 放开它，三列才能真正等宽。
        try:
            self.canvas.setMinimumSize(40, 40)
        except Exception:
            pass
        cols, rows = self._grid_shape(len(self._items))
        slots = [s for s in [(c, r) for r in range(rows) for c in range(cols)] if s != (0, 0)][:n_ref]
        for (c, r), (path, name) in zip(slots, self._items[1:]):
            panel = _RefPanel()
            # 每个面板绑定到"自己"：这样滚轮坐标才能换算成该面板内的光标位置
            panel._on_wheel = (lambda d, pos, p=panel: self._zoom_at(d, p, pos))
            # 拖动平移：三个窗口都能拖，拖任意一个其余两个同步
            panel._on_pan = (lambda d, p=panel: self._pan_by(d, p))
            panel.set_image(path, name)
            panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            panel.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # 焦点落到参考窗时按键不会传给画布，装过滤器自己接住 Ctrl+Space
            panel.installEventFilter(self)
            self._panels.append(panel)
            self.grid.addWidget(panel, r, c)
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)
        for c in range(cols, 5):
            self.grid.setColumnStretch(c, 0)
        self._last_sig = None
        self._sync_from_canvas()

    # ────────── 缩放同步 ──────────
    def _pan_by(self, delta_px, panel=None):
        """拖动平移：把面板上的像素增量换算到画布坐标后累加到画布 pan。

        画布的 pan_offset 就是屏幕像素，所以同宽时 1:1；不同宽时按比例换算。
        """
        c = self.canvas
        if c.image is None:
            return
        scale = 1.0
        try:
            if panel is not None and panel.view.width() > 0:
                scale = float(c.width()) / float(panel.view.width())
        except Exception:
            scale = 1.0
        c.pan_offset_x = float(getattr(c, "pan_offset_x", 0.0)) + delta_px.x() * scale
        c.pan_offset_y = float(getattr(c, "pan_offset_y", 0.0)) + delta_px.y() * scale
        try:
            c.update()
        except Exception:
            pass
        self._sync_from_canvas(force=True)

    def _zoom_at(self, delta, panel=None, pos=None):
        c = self.canvas
        got = self._canvas_center_in_image()
        if got is None:
            return
        z, cx, cy = got
        # ⚠️ AnnotationCanvas.image 是 **PIL** 图像：只有 .size / .width 属性，
        #    没有 .width() 方法。早期版本写成 c.image.width()，抛 AttributeError
        #    被 except 静默吞掉 -> 看着"锚定完美"，其实是根本没缩放（已踩过）。
        try:
            iw, ih = c.image.size
            iw, ih = int(iw), int(ih)
        except Exception as e:
            self._last_err = f"取图像尺寸失败: {type(e).__name__}: {e}"
            return
        factor = WHEEL_FACTOR if delta > 0 else 1.0 / WHEEL_FACTOR
        lo = float(getattr(c, "MIN_ZOOM", 0.05) or 0.05)
        hi = float(getattr(c, "MAX_ZOOM", 30.0) or 30.0)
        nz = max(lo, min(hi, z * factor))
        if abs(nz - z) < 1e-9:
            return
        actual = nz / z
        # 光标相对所在窗口中心的偏移（窗口像素）
        if panel is not None and pos is not None:
            vw, vh = panel.view.width(), panel.view.height()
            dx = pos.x() - vw / 2.0
            dy = pos.y() - vh / 2.0
        else:
            vw, vh = c.width(), c.height()
            dx = (pos.x() if pos is not None else vw / 2.0) - vw / 2.0
            dy = (pos.y() if pos is not None else vh / 2.0) - vh / 2.0
        # 光标下的图像点
        a_x = cx + dx / z
        a_y = cy + dy / z
        # 缩放后让它仍落在同一屏幕位置 -> 新的窗口中心图像坐标
        n_cx = a_x - dx / nz
        n_cy = a_y - dy / nz
        c.zoom_level = nz
        # 画布变换: cx_img = iw/2 - pan/zoom  =>  pan = (iw/2 - cx_img) * zoom
        c.pan_offset_x = (iw / 2.0 - n_cx) * nz
        c.pan_offset_y = (ih / 2.0 - n_cy) * nz
        try:
            c.update()
        except Exception:
            pass
        self._sync_from_canvas(force=True)

    def _canvas_center_in_image(self):
        c = self.canvas
        try:
            z = float(c.zoom_level)
            if z <= 0 or c.image is None:
                return None
            rect = c._image_to_widget_rect()
            cx = (c.width() / 2.0 - rect.x()) / z
            cy = (c.height() / 2.0 - rect.y()) / z
            return z, cx, cy
        except Exception:
            return None

    def _sync_from_canvas(self, force=False):
        # 顶栏状态信息每轮都刷一次（只是字符串比较，没有变化就不动控件）
        self._refresh_info()
        if not self._panels:
            return
        got = self._canvas_center_in_image()
        if got is None:
            return
        z, cx, cy = got
        sig = (round(z, 6), round(cx, 2), round(cy, 2))
        # 「缩放签名没变」不代表不用重画：切换图片/布局刚做完时控件还没落到最终尺寸，
        # 那时渲染出来的是小图；必须等面板变脏(resize/set_image)后补一次渲染，
        # 否则会一直停在"1 大 2 小"，直到用户手动缩放才恢复。
        dirty = any((getattr(p, "_dirty", True) or p.needs_render()) for p in self._panels)
        # 两个独立开关都要计入变化检测：
        #   Ctrl+Space -> canvas._show_annotation（缺陷类别）
        #   Ctrl+Alt   -> canvas._show_ignore（忽略区 __ignore__/255）
        # 少算任何一个，关掉之后参考窗都还会照旧显示。
        show_ann = bool(getattr(self.canvas, "_show_annotation", True))
        show_ign = bool(getattr(self.canvas, "_show_ignore", True))
        if not force and sig == self._last_sig and not dirty \
                and show_ann == getattr(self, "_ov_show_ann", None) \
                and show_ign == getattr(self, "_ov_show_ignore", None):
            # 掩膜可能刚被编辑过：叠加图变了也要重画（否则参考窗不显示新标注）
            try:
                self.canvas._build_overlay()
            except Exception:
                pass
            ov = getattr(self.canvas, "_cached_overlay_qimage", None)
            tag = None
            try:
                tag = ov.cacheKey() if ov is not None else None
            except Exception:
                tag = None
            if all(getattr(p, "_ov_tag", None) == tag for p in self._panels):
                return
        self._last_sig = sig
        # 掩膜叠加层：只在掩膜变化时重建。
        # 用主画布叠加图的 cacheKey 当"掩膜已变"的信号（掩膜一改，它必然重建）。
        try:
            self.canvas._build_overlay()
        except Exception:
            pass
        cq = getattr(self.canvas, "_cached_overlay_qimage", None)
        try:
            ctag = cq.cacheKey() if cq is not None else None
        except Exception:
            ctag = None
        # 重建条件要看两件事：
        #   ① 掩膜数组换对象了（切图 -> canvas.mask 重新赋值）
        #   ② 掩膜被就地编辑（canvas 的叠加图 cacheKey 变化）
        # 只看 ② 会漏掉"首轮同步时掩膜还是 None、之后再加载"的情况（已踩过：
        # 结果 _mask_qimg 一直是 None，参考窗永远不显示标注）。
        mask_obj = getattr(self.canvas, "mask", None)
        # ⚠️ 两个开关必须同时进 need，不能只放进"提前返回"判断：
        # force=True 会跳过提前返回，但 need 仍为 False 时会**跳过重建**、
        # 复用旧掩膜层，于是关掉之后参考窗照旧显示（已踩过）。
        need = (mask_obj is not getattr(self, "_mask_obj", "__unset__")
                or ctag != getattr(self, "_ov_ctag", "__unset__")
                or show_ann != getattr(self, "_ov_show_ann", "__unset__")
                or show_ign != getattr(self, "_ov_show_ignore", "__unset__"))
        if need:
            self._mask_obj = mask_obj
            self._ov_ctag = ctag
            self._ov_show_ann = show_ann
            self._ov_show_ignore = show_ign
            self._mask_qimg = None
            # Ctrl+Space / Ctrl+Alt 关掉对应内容时，参考窗也要跟着不画
            # 两个开关全关 -> 整层透明，直接不发
            if mask_obj is not None and (show_ann or show_ign):
                rgba = build_mask_overlay_rgba(mask_obj, show_classes=show_ann,
                                               show_ignore=show_ign)
                if rgba is None:
                    self._ov_err = "build_mask_overlay_rgba 返回 None"
                else:
                    try:
                        h, w = rgba.shape[:2]
                        # 用 tobytes() 保证连续内存，QImage 才不会踩到 stride
                        self._mask_qimg = QImage(rgba.data.tobytes(), w, h, 4 * w,
                                                 QImage.Format_RGBA8888).copy()
                        self._ov_err = None
                    except Exception as e:
                        self._ov_err = f"QImage: {type(e).__name__}: {e}"
            else:
                self._ov_err = ("canvas.mask 为 None" if mask_obj is None
                                else "类别与忽略区都被关掉了（Ctrl+Space / Ctrl+Alt）")
        for p in self._panels:
            # 注意：传的是**仅掩膜**的透明层，各面板仍画各自的视图原图
            p.set_overlay(self._mask_qimg, ctag)
            p.render_view(z, cx, cy)
        try:
            fit = float(getattr(self.canvas, "_fit_zoom", 1.0) or 1.0)
            self.zoom_lbl.setText(f"缩放 {int(z / fit * 100)}%（三窗同步）")
        except Exception:
            pass
