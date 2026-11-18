# -*- coding: utf-8 -*-
"""多视图并排预览（对标友商 友商平台 的 3*1 横排预览）

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
from core.i18n import tr

import numpy as np

from PyQt5.QtCore import Qt, QRectF, QEvent, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (QFrame, QGridLayout, QLabel,
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

    # 光标在本面板里的像素坐标（**主视图图像坐标系**，见 `_view_to_image`）。
    # 无坐标（出图 / 离开控件 / 映射拿不到）发 (-1, -1) 表示"清空"。
    # 只在整数像素真的变了才发：鼠标每动一下都发信号 = 让底部信息栏跟着重排，
    # 比重绘还贵（主画布那条信号定的也是这个规矩）。
    cursor_moved = pyqtSignal(int, int)

    def __init__(self, on_wheel=None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border:1px solid #2A3A5C; border-radius:4px; }")
        self._on_wheel = on_wheel
        self._on_pan = None       # 由 MultiViewPreview 绑定：拖动平移画布
        self.setFocusPolicy(Qt.StrongFocus)
        # ⚠️ **鼠标跟踪必须显式打开**，而且面板与内层 QLabel 两个都要开。
        #    QLabel 默认只在**按住鼠标键**时才产生 MouseMove —— 于是"只是把光标停在
        #    形状图1 / 形状图2 上"一个事件都没有，底部信息栏的「坐标 / RGB」
        #    就一直停在 `--`（用户 2026-10-14 报的正是这个）。
        #    主画布那边之所以"一直是对的"，是因为 AnnotationCanvas 自己开了跟踪。
        self.setMouseTracking(True)
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
        self.view.setMouseTracking(True)        # 见 __init__ 里的说明：不开就收不到 MouseMove
        self.view.installEventFilter(self)      # 在图上滚动也要能缩放
        lay.addWidget(self.view, 1)
        self._src = None          # 本视图自己的原图（形状图就是形状图，绝不能换成别的）
        self._ov = None           # **仅掩膜**的 RGBA 叠加层（透明背景）
        self._ov_tag = None
        # 标注标签（IGNORE / 类别名）——与主画布**同一份** shapes 与同一套绘制实现。
        # 早期版本参考窗只画掩膜色块、不画标签，于是"只有漫反射图有 IGNORE/Ys 字样，
        # 形状图1/形状图2 光有色块"（用户一眼就看出来了）。
        self._shapes = None
        self._label_tag = None
        self._lab_ann = True      # Ctrl+Space：缺陷类别标签
        self._lab_ign = True      # Ctrl+Alt：忽略区 IGNORE 标签
        self._state = None        # 上次渲染用的 (zoom, cx, cy, w, h, ...)
        self._buf = None
        self._dirty = True        # 需要重画（尺寸/图像变化时置位）
        self._drag = None
        # ---- 光标上报（底部「图片信息栏」的坐标 / RGB 用的就是这几个）----
        self._path = None
        self._srcimg = None       # 懒取 QImage（见 _src_image 的性能说明）
        self._hover_pt = None     # 上次**上报过**的图像坐标（去重用），None = 当前没上报
        self._hover_vpos = None   # 光标在 view 内的最后位置（换图后按它补报一次）
        self._hovering = False
        # 兜底：面板自己还没渲染过（_state 为 None）时，向容器要画布当前的中心点
        self._state_provider = None
        self.setCursor(Qt.OpenHandCursor)   # 提示可拖动

    # ---- 数据 ----
    def set_image(self, path, name):
        self.title.setText(name)
        self._path = path
        self._src = _load_pixmap(path)
        self._srcimg = None          # 换图必须让懒取的 QImage 失效
        self._state = None
        self._dirty = True
        if self._src is None or self._src.isNull():
            self.view.setPixmap(QPixmap())
            self.view.setText(tr("加载失败"))
        else:
            self.view.setText("")
            # 换图后光标还停在这一格上：按同一个窗口位置补报一次。
            # 不补的话映射（_state）刚被清空，读数要等用户再动一下鼠标才回来 ——
            # 那就又变成"看起来是空的"了。
            if self._hovering and self._hover_vpos is not None:
                self._hover_pt = None
                self._emit_hover_xy(self._hover_vpos[0], self._hover_vpos[1])

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

    def set_labels(self, shapes, tag=None, show_classes=True, show_ignore=True):
        """设置标注标签的数据源（shapes 来自**主画布当前图**的 json，与掩膜同源）。

        为什么 labels 跟掩膜同源而不是各视图自己的 json：三视图工程里标注是一份
        （掩膜就是这么来的），参考窗若去读 `xxx_te_1.json` 只会读到不存在 —— 于是
        "掩膜有、标签没有"，正是这次要修的现象。

        tag 用来做变化检测（这里传 json 路径 + mtime）；标签内容或两个开关变了都要重画。
        """
        show_classes = bool(show_classes)
        show_ignore = bool(show_ignore)
        if (tag == self._label_tag and show_classes == self._lab_ann
                and show_ignore == self._lab_ign):
            return
        self._shapes = shapes
        self._label_tag = tag
        self._lab_ann = show_classes
        self._lab_ign = show_ignore
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
               self._ov_tag if self._ov is not None else None,
               # 标签也必须进 key：否则掩膜没变时这里直接 return，
               # 改了标注/按了 Ctrl+Space/Ctrl+Alt 参考窗都不会重画（就是"看起来没反应"）。
               self._label_tag, self._lab_ann, self._lab_ign,
               id(self._shapes) if self._shapes is not None else None)
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
        # ③ 最后画标注标签（IGNORE / 类别名）
        #    与主画布共用 core/label_overlay.paint_shape_labels —— 字号公式
        #    （max(8, 10*zoom)）与配色完全一致，三窗同款；各窗变换不同而已。
        #    这一层以前缺失，就是"只有漫反射图有 IGNORE/Ys 字样"的直接原因。
        if self._shapes and (self._lab_ann or self._lab_ign):
            try:
                from core.label_overlay import paint_shape_labels
                paint_shape_labels(
                    p, self._shapes,
                    lambda ix, iy: (x + ix * zoom, y + iy * zoom),
                    zoom, show_classes=self._lab_ann, show_ignore=self._lab_ign)
            except Exception:
                pass
        p.end()
        self.view.setPixmap(self._buf)
        self._dirty = False

    # ---- 光标坐标 / 像素（底部「图片信息栏」的数据源）----
    def _view_state(self):
        """返回 (zoom, cx_img, cy_img, vw, vh)；拿不到就返回 None。"""
        vw, vh = self.view.width(), self.view.height()
        st = self._state
        if st is not None and len(st) >= 5:
            return float(st[0]), float(st[1]), float(st[2]), int(st[3]), int(st[4])
        # 还没渲染过（刚切图 / 刚布局 / 尺寸还没落定）：向容器要主画布当前的中心点。
        # 没有这条兜底的话，这段瞬间坐标就一直是 `--` —— 又回到"看起来是空的"。
        fn = self._state_provider
        if fn is None or vw < 8 or vh < 8:
            return None
        got = fn()
        if got is None:
            return None
        z, cx, cy = got
        return float(z), float(cx), float(cy), vw, vh

    def _view_to_image(self, vx, vy):
        """面板内坐标 -> **主视图图像坐标**（`render_view` 那套变换的精确反解）。

        为什么用主视图的坐标系：三窗同步干的就是"把画布中心对应的图像点搬到各自中心"
        （见模块 docstring），所以三格里同一个图像坐标指的就是同一处 —— 坐标在
        漫反射 / 形状图1 / 形状图2 上一致，这正是这个字段的意义。

        ⚠️ 反解必须和 `render_view` 里**同一个取整**（那里画的目标矩形是 round 过的）。
            直接用未取整的公式，缩小时会系统性偏 1 个像素 —— 这种偏差肉眼看不出，
            但拿它去对标注坐标就会"差一行"。
        """
        g = self._view_state()
        if g is None:
            return None
        z, cx, cy, vw, vh = g
        if z <= 0:
            return None
        x = vw / 2.0 - cx * z
        y = vh / 2.0 - cy * z
        sx = max(0.0, -x / z)
        sy = max(0.0, -y / z)
        dx = float(round(x + sx * z))
        dy = float(round(y + sy * z))
        return sx + (vx - dx) / z, sy + (vy - dy) / z

    def _src_image(self):
        """懒取本视图的 QImage（给 pixel_at 用）。

        实测 `QPixmap.toImage()` 对 1450×3940 只要 **0.1~1.5ms**（Qt5 光栅后端下
        与 pixmap 共用缓冲，几乎没有拷贝开销），所以不必担心"鼠标一动就解一张大图"。
        """
        if self._srcimg is not None:
            return self._srcimg
        src = self._src
        if src is None or src.isNull():
            return None
        try:
            self._srcimg = src.toImage()
        except Exception:
            return None
        return self._srcimg

    def pixel_at(self, x, y):
        """读本视图 (x, y) 处的像素，**一律返回 (r, g, b) 三元组**。

        参考窗 = 形状图1 / 形状图2。产品口径（维护者 2026-10-14 明确：
        "形状1和形状2都要显示RGB"）：这两格**固定按 RGB 呈现**，
        即使源文件是 L 灰度（此时 r == g == b，也是数据本来如此，不是读错）。

        为什么要这样：主画布那格（漫反射）走的是 `AnnotationCanvas.image`，而它在加载时
        就 `.convert("RGB")` 了（`mask_editor.py`）⇒ 底栏一直显示「RGB 169,169,169」。
        形状图那两格早先按**文件 mode** 走，L 灰度就显示「灰度 127」—— 于是同一条信息栏上
        两种形态并存（用户就是看到这个才提的）。现在三格**统一 RGB**。

        早先那段判定（`Image.open()` 只读文件头拿 `mode`，比"逐像素 r==g==b"猜可靠 ——
        真彩图里的灰色像素会被后者误判）已删：形态由**视图角色**定，不由文件 mode 定。

        ⚠️ 实测（客户现场 1175×3739 L 灰度 JPEG）：Qt 解出来的 r/g/b **严格相等**，
           且与 PIL 逐点完全一致 ⇒ 形状图那两格的三值相等是数据本身如此。
        """
        img = self._src_image()
        if img is None or img.isNull():
            return None
        x, y = int(x), int(y)
        if x < 0 or y < 0 or x >= img.width() or y >= img.height():
            return None
        c = img.pixelColor(x, y)
        return (c.red(), c.green(), c.blue())

    def _emit_hover_xy(self, px, py):
        """把"面板内坐标"换算成图像坐标并上报（越界 -> 上报清空）。"""
        self._hovering = True
        self._hover_vpos = (int(px), int(py))
        vw, vh = self.view.width(), self.view.height()
        if vw <= 0 or vh <= 0 or not (0 <= px < vw and 0 <= py < vh):
            self._drop_hover()
            return
        got = self._view_to_image(px, py)
        if got is None:
            self._drop_hover()
            return
        ix, iy = int(got[0]), int(got[1])
        src = self._src
        if src is not None and not src.isNull() \
                and not (0 <= ix < src.width() and 0 <= iy < src.height()):
            # 落在本视图图像之外（三图尺寸不一致时才会发生）：报"无坐标"，
            # 别报一个越界的数字 —— 那比没有更误导。
            self._drop_hover()
            return
        if (ix, iy) == self._hover_pt:
            return
        self._hover_pt = (ix, iy)
        self.cursor_moved.emit(ix, iy)

    def _drop_hover(self):
        """上报"无坐标"。只在从"有坐标"变过来时发一次，免得鼠标一出界就狂发信号。

        ⚠️ 特意**不清** `_hover_vpos`：换图后还要按这个窗口位置补报一次。
        清不清由 `_hovering` 决定。
        """
        if self._hover_pt is not None:
            self._hover_pt = None
            self.cursor_moved.emit(-1, -1)

    def leaveEvent(self, e):
        self._hovering = False
        self._drop_hover()
        super().leaveEvent(e)

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
        # 这里**不**上报坐标：面板级的 MouseMove 要么来自标题条/边距（本来就不是图像区，
        # 该走"清空"），要么是 eventFilter 从 `view` **原样转发**过来的平移事件 ——
        # 后者的 `e.pos()` 是 view 坐标，而本函数看到的面板坐标还含标题条与边距，
        # 在这里换算会**再报一个偏了一整条标题条高度（~20px）的值**，把 eventFilter
        # 刚报对的那个盖掉。光标在图像区时的事件一律由 eventFilter（挂在 self.view 上）
        # 处理，足够覆盖 MouseMove / Enter / Leave 三种情形。
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
        if obj is self.view:
            t = ev.type()
            # ★ 光标上报放在**滚轮/平移开关判断之外**：否则没绑定平移时
            #   （比如只读参考预览）坐标就永远收不到 —— 又变成空的。
            if t == QEvent.MouseMove:
                try:
                    p = ev.pos()
                    self._emit_hover_xy(p.x(), p.y())
                except Exception:
                    pass
            elif t == QEvent.Enter:
                # 光标从画布挪进来时按**当前真实位置**报一次，免得非等用户再动一下
                # 才出现读数（Enter 事件本身不带坐标）。
                try:
                    p = self.view.mapFromGlobal(QCursor.pos())
                    self._emit_hover_xy(p.x(), p.y())
                except Exception:
                    pass
            elif t == QEvent.Leave:
                self._hovering = False
                self._drop_hover()
            if self._on_wheel is not None:
                if t == QEvent.Wheel:
                    self._on_wheel(ev.angleDelta().y(), self._wheel_pos_in_view(self.view, ev))
                    ev.accept()
                    return True
                if self._on_pan is not None and t in (QEvent.MouseButtonPress,
                                                      QEvent.MouseMove,
                                                      QEvent.MouseButtonRelease):
                    self.event(ev)      # 交给本面板的鼠标处理（坐标已按 view 平移）
                    ev.accept()
                    return True
        return super().eventFilter(obj, ev)


class MultiViewPreview(QWidget):
    """N 视图并排预览容器（viewer stack 的 index 0）。"""

    # 光标上报的**唯一出口**：`(x, y, source)`。
    #   x, y   = 主视图图像坐标系里的像素坐标；无坐标时 (-1, -1)
    #   source = 光标所在的那一格（主画布，或某个 _RefPanel）
    # 为什么要带 source：三格各自显示自己的图（漫反射 / 形状图1 / 形状图2），
    # 底栏的「RGB」必须读**光标底下那一格**的像素 —— 否则对着形状图报漫反射灰度，
    # 这种读数比没有读数更误导。主画布与参考窗的原始信号都在这里汇成一路，
    # 上层（底部信息栏）只挂一个钩子，不用知道有几格。
    cursor_moved = pyqtSignal(int, int, object)

    # 主画布状态汇总：三个图层开关 + 当前类别 + 缩放百分比 + 视图数。
    #
    # 顶栏删掉之后，这些值**没有第二个人**可以问 —— 所以必须有个出口。
    # 为什么是一个 dict 而不是五个信号：这些值由同一个 40ms 轮询里的一次比较得出
    # （见 `_emit_state`），"任意一项变了"就该刷新工具栏/底栏；
    # 拆成五个信号就得在五处各写一遍去重，还没法保证"三开关 + 类别"是同一瞬间的快照。
    state_changed = pyqtSignal(dict)

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self._items = []
        self._panels = []
        self._last_sig = None
        # 视图布局模式（原先是顶栏那个下拉框，下拉已搬到标注工具栏）。
        # 默认取 LAYOUT_MODES[0] = "自动（N*1）"，与搬走前的默认一致。
        self._layout_mode = LAYOUT_MODES[0]
        self._build()
        # 轮询计时器在 _build() 里创建（装完画布事件过滤器之后），此处不要再建一个，
        # 否则会同时跑两个定时器做重复同步。

    # ────────── 构建 ──────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # ⚠️ 图像区上方那一整行「顶栏」已按 维护者 2026-10-14 的决定**拆掉**
        #    （原话："我感觉咱们顶栏和底栏可以合二为一，你可以参考一下友商的"）。
        #    关键是**不能只搬位置**：实测（tmp/measure_bars2.py，SimSun 9pt）
        #    这条栏三段文本 726px + 下拉 150px ≈ **976px**，而中央列可用只有
        #    757~950px ⇒ 一条底栏根本装不下两行的信息，硬搬只会挤爆。
        #    所以按"这项有没有别处显示"重新分派：
        #      · 缩放 X%          -> 底部信息栏最右端（照友商：那里放 −/滑块/+ 控件，
        #                            读数与控件同处，不用"上面看数、下面调"）
        #      · 视图布局下拉      -> 标注工具栏右侧（QToolButton + 菜单，仅多视图可见）
        #      · Annot/Ignore/Pred -> 标注工具栏上三个可点开关（**原来全屏只有这一处显示**，
        #                            只能靠 Ctrl+Space / Ctrl+Alt 切，删了就再也看不见）
        #      · Class: xx        -> 标注工具栏上的类别格（点开可换类别）
        #      · 当前工具 Pan / Brush: 15 -> **删**：工具栏上工具按钮本来就高亮、
        #                            最左边那个控件就是「宽度 15 px」，同屏写两遍纯冗余
        #      · "3 视图：漫反射/形状图1/形状图2" -> **删**：每格标题上已经写着
        #    删掉后状态文本的出口是 state_changed 信号（见类属性与 _emit_state），
        #    主窗口接住后分发给工具栏按钮与底栏读数 —— 单一出口，不再各处挂钩子。

        self.host = QWidget()
        self.grid = QGridLayout(self.host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        root.addWidget(self.host, 1)

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
        # 主视图（第 0 格）的光标上报也汇进本容器那一路：底部信息栏只挂一个钩子。
        # 用 lambda 绑定 source，别依赖槽的默认参数（信号 2 个参数、槽 3 个参数，
        # 靠默认值能不能连上要看 PyQt 的签名推断，写成 lambda 就没这个悬念）。
        try:
            self.canvas.cursor_moved.connect(self._relay_cursor_from_canvas)
        except Exception as e:
            print(f"[多视图预览] 画布无 cursor_moved 信号，底栏坐标只能靠参考窗那两格: {e}")
        self._sync_timer = QTimer(self)
        # 轮询只兜底：拖动平移这类高频连续动作不逐个信号推送（每移动一次就重画两个窗
        # 反而更卡），交给 40ms 轮询跟进。数值比较开销可忽略。
        self._sync_timer.setInterval(40)
        self._sync_timer.timeout.connect(self._sync_from_canvas)
        self._sync_timer.start()

    def _on_canvas_view_changed(self):
        """画布 zoom/pan/标注开关变化 -> 立即同步（不等轮询）。"""
        self._sync_from_canvas(force=True)

    def _relay_cursor_from_canvas(self, x, y):
        self.cursor_moved.emit(int(x), int(y), self.canvas)

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

    def _emit_state(self):
        """把画布状态汇成一个 dict 推给主窗口（**只在真的变了**才发）。

        这是顶栏拆掉之后这些状态的**唯一出口**。为什么挂在 40ms 轮询里、而不是全靠信号：
        这些值有三个来源 —— ① 快捷键（画布自己改，`toggle_*` / 数字键选类别）；
        ② 工具栏上的开关与类别格（主窗口改）；③ 打开工程/换图时的一次性初始化。
        靠信号就得三处都记得发、还得各自去重；轮询里比较一个 5 元组几乎不要钱，
        而且**漏掉任何一条路径也不会出现"按钮亮着、实际是关的"**这种假状态。
        这一条比省几个 CPU 周期重要得多。

        ⚠️ 检测模式下"标注层"其实是 `_show_annotation_boxes`（Ctrl+Space 切的是框），
        分割模式下才是 `_show_annotation` —— 回读时也必须分模式，
        否则按钮的亮灭跟开关实际控制的东西不是同一个（`status_info_text` 踩过同一个坑）。
        """
        c = self.canvas
        if c is None:
            return
        det = getattr(c, "_det_overlay", None) is not None
        try:
            annot = (bool(getattr(c, "_show_annotation_boxes", True)) if det
                     else bool(getattr(c, "_show_annotation", True)))
        except Exception:
            annot = True
        try:
            # 三个模式各管一个字段，跟 `toggle_prediction_visibility` / `keyPressEvent`
            # 里的分支必须一一对应，否则按钮的亮灭跟它实际控制的东西不是一回事。
            if det:
                pred = bool(getattr(c, "_show_pred_boxes", True))
            elif getattr(c, "_cls_mode", False):
                pred = bool(getattr(c, "_show_heatmap", True))
            else:
                pred = bool(getattr(c, "_show_prediction", False))
        except Exception:
            pred = False
        try:
            ignore = bool(getattr(c, "_show_ignore", True))
        except Exception:
            ignore = True
        cls_name = "?"
        try:
            lm = getattr(c, "label_manager", None)
            if lm is not None and 0 <= int(c.current_class_id) < len(lm.classes):
                cls_name = lm.classes[int(c.current_class_id)]
        except Exception:
            cls_name = "?"
        st = {
            "annot": annot,
            "ignore": ignore,
            "pred": pred,
            "cls": cls_name,
            "zoom_pct": self.zoom_percent(),
            "n_views": len(self._items or []),
        }
        # ⚠️ `n_views` 必须进签名。视图数从 1 变成 3 时另外五项一个都没变，
        #    不进签名就**永远不发这一拍** —— 工具栏右侧那三个开关与类别格只在
        #    多视图模式下露面，于是它们永远不显示，而界面上完全看不出"少了东西"
        #    （2026-10-14 冒烟实测：layers_visible 一直是 False）。
        sig = (annot, ignore, pred, cls_name, st["zoom_pct"], st["n_views"])
        if sig == getattr(self, "_state_sig", None):
            return
        self._state_sig = sig
        self.state_changed.emit(st)

    def zoom_percent(self):
        """当前缩放，单位是"相对适应窗口的百分比"（界面上那个 `140%`）。

        算式只留这一处：`_emit_state` 与底栏都从这里取 ——
        两处各算一遍迟早会漂移成"上面的数字和下面滑块对不上"。
        """
        c = self.canvas
        try:
            fit = float(getattr(c, "_fit_zoom", 1.0) or 1.0)
            if fit <= 0:
                return 100
            return int(round(float(c.zoom_level) / fit * 100))
        except Exception:
            return 100

    def set_zoom_percent(self, pct):
        """按"相对适应窗口的百分比"设定缩放（底栏 −/滑块/+ 走这条路）。

        锚点是**窗口中心**。滚轮缩放锚的是光标，但拖滑块时光标在滑块上、不在图上，
        锚光标没有意义 ⇒ 这里锚中心，等价于"窗口中心那个图像点保持不动"。
        中心锚定的好处是 n_cx/n_cy 直接就是当前的 cx/cy，不用像 `_zoom_at` 那样解方程；
        但 **zoom 变了 pan 必须跟着重算**，否则图像会整体跳一下。

        返回是否真的改了（滑块回写时要用它决定要不要动滑块，避免来回打架）。
        """
        c = self.canvas
        got = self._canvas_center_in_image()
        if got is None:
            return False
        z, cx, cy = got
        try:
            iw, ih = c.image.size
            iw, ih = int(iw), int(ih)
        except Exception:
            return False
        fit = float(getattr(c, "_fit_zoom", 0.0) or 0.0)
        if fit <= 0:
            return False
        lo = float(getattr(c, "MIN_ZOOM", 0.05) or 0.05)
        hi = float(getattr(c, "MAX_ZOOM", 30.0) or 30.0)
        nz = max(lo, min(hi, fit * float(pct) / 100.0))
        if abs(nz - z) < 1e-9:
            return False
        c.zoom_level = nz
        c.pan_offset_x = (iw / 2.0 - cx) * nz
        c.pan_offset_y = (ih / 2.0 - cy) * nz
        try:
            c.update()
        except Exception:
            pass
        self._sync_from_canvas(force=True)
        return True

    def layout_modes(self):
        """可选视图布局（工具栏那个按钮的菜单项来源，别再抄一份常量）。"""
        return list(LAYOUT_MODES)

    def layout_mode(self):
        return getattr(self, "_layout_mode", LAYOUT_MODES[0])

    def set_layout_mode(self, text):
        """由工具栏的「视图布局」按钮调用（这个下拉原先在顶栏，已搬到工具栏）。"""
        if text not in LAYOUT_MODES or text == getattr(self, "_layout_mode", None):
            return False
        self._layout_mode = text
        self.apply_layout()
        return True

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
        # 布局模式原先读顶栏那个下拉框；下拉搬到工具栏后由 set_layout_mode 写入。
        m = getattr(self, "_layout_mode", LAYOUT_MODES[0])
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
            # 光标坐标：面板自己还没渲染过时向本容器要画布的中心点（见 _view_state）
            panel._state_provider = self._canvas_center_in_image
            # 默认参数 p=panel 把"是哪一格"绑死 —— 不要用 self.sender()，
            # 那要依赖连接方式（直连/排队）才成立。
            panel.cursor_moved.connect(
                lambda x, y, p=panel: self.cursor_moved.emit(int(x), int(y), p))
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

    # ────────── 标注标签数据源（与主画布同源） ──────────
    def _current_shapes(self):
        """取主画布当前图对应的 LabelMe shapes，返回 `(shapes, tag)`。

        `tag = (json 路径, mtime)`：保存标注后 mtime 变 -> 参考窗自动补上新标签。

        为什么用**主画布**这份 json、而不是各参考视图自己的：
        三视图工程的标注只有一份（掩膜就是这么加载的，见 `load_mask_from_json`），
        参考窗若去读 `xxx_te_1.json` 通常读到的是"不存在" —— 结果就是"掩膜有、
        标签没有"，正是这次修的现象。标签与掩膜同源，两者永远一致。

        带缓存：这里每 40ms 被轮询一次，不能每次都读盘 + 解析 json。
        """
        path = getattr(self.canvas, "current_image_path", None)
        if not path:
            return None, None
        try:
            from annotation.labelme_io import get_json_path
            jp = get_json_path(path)
        except Exception:
            import os.path as _osp
            jp = _osp.splitext(path)[0] + ".json"
        try:
            import os
            tag = (jp, os.path.getmtime(jp))
        except OSError:
            tag = (jp, None)          # json 还不存在（未保存过标注）
        if tag == getattr(self, "_lab_tag_cache", None):
            return getattr(self, "_lab_shapes_cache", None), tag
        shapes = None
        if tag[1] is not None:
            try:
                from annotation.labelme_io import load_labelme_json
                shapes, _ = load_labelme_json(jp)
            except Exception:
                shapes = None
        self._lab_tag_cache = tag
        self._lab_shapes_cache = shapes
        return shapes, tag

    def _sync_from_canvas(self, force=False):
        # 状态汇总每轮都做一次（比较一个 5 元组，没变就不发信号）。
        # 必须放在它下面那个 `if not self._panels: return` **之前** ——
        # 单图模式同样要用工具栏上的三个图层开关与底栏的缩放滑块。
        self._emit_state()
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
        # 标注标签（IGNORE / 类别名）与掩膜**同源**：都取主画布当前图的 json。
        # 参考窗以前不画标签，于是"只有漫反射图有 IGNORE/Ys 字样"——这里补上。
        shapes, lab_tag = self._current_shapes()
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
            if all(getattr(p, "_ov_tag", None) == tag for p in self._panels) \
                    and all(getattr(p, "_label_tag", None) == lab_tag for p in self._panels):
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
            # 标签与掩膜同源（主画布的 json），两个开关各管一类
            p.set_labels(shapes, lab_tag, show_classes=show_ann, show_ignore=show_ign)
            p.render_view(z, cx, cy)
        # 原来这里往顶栏那个青色标签写 "缩放 X%（三窗同步）"。顶栏已拆，
        # 缩放改由底栏右端显示（与 −/滑块/+ 控件同处一栏，读数就在手指边）。
