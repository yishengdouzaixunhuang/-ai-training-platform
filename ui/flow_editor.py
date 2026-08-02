# -*- coding: utf-8 -*-
"""流程编辑器：拖拽建节点、端口连线、参数面板、保存/加载/运行。

P2 目标：对齐海康 VisionMaster 的「方案」编辑体验。
- 左侧节点库：按类别分组，拖拽到画布创建节点
- 中间画布：节点可拖动，输出口 -> 输入口 连线，Delete 删除
- 右侧参数面板：选中节点后按参数 Schema 自动生成控件
- 顶部工具栏：打开/保存/运行/停止，运行结果写入日志区
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath,
                         QPen)
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                             QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGraphicsItem, QGraphicsPathItem, QGraphicsScene,
                             QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMainWindow,
                             QMessageBox, QPushButton, QSpinBox, QSplitter,
                             QTextEdit, QVBoxLayout, QWidget,
                             QGraphicsSceneMouseEvent)

from flow.registry import create_node, list_nodes
from flow.runner import Flow, Runner


# ============ 常量 ============

CATEGORY_COLORS = {
    "采集": QColor(0, 150, 136),
    "预处理": QColor(33, 150, 243),
    "算法": QColor(156, 39, 176),
    "可视化": QColor(76, 175, 80),
    "判定": QColor(255, 87, 34),
    "输出": QColor(97, 97, 97),
}

NODE_W = 170
NODE_H = 64
TITLE_H = 24
PORT_R = 5


def _category_color(category: str) -> QColor:
    return CATEGORY_COLORS.get(category, QColor(120, 120, 120))


# ============ 节点图元 ============

class FlowNodeItem(QGraphicsItem):
    """画布上的算子节点：标题栏 + 左右端口。"""

    def __init__(self, node, pos: QPointF):
        super().__init__()
        self.node = node
        self.node_type = node.node_type
        self.node_id = node.node_id
        self._dragging_out = False
        self.setPos(pos)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(10)

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_R - 2, -PORT_R - 2, NODE_W + 2 * PORT_R + 4, NODE_H + 2 * PORT_R + 4)

    def input_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0, NODE_H / 2))

    def output_pos(self) -> QPointF:
        return self.mapToScene(QPointF(NODE_W, NODE_H / 2))

    def paint(self, painter: QPainter, option, widget=None):
        color = _category_color(self.node.category)
        painter.setRenderHint(QPainter.Antialiasing, True)

        body = QRectF(0, 0, NODE_W, NODE_H)
        painter.setBrush(QColor(30, 30, 34))
        painter.setPen(QPen(QColor(90, 90, 100), 1))
        painter.drawRoundedRect(body, 6, 6)

        title = QRectF(0, 0, NODE_W, TITLE_H)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(title, 6, 6)
        painter.drawRect(QRectF(0, TITLE_H - 6, NODE_W, 6))

        painter.setPen(Qt.white)
        painter.setFont(QFont("Microsoft YaHei", 8, QFont.Bold))
        painter.drawText(title.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, self.node.display_name)

        painter.setPen(QColor(180, 180, 190))
        painter.setFont(QFont("Microsoft YaHei", 7))
        painter.drawText(QRectF(6, TITLE_H + 6, NODE_W - 12, NODE_H - TITLE_H - 8),
                         Qt.AlignVCenter | Qt.AlignLeft, f"[{self.node_type}]")

        painter.setBrush(QColor(220, 220, 220))
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.drawEllipse(QPointF(0, NODE_H / 2), PORT_R, PORT_R)
        painter.drawEllipse(QPointF(NODE_W, NODE_H / 2), PORT_R, PORT_R)

    # ---- 交互：从输出口拖出连线 ----
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.LeftButton and self._hit_output(event.pos()):
            self._dragging_out = True
            scene = self.scene()
            if hasattr(scene, "begin_connection"):
                scene.begin_connection(self.node_id, event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self._dragging_out:
            scene = self.scene()
            if hasattr(scene, "update_connection"):
                scene.update_connection(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if self._dragging_out:
            self._dragging_out = False
            scene = self.scene()
            if hasattr(scene, "end_connection"):
                scene.end_connection(event.scenePos())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            scene = self.scene()
            if scene is not None and hasattr(scene, "refresh_edges"):
                scene.refresh_edges(self.node_id)
        return super().itemChange(change, value)

    def _hit_output(self, pos: QPointF) -> bool:
        return (pos - QPointF(NODE_W, NODE_H / 2)).manhattanLength() <= PORT_R + 6


# ============ 连线图元 ============

class FlowEdgeItem(QGraphicsPathItem):
    """连接两个端口的贝塞尔曲线。"""

    def __init__(self, from_id: str, to_id: str):
        super().__init__()
        self.from_id = from_id
        self.to_id = to_id
        pen = QPen(QColor(0, 188, 212), 2)
        pen.setCapStyle(Qt.RoundCap)
        self.setPen(pen)
        self.setZValue(5)

    def set_endpoints(self, p1: QPointF, p2: QPointF):
        path = QPainterPath(p1)
        dx = max(40.0, abs(p2.x() - p1.x()) * 0.5)
        path.cubicTo(p1.x() + dx, p1.y(), p2.x() - dx, p2.y(), p2.x(), p2.y())
        self.setPath(path)


# ============ 场景 ============

class FlowScene(QGraphicsScene):
    """管理节点与连线的画布场景。"""

    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.setBackgroundBrush(QColor(24, 24, 28))
        self._node_items: Dict[str, FlowNodeItem] = {}
        self._edges: Dict[tuple, FlowEdgeItem] = {}
        self._temp_edge = None
        self._drag_from = None

    # ---- 节点 ----
    def add_node_item(self, node_type: str, pos: QPointF) -> FlowNodeItem:
        node = create_node(node_type, node_id=self._new_id())
        item = FlowNodeItem(node, pos)
        self.addItem(item)
        self._node_items[node.node_id] = item
        self.log_signal.emit(f"已创建节点 {node.display_name} ({node.node_id})")
        return item

    def _new_id(self) -> str:
        used = set(self._node_items)
        i = 1
        while f"n{i}" in used:
            i += 1
        return f"n{i}"

    def get_node_item(self, node_id: str):
        return self._node_items.get(node_id)

    def remove_node_item(self, node_id: str):
        item = self._node_items.pop(node_id, None)
        if item is None:
            return
        self.removeItem(item)
        for (f, t) in [k for k in self._edges if k[0] == node_id or k[1] == node_id]:
            edge = self._edges.pop((f, t))
            self.removeItem(edge)
        self.log_signal.emit(f"已删除节点 {node_id}")

    def selected_node_id(self):
        for item in self.selectedItems():
            if isinstance(item, FlowNodeItem):
                return item.node_id
        return None

    # ---- 连线 ----
    def begin_connection(self, from_id: str, scene_pos: QPointF):
        self._drag_from = from_id
        self._temp_edge = QGraphicsPathItem()
        pen = QPen(QColor(0, 188, 212), 2, Qt.DashLine)
        pen.setCapStyle(Qt.RoundCap)
        self._temp_edge.setPen(pen)
        self._temp_edge.setZValue(4)
        self.addItem(self._temp_edge)
        self.update_connection(scene_pos)

    def update_connection(self, scene_pos: QPointF):
        if self._temp_edge is None or self._drag_from is None:
            return
        p1 = self._node_items[self._drag_from].output_pos()
        path = QPainterPath(p1)
        dx = max(40.0, abs(scene_pos.x() - p1.x()) * 0.5)
        path.cubicTo(p1.x() + dx, p1.y(), scene_pos.x() - dx, scene_pos.y(), scene_pos.x(), scene_pos.y())
        self._temp_edge.setPath(path)

    def end_connection(self, scene_pos: QPointF):
        if self._temp_edge is not None:
            self.removeItem(self._temp_edge)
            self._temp_edge = None
        from_id = self._drag_from
        self._drag_from = None
        if not from_id:
            return
        to_id = self._hit_input_node(scene_pos)
        if to_id is None or to_id == from_id:
            return
        # 输入口已有连线则替换
        for (f, t) in [k for k in self._edges if k[1] == to_id]:
            self.removeItem(self._edges.pop((f, t)))
        if (from_id, to_id) in self._edges:
            return
        self._add_edge(from_id, to_id)
        self.log_signal.emit(f"已连线 {from_id} -> {to_id}")

    def _add_edge(self, from_id: str, to_id: str):
        from_item = self._node_items.get(from_id)
        to_item = self._node_items.get(to_id)
        if from_item is None or to_item is None:
            return
        edge = FlowEdgeItem(from_id, to_id)
        edge.set_endpoints(from_item.output_pos(), to_item.input_pos())
        self.addItem(edge)
        self._edges[(from_id, to_id)] = edge

    def _hit_input_node(self, scene_pos: QPointF):
        for nid, item in self._node_items.items():
            if (scene_pos - item.input_pos()).manhattanLength() <= PORT_R + 8:
                return nid
        return None

    def refresh_edges(self, node_id: str):
        for (f, t), edge in self._edges.items():
            if f != node_id and t != node_id:
                continue
            f_item = self._node_items.get(f)
            t_item = self._node_items.get(t)
            if f_item is not None and t_item is not None:
                edge.set_endpoints(f_item.output_pos(), t_item.input_pos())

    # ---- 拖放 ----
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-flow-node"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-flow-node"):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        node_type = bytes(event.mimeData().data("application/x-flow-node")).decode("utf-8")
        if node_type:
            pos = event.scenePos()
            self.add_node_item(node_type, QPointF(pos.x() - NODE_W / 2, pos.y() - NODE_H / 2))
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    # ---- 序列化 ----
    def to_flow(self, name: str = "unnamed") -> Flow:
        flow = Flow(name=name)
        for item in self._node_items.values():
            flow.nodes.append(item.node)
        for (f, t) in self._edges:
            flow.edges.append((f, t))
        return flow

    def load_flow(self, flow: Flow):
        self.clear()
        self._node_items.clear()
        self._edges.clear()
        pos = QPointF(-300, -200)
        dx = NODE_W + 60
        for i, node in enumerate(flow.nodes):
            item = FlowNodeItem(node, QPointF(pos.x() + i * dx, pos.y()))
            self.addItem(item)
            self._node_items[node.node_id] = item
        for (f, t) in flow.edges:
            self._add_edge(f, t)
        self.log_signal.emit(f"已加载流程: {flow.name} ({len(flow.nodes)} 节点, {len(flow.edges)} 连线)")


# ============ 参数面板 ============

class ParamPanel(QWidget):
    """选中节点的参数编辑区，按 NodeParam Schema 自动生成控件。"""

    changed = pyqtSignal(object, str, object)  # (node, param_name, value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._node = None
        self._widgets = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.title = QLabel("未选择节点")
        self.title.setStyleSheet("font-weight: bold; color: #ddd;")
        layout.addWidget(self.title)
        self._form_host = QWidget()
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(self._form_host)
        layout.addStretch(1)
        self.setMinimumWidth(220)

    def show_node(self, node):
        self._node = node
        self._widgets.clear()
        while self._form.rowCount() > 0:
            self._form.removeRow(0)
        if node is None:
            self.title.setText("未选择节点")
            return
        self.title.setText(f"{node.display_name}\n[{node.node_type}]")
        for p in node.PARAMS:
            label = p.label + (f" ({p.unit})" if p.unit else "")
            w = self._make_control(node, p)
            self._widgets[p.name] = w
            self._form.addRow(QLabel(label), w)
        if not node.PARAMS:
            self._form.addRow(QLabel("无参数"))

    def _make_control(self, node, p):
        value = node.params.get(p.name, p.default)
        if p.ptype == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(value))
            cb.toggled.connect(lambda v, n=node, name=p.name: self.changed.emit(n, name, v))
            return cb
        if p.ptype == "choice":
            cb = QComboBox()
            cb.addItems(p.choices or [])
            idx = cb.findText(str(value))
            cb.setCurrentIndex(max(0, idx))
            cb.currentTextChanged.connect(lambda v, n=node, name=p.name: self.changed.emit(n, name, v))
            return cb
        if p.ptype == "int":
            sb = QSpinBox()
            lo = int(p.min) if p.min is not None else -100000
            hi = int(p.max) if p.max is not None else 100000
            sb.setRange(lo, hi)
            sb.setValue(int(value) if value is not None else lo)
            if p.step is not None:
                sb.setSingleStep(int(p.step))
            sb.valueChanged.connect(lambda v, n=node, name=p.name: self.changed.emit(n, name, v))
            return sb
        if p.ptype == "float":
            sb = QDoubleSpinBox()
            lo = float(p.min) if p.min is not None else -1e9
            hi = float(p.max) if p.max is not None else 1e9
            sb.setRange(lo, hi)
            try:
                sb.setValue(float(value) if value is not None else 0.0)
            except (TypeError, ValueError):
                sb.setValue(0.0)
            if p.step is not None:
                sb.setSingleStep(float(p.step))
            if p.unit:
                sb.setSuffix(" " + p.unit)
            sb.valueChanged.connect(lambda v, n=node, name=p.name: self.changed.emit(n, name, v))
            return sb
        le = QLineEdit()
        le.setText("" if value is None else str(value))
        le.textChanged.connect(lambda v, n=node, name=p.name: self.changed.emit(n, name, v))
        return le


# ============ 节点库 / 画布视图 ============

class PaletteList(QListWidget):
    """节点库：拖拽时附带节点类型的自定义 MIME。"""

    def mimeData(self, items):
        md = super().mimeData(items)
        if items:
            node_type = items[0].data(Qt.UserRole)
            if node_type:
                md.setData("application/x-flow-node", node_type.encode("utf-8"))
        return md


class FlowGraphicsView(QGraphicsView):
    """画布视图：拦截 Delete 键删除选中节点。"""

    delete_requested = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


# ============ 主编辑器 ============

class FlowEditorWidget(QWidget):
    """流程编辑器主组件。"""

    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_signal.connect(self._append_log)
        self._build_ui()
        self._load_palette()
        self._current_runner = None
        self._stop_flag = False
        self._input_path = ""
        self._last_dir = os.getcwd()

    # ---- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        tb = QWidget()
        tb.setStyleSheet("QWidget { background: #2b2b30; }")
        tbh = QHBoxLayout(tb)
        tbh.setContentsMargins(6, 4, 6, 4)
        tbh.setSpacing(6)
        b_open = QPushButton("打开方案")
        b_save = QPushButton("保存方案")
        b_run = QPushButton("▶ 运行")
        b_stop = QPushButton("■ 停止")
        b_pick = QPushButton("选择输入图...")
        for b in (b_open, b_save, b_run, b_stop, b_pick):
            b.setStyleSheet("QPushButton { background:#3a3a42; color:#eee; border:1px solid #555; border-radius:3px; padding:4px 10px; }"
                            "QPushButton:hover { background:#4a4a55; }")
            tbh.addWidget(b)
        tbh.addStretch(1)
        root.addWidget(tb)

        splitter = QSplitter(Qt.Horizontal)

        self.palette = PaletteList()
        self.palette.setDragEnabled(True)
        self.palette.setDragDropMode(QAbstractItemView.DragOnly)
        self.palette.setSelectionMode(QAbstractItemView.SingleSelection)
        self.palette.setStyleSheet("QListWidget { background: #26262b; color: #ddd; }")
        splitter.addWidget(self.palette)

        self.scene = FlowScene(self)
        self.scene.log_signal.connect(self.log)
        self.view = FlowGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        self.view.setStyleSheet("QGraphicsView { background: #18181c; }")
        self.view.delete_requested.connect(self._delete_selected)
        splitter.addWidget(self.view)

        self.param_panel = ParamPanel()
        self.param_panel.changed.connect(self._on_param_changed)
        splitter.addWidget(self.param_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([180, 720, 260])
        root.addWidget(splitter, 1)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(160)
        self.log_view.setStyleSheet("QTextEdit { background: #101014; color: #9fe08a; font-family: Consolas; }")
        root.addWidget(self.log_view)

        b_open.clicked.connect(self._open_flow)
        b_save.clicked.connect(self._save_flow)
        b_run.clicked.connect(self._run_flow)
        b_stop.clicked.connect(self._stop_run)
        b_pick.clicked.connect(self._pick_input)
        self.scene.selectionChanged.connect(self._on_selection_changed)

    def _load_palette(self):
        by_cat = {}
        for info in list_nodes():
            by_cat.setdefault(info["category"], []).append(info)
        for cat in ["采集", "预处理", "算法", "可视化", "判定", "输出"]:
            infos = by_cat.pop(cat, [])
            if infos:
                self._palette_add_group(cat, infos)
        for cat, infos in by_cat.items():
            self._palette_add_group(cat, infos)

    def _palette_add_group(self, cat, infos):
        head = QListWidgetItem(f"── {cat} ──")
        head.setFlags(Qt.NoItemFlags)
        head.setForeground(QBrush(QColor(140, 140, 150)))
        self.palette.addItem(head)
        for info in sorted(infos, key=lambda x: x["type"]):
            item = QListWidgetItem(info["name"])
            item.setData(Qt.UserRole, info["type"])
            item.setToolTip(info["type"])
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
            item.setForeground(QBrush(QColor(230, 230, 235)))
            self.palette.addItem(item)

    # ---- 选中/参数 ----
    def _on_selection_changed(self):
        nid = self.scene.selected_node_id()
        item = self.scene.get_node_item(nid) if nid else None
        self.param_panel.show_node(item.node if item else None)

    def _on_param_changed(self, node, name, value):
        for p in node.PARAMS:
            if p.name == name:
                node.params[name] = p.coerce(value)
                break
        else:
            node.params[name] = value
        self.log(f"参数更新 [{node.node_id}] {name} = {node.params[name]}")

    # ---- 保存/加载 ----
    def _open_flow(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开方案", self._last_dir, "Flow JSON (*.json)")
        if not path:
            return
        try:
            flow = Flow.load(path)
            self.scene.load_flow(flow)
            self._last_dir = os.path.dirname(path)
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))

    def _save_flow(self):
        if not self.scene._node_items:
            QMessageBox.information(self, "提示", "画布上没有节点")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存方案",
                                              os.path.join(self._last_dir, "flow.json"), "Flow JSON (*.json)")
        if not path:
            return
        try:
            flow = self.scene.to_flow(name=os.path.splitext(os.path.basename(path))[0])
            flow.save(path)
            self._last_dir = os.path.dirname(path)
            self.log(f"已保存方案: {path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    # ---- 运行 ----
    def _pick_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择输入图像", self._last_dir,
                                              "Images (*.bmp *.png *.jpg *.jpeg *.tif)")
        if path:
            self._input_path = path
            self._last_dir = os.path.dirname(path)
            self.log(f"输入图: {path}")

    def _run_flow(self):
        if not self.scene._node_items:
            QMessageBox.information(self, "提示", "画布上没有节点")
            return
        input_path = self._input_path
        if not input_path:
            input_path, _ = QFileDialog.getOpenFileName(self, "选择输入图像", self._last_dir,
                                                        "Images (*.bmp *.png *.jpg *.jpeg *.tif)")
            if not input_path:
                return
            self._input_path = input_path
            self._last_dir = os.path.dirname(input_path)
        flow = self.scene.to_flow(name="editor")
        self._stop_flag = False
        self.log("=" * 40)
        self.log(f"开始运行流程: {flow.name}")
        threading.Thread(target=self._run_worker, args=(flow, input_path), daemon=True).start()

    def _run_worker(self, flow: Flow, input_path: str):
        from flow.frame import Frame
        from flow.runner import Runner
        try:
            in_frame = Frame(path=input_path)
            runner = Runner(flow)
            self._current_runner = runner
            out = runner.run(
                in_frame,
                on_node=lambda n, f: self.log_signal.emit(f"  → {n.node_type} ({n.node_id})"),
            )
            self.log_signal.emit("运行完成")
            self.log_signal.emit(f"结果: {_brief(out.result)}")
        except Exception as e:
            import traceback
            self.log_signal.emit(f"运行失败: {e}")
            self.log_signal.emit(traceback.format_exc())

    def _stop_run(self):
        self._stop_flag = True
        if self._current_runner is not None:
            self._current_runner.stop()
        self.log("已请求停止")

    # ---- 日志 / 删除 ----
    def log(self, text: str):
        self.log_view.append(text)

    def _append_log(self, text: str):
        self.log_view.append(text)

    def _delete_selected(self):
        nid = self.scene.selected_node_id()
        if nid:
            self.scene.remove_node_item(nid)
            self.param_panel.show_node(None)


def _brief(result: dict) -> str:
    """把结果精简成一行文本（跳过 numpy 大数组）。"""
    import numpy as np
    parts = []
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            parts.append(f"{k}=<ndarray {v.shape}>")
        elif isinstance(v, (list, tuple)) and len(v) > 4:
            parts.append(f"{k}=[{len(v)} items]")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "(空)"


class FlowEditorDialog(QDialog):
    """以独立窗口打开的流程编辑器（QDialog 才有 exec_()）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("流程编辑器 - Vision Flow")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(FlowEditorWidget(self))
        self.resize(1280, 820)
