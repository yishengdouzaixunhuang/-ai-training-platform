# -*- coding: utf-8 -*-
"""任务队列表（左栏「任务队列」页的内容区）。

参照友商 iCogtiveFlow 的同名模块：每行一个任务，显示 类型/标题/进度/剩余时间 +取消（×），
顶部有筛选与「清空」。

三条UI 侧的纪律（都是今天踩出来的）：
1. **刷新必须回到 UI 线程**：队列跑在后台线程里，`on_change` 回调会被工作线程直接调用，
   在那里碰控件会随机崩（Qt 只允许 GUI 线程操作控件）。⇒ 用 `QMetaObject.invokeMethod`
   把刷新**投递**回 UI 线程，绝不在回调里直接改控件。
2. **刷新要合并**：训练每个 batch 都会推进度，若每次都重建整张表，几千行/s 会把界面拖死。
   ⇒ 只在"行的集合/状态"变化时重建行；进度文字用 `setText` 原地更新。
3. **按钮尺寸写在样式里**：主题清洗只剥颜色不剥度量，但控件本身的内联色会被剥；
   这里干脆只用 objectName + 主题表（见 themes.py），不在控件上写颜色。
"""
from __future__ import annotations
from core.i18n import tr

from PyQt5.QtCore import Q_ARG, QMetaObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from ui.task_queue import KIND_LABEL, TaskQueue, TaskState

#: 状态下拉的可选项（"全部" 之外的每一项 = 只看该状态）
_FILTER_ITEMS = [("全部", None), ("排队中", TaskState.QUEUED), ("执行中", TaskState.RUNNING),
                 ("已完成", TaskState.DONE), ("失败", TaskState.FAILED),
                 ("已取消", TaskState.CANCELED)]

_STATE_TEXT = {
    TaskState.QUEUED: "排队中",
    TaskState.RUNNING: "执行中",
    TaskState.DONE: "已完成",
    TaskState.FAILED: "失败",
    TaskState.CANCELED: "已取消",
}


def _localize_status(text):
    """把 `剩余 1小时02分03秒` 这类核心状态文本按当前语言显示。

    任务队列核心（ui/task_queue.py）刻意只用中文，避免把 i18n 塞进无 Qt 的
    队列逻辑；显示层在这里做单位替换。中文模式 tr() 恒等，不会改动原串。
    """
    if not text:
        return text
    return (text.replace("剩余 ", tr("剩余 "))
                .replace("小时", tr("小时"))
                .replace("分", tr("分"))
                .replace("秒", tr("秒")))


class TaskQueuePanel(QWidget):
    """任务队列面板。`queue` 由主窗口创建并传入（队列核心不依赖 Qt）。"""

    #: 用户点了某一行的 ×（传 task_id）
    cancel_requested = pyqtSignal(str)
    #: 用户点了「清空」（清历史）
    clear_requested = pyqtSignal()
    #: 队列状态变了。**由工作线程 emit**，Qt 自动排队到 UI 线程（跨线程投递的正确姿势）。
    _queue_changed = pyqtSignal(object)

    def __init__(self, queue, parent=None):
        super().__init__(parent)
        self._queue = queue
        self._rows = {}                # task_id -> dict(容器/进度条/状态标签/取消按钮)
        self._filter_state = None

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(4)

        # ── 顶部：筛选 + 清空（照友商那一行的位置）──
        top = QHBoxLayout(); top.setSpacing(4)
        self.filter_combo = QComboBox()
        for text, st in _FILTER_ITEMS:
            self.filter_combo.addItem(tr(text), st)
        self.filter_combo.setToolTip(tr("只看某一状态的任务"))
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        top.addWidget(self.filter_combo, 1)
        self.btn_clear = QPushButton(tr("清空"))
        self.btn_clear.setObjectName("taskClearBtn")
        self.btn_clear.setToolTip(tr("清掉已结束的任务（正在跑的会先请求取消）"))
        self.btn_clear.clicked.connect(self.clear_requested.emit)
        top.addWidget(self.btn_clear)
        root.addLayout(top)

        # ── 任务行容器 ──
        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(3)
        # ⚠️ 末尾必须加弹簧：否则每一行都会**均分**多余高度（实测一行被拉到 480px，
        #   看起来像"卡片里大片空白"）。有了弹簧，余量落在列表底部。
        self._rows_box.addStretch(1)
        root.addLayout(self._rows_box, 1)

        self.empty_label = QLabel(tr("暂无任务\n\n在「标注管理 / 参数管理」里点训练或批量推理，\n任务会登记到这里（串行执行，可单独取消）"))
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setObjectName("taskEmpty")
        # 构造期就是**隐藏**（不是"等refresh 再藏"）——之前先建后藏的那一帧里，
        # 它会按 stretch=1 吃掉半个面板，且行被挤成 0 高（实测截图只剩标题）。
        self.empty_label.setVisible(False)
        root.addWidget(self.empty_label, 1)

        # 队列核心 -> 面板：只在 UI 线程刷新，且 200ms 合并一次
        self._pending = False
        # ★ 跨线程投递走 **Qt 信号**：工作线程 emit，自动排队到 UI 线程。
        #   （曾试过在工作线程里 `QTimer.start(0)` —— 那是未定义行为，
        #     定时器常常根本没排上，而 _pending 已经置 True ⇒ 之后所有刷新被永久挡掉，
        #     症状是"清空后行还在"。信号是唯一可靠的做法。）
        self._queue_changed.connect(self._on_queue_changed)
        self._queue.on_change = self._queue_changed.emit
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush)
        # ⚠️ 必须连 `_flush`（它会复位 `_pending`），**不能**直接连 `refresh`：
        #    那样第一次刷新后 `_pending` 永远是 True ⇒ 之后所有刷新都被"已排着"挡掉，
        #    表现为"只有第一行/之后再提交的任务不出现"（实测：探针 KeyError 找不到行）。
        self.refresh()

    # ── 队列回调（**可能在后台线程**，只投递不碰控件）──
    def _on_queue_changed(self, *_args):
        """（**已在 UI 线程**）排一个 0ms 单次定时器，顺带合并高频回调。

        形参 `*_args`：队列回调是 `on_change(tasks)`，带一个参数；
        信号也必须声明成 `pyqtSignal(object)`，否则 emit 时 TypeError
        会被队列里的 `except Exception: pass` **静默吞掉**（症状：刷新一次都不发生）。

        真正保证线程正确的是上一层：工作线程只 `emit` 一个信号，
        Qt 用QueuedConnection 把它排到 UI 线程，这里才碰得到 QTimer/控件。
        """
        if self._pending:
            return                      # 已有一次刷新排着，等它带上最新状态即可
        self._pending = True
        self._flush_timer.start(0)

    def _flush(self):
        self._pending = False
        self.refresh()

    def _on_filter_changed(self, idx):
        self._filter_state = _FILTER_ITEMS[idx][1] if 0 <= idx < len(_FILTER_ITEMS) else None
        self.refresh()

    def visible_tasks(self):
        ts = self._queue.tasks
        if self._filter_state is None:
            return ts
        return [t for t in ts if t.state is self._filter_state]

    # ── 刷新（**只在 UI 线程**）──
    def refresh(self):
        """幂等刷新：行按 task_id 复用（缺的建、多余的删），已存在的只更新文字。

        ⚠️ 带**重入保护**：队列是后台线程，`on_change` 可能在上一轮 refresh 还没走完时
        又投递一次刷新；不加锁的话两个入口同时改 `self._rows` 与布局，
        实测出现过"DOM 里只剩一行、字典里却有三条"的悬空状态。

        为什么"幂等复用"而不是"结构变了就整体重建"：重建会在**后台线程高频回调**
        （训练每个 batch 一次）下反复 deleteLater/addWidget，行的增删会跟着抖，
        实测还会把布局搞乱（行数与 rows_box 子控件数对不上）。
        """
        if getattr(self, "_in_refresh", False):
            return                       # 已在刷新中：这一轮直接跳过（下一次刷新会带上最新状态）
        self._in_refresh = True
        try:
            self._refresh_impl()
        finally:
            self._in_refresh = False

    def _refresh_impl(self):
        want = self.visible_tasks()
        want_ids = [t.id for t in want]
        have_ids = list(self._rows.keys())

        if want_ids != have_ids:
            # ① 删掉不再需要的（倒序删，避免遍历中改字典）
            for tid in [t for t in have_ids if t not in want_ids]:
                r = self._rows.pop(tid)
                self._rows_box.removeWidget(r["box"])
                # ⚠️ 必须**立即** setParent(None)，不能只 deleteLater：
                #    deleteLater 要等事件循环才销毁，而布局里还留着那个 item
                #    ⇒ 实测出现"已删但仍在布局里"的幽灵行（几何停在 0,0,640,480，
                #    后面新建的真行反而被挤掉，看起来就是"只有一行 + 大片空白"）。
                r["box"].setParent(None)
                r["box"].deleteLater()
            # ② 按当前顺序补齐缺失的（新建的行插在末尾，顺序由列表顺序近似保证）
            for t in want:
                if t.id not in self._rows:
                    row = self._make_row(t)
                    self._rows[t.id] = row
                    self._rows_box.insertWidget(self._rows_box.count() - 1, row["box"])
        for t in want:
            self._update_row(t)
        self.empty_label.setVisible(not want)
        # 行的增删会打乱布局缓存，强制重算一次（否则新行的几何停在初始值）
        self._rows_box.invalidate()
        self._rows_box.activate()

    def _make_row(self, task):
        # ⚠️ 必须**创建时就给父级**：无父级的 QWidget 由 Python 拥有，
        #    一旦 dict 引用被替换就可能被 GC 掉，而布局里还留着它的 item
        #    ⇒ 出现"DOM 里找不到、行数据却还在"的悬空行（实测 taskB/taskC 消失）。
        box = QWidget(self)
        box.setObjectName("taskRow")
        # 行必须钉住高度：默认 Preferred 会让每一行均分多余空间（实测 480px 高的"空卡片"）
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 3, 4, 3)
        lay.setSpacing(2)

        line1 = QHBoxLayout(); line1.setSpacing(4)
        kind = QLabel(tr(KIND_LABEL.get(task.kind, "任务")))
        kind.setObjectName("taskKind")
        title = QLabel(task.title)
        title.setObjectName("taskTitle")
        title.setToolTip(task.title + (("\n" + task.project) if task.project else ""))
        state = QLabel(tr(_STATE_TEXT.get(task.state, task.state.value)))
        state.setObjectName("taskState")
        btn_x = QPushButton("×")
        btn_x.setObjectName("taskCancelBtn")
        btn_x.setFixedSize(20, 20)
        btn_x.setToolTip(tr("取消这个任务（正在跑的会在下一个检查点停下）"))
        btn_x.clicked.connect(lambda _c=False, tid=task.id: self.cancel_requested.emit(tid))
        line1.addWidget(kind)
        line1.addWidget(title, 1)
        line1.addWidget(state)
        line1.addWidget(btn_x)
        lay.addLayout(line1)

        line2 = QHBoxLayout(); line2.setSpacing(4)
        bar = QProgressBar()
        bar.setObjectName("taskProgress")
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        status = QLabel("")
        status.setObjectName("taskStatus")
        detail = QLabel("")
        detail.setObjectName("taskDetail")
        detail.setToolTip(task.detail or "")
        line2.addWidget(bar, 1)
        line2.addWidget(status)
        lay.addLayout(line2)
        if task.detail:
            lay.addWidget(detail)

        row = {"box": box, "kind": kind, "title": title, "state": state,
               "bar": bar, "status": status, "detail": detail, "btn_x": btn_x}
        self._update_row(task, row)
        return row

    def _update_row(self, task, row=None):
        r = row or self._rows.get(task.id)
        if r is None:
            return
        r["state"].setText(tr(_STATE_TEXT.get(task.state, task.state.value)))
        if task.total > 0:
            r["bar"].setMaximum(task.total)
            r["bar"].setValue(task.done)
            r["bar"].setVisible(True)
        else:
            # 总量未知（比如总图数还没统计出来）：转圈表示"在动"，
            # 刻意**不显示 0%** —— 那会让用户以为卡住了
            r["bar"].setRange(0, 0) if task.state is TaskState.RUNNING else r["bar"].setRange(0, 1)
            r["bar"].setValue(0)
        r["status"].setText(_localize_status(task.status_line))
        if task.error:
            r["detail"].setText("⚠ " + task.error)
            r["detail"].setToolTip(task.error)
        elif r["detail"].text() != (task.detail or ""):
            r["detail"].setText(task.detail or "")
            r["detail"].setToolTip(task.detail or "")
        # 已结束的项不再需要取消按钮
        r["btn_x"].setVisible(task.state in (TaskState.QUEUED, TaskState.RUNNING))