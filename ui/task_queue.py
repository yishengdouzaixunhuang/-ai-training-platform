# -*- coding: utf-8 -*-
"""任务队列（参照友商 iCogtiveFlow 的「任务队列」模块）。

**为什么需要**：平台原本是"单槽"训练（`self.trainer` 一个槽、`_stop_flag` 一个全局标志），
同一时间只能跑一个任务，第二个任务会把第一个顶掉；进度只显示在一个进度条 + 一个 Loss 窗口里。
友商的做法是把**训练与推理都登记成队列项**，每行看得到进度、剩余时间，并能单独取消。

**三条设计决定（都写在这儿，别后来人"优化"掉）**：

1. **串行，不并发**。单卡机器上并发训练只会 OOM；GPU 本来就该排队。
   队列 = 先进先出，**同一时刻只有一个 run() 在跑**（`self._running` 标记）。
2. **持久化终态任务**（2026-10-29 改）。原设计是"不持久化，关平台即丢，
   痕迹看 `_train_logs/*.log` 与工程 `outputs/`"。**实测这个判断站不住**：
   用户重启后打开「任务队列」面板只看到"暂无任务"，筛「已完成」也是空的 ——
   他不知道刚才跑过什么、跑成功没、耗时多久，还得去翻日志文件。
   现在把**终态任务**（done/failed/canceled）写进 `.task_history.json`，
   最多留 200 条；运行中/排队的**不存**（重启后它们已经不存在，存了反而误导）。
   加载时若有历史 RUNNING 记录，一律降级成 CANCELED —— 那是上次没跑完就退出的残影。
3. **取消是"请求式"**：`request_cancel()` 只置标志，真正停下来要等任务自己检查到。
   训练在 epoch 边界检查、推理在每张图后检查 ⇒ 最坏等一个 epoch/一张图。
   **绝不在队列线程里强杀线程** —— 那样会留下半个写坏的权重文件。
4. **停止标志挂在任务上，不是全局**（2026-11-11，重构清单 P0-3）。原来平台只有一个
   `self._stop_flag`：取消任务 A 会连坐任务 B；"取消一次"之后后面每个任务都会秒停
   （于是每处 `finally` 都得记得复位，漏一处就复发）。现在 `task.cancel_requested`
   就是那个标志，任务结束随对象消失，**没有任何地方需要复位**。
   交接文档第 20 条那个 `self.trainer` 单槽崩溃，根因也是同一个"共享可变状态"。

ETA 估算：`已完成步数 / 已用时` 线性外推。对训练用 epoch 级进度，对推理用图片级进度。
早期样本太少（<2 个进度点或<3%完成度）时**故意不显示 ETA** —— 那一刻的估算纯属噪声，
显示一个"剩余 6 小时"再变成"剩余 20 分钟"，比不给更让人不信任。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

# ── 历史任务持久化 ────────────────────────────────────────────────
# 存在项目根目录（不是工作区）：任务队列是平台级的面板，跨工程共用一份历史，
# 每条记录自带 project 字段用于区分。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HISTORY_PATH = os.path.join(_ROOT, ".task_history.json")
MAX_HISTORY = 200          # 只留最近 200 条，再老的自动丢弃


class TaskKind(str, Enum):
    TRAIN = "train"
    INFER = "infer"
    OTHER = "other"


class TaskState(str, Enum):
    QUEUED = "queued"        # 排队中
    RUNNING = "running"      # 执行中
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"


#: 显示名（跟友商一致的口语化写法）
KIND_LABEL = {TaskKind.TRAIN: "训练", TaskKind.INFER: "推理", TaskKind.OTHER: "任务"}


def fmt_eta(seconds):
    """把秒数写成 `1小时02分03秒` / `59分42秒` / `42秒`；不可估返回 ''。"""
    if seconds is None:
        return ""
    try:
        s = int(max(0, round(float(seconds))))
    except Exception:
        return ""
    if s <= 0:
        return "0秒"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return "%d小时%02d分%02d秒" % (h, m, sec)
    if m:
        return "%d分%02d秒" % (m, sec)
    return "%d秒" % sec


@dataclass
class Task:
    """一个队列项。`run(ctx)` 由调用方给，ctx 提供 progress/log/is_canceled 三个回调。"""

    kind: TaskKind
    title: str
    project: str = ""
    # ⚠️ dataclass **只把带注解的属性当字段**：`run=None`（无注解）会变成普通类属性，
    #    于是 `Task(run=...)` 直接报 unexpected keyword argument。注解不能省。
    run: object = None               # 可调用对象：fn(ctx) -> 任意返回值；None 表示只登记不执行
    total: int = 0                # 总步数（epoch 数 / 图片数），0 = 未知
    done: int = 0
    state: TaskState = TaskState.QUEUED
    detail: str = ""              # 副标题（模型名、批量推理的目录…）
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    cancel_requested: bool = False
    #: 当前阶段（""|"preparing"|"training"|"validating"|"saving"|"inferring"），
    #: 由任务自己用 `ctx.phase("validating")` 上报，UI 只读不算。
    phase: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # ── 停止标志（**每个任务自己一份**）──
    # 交接文档第 20 条那个 bug 的根因：平台原来是**一个全局 `_stop_flag`**，
    # 任务 A 的取消会误伤任务 B，而且"取消一次"之后后面每个任务都会秒停。
    # 现在标志挂在 Task 上，任务结束随对象消失，不需要任何地方"记得复位"。
    # ⚠️ 刻意**不新增字段**：字段名就叫 `cancel_requested`（取消请求），
    #    `stop_flag` 只是它的读写别名 —— 两份状态迟早会打架。
    @property
    def stop_flag(self):
        """这个任务自己的停止标志（`cancel_requested` 的别名，读写同一份）。"""
        return bool(self.cancel_requested)

    @stop_flag.setter
    def stop_flag(self, value):
        self.cancel_requested = bool(value)

    # ── 进度与 ETA ──
    @property
    def percent(self):
        if self.total <= 0:
            return 0.0
        return max(0.0, min(100.0, self.done * 100.0 / self.total))

    @property
    def eta_seconds(self):
        """剩余秒数；样本不足时返回 None（**宁可不显示，也不给噪声数字**）。"""
        if self.state is not TaskState.RUNNING or self.total <= 0:
            return None
        if self.done < 1:
            return None
        elapsed = time.time() - (self.started_at or time.time())
        if elapsed < 3:                       # 太短，速率还没稳
            return None
        per_step = elapsed / self.done
        remain = (self.total - self.done) * per_step
        # 完成度 <3% 时外推极不可靠
        if self.percent < 3 and self.done < 3:
            return None
        return remain

    @property
    def eta_text(self):
        return fmt_eta(self.eta_seconds)

    @property
    def elapsed_text(self):
        if not self.started_at:
            return ""
        end = self.finished_at or time.time()
        return fmt_eta(end - self.started_at)

    # ── 行文本（给 UI 用；UI 只读这里，不自己算）──
    @property
    def status_line(self):
        """`进度 12.3%  剩余 12分30秒` —— ETA 不可估时只给进度。"""
        pct = "%.2f%%" % self.percent if self.total > 0 else "--"
        eta = self.eta_text
        return ("%s  %s" % (pct, ("剩余 " + eta) if eta else "")).strip()


class TaskQueue:
    """串行任务队列：先进先出，同一时刻只跑一个。

    刻意**不含任何Qt 代码** —— 这样它能被单测直接驱动（不建窗口也能跑完整生命周期）。
    UI 侧通过 `on_change` 回调刷新。
    """

    def __init__(self, on_change=None, history_path=None):
        self._tasks = []                      # 全部（含已完成），按创建顺序
        self._running = None                 # 当前正在跑的 Task
        self.on_change = on_change           # callable(task_list) -> None
        self._idle_hook = None               # 队列空时调用（UI 可用来关进度条）
        self._history_path = history_path or DEFAULT_HISTORY_PATH
        self._load_history()                 # 把上次的终态任务读回来

    # ── 只读 ──
    @property
    def tasks(self):
        return list(self._tasks)

    def active_tasks(self):
        return [t for t in self._tasks if t.state in (TaskState.QUEUED, TaskState.RUNNING)]

    def is_busy(self):
        return self._running is not None or any(
            t.state is TaskState.QUEUED for t in self._tasks)

    def get(self, task_id):
        for t in self._tasks:
            if t.id == task_id:
                return t
        return None

    @property
    def running(self):
        """当前正在跑的那个任务（没有就 None）。「停止」按钮靠它找目标。"""
        return self._running

    # ── 变更通知 ──
    def _notify(self):
        try:
            if self.on_change:
                self.on_change(self.tasks)
        except Exception:
            pass

    # ── 登记 ──
    def submit(self, kind, title, project="", run=None, total=0, detail=""):
        t = Task(kind=kind, title=title, project=project, run=run, total=total, detail=detail)
        self._tasks.append(t)
        self._notify()
        self._pump()
        return t

    # ── 取消 ──
    def request_cancel(self, task_id):
        """请求取消：**置标志，不强杀**。正在跑的那个要等它自己检查到。"""
        t = self.get(task_id)
        if t is None or t.state in (TaskState.DONE, TaskState.FAILED, TaskState.CANCELED):
            return False
        t.cancel_requested = True
        if t.state is TaskState.QUEUED:
            #还没开始，直接置取消（不会去跑它）
            t.state = TaskState.CANCELED
            t.finished_at = time.time()
        self._notify()
        return True

    def cancel_all(self):
        for t in self.active_tasks():
            self.request_cancel(t.id)
        self._notify()

    def clear_finished(self):
        """清空已完成/失败/取消的项；**正在跑的不动**。"""
        self._tasks = [t for t in self._tasks
                       if t.state in (TaskState.QUEUED, TaskState.RUNNING)]
        self._save_history()                 # 用户点了"清空"，历史文件也得清
        self._notify()

    def clear_all(self):
        """清空所有（先请求取消在跑的，再删历史）。"""
        self.cancel_all()
        self._tasks = []
        self._save_history()
        self._notify()

    # ── 执行 ──
    # ── 历史持久化 ──────────────────────────────────────────────
    def _load_history(self):
        """把上次的终态任务读回队列。

        ⚠️ 运行中/排队中的记录一律降级成 CANCELED：
           那是"上次没跑完就退出"的残影，若原样恢复，UI 会以为它还在跑。
        """
        try:
            if not os.path.exists(self._history_path):
                return
            with open(self._history_path, "r", encoding="utf-8") as f:
                rows = json.load(f) or []
        except Exception:
            return                      # 文件坏了就当没有，绝不能让队列起不来
        if not isinstance(rows, list):
            return
        for r in rows:
            try:
                st = TaskState(str(r.get("state", "done")))
                if st in (TaskState.RUNNING, TaskState.QUEUED):
                    st = TaskState.CANCELED
                t = Task(
                    kind=TaskKind(str(r.get("kind", "other"))),
                    title=str(r.get("title", "")),
                    project=str(r.get("project", "")),
                    total=int(r.get("total", 0) or 0),
                    done=int(r.get("done", 0) or 0),
                    state=st,
                    detail=str(r.get("detail", "") or ""),
                    error=str(r.get("error", "") or ""),
                    created_at=float(r.get("created_at", 0) or 0),
                    started_at=float(r.get("started_at", 0) or 0),
                    finished_at=float(r.get("finished_at", 0) or 0),
                )
                if r.get("id"):
                    t.id = str(r["id"])
                self._tasks.append(t)
            except Exception:
                continue                # 单条坏了跳过，不连累其它
        self._tasks.sort(key=lambda x: x.created_at)

    def _save_history(self):
        """只写终态任务（运行中的不写）。原子替换，写一半断电也不会坏文件。"""
        try:
            rows = []
            for t in self._tasks:
                if t.state in (TaskState.RUNNING, TaskState.QUEUED):
                    continue
                rows.append({
                    "id": t.id,
                    "kind": t.kind.value,
                    "title": t.title,
                    "project": t.project,
                    "total": t.total,
                    "done": t.done,
                    "state": t.state.value,
                    "detail": t.detail,
                    "error": t.error,
                    "created_at": t.created_at,
                    "started_at": t.started_at,
                    "finished_at": t.finished_at,
                })
            rows = rows[-MAX_HISTORY:]
            tmp = self._history_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self._history_path)      # 原子替换
        except Exception:
            pass                        # 存不上不影响训练，静默即可

    def _pump(self):
        """有空闲且有排队项就启动下一个（**只启动一个**）。"""
        if self._running is not None:
            return
        nxt = next((t for t in self._tasks if t.state is TaskState.QUEUED), None)
        if nxt is None:
            if self._idle_hook:
                try:
                    self._idle_hook()
                except Exception:
                    pass
            return
        self._start(nxt)

    def _start(self, task):
        import threading

        task.state = TaskState.RUNNING
        task.started_at = time.time()
        task.cancel_requested = False
        self._running = task
        self._notify()

        def _run():
            try:
                if task.run is not None:
                    task.run(_Ctx(self, task))
                task.state = TaskState.DONE
            except _Canceled:
                task.state = TaskState.CANCELED
            except Exception as e:                # 任务失败**不许炸掉整个平台**
                task.state = TaskState.FAILED
                task.error = "%s: %s" % (type(e).__name__, e)
            finally:
                task.finished_at = time.time()
                self._running = None
                self._save_history()             # 终态落盘：重启后还能看到跑过什么
                self._notify()
                self._pump()                     # 接力下一个

        threading.Thread(target=_run, daemon=True,
                         name="aitp-task-%s" % task.kind.value).start()


class _Canceled(Exception):
    """任务内部检查到取消标志时抛这个，让队列把它记成 canceled 而不是 failed。"""


class _Ctx:
    """交给任务 run() 的上下文：推进度、写日志、查取消。"""

    def __init__(self, queue, task):
        self._q = queue
        self._t = task

    def progress(self, done, total=None):
        if total is not None and total > 0:
            self._t.total = int(total)
        self._t.done = int(done)
        self._q._notify()

    def log(self, msg):
        self._t.detail = str(msg)[-200:]
        self._q._notify()

    def phase(self, name):
        """上报当前阶段（"preparing"/"training"/"validating"/"saving"/"inferring"）。

        只是给 UI 显示用；**不参与任何判断**，所以传什么都不会改变执行流程。
        """
        self._t.phase = str(name or "")
        self._q._notify()

    @property
    def canceled(self):
        return bool(self._t.cancel_requested)

    @property
    def stop_requested(self):
        """同 `canceled`，但读起来更像"要不要停" —— 给 `stop_check=` 用。

        ⚠️ 别把 `self._t.cancel_requested` 存成局部 bool 再传出去：
           那是一次性快照，之后用户点取消它就永远看不到。
        """
        return bool(self._t.cancel_requested)

    def check_cancel(self):
        """在循环里调用：被请求取消就抛 `_Canceled`，队列会记成 canceled。"""
        if self._t.cancel_requested:
            raise _Canceled()

    @property
    def task(self):
        return self._t