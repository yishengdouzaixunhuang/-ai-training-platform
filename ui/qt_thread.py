# -*- coding: utf-8 -*-
"""Qt 线程安全工具：把后台线程的 UI 更新排队回主线程。

## 为什么需要它

Qt 的控件**不是线程安全的**：任何 QWidget 的方法（setText / setVisible /
setEnabled / setItem …）都必须在 GUI 主线程里调用。在工作线程里直接调用，
行为是未定义的 —— 大多数时候看着没事，负载一高就随机崩溃、白屏、或者卡死。

本项目里 `ui/main_window.py` 曾经有 21 处 `threading.Thread(daemon=True)`
直接起裸线程，其中 16 个函数体内又直接操作控件，比如：

    threading.Thread(target=run, daemon=True).start()

    def run():
        ...
        self.cls_train_status.setText(f"Epoch {epoch}/{total}")   # ⚠️ 子线程写控件

项目原本已经定义了 `log_signal / progress_signal` 等信号，思路是对的，
但只是零星用了几个，其余仍是裸调用。这里把「回主线程」这件事收成一个工具，
以后所有跨线程 UI 更新统一走 `ui_call(...)`，不再各写各的信号。

## 用法

    from ui.qt_thread import ui_call, run_bg

    # 1) 后台线程里更新界面
    ui_call(self.status_label.setText, "正在训练…")

    # 2) 更省事：整段搬到后台，里面的 UI 调用自己用 ui_call 兜住
    run_bg(self._do_heavy_work, on_error=lambda e: ui_call(self.log, str(e)))

注意：`init_ui_invoker()` 必须在主线程、QApplication 之后调用一次（main.py 已调用）。
"""
from __future__ import annotations

import functools
import threading
import traceback

from PyQt5.QtCore import QObject, Qt, pyqtSignal


class _UiInvoker(QObject):
    """住在主线程的转发器：收到任何线程发来的可调用对象，就在主线程执行。"""

    _invoke = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        # QueuedConnection 是关键：即使 emit 来自工作线程，
        # 槽也会被排进主线程的事件队列，而不是当场执行。
        self._invoke.connect(self._execute, Qt.QueuedConnection)

    def _execute(self, fn):
        try:
            fn()
        except Exception:
            # 这里不能静默：UI 更新出错往往意味着状态已经坏了
            traceback.print_exc()


_INVOKER: _UiInvoker | None = None
_BG_POOL = None


def init_ui_invoker():
    """在主线程、QApplication 创建之后调用一次。返回转发器实例。"""
    global _INVOKER
    if _INVOKER is None:
        _INVOKER = _UiInvoker()
    return _INVOKER


def ui_call(fn, *args, **kwargs):
    """从任意线程安全地在主线程执行 `fn(*args, **kwargs)`。

    - 已在主线程：直接同步执行，保持原来的调用时序（这点很重要，
      有些代码依赖 setText 之后立刻读到值）。
    - 在子线程：排队到主线程事件循环，立即返回。
    - 转发器未初始化（没跑 init_ui_invoker）：退化为直接调用，并打一次警告，
      避免因为没初始化而把功能整个卡死。
    """
    inv = _INVOKER
    if inv is None:
        if threading.current_thread() is threading.main_thread():
            return fn(*args, **kwargs)
        if not getattr(ui_call, "_warned", False):
            ui_call._warned = True
            print("[qt_thread] 警告：未调用 init_ui_invoker()，跨线程 UI 调用未受保护")
        return fn(*args, **kwargs)

    if threading.current_thread() is threading.main_thread():
        return fn(*args, **kwargs)

    inv._invoke.emit(functools.partial(fn, *args, **kwargs))
    return None


def ui_call_sync(fn, *args, timeout=15.0, **kwargs):
    """和 ui_call 一样，但**阻塞等待**主线程执行完成并返回结果。

    什么时候必须要它：子线程算完一批数据，紧接着要拿到主线程里控件的返回值，
    或者「先把表格填好，再基于表格内容继续算」。用异步 ui_call 会读到空值。

    ⚠️ 死锁风险：如果主线程此刻正在 join 这个子线程，就会互相等。
      本项目的主线程从不 join 后台任务（只用 daemon 线程），所以是安全的。
    """
    inv = _INVOKER
    if inv is None or threading.current_thread() is threading.main_thread():
        return fn(*args, **kwargs)

    box: dict = {}
    done = threading.Event()

    def _wrap():
        try:
            box["r"] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            box["e"] = exc
        finally:
            done.set()

    inv._invoke.emit(_wrap)
    if not done.wait(timeout):
        raise TimeoutError("ui_call_sync 超时（%.0fs）：主线程可能被长任务阻塞" % timeout)
    if "e" in box:
        raise box["e"]
    return box.get("r")


def run_bg(fn, *args, on_done=None, on_error=None, name=None, **kwargs):
    """在后台线程跑 `fn`，并把结果/异常安全地送回主线程。

    on_done(result) 与 on_error(exc) 都在主线程执行，可以直接碰控件。
    返回值是 threading.Thread，需要时可 join（一般不用）。
    """

    def _wrapper():
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            if on_error is not None:
                ui_call(on_error, exc)
            return
        if on_done is not None:
            ui_call(on_done, result)

    t = threading.Thread(target=_wrapper, daemon=True, name=name or getattr(fn, "__name__", "bg"))
    t.start()
    return t


def is_main_thread():
    return threading.current_thread() is threading.main_thread()
