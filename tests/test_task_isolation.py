# -*- coding: utf-8 -*-
"""任务级停止标志的回归测试（重构清单 P0-3）。

## 为什么不能"跑一遍不崩"就算过

原来平台只有一个全局 `self._stop_flag`：

- 取消**排队中的推理 B**，会把**正在跑的训练 A** 一起停掉（连坐）；
- A 被取消之后，那个标志要靠各处 `finally` 复位，漏一处就是
  "后面每个任务都在第一步就停"（秒停）；
- 分类训练当时压根不走队列，和分割训练共用这个标志 —— 互相抹状态。

这类 bug 的复现窗口是"A 正在跑、B 还在排队"的**那一瞬间**，正常点一遍根本碰不到。
所以这里的用例全部显式构造时序：起一个卡住的任务当"正在跑的那个"，
再往队列里塞第二个，然后对第二个动手。

## 两条底线

1. 取消一个任务，**不许**影响别的任务的标志（`stop_flag` 各归各）；
2. 任务结束/新任务开始，**没有任何地方需要复位**（标志随对象消失）。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_task_isolation.py
"""
import os
import sys
import tempfile
import threading
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def wait_until(pred, timeout=10.0, desc="条件"):
    """轮询等到 pred() 为真；超时抛异常（带描述，避免"卡住"被当成"通过"）。"""
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("等待超时：%s" % desc)


def tmp_history():
    fd, p = tempfile.mkstemp(prefix="aitp_tq_", suffix=".json")
    os.close(fd)
    os.unlink(p)                        # 让队列自己新建，别读进真实历史
    return p


# ---------------------------------------------------------------- 队列层

def test_two_tasks_flags_independent():
    """A 正在跑、B 在排队；取消 B 不许影响 A。"""
    from ui.task_queue import TaskQueue, TaskKind, TaskState

    q = TaskQueue(history_path=tmp_history())
    gate = threading.Event()
    seen = {}

    def run_a(ctx):
        seen["ctx_a"] = ctx
        ctx.phase("training")
        gate.wait(5)                    # 卡住，模拟"正在训练"
        # 被要求停就在这里抛（真实训练在 epoch 边界做同样的事）
        ctx.check_cancel()

    def run_b(ctx):
        seen["ctx_b"] = ctx

    a = q.submit(TaskKind.TRAIN, "A", run=run_a)
    wait_until(lambda: a.state is TaskState.RUNNING, desc="A 开始运行")
    b = q.submit(TaskKind.INFER, "B", run=run_b)
    assert b.state is TaskState.QUEUED, "B 应该还在排队（队列串行）"

    q.request_cancel(b.id)
    assert b.state is TaskState.CANCELED, "B 应立刻变 canceled（还没开始）"
    assert a.state is TaskState.RUNNING, "取消 B 不该动正在跑的 A"
    assert a.stop_flag is False, "A 的停止标志被 B 的取消污染了"
    assert seen["ctx_a"].stop_requested is False, "A 的 ctx.stop_requested 被污染"

    gate.set()
    wait_until(lambda: a.state is TaskState.DONE, desc="A 正常跑完")


def test_cancel_running_then_next_not_poisoned():
    """取消正在跑的 A 之后，**下一个任务不许秒停**（原全局标志的经典复发点）。"""
    from ui.task_queue import TaskQueue, TaskKind, TaskState, _Canceled

    q = TaskQueue(history_path=tmp_history())
    gate = threading.Event()
    started = {"a": False}

    def run_a(ctx):
        started["a"] = True
        gate.wait(5)
        while True:                     # 模拟 epoch 循环：只在边界检查
            ctx.check_cancel()
            time.sleep(0.01)

    a = q.submit(TaskKind.TRAIN, "A", run=run_a)
    wait_until(lambda: started["a"], desc="A 开始")
    q.request_cancel(a.id)
    gate.set()
    wait_until(lambda: a.state is TaskState.CANCELED, desc="A 被取消")

    ran = {"c": False}

    def run_c(ctx):
        ran["c"] = True
        ctx.check_cancel()              # 新任务第一次检查就必须是"没被要求停"

    c = q.submit(TaskKind.TRAIN, "C", run=run_c)
    wait_until(lambda: c.state is TaskState.DONE, desc="C 正常完成")
    assert ran["c"], "C 根本没跑起来"
    assert c.stop_flag is False, "新任务继承了上一个任务的取消状态（秒停复发）"


def test_phase_reporting():
    """`ctx.phase()` 只给 UI 看，必须原样落到 task.phase 上。"""
    from ui.task_queue import TaskQueue, TaskKind, TaskState

    q = TaskQueue(history_path=tmp_history())
    seen = {}

    def run(ctx):
        ctx.phase("preparing")
        seen["mid"] = ctx.task.phase
        ctx.phase("validating")

    t = q.submit(TaskKind.TRAIN, "P", run=run)
    wait_until(lambda: t.state is TaskState.DONE, desc="任务完成")
    assert seen["mid"] == "preparing", "phase 没落到任务上：%r" % seen.get("mid")
    assert t.phase == "validating", "最后一次 phase 应保留：%r" % t.phase


def test_stop_flag_single_source():
    """`stop_flag` 是 `cancel_requested` 的别名，读写同一份（不能有两份状态）。"""
    from ui.task_queue import Task

    t = Task(kind=None, title="x")
    assert t.stop_flag is False and t.cancel_requested is False
    t.stop_flag = True
    assert t.cancel_requested is True, "写 stop_flag 没同步到 cancel_requested"
    t.cancel_requested = False
    assert t.stop_flag is False, "写 cancel_requested 没同步到 stop_flag"


def test_running_property():
    """`queue.running` 指向正在跑的那个；空闲时是 None。「停止」按钮靠它找目标。"""
    from ui.task_queue import TaskQueue, TaskKind, TaskState

    q = TaskQueue(history_path=tmp_history())
    gate = threading.Event()

    def run(ctx):
        gate.wait(5)

    assert q.running is None, "空队列的 running 应该是 None"
    t = q.submit(TaskKind.TRAIN, "R", run=run)
    wait_until(lambda: t.state is TaskState.RUNNING, desc="开始运行")
    assert q.running is t, "running 没指向正在跑的任务"
    gate.set()
    wait_until(lambda: t.state is TaskState.DONE, desc="完成")
    assert q.running is None, "跑完后 running 应复位"


# ---------------------------------------------------------------- 接线层（真窗口）

_APP = None
_WIN = None


def _window():
    """构造一个真 MainWindow（offscreen），并把任务历史指到临时文件。"""
    global _APP, _WIN
    if _WIN is not None:
        return _WIN
    try:
        import torch  # noqa: F401  必须先于 PyQt5
    except OSError:
        pass
    from PyQt5.QtWidgets import QApplication
    import ui.task_queue as tq

    tq.DEFAULT_HISTORY_PATH = tmp_history()     # 别写真实的 .task_history.json
    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    _APP = QApplication.instance() or QApplication([])
    init_ui_invoker()
    _WIN = MainWindow()
    _WIN._task_queue.clear_all()
    return _WIN


def _drain(w):
    """把窗口队列排干后再开始下一组时序。

    ⚠️ 五个接线用例共用一个 MainWindow（建窗口要 1 秒，不想每题重建）。
       上一个用例留下的任务若还在收尾，下一个用例 submit 的新任务会排在它后面，
       于是 `wait_until(state is RUNNING)` 等的是**别人**的任务 ——
       表现为"偶尔超时"，而单跑一遍永远复现不了。
    """
    q = w._task_queue
    q.cancel_all()
    wait_until(lambda: q.running is None and not q.active_tasks(),
               desc="队列排干（等上一个用例收尾）")
    q.clear_finished()


def _starting_pair(w, kind_running, kind_queued):
    """在真窗口的队列里造"A 正在跑、B 在排队"的时序。

    A 会像真实训练那样**在循环里检查取消并抛 `_Canceled`**（不是干等），
    否则"取消后应变成 canceled"这条断言必然超时 —— 队列只在 run 抛
    `_Canceled` 时才记 canceled，这是它的设计（不强杀线程）。
    """
    from ui.task_queue import TaskKind, TaskState, _Canceled

    _drain(w)
    gate = threading.Event()
    seen = {}

    def run_running(ctx):
        seen["running_ctx"] = ctx
        end = time.time() + 8
        while time.time() < end:
            if ctx.stop_requested:
                raise _Canceled()          # 真实训练在 epoch 边界做同样的事
            if gate.wait(0.005):
                return

    def run_queued(ctx):
        seen["queued_ctx"] = ctx

    a = w._task_queue.submit(kind_running, "running-task", run=run_running)
    wait_until(lambda: a.state is TaskState.RUNNING, desc="running-task 开始")
    b = w._task_queue.submit(kind_queued, "queued-task", run=run_queued)
    wait_until(lambda: b.state is TaskState.QUEUED, desc="queued-task 进队列")
    return a, b, gate, seen


def test_on_task_cancel_does_not_touch_global():
    """★ 主 bug：取消排队中的推理，不许把正在跑的训练停掉，也不许动全局标志。"""
    from ui.task_queue import TaskKind, TaskState

    w = _window()
    a, b, gate, seen = _starting_pair(w, TaskKind.TRAIN, TaskKind.INFER)
    try:
        w._stop_flag = False
        w._on_task_cancel(b.id)
        assert b.state is TaskState.CANCELED, "B 应被取消"
        assert a.stop_flag is False, "★ 取消 B 把正在跑的 A 一起停了"
        assert a.state is TaskState.RUNNING, "A 不该被取消"
        assert w._stop_flag is False, "_on_task_cancel 又去动了全局标志"
    finally:
        gate.set()
        wait_until(lambda: a.state is TaskState.DONE, desc="A 收尾")


def test_stop_buttons_only_hit_matching_kind():
    """「停止推理」不许停训练；「停止训练」才停训练。"""
    from ui.task_queue import TaskKind, TaskState

    w = _window()
    a, b, gate, seen = _starting_pair(w, TaskKind.TRAIN, TaskKind.INFER)
    try:
        w._stop_flag = False
        w._stop_batch()                       # 推理按钮：目标是 INFER
        assert a.stop_flag is False, "★ 点「停止推理」把训练 A 停了"
        w._stop_training()                    # 训练按钮：目标是 TRAIN
        assert a.stop_flag is True, "点「停止训练」没停到正在跑的训练"
        wait_until(lambda: a.state is TaskState.CANCELED, desc="A 被取消")
    finally:
        gate.set()
        w._task_queue.clear_finished()


def test_stop_training_does_not_stop_bare_infer():
    """★ 审核实测的那条：点「停止训练」，不许把**裸线程**批量推理一起停掉。

    分类批量 / 三视图 / 检测批量 / OCR 都不走队列，各自持一个令牌（`_infer_stop`）。
    改动前它们读全局 `_stop_flag`，而 `_stop_training` 无条件置它 ⇒ 静默误停。
    """
    w = _window()
    _drain(w)
    w._stop_flag = False
    token = w._infer_stop()               # 模拟一条裸线程推理拿到的判据
    assert token() is False, "新令牌不该是「已停」"

    w._stop_training()                     # 点「停止训练」
    assert token() is False, "★ 点「停止训练」把正在跑的非队列推理一起停了"

    w._stop_batch()                        # 点「停止推理」才该停它
    assert token() is True, "「停止推理」没停到裸线程推理"


def test_stop_infer_does_not_touch_training_flag():
    """反过来也一样：「停止推理」不许动训练的全局标志（那是老路径训练的兜底判据）。"""
    w = _window()
    _drain(w)
    w._stop_flag = False
    w._stop_batch()
    assert w._stop_flag is False, "★ 点「停止推理」把训练的全局标志也置了"


def test_infer_tokens_are_per_run():
    """两条推理各持一个令牌：取消第一条，第二条的判据必须还是"没停"。"""
    w = _window()
    _drain(w)
    first = w._infer_stop()
    second = w._infer_stop()
    assert first is not second, "两次推理拿到了同一个判据（又变成共享状态）"
    w._stop_batch()                        # 取消所有在跑的
    assert first() is True and second() is True, "停止请求应覆盖所有在跑的推理"
    third = w._infer_stop()                # 下一轮推理是新令牌
    assert third() is False, "★ 新一轮推理继承了上一轮的停止状态"


def test_dispatch_targets_exist():
    """`_dispatch_batch_inference` 里 `self._xxx(` 调的每个方法都必须真的存在。

    ★ 实际抓到的 bug：它原来调 `self._start_det_batch_inference()` —— 这个方法
      **全项目不存在**，于是"检测工程的批量推理"一进队列就 AttributeError（任务变 failed）。
      队列会把异常记成 failed 而不弹窗，用户只看到"跑了一下就说失败了"。
    """
    import inspect
    import re
    from ui.main_window import MainWindow

    for fn in (MainWindow._dispatch_batch_inference, MainWindow._batch_inference):
        src = inspect.getsource(fn)
        names = set(re.findall(r"self\.(_[A-Za-z0-9_]+)\(", src))
        missing = sorted(n for n in names if not hasattr(MainWindow, n))
        assert not missing, "%s 里调了不存在的方法：%s" % (fn.__name__, missing)


def test_no_reset_needed_anywhere():
    """跑完一个被取消的任务后，新任务直接开跑，谁都不用"记得复位"。"""
    from ui.task_queue import TaskKind, TaskState

    w = _window()
    _drain(w)
    q = w._task_queue
    seen = {}

    def run1(ctx):
        ctx.task.stop_flag = True
        ctx.check_cancel()

    t1 = q.submit(TaskKind.TRAIN, "cancel-me", run=run1)
    wait_until(lambda: t1.state is TaskState.CANCELED, desc="第一个被取消")

    def run2(ctx):
        seen["flag"] = ctx.task.stop_flag
        seen["global"] = w._stop_flag

    t2 = q.submit(TaskKind.TRAIN, "next", run=run2)
    wait_until(lambda: t2.state is TaskState.DONE, desc="第二个正常完成")
    assert seen["flag"] is False, "★ 新任务继承了取消状态（秒停）"
    q.clear_finished()


def test_cls_training_goes_through_queue():
    """★ 分类训练必须登记成队列任务，并且把**任务自己的**标志传下去。

    它原来是裸起线程（不走队列）：队列的串行保证管不到它，它和分割训练共用一个
    全局 `_stop_flag`。这条用例用一个假训练器跑通整条链 ——
    真训练太慢，但"标志从哪来"这件事必须被钉住。

    断言四件事：
      1. 提交的是 TRAIN 任务（能排队、能在队列页看到）
      2. `stop_check()` 读的是**这个任务**的标志（改任务标志就变 True）
      3. 提交时没有去动全局 `_stop_flag`（那正是当年抹掉别人取消状态的那句）
      4. 训练器是在任务线程里构造的（每个任务一份实例）
    """
    import classification.trainer as ctr
    from ui.task_queue import TaskKind, TaskState

    w = _window()
    _drain(w)
    tmp = tempfile.mkdtemp(prefix="cls_")
    os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
    w.current_project = {"name": "stub"}
    w.pm.get_project_dir = lambda name: tmp
    w._save_cls_train_settings = lambda: None

    seen = {}

    class FakeTrainer:
        def __init__(self, project_dir, model_name=None, **kw):
            self.history = {}
            seen["init_thread"] = threading.current_thread().name

        def train(self, **kw):
            seen["stop_check"] = kw.get("stop_check")
            seen["stop_before"] = kw["stop_check"]()
            seen["task_flag"] = w._task_queue.running.stop_flag
            kw["stop_check"] = None            # 别在假训练里真停
            seen["stop_after_cancel"] = None

    orig = ctr.ClassificationTrainer
    ctr.ClassificationTrainer = FakeTrainer
    try:
        w._stop_flag = False
        w._start_cls_training()
        q = w._task_queue
        t = next((x for x in q.tasks if x.kind is TaskKind.TRAIN
                  and x.title.startswith("cls:")), None)
        assert t is not None, "分类训练没有登记成 TRAIN 队列任务"
        wait_until(lambda: t.state in (TaskState.RUNNING, TaskState.DONE),
                   desc="分类训练任务开始")
        wait_until(lambda: t.state is TaskState.DONE, desc="假训练跑完")

        assert seen.get("stop_check") is not None, "train() 没收到 stop_check"
        assert seen["stop_before"] is False, "刚开跑 stop_check 就该是 False"
        assert seen["task_flag"] is False, "拿到的不是任务自己的标志"
        assert w._stop_flag is False, "★ 提交分类训练时又动了全局标志"
        assert "aitp-task" in str(seen.get("init_thread", "")), \
            "训练器不是任务线程里构造的（还是主线程先建好再塞进去）：%r" % seen.get("init_thread")
        # 真取消一次：任务自己的标志必须能被 stop_check 看到（接线通了才算数）
        t.stop_flag = True
        assert seen["stop_check"]() is True, "stop_check 没跟着任务标志变（接线断了）"
    finally:
        ctr.ClassificationTrainer = orig
        w.current_project = None
        w._task_queue.clear_finished()


def test_ocv_batch_path_wires_stop():
    """★ OCV 批量检测这条路径必须真的能跑起来，且停止判据接的是令牌。

    2026-11-11 那轮把 `stop_check=self._active_stop` 批量改成 `stop_check=stop` 时，
    给 `_run_ocr_batch` 补了 `stop = self._infer_stop(stop)`，**漏了 `_ocv_batch_inspect`**
    —— 那个方法里的 `stop` 从未定义，一点"Batch Inspect"就 NameError，
    而且抛在子线程里被吞掉，用户只看到状态栏卡住。
    队列层/标志层的用例一条都不走这条路径，所以 87 项全绿也没抓到。
    """
    import flow.batch as fb

    w = _window()
    _drain(w)
    tmp = tempfile.mkdtemp(prefix="ocv_")
    os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "models"), exist_ok=True)
    with open(os.path.join(tmp, "models", "ocv_model.pkl"), "wb") as f:
        f.write(b"stub")
    w.current_project = {"name": "stub"}
    w.pm.get_project_dir = lambda name: tmp

    seen = {}

    class FakeRunner:
        def __init__(self, flow, project_dir=None, stop_check=None):
            seen["stop"] = stop_check

        def run(self, images, on_progress=None):
            seen["ran"] = True

    orig = (fb.BatchRunner, fb.list_images, fb.build_ocv_flow)
    fb.BatchRunner = FakeRunner
    fb.list_images = lambda pd: ["a.bmp"]
    fb.build_ocv_flow = lambda **kw: object()
    try:
        w._ocv_batch_inspect()
        wait_until(lambda: "stop" in seen, desc="OCV 批量启动（NameError 会让这一步超时）")
        assert callable(seen["stop"]), "BatchRunner 没拿到可调用的 stop_check"
        assert seen["stop"]() is False, "新推理的判据不该是「已停」"
        w._stop_batch()
        assert seen["stop"]() is True, "「停止推理」没停到 OCV 批量"
    finally:
        fb.BatchRunner, fb.list_images, fb.build_ocv_flow = orig
        w.current_project = None


def test_infer_token_store_is_bounded():
    """令牌容器不能无限涨 —— 只保留最新 32 个。

    ★ 审核 P2：原来写的是"超过 32 个就清掉**已置位**的"，而正常跑完的推理
      **永远不会被 set**，那个过滤等于什么都清不掉，长度一直涨、每次还白重建一遍。
      现在用 `deque(maxlen=32)`，按添加顺序丢最老的。
    """
    w = _window()
    _drain(w)
    w._infer_stops.clear()
    latest = None
    for _ in range(40):
        latest = w._infer_stop()
    assert len(w._infer_stops) <= 32, "令牌容器涨到 %d 个（上限没生效）" % len(w._infer_stops)
    assert latest() is False, "最新一个令牌必须是干净的"
    w._stop_batch()
    assert latest() is True, "窗口内的令牌应被「停止推理」点亮"


def main():
    print("=== 任务级停止标志 回归测试（P0-3）===")
    checks = [
        ("两个任务标志互不影响", test_two_tasks_flags_independent),
        ("取消一个后下一个不秒停", test_cancel_running_then_next_not_poisoned),
        ("phase 上报", test_phase_reporting),
        ("stop_flag 单一来源", test_stop_flag_single_source),
        ("queue.running", test_running_property),
        ("★ 取消排队项不动全局标志", test_on_task_cancel_does_not_touch_global),
        ("★ 停止按钮只打同类任务", test_stop_buttons_only_hit_matching_kind),
        ("★ 停止训练不误伤裸线程推理", test_stop_training_does_not_stop_bare_infer),
        ("★ 停止推理不动训练的全局标志", test_stop_infer_does_not_touch_training_flag),
        ("推理令牌各自独立", test_infer_tokens_are_per_run),
        ("分派目标方法都存在", test_dispatch_targets_exist),
        ("★ 新任务不需要谁复位", test_no_reset_needed_anywhere),
        ("★ 分类训练走队列且标志接对", test_cls_training_goes_through_queue),
        ("★ OCV 批量路径真的能跑且接对判据", test_ocv_batch_path_wires_stop),
        ("令牌容器有上限", test_infer_token_store_is_bounded),
    ]
    for name, fn in checks:
        check(name, fn)

    print()
    if failures:
        print(f"❌ {len(failures)}/{len(checks)} 项失败")
        return 1
    print(f"✅ {len(checks)}/{len(checks)} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
