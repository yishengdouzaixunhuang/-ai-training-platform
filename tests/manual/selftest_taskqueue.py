# -*- coding: utf-8 -*-
"""任务队列自测（2026-10-16）—— 队列核心不依赖 UI，可独立驱动。

要钉住的**语义**（每条都对应一个真实会踩的坑）：
  ① **串行**：同一时刻只跑一个（单卡并发 = OOM）
  ② **取消是请求式**：置标志，不强杀线程；排队中的项直接取消、不会去跑它
  ③ **ETA 不给噪声**：样本不足时不显示，而不是显示一个马上要变的假数字
  ④ **接力**：一个跑完自动接下一个
  ⑤ **失败不炸平台**：任务抛异常只记 failed
  ⑥ **清空只清历史**，在跑的不动

跑法：`C:/aitp_venv/Scripts/python.exe tests/manual/selftest_taskqueue.py`（在仓库根）

⚠️ 2026-11-17：本脚本含「清空历史」「清空全部」两条用例，会用**默认**历史路径
   `TaskQueue()` 真删 `.task_history.json`。审核实测跑一次就把仓库根的 3234 字节
   历史写成 `[]`（真实记录全丢，且该文件在 .gitignore 里、版本库救不回来）。
   所以下面**强制**把默认历史路径指到临时目录 —— 这条自测不许再碰真实数据。
"""
import os
import sys
import tempfile
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 必须在构造 TaskQueue() 之前改默认值：它只在 __init__ 里读这一次。
import ui.task_queue as _task_queue
_task_queue.DEFAULT_HISTORY_PATH = os.path.join(
    tempfile.mkdtemp(prefix="aitp_tq_selftest_"), ".task_history.json")

from ui.task_queue import (TaskKind, TaskQueue, TaskState, _Canceled, fmt_eta)

_R = []


def ck(name, ok, detail=""):
    _R.append((bool(ok), name))
    print("  %-4s %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  <- %s" % detail) if not ok else ""), flush=True)


def wait_until(pred, timeout=6.0, step=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    return False


def main():
    # ── ① 串行：两个任务，第一个跑完第二个才开跑 ──
    order = []
    q = TaskQueue()
    def mk(tag, dur=0.25):
        def run(ctx):
            for i in range(5):
                ctx.progress(i + 1, 5)
                ctx.check_cancel()
                time.sleep(dur / 5)
            order.append(tag)
        return run
    t1 = q.submit(TaskKind.TRAIN, "jobA", run=mk("A"))
    t2 = q.submit(TaskKind.TRAIN, "jobB", run=mk("B"))
    ck("提交后立刻开跑第一个", wait_until(lambda: t1.state is TaskState.RUNNING))
    ck("**串行**：第二个还在排队（同一时刻只跑一个）", t2.state is TaskState.QUEUED,
       t2.state.value)
    ck("第一个跑完自动接力第二个", wait_until(lambda: order == ["A", "B"], 8),
       order)
    ck("两个任务都标记完成",
       wait_until(lambda: all(t.state is TaskState.DONE for t in (t1, t2)), 8))

    # ── ② 取消：跑着的置标志；排队的直接取消 ──
    gate = threading.Event()
    q2 = TaskQueue()
    def slow(ctx):
        for i in range(200):
            ctx.progress(i + 1, 200)
            ctx.check_cancel()
            time.sleep(0.05)
    a = q2.submit(TaskKind.TRAIN, "slow", run=slow)
    b = q2.submit(TaskKind.INFER, "queued-one", run=lambda c: None)
    ck("慢任务开跑", wait_until(lambda: a.state is TaskState.RUNNING))
    ck("取消排队中的项 => 直接 canceled、不开跑",
       q2.request_cancel(b.id) and b.state is TaskState.CANCELED, b.state.value)
    ck("取消正在跑的 => 置标志，状态仍是 running（不强杀）",
       q2.request_cancel(a.id) and a.state is TaskState.RUNNING and a.cancel_requested)
    ck("被取消的任务最终记为 canceled（不是 failed）",
       wait_until(lambda: a.state is TaskState.CANCELED, 6), a.state.value)
    gate.set()

    # ── ③ ETA：不给噪声 ──
    ck("fmt_eta 小时级", fmt_eta(3723) == "1小时02分03秒", fmt_eta(3723))
    ck("fmt_eta 分钟级（友商那种 59分42秒）", fmt_eta(3582) == "59分42秒", fmt_eta(3582))
    ck("fmt_eta 秒级", fmt_eta(42) == "42秒", fmt_eta(42))
    q3 = TaskQueue()
    holder = {}
    def eta_job(ctx):
        holder["t"] = ctx.task
        ctx.progress(0, 100)
        time.sleep(0.2)
        ctx.progress(1, 100)          # 只有 1% ⇒ 不该给 ETA
        holder["eta_early"] = ctx.task.eta_text
        time.sleep(0.3)
        for i in range(2, 100):
            ctx.progress(i, 100)
            ctx.check_cancel()
            time.sleep(0.01)
    e = q3.submit(TaskKind.TRAIN, "eta", run=eta_job)
    ck("进度 <3% 时**不显示 ETA**（宁可不给，也不给噪声数字）",
       wait_until(lambda: "eta_early" in holder, 6) and holder.get("eta_early") == "",
       repr(holder.get("eta_early")))
    ck("进度上来后 ETA 出现且为正数",
       wait_until(lambda: e.eta_text not in ("", None) or e.state is TaskState.DONE, 10)
       and (e.state is TaskState.DONE or float(e.eta_text.split("剩余")[1][:-1] or 0) >= 0),
       e.eta_text)
    # 任务结束后**不再显示 ETA**（eta_seconds 只在 RUNNING 时有值）
    ck("任务结束后 ETA 消失（不再拿一个陈旧数字吓人）",
       wait_until(lambda: e.state is TaskState.DONE, 12) and e.eta_text == "",
       "state=%s eta=%r" % (e.state.value, e.eta_text))

    # ── ④ 失败不炸平台 ──
    q4 = TaskQueue()
    def boom(ctx):
        ctx.progress(1, 4)
        raise ValueError("模拟训练崩了")
    f = q4.submit(TaskKind.TRAIN, "boom", run=boom)
    nxt = q4.submit(TaskKind.INFER, "after-boom", run=lambda c: None)
    ck("任务抛异常 => 记 failed（平台不炸）",
       wait_until(lambda: f.state is TaskState.FAILED, 5), f.state.value)
    ck("失败信息被记录", "ValueError" in f.error, f.error)
    ck("失败后队列继续接下一个（不卡死）",
       wait_until(lambda: nxt.state is TaskState.DONE, 5), nxt.state.value)

    # ── ⑤ 清空只清历史 ──
    q5 = TaskQueue()
    def hold(ctx):
        for _ in range(200):
            ctx.check_cancel()
            time.sleep(0.05)
    # ⚠️ 顺序很关键：**串行**队列里，先排的先跑完。所以要先让 done1 跑完，
    #    再提交那个会一直占着的 hold —— 否则 done1 永远在排队（我第一次就写错了）。
    d1 = q5.submit(TaskKind.INFER, "done1", run=lambda c: None)
    ck("历史项已完成", wait_until(lambda: d1.state is TaskState.DONE, 5))
    h = q5.submit(TaskKind.TRAIN, "hold", run=hold)
    ck("长任务占住队列", wait_until(lambda: h.state is TaskState.RUNNING, 5))
    q5.clear_finished()
    ck("清空历史后只剩在跑的那一个",
       [t.id for t in q5.tasks] == [h.id], [t.title for t in q5.tasks])
    q5.request_cancel(h.id)
    wait_until(lambda: h.state is TaskState.CANCELED, 6)
    q5.clear_all()
    ck("清空全部后队列为空", q5.tasks == [], q5.tasks)

    n_pass = sum(1 for ok, _ in _R if ok)
    n_fail = len(_R) - n_pass
    print("=" * 68)
    print("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (n_pass, n_fail, len(_R)))
    print("=" * 68)
    return n_fail


if __name__ == "__main__":
    sys.exit(main())