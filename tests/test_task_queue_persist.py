# -*- coding: utf-8 -*-
"""任务队列持久化回归测试（对应 2026-10-29 新增的历史落地）。

背景：用户重启后打开「任务队列」看到"暂无任务"，筛「已完成」也是空的，
      不知道刚才跑过什么。原设计是"不持久化，痕迹看日志"，实测不成立。

这里钉住四条语义：
    ① 终态任务能存能读，字段不丢
    ② 运行中/排队中的**不存**；若历史里残留 RUNNING，加载时降级成 CANCELED
    ③ 超过 MAX_HISTORY 只留最近的
    ④ 历史文件损坏时队列照样能起来（绝不因为一个 json 崩掉整个平台）

运行：python tests/test_task_queue_persist.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.task_queue import (  # noqa: E402
    MAX_HISTORY, Task, TaskKind, TaskQueue, TaskState,
)


def _tmp_path():
    d = tempfile.mkdtemp()
    return os.path.join(d, ".task_history.json")


def _mk(state=TaskState.DONE, title="t", total=10, done=10, **kw):
    return Task(kind=TaskKind.TRAIN, title=title, total=total, done=done,
                state=state, **kw)


def test_roundtrip():
    """终态任务存进去，新队列能原样读回来。"""
    p = _tmp_path()
    q = TaskQueue(history_path=p)
    t = _mk(title="训练 unet", project="proj-A", detail="best_model.pth")
    t.started_at = t.created_at + 1
    t.finished_at = t.created_at + 61
    q._tasks.append(t)
    q._save_history()

    q2 = TaskQueue(history_path=p)
    assert len(q2.tasks) == 1, q2.tasks
    r = q2.tasks[0]
    assert r.title == "训练 unet"
    assert r.project == "proj-A"
    assert r.detail == "best_model.pth"
    assert r.state == TaskState.DONE
    assert r.total == 10 and r.done == 10
    assert r.id == t.id, "任务 id 应当保留"
    # 耗时能算出来（说明 started/finished 都存住了）
    assert r.elapsed_text != ""


def test_running_not_saved():
    """运行中的任务不落盘（重启后它已经不存在了）。"""
    p = _tmp_path()
    q = TaskQueue(history_path=p)
    q._tasks.append(_mk(state=TaskState.DONE, title="done"))
    q._tasks.append(_mk(state=TaskState.RUNNING, title="running", done=3))
    q._tasks.append(_mk(state=TaskState.QUEUED, title="queued", done=0))
    q._save_history()

    rows = json.load(open(p, encoding="utf-8"))
    titles = [r["title"] for r in rows]
    assert titles == ["done"], titles


def test_running_downgraded_on_load():
    """历史里残留 RUNNING → 加载时降级为 CANCELED（上次没跑完就退出的残影）。"""
    p = _tmp_path()
    json.dump([{
        "id": "abc12345", "kind": "train", "title": "上次的训练",
        "project": "p", "total": 100, "done": 42,
        "state": "running", "detail": "", "error": "",
        "created_at": 1000.0, "started_at": 1001.0, "finished_at": 0.0,
    }], open(p, "w", encoding="utf-8"), ensure_ascii=False)

    q = TaskQueue(history_path=p)
    assert len(q.tasks) == 1
    assert q.tasks[0].state == TaskState.CANCELED, q.tasks[0].state
    assert q.tasks[0].done == 42, "进度应当保留，便于判断断在哪"


def test_max_history_truncate():
    """超过上限只留最近 MAX_HISTORY 条。"""
    p = _tmp_path()
    q = TaskQueue(history_path=p)
    for i in range(MAX_HISTORY + 30):
        t = _mk(title="t%d" % i)
        t.created_at = 1000.0 + i
        q._tasks.append(t)
    q._save_history()

    rows = json.load(open(p, encoding="utf-8"))
    assert len(rows) == MAX_HISTORY, len(rows)
    # 留下的是最新的那批
    assert rows[-1]["title"] == "t%d" % (MAX_HISTORY + 29)


def test_clear_finished_clears_file():
    """点"清空已完成"后，历史文件也要跟着清空（否则下次启动又冒出来）。"""
    p = _tmp_path()
    q = TaskQueue(history_path=p)
    q._tasks.append(_mk(title="a"))
    q._tasks.append(_mk(title="b"))
    q._save_history()
    assert len(json.load(open(p, encoding="utf-8"))) == 2

    q.clear_finished()
    rows = json.load(open(p, encoding="utf-8"))
    assert rows == [], rows


def test_corrupt_file_survives():
    """历史文件坏了（不是合法 JSON），队列照样能起来。"""
    p = _tmp_path()
    open(p, "w", encoding="utf-8").write("{这不是合法json")
    q = TaskQueue(history_path=p)          # 不能抛异常
    assert q.tasks == []


def test_missing_file_survives():
    """没有历史文件时安静从空开始。"""
    q = TaskQueue(history_path=os.path.join(tempfile.mkdtemp(), "nope.json"))
    assert q.tasks == []


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
