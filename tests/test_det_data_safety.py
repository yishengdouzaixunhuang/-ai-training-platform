# -*- coding: utf-8 -*-
"""两条"点一下就可能丢数据"的保护（2026-11-16 用户复核提出）。

## 1. 单图检测推理不许碰标注

`_run_det_inference_single()` 原来是：

```python
delete_all()          # 里面直接 save() → images/<base>_det.json 立刻被写成 []
for b in boxes:
    add_box(...)      # 每个框一次 save()（6 个框 = 6 次写盘）
```

`save()` 写的是**标注的权威文件**，而批量推理写 `outputs/`。用户的心理是
"我标完了，跑一下模型看看效果"，不会预期 10 个 GT 框被换成预测。

而且 `delete_all()` 第一次 save 就已经落盘了 —— 撤销栈救不回来（undo 只是把内存
列表写回去，前提是文件没被覆盖过）。

现在：预测写 `outputs/<base>_det.json`（含 `meta.infer_ms`）→ `load_predictions()`
读进**预测层**显示。`_det.json` 只在用户手动改框时写。

## 2. 坏标注文件先留档，再允许写

`BoxManager.load()` 容错后（坏文件按 0 个框显示），风险变成
"打得开 → 一编辑就覆盖"：坏文件 → 显示 0 个框 → 用户随手加一个框 → `save()`
把原文件整体替换。哪怕原文只是被截断 2 字节、还能手工救回 9 个框，也没了。

现在加载失败先把原文件改名 `<base>_det.json.bad-<时间戳>`；改名失败也不挡加载，
但要把原因记进 `load_errors`。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_det_data_safety.py
"""
import _testenv  # noqa: F401  无显示器 + 配置目录隔离（见 tests/_testenv.py）
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:                                  # 与 main.py 一致：torch 先于 PyQt5（WinError 1114）
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

from detection.box_manager import BoxManager  # noqa: E402
from detection.coco_io import save_coco_json  # noqa: E402

failures = []
_APP = None
_WIN = None
_TMP = None
_MADE = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def app():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication
        _APP = QApplication.instance() or QApplication([])
    return _APP


def window():
    global _WIN, _TMP
    if _WIN is not None:
        return _WIN
    import ui.task_queue as tq

    fd, p = tempfile.mkstemp(prefix="aitp_dsafe_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    app()
    init_ui_invoker()
    _TMP = tempfile.mkdtemp(prefix="aitp_dsafe_proj_")
    _WIN = MainWindow()
    return _WIN


def wait_until(pred, timeout=40.0, desc="条件"):
    end = time.time() + timeout
    while time.time() < end:
        app().processEvents()
        time.sleep(0.02)
        if pred():
            return
    raise AssertionError("等待超时：%s" % desc)


def _png(path, w=400, h=80):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), (30, 30, 30)).save(path)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _boxes(n, cat="0"):
    return [{"category": cat, "bbox": [4 + i * 8, 4, 6, 20]} for i in range(n)]


# ---------------------------------------------------------------- 单图推理

class _StubTrainer:
    """替掉真 YOLO：返回固定 2 个框，并记录被调了几次。"""

    calls = []

    def __init__(self, project_dir=None):
        self.project_dir = project_dir

    def predict(self, image_path, weights=None, conf=0.25, iou=0.45):
        _StubTrainer.calls.append(image_path)
        return [
            {"bbox": [10.0, 10.0, 30.0, 40.0], "confidence": 0.91,
             "category": "1", "category_id": 1},
            {"bbox": [60.0, 10.0, 30.0, 40.0], "confidence": 0.77,
             "category": "A", "category_id": 2},
        ]


class _MB:
    """接住 QMessageBox：成功路径上不该弹任何东西（offscreen 下真弹会 segfault）。"""

    calls = []

    @staticmethod
    def warning(*a, **k):
        _MB.calls.append(("warning", a, k))

    @staticmethod
    def information(*a, **k):
        _MB.calls.append(("information", a, k))

    @staticmethod
    def critical(*a, **k):
        _MB.calls.append(("critical", a, k))

    @staticmethod
    def question(*a, **k):
        _MB.calls.append(("question", a, k))
        return 0


def make_det_project(n_boxes=10, with_det=True):
    w = window()
    name = "_dsafe_%d" % (len(_MADE) + 1)
    w.pm.create_project(name, task_type="目标检测", parent_dir=_TMP)
    _MADE.append(name)
    pd = str(w.pm.get_project_dir(name))
    img = os.path.join(pd, "images", "img1.png")
    _png(img)
    if with_det:
        save_coco_json(os.path.join(pd, "images", "img1_det.json"), img, 400, 80,
                       _boxes(n_boxes), ["0"])
    # 假模型文件：只要路径存在，函数就会往下走（真推理被 _StubTrainer 替掉）
    mdir = os.path.join(pd, "models", "detection")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "best.pt"), "wb") as f:
        f.write(b"stub")
    return name, pd, img


def open_det(name, pd):
    """打开工程 + 等首图 + 手动加载一次标注（保证 BoxManager 里有框）。"""
    w = window()
    w._open_project_by_name(name)
    wait_until(lambda: w._project_dir == pd and getattr(w.canvas, "current_image_path", None),
               desc="工程已打开且首图显示")
    w._load_det_annotations()
    assert w._box_manager is not None, "检测模式的 BoxManager 没建起来"
    return w


def _run_single(w, pd):
    import detection.trainer as _tr
    import ui.main_window as mw
    orig_tr, orig_mb = _tr.DetectionTrainer, mw.QMessageBox
    _StubTrainer.calls = []
    _MB.calls = []
    _tr.DetectionTrainer, mw.QMessageBox = _StubTrainer, _MB
    try:
        w._run_det_inference_single(pd)
    finally:
        _tr.DetectionTrainer, mw.QMessageBox = orig_tr, orig_mb
    assert _StubTrainer.calls, "★ 推理根本没跑（stub 没被调用）"
    assert not _MB.calls, "★ 成功路径弹了框：%r" % (_MB.calls[:1],)


def test_single_infer_keeps_annotations():
    """★ 推理前后 `_det.json` **字节不变**，标注框数仍是 10。"""
    name, pd, img = make_det_project(n_boxes=10)
    w = open_det(name, pd)
    det = os.path.join(pd, "images", "img1_det.json")
    sha_before = _sha(det)
    mtime_before = os.path.getmtime(det)

    _run_single(w, pd)

    assert os.path.exists(det), "★ 标注文件不见了"
    assert _sha(det) == sha_before, "★ 单图推理改写了标注文件（字节变了）"
    assert os.path.getmtime(det) == mtime_before, "★ 标注文件被重写过（mtime 变了）"
    with open(det, "r", encoding="utf-8") as f:
        assert len(json.load(f)["annotations"]) == 10
    assert w._box_manager.count == 10, \
        "★ 内存里的标注被清了/换了：%r" % w._box_manager.count


def test_single_infer_writes_outputs_and_pred_layer():
    """预测落到 `outputs/`（含 meta.infer_ms），并显示在预测层。"""
    name, pd, img = make_det_project(n_boxes=10)
    w = open_det(name, pd)
    _run_single(w, pd)

    out = os.path.join(pd, "outputs", "img1_det.json")
    assert os.path.exists(out), "★ 预测没写到 outputs/"
    with open(out, "r", encoding="utf-8") as f:
        d = json.load(f)
    assert len(d["annotations"]) == 2, d["annotations"]
    assert [c["name"] for c in d["categories"]] == ["1", "A"], d["categories"]
    scores = [a.get("score") for a in d["annotations"]]
    assert scores == [0.91, 0.77], scores
    ms = (d.get("meta") or {}).get("infer_ms")
    assert isinstance(ms, (int, float)), "★ 没落 meta.infer_ms：%r" % (d.get("meta"),)

    assert len(w._box_manager.pred_boxes) == 2, \
        "★ 预测层没更新：%r" % len(w._box_manager.pred_boxes)
    assert w._box_manager.count == 10, "预测不许进标注层"


def test_single_infer_does_not_create_det_file():
    """本来没有标注文件的图：推理后仍然没有 —— 不许凭空造一个"标注"。"""
    name, pd, img = make_det_project(with_det=False)
    w = open_det(name, pd)
    det = os.path.join(pd, "images", "img1_det.json")
    assert not os.path.exists(det), "前提：起始不该有 _det.json"

    _run_single(w, pd)

    assert not os.path.exists(det), "★ 推理给没有标注的图造了个 _det.json"
    assert os.path.exists(os.path.join(pd, "outputs", "img1_det.json")), "预测还是该落盘"


# ------------------------------------------------------------ 坏文件留档

def _corrupt(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write('{ "annotations": [ {"id": 1')      # 截断：还能救回一部分


def _bad_files(tmp):
    return sorted(f for f in os.listdir(tmp) if ".bad-" in f)


def test_corrupt_file_is_archived():
    """★ 坏文件被改名留档，原文一个字节不少。"""
    tmp = tempfile.mkdtemp()
    try:
        img = os.path.join(tmp, "a.png")
        _png(img)
        det = os.path.join(tmp, "a_det.json")
        _corrupt(det)
        raw = open(det, "rb").read()

        bm = BoxManager()
        bm.load(img)

        assert bm.count == 0
        assert not os.path.exists(det), "★ 坏文件还在原地：加一个框就会把它覆盖掉"
        bad = _bad_files(tmp)
        assert len(bad) == 1, bad
        assert bad[0].startswith("a_det.json.bad-"), bad[0]
        with open(os.path.join(tmp, bad[0]), "rb") as f:
            assert f.read() == raw, "★ 留档内容与原文件不一致"
        assert "留档" in (bm.load_errors.get("annot") or ""), \
            "★ 没在错误信息里告诉用户文件去哪了：%r" % bm.load_errors.get("annot")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_after_corrupt_keeps_archive():
    """留档之后再 add_box：新文件是新的，`.bad` 里仍是原文。"""
    tmp = tempfile.mkdtemp()
    try:
        img = os.path.join(tmp, "b.png")
        _png(img)
        det = os.path.join(tmp, "b_det.json")
        _corrupt(det)
        raw = open(det, "rb").read()

        bm = BoxManager()
        bm.load(img)
        bm.add_box([1, 2, 3, 4], category="0")

        assert os.path.exists(det), "新文件该被写出来"
        with open(det, "r", encoding="utf-8") as f:
            assert len(json.load(f)["annotations"]) == 1
        bad = _bad_files(tmp)
        assert len(bad) == 1, bad
        with open(os.path.join(tmp, bad[0]), "rb") as f:
            assert f.read() == raw, "★ 写新文件把留档覆盖了"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_valid_file_not_archived():
    """正常文件不许留档（否则每打开一次就产一个 .bad）。"""
    tmp = tempfile.mkdtemp()
    try:
        img = os.path.join(tmp, "c.png")
        _png(img)
        save_coco_json(os.path.join(tmp, "c_det.json"), img, 400, 80, _boxes(3), ["0"])
        bm = BoxManager()
        bm.load(img)
        bm.load(img)
        # 先判留档：变异"正常文件也留档"时，框数会变成 0，那是**结果**；
        # 真正要说的是"正常文件不该留在 .bad 里"。断言顺序决定报错说的是哪件事。
        assert not _bad_files(tmp), "★ 正常文件被留档了：%r" % _bad_files(tmp)
        assert bm.count == 3, "★ 正常文件读不回框：%r" % bm.count
        assert not bm.load_errors.get("annot"), bm.load_errors
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_archive_failure_does_not_block_load():
    """改名失败（文件被占用）也必须继续按 0 个框加载，但要说清没留成档。"""
    tmp = tempfile.mkdtemp()
    try:
        img = os.path.join(tmp, "d.png")
        _png(img)
        det = os.path.join(tmp, "d_det.json")
        _corrupt(det)

        real = os.replace
        os.replace = lambda *a, **k: (_ for _ in ()).throw(PermissionError(32, "占用"))
        try:
            bm = BoxManager()
            bm.load(img)                      # 不许抛
        finally:
            os.replace = real

        assert bm.count == 0
        err = bm.load_errors.get("annot") or ""
        assert "留档失败" in err, "★ 留档失败没说：%r" % err
        assert os.path.exists(det), "前提：改名失败时原文件应还在原地"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _adopt(w, yes=True):
    """走真实入口 `_det_adopt_predictions()`，把确认框固定成 Yes/No。"""
    import ui.main_window as mw
    real = mw.QMessageBox

    class _MB2:
        Yes, No = real.Yes, real.No

        @staticmethod
        def question(*a, **k):
            return real.Yes if yes else real.No

        @staticmethod
        def warning(*a, **k):
            raise AssertionError("不该弹错误框：%r" % (a[:3],))

        @staticmethod
        def information(*a, **k):
            raise AssertionError("不该弹信息框：%r" % (a[:3],))

        @staticmethod
        def critical(*a, **k):
            raise AssertionError("不该弹错误框：%r" % (a[:3],))

    mw.QMessageBox = _MB2
    try:
        w._det_adopt_predictions()
    finally:
        mw.QMessageBox = real


def test_adopt_predictions_appends():
    """★ 预测转标注：追加 2 个、跳过重复，且**一次撤销整体回退**。"""
    name, pd, img = make_det_project(n_boxes=3)
    w = open_det(name, pd)
    det = os.path.join(pd, "images", "img1_det.json")
    assert w._box_manager.count == 3, "前提：起点 3 个标注"

    _run_single(w, pd)                       # 预测层 2 个框
    _adopt(w)
    assert w._box_manager.count == 5, "★ 预测没被追加成标注：%r" % w._box_manager.count
    with open(det, "r", encoding="utf-8") as f:
        assert len(json.load(f)["annotations"]) == 5, "★ 没落盘"

    # 幂等：再点一次，两个预测都已有同位置标注 ⇒ 一个都不加
    _adopt(w)
    assert w._box_manager.count == 5, "★ 重复点翻倍了：%r" % w._box_manager.count

    # ★ 批量快照：一次撤销回到追加之前（按框快照的话这里会是 4）
    assert w._box_manager.undo() is True
    assert w._box_manager.count == 3, \
        "★ add_boxes 没做成「一次快照」（撤销后剩 %r 个，应为 3）" % w._box_manager.count
    with open(det, "r", encoding="utf-8") as f:
        assert len(json.load(f)["annotations"]) == 3, "★ 撤销没写回文件"


def test_adopt_without_predictions_does_nothing():
    """没有预测时点「转标注」：不许改标注文件，也不许弹错误框。"""
    name, pd, img = make_det_project(n_boxes=3)
    w = open_det(name, pd)
    det = os.path.join(pd, "images", "img1_det.json")
    sha = _sha(det)
    _adopt(w)
    assert w._box_manager.count == 3, w._box_manager.count
    assert _sha(det) == sha, "★ 没预测也把标注文件改了"


def cleanup():
    w = _WIN
    if w is not None:
        for name in _MADE:
            try:
                w.pm.delete_project(name)
            except Exception:  # noqa: BLE001
                pass
    if _TMP:
        shutil.rmtree(_TMP, ignore_errors=True)


def main():
    checks = [
        ("★ 单图推理不改标注（字节不变、框数不变）", test_single_infer_keeps_annotations),
        ("单图推理落 outputs/ + 预测层", test_single_infer_writes_outputs_and_pred_layer),
        ("无标注的图推理后仍无 _det.json", test_single_infer_does_not_create_det_file),
        ("★ 坏文件先改名留档", test_corrupt_file_is_archived),
        ("留档后写新文件不覆盖留档", test_write_after_corrupt_keeps_archive),
        ("正常文件不留档", test_valid_file_not_archived),
        ("留档失败也不挡加载", test_archive_failure_does_not_block_load),
        ("★ 预测转标注：追加 + 幂等 + 一次撤销", test_adopt_predictions_appends),
        ("无预测时转标注不动文件", test_adopt_without_predictions_does_nothing),
    ]
    try:
        for name, fn in checks:
            check(name, fn)
    finally:
        cleanup()

    print()
    if failures:
        print("❌ 有失败：%d / %d" % (len(failures), len(checks)))
        for name, exc in failures:
            print("   - %s: %s" % (name, exc))
        return 1
    print("✅ 全部通过  %d/%d 项通过" % (len(checks), len(checks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
