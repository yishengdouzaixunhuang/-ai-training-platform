# -*- coding: utf-8 -*-
"""界面这条线：右键「Import JSON for Selected…」之后，检测工程到底被写进了什么。

单元测试（`tests/test_det_import.py`）只保证转换函数对；这里钉的是**接线**：
真的调 `_import_json_for_selected()`（用户点的那个动作），断言

- 目标检测工程 → 产出 `images/<base>_det.json`，且**不再**生成分割掩膜；
- `get_annotated_count()` 从 0 变正数（训练的前置闸门）；
- `build_yolo_dataset()` 能建出数据集 —— 这一步过了，才叫"设参数就能训"。

⚠️ 这里**不许**直接调 `_rebuild_det_for_imported()` 了事：那样把
`if self._is_detection_project():` 改成 `if False:` 测试照样全绿
（第一版就这么写的，变异测试实拍漏掉）。必须走真实入口。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_det_import_ui.py
"""
import _testenv  # noqa: F401  无显示器 + 配置目录隔离（见 tests/_testenv.py）
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

# ⚠️ 与 main.py 一致：torch 必须先于 PyQt5（否则 WinError 1114 c10.dll）
try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

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


def window():
    global _APP, _WIN, _TMP
    if _WIN is not None:
        return _WIN
    from PyQt5.QtWidgets import QApplication
    import ui.task_queue as tq

    fd, p = tempfile.mkstemp(prefix="aitp_detui_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p            # 别写真实的 .task_history.json

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    _APP = QApplication.instance() or QApplication([])
    init_ui_invoker()
    _TMP = tempfile.mkdtemp(prefix="aitp_detui_proj_")
    _WIN = MainWindow()
    return _WIN


def wait_until(pred, timeout=40.0, desc="条件"):
    end = time.time() + timeout
    while time.time() < end:
        _APP.processEvents()
        time.sleep(0.02)
        if pred():
            return
    raise AssertionError("等待超时：%s" % desc)


def _png(path, w=400, h=80):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), (30, 30, 30)).save(path)


def _labelme(pairs, img_name, w=400, h=80):
    return {"version": "5.0.1", "flags": {},
            "shapes": [{"label": lb, "shape_type": "rectangle",
                        "points": [[p[0], p[1]], [p[2], p[3]]]} for lb, p in pairs],
            "imagePath": img_name, "imageData": None,
            "imageWidth": w, "imageHeight": h}


def make_project(task_type="目标检测", n=2):
    """建工程 → 只放图（模拟"导入图片"）→ JSON 放在**另一个目录**（模拟"标注文件夹"）。

    ⚠️ JSON 不能直接放 images/：`_import_json_for_selected` 会 copy2 到 images/，
       源和目标是同一个文件时抛 SameFileError，那样测的就不是真实路径了。
    """
    w = window()
    name = "_detui_%d" % (len(_MADE) + 1)
    w.pm.create_project(name, task_type=task_type, parent_dir=_TMP)
    _MADE.append(name)
    pd = str(w.pm.get_project_dir(name))
    labels = os.path.join(_TMP, "labels_" + name)
    os.makedirs(labels, exist_ok=True)
    bases = []
    for i in range(n):
        base = "img%d" % (i + 1)
        _png(os.path.join(pd, "images", base + ".png"))
        with open(os.path.join(labels, base + ".json"), "w", encoding="utf-8") as f:
            json.dump(_labelme([("1", [10, 10, 30, 50]), ("A", [60, 8, 92, 46])],
                               base + ".png"), f)
        bases.append(base)
    return name, pd, bases, labels


class _FD:
    """替身：`QFileDialog.getExistingDirectory` 直接返回指定目录。"""
    _folder = ""

    @classmethod
    def getExistingDirectory(cls, *a, **k):
        return cls._folder


class _MB:
    """替身：所有弹窗一律"知道了"，否则 offscreen 下会卡住等人点。"""
    Ok = 1
    Yes = 2
    No = 4

    @staticmethod
    def information(*a, **k):
        return 1

    @staticmethod
    def warning(*a, **k):
        return 1

    @staticmethod
    def critical(*a, **k):
        return 1

    @staticmethod
    def question(*a, **k):
        return 2


def run_import(w, pd, bases, labels, spy=None):
    """走用户真实动作：选中全部行 → 右键 Import JSON for Selected…

    spy: 传一个 dict 进来就记录"这次走的是哪条分支"（det / mask）。
         不盯文件系统 —— 画布预览首图时自己就会写一张 `_mask.png`，
         拿"有没有掩膜文件"当判据必然误判（第一版就这么栽的）。
    """
    import ui.main_window as mw
    wait_until(lambda: len(getattr(w, "_filtered_data", []) or []) >= len(bases),
               desc="图片列表就绪")
    rows = list(range(len(w._filtered_data)))
    orig_fd, orig_mb = mw.QFileDialog, mw.QMessageBox
    orig_det = w._rebuild_det_for_imported
    orig_msk = w._rebuild_masks_for_imported
    if spy is not None:
        # 包一层：既记分支、又让真流程照跑（不能替身掉，否则断言不到产物）
        def spy_det(bs):
            spy.setdefault("det", list(bs))
            return orig_det(bs)

        def spy_mask(bs):
            spy.setdefault("mask", list(bs))
            return orig_msk(bs)

        w._rebuild_det_for_imported = spy_det
        w._rebuild_masks_for_imported = spy_mask
    _FD._folder = labels
    mw.QFileDialog, mw.QMessageBox = _FD, _MB
    try:
        w._import_json_for_selected(rows, pd)
    finally:
        mw.QFileDialog, mw.QMessageBox = orig_fd, orig_mb
        w._rebuild_det_for_imported = orig_det
        w._rebuild_masks_for_imported = orig_msk


def test_detection_project_flag():
    w = window()
    name, pd, _bases, _lbl = make_project("目标检测")
    w._open_project_by_name(name)
    wait_until(lambda: w._project_dir == pd, desc="工程已切到临时检测工程")
    assert w._is_detection_project() is True, "检测工程没被判成 detection"
    assert w._detection_mode is True, "检测工程打开后 _detection_mode 应为 True"


def test_segmentation_project_flag():
    w = window()
    name, pd, _bases, _lbl = make_project("语义分割")
    w._open_project_by_name(name)
    wait_until(lambda: w._project_dir == pd, desc="工程已切到临时分割工程")
    assert w._is_detection_project() is False, "分割工程被判成了 detection"


def test_import_writes_det_json_not_mask():
    """★ 导入标注（真入口）后：`_det.json` 必须有，分割掩膜必须没有。"""
    from detection.dataset import get_annotated_count
    w = window()
    name, pd, bases, labels = make_project("目标检测", n=3)
    w._open_project_by_name(name)
    wait_until(lambda: w._project_dir == pd, desc="工程已切到临时检测工程")

    img_dir = os.path.join(pd, "images")
    assert get_annotated_count(pd) == 0, "前置条件：造工程时不该有 _det.json"

    spy = {}
    run_import(w, pd, bases, labels, spy=spy)

    assert "det" in spy, "★ 检测工程导入后没走 _rebuild_det_for_imported"
    assert "mask" not in spy, "★ 检测工程还在跑分割那条路（生成掩膜）"
    assert sorted(spy["det"]) == sorted(bases), spy["det"]

    for b in bases:
        det = os.path.join(img_dir, b + "_det.json")
        assert os.path.exists(det), "★ 导入后没生成 %s_det.json" % b
        with open(det, "r", encoding="utf-8") as f:
            d = json.load(f)
        assert len(d["annotations"]) == 2, d
    assert get_annotated_count(pd) == len(bases), \
        "★ 训练前置计数没跟上（get_annotated_count=%d）" % get_annotated_count(pd)

    from detection.dataset import build_yolo_dataset
    ds = build_yolo_dataset(pd, val_split=0.34, seed=1)
    assert ds["num_classes"] == 2 and ds["class_names"] == ["1", "A"], ds


def test_segmentation_import_still_makes_mask():
    """对照组：分割工程还是那条老路（掩膜），不能被这次改动带偏。"""
    w = window()
    name, pd, bases, labels = make_project("语义分割", n=2)
    w._open_project_by_name(name)
    wait_until(lambda: w._project_dir == pd, desc="工程已切到临时分割工程")

    spy = {}
    run_import(w, pd, bases, labels, spy=spy)

    assert "mask" in spy, "★ 分割工程没走掩膜那条路"
    assert "det" not in spy, "★ 分割工程被误当成检测，去生成 _det.json 了"

    ann_dir = os.path.join(pd, "annotations")
    masks = [f for f in os.listdir(ann_dir) if f.endswith("_mask.png")] \
        if os.path.isdir(ann_dir) else []
    assert len(masks) >= len(bases), "★ 分割工程掩膜不全：%r" % masks
    dets = [f for f in os.listdir(os.path.join(pd, "images")) if f.endswith("_det.json")]
    assert not dets, "★ 分割工程不该产生检测标注：%r" % dets[:3]


def test_import_twice_is_idempotent():
    """重复导入同一批：不能报错、不能把框翻倍。"""
    w = window()
    name, pd, bases, labels = make_project("目标检测", n=1)
    w._open_project_by_name(name)
    wait_until(lambda: w._project_dir == pd, desc="工程已切到临时检测工程")
    run_import(w, pd, bases, labels)
    run_import(w, pd, bases, labels)
    with open(os.path.join(pd, "images", bases[0] + "_det.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    assert len(d["annotations"]) == 2, "重复导入把框翻倍了：%d" % len(d["annotations"])


def test_det_training_guard_blocks_double_start():
    """★ 检测训练连点两次：第二次必须被拦下，不许起第二个训练线程。

    为什么这条重要：检测训练是直接 `threading.Thread(...)`，不走任务队列。
    两份训练同时跑会抢显存，而且第二份在重建数据集时会把第一份正在读的
    `yolo_dataset/` 删掉 —— 第一份读着读着文件没了。
    """
    import ui.main_window as mw
    w = window()
    name, pd, bases, labels = make_project("目标检测", n=2)
    w._open_project_by_name(name)
    wait_until(lambda: w._project_dir == pd, desc="工程已切到临时检测工程")

    calls = []

    class MB(_MB):
        @staticmethod
        def information(*a, **k):
            calls.append(a[1] if len(a) > 1 else "")
            return 1

    class _NoThread:
        class Thread:
            def __init__(self, *a, **k):
                raise AssertionError("★ 第二次点击仍然起了训练线程")

    w._det_training_running = True            # 模拟"上一份还在跑"
    orig_mb, orig_th = mw.QMessageBox, mw.threading
    mw.QMessageBox, mw.threading = MB, _NoThread
    try:
        w._start_det_training()
    finally:
        mw.QMessageBox, mw.threading = orig_mb, orig_th
        # ⚠️ 这里**不能**顺手清 `_det_training_running`：清了之后下面那条
        #   "训练结束要复位"的断言就永远为真（第一版就这么写成空转的）。

    assert w._det_training_running is True, "★ 被挡下的那次把标志清了？"
    assert calls, "★ 没提示用户（会被当成点了没反应）"
    assert w.det_train_status.text() != "Preparing...", "★ 状态栏仍被改成 Preparing"

    # 反向：训练结束必须复位，否则"挡一次"变成"永远挡住"，下次点训练毫无反应
    w._on_det_training_done()
    assert w._det_training_running is False, \
        "★ 训练结束后标志没复位 —— 下一次点训练会被误挡"


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
        ("★ 检测工程被识别成 detection", test_detection_project_flag),
        ("分割工程不被误判", test_segmentation_project_flag),
        ("★ 导入标注走真入口：出 _det.json、不出掩膜", test_import_writes_det_json_not_mask),
        ("★ 对照组：分割工程仍然只出掩膜", test_segmentation_import_still_makes_mask),
        ("重复导入幂等（框不翻倍）", test_import_twice_is_idempotent),
        ("★ 检测训练连点两次被挡下", test_det_training_guard_blocks_double_start),
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
