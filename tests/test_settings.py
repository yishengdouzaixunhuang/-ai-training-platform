# -*- coding: utf-8 -*-
"""配置统一入口 `core/settings.py` 的回归测试（重构清单 P0-2）。

## 验收标准（审核定的）

> 不能是"加了 schema 但没人用"。至少**一个真实调用点**从旧写法换过来，
> 而且要能证明"**配置字段写错时会报出来，而不是静默用默认值**"。

所以这里分两层：

- **引擎层**：`validate()` 逐字段判类型/范围，`merge()` 只认它的结论；
- **真实调用点**：`MainWindow._load_panel_train_settings()` / `_save_panel_train_settings()`。
  断言不只是"面板值对了"，还包括"**坏字段会在日志里被点名**" ——
  这一条才能抓住"校验接了但调用方把结果丢了"那种半吊子实现。

## 这个模块要解决的原始 bug

改前 `_load_panel_train_settings()` 是：

```python
self.epochs_spin.setValue(int(s.get("epochs", 200)))   # "abc" 就抛
...
except Exception:
    pass                                               # 整段静默放弃
```

一个字段写坏 → `int()` 抛 → 后面所有字段都不再加载，面板一半旧值一半文件值，
日志一个字都没有。用例 `test_panel_load_reports_bad_field` 钉的就是这个场景。

运行：
    C:\\aitp_venv\\Scripts\\python.exe tests\\test_settings.py
"""
import io
import json
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ⚠️ 与 main.py 一致：torch 必须先于 PyQt5（交接说明地雷 #2）
try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

from core import settings  # noqa: E402

failures = []
_APP = None
_WIN = None


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def _tmpdir():
    return tempfile.mkdtemp(prefix="cfg_")


def _write(path, payload, raw=False):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        if raw:
            f.write(payload)
        else:
            json.dump(payload, f, ensure_ascii=False)


def _read(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _audit_log(w):
    """把窗口的日志接出来 —— 用来断言"坏字段有没有被报出来"。"""
    got = []
    try:
        w.log_signal.disconnect()
    except Exception:
        pass
    w.log_signal.connect(lambda msg: got.append(str(msg)))
    return got


def window():
    global _APP, _WIN
    if _WIN is not None:
        return _WIN
    from PyQt5.QtWidgets import QApplication
    import ui.task_queue as tq

    fd, p = tempfile.mkstemp(prefix="aitp_cfg_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p            # 别写真实的 .task_history.json

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    _APP = QApplication.instance() or QApplication([])
    init_ui_invoker()
    _WIN = MainWindow()
    return _WIN


# ---------------------------------------------------------------- 引擎层

def test_schema_covers_real_file():
    """真实 `.train_settings.json` 里的每个键，schema 都必须声明过。

    没声明的键会在 `save()` 时被当"未知键"保留下来，但 `load()` 不认 ——
    表现是"设置了但重启就丢"。这条挡的是 schema 漏项。
    """
    real = ".train_settings.json"
    if not os.path.exists(real):
        return                              # 新环境没有这个文件，跳过
    keys = set(_read(real).keys())
    declared = set(settings.schema("train").keys())
    missing = sorted(keys - declared)
    assert not missing, (
        "schema 漏了这些真实存在的键：%r\n"
        "（漏了就会被当成未知键：读不回来、面板上表现为\"设置没保存住\"）" % missing)


def test_defaults_come_from_schema():
    d = settings.defaults("train")
    assert d["epochs"] == 200, d["epochs"]
    assert d["batch_size"] == 8, d["batch_size"]
    assert d["k_folds"] == 1, d["k_folds"]
    assert d["tile_stride"] == 75, d["tile_stride"]
    assert d["model"] == "deeplabv3", d["model"]
    assert d["infer_mode"] == "fine", d["infer_mode"]
    assert isinstance(d["augment_flags"], dict), "augment_flags 的默认值要是 dict"
    # 默认值必须是**拷贝**，否则调用方改一下返回值就污染了 schema
    d["augment_flags"]["enabled"] = "污染"
    assert settings.defaults("train")["augment_flags"]["enabled"] is True, \
        "默认值被上一次返回的字典污染了（deepcopy 没生效）"


def test_load_valid_file():
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": 45, "batch_size": 4, "model": "unet",
               "infer_mode": "fast", "tile": True})
    r = settings.load("train", p)
    assert r.ok, "完整合法的文件应返回 ok，实际 %s" % r.code
    v = r.value.values
    assert v["epochs"] == 45, v["epochs"]
    assert v["batch_size"] == 4, v["batch_size"]
    assert v["model"] == "unet", v["model"]
    assert v["infer_mode"] == "fast", v["infer_mode"]
    assert v["tile"] is True, v["tile"]
    # 文件里没写的键 → 默认值，且**不算问题**
    assert v["tile_size"] == 256, v["tile_size"]
    assert not r.value.problems, r.value.problems


def test_load_missing_file():
    d = _tmpdir()
    r = settings.load("train", os.path.join(d, "nope.json"))
    assert r.code == "not_found", r.code
    assert r.value is None, "文件不存在时 value 必须是 None（调用方据此\"什么都不改\"）"
    assert r.benign is True, "文件不存在是正常情况，必须 benign"


def test_load_corrupt_json():
    d = _tmpdir()
    p = os.path.join(d, "bad.json")
    _write(p, "{ 这不是 json", raw=True)
    r = settings.load("train", p)
    assert r.code == "corrupt", r.code
    assert r.value is None, "坏了就别给半份数据"
    assert r.benign is False, "坏了不是\"本来就没有\"，必须能被分辨"


def test_load_root_not_object():
    d = _tmpdir()
    p = os.path.join(d, "list.json")
    _write(p, [1, 2, 3])
    r = settings.load("train", p)
    assert r.code == "corrupt", r.code
    assert r.value is None


def test_bad_type_falls_back_and_reports():
    """★ 本轮的核心：字段写错要**报出来并指名**，同时不能拖垮其它字段。"""
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": "200 次", "batch_size": 4})
    r = settings.load("train", p)
    assert r.code == "invalid", "坏字段必须给 invalid，实际 %s" % r.code
    assert "epochs" in r.message, "报错必须指名是哪个键：%r" % r.message
    v = r.value.values
    assert v["epochs"] == 200, "坏字段应退回默认值 200，实际 %r" % (v["epochs"],)
    assert v["batch_size"] == 4, "★ 一个坏字段不能拖垮其它字段（改前就是这个 bug）"
    assert len(r.value.problems) == 1, r.value.problems


def test_out_of_range_is_clamped():
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": 0, "tile_stride": 999, "batch_size": -3})
    r = settings.load("train", p)
    assert r.code == "invalid", r.code
    v = r.value.values
    assert v["epochs"] == 1, "epochs 下界 1，实际 %r" % (v["epochs"],)
    assert v["tile_stride"] == 100, "tile_stride 上界 100，实际 %r" % (v["tile_stride"],)
    assert v["batch_size"] == 1, "batch_size 下界 1，实际 %r" % (v["batch_size"],)


def test_bool_is_not_int():
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": True})
    r = settings.load("train", p)
    assert r.code == "invalid", "True 当 epochs 应被拒绝（bool 是 int 子类，容易漏）"
    assert "epochs" in r.message, r.message


def test_integral_float_accepted_as_int():
    """手改 JSON 写成 `200.0` 很常见，应当按 200 接受，而不是报错。"""
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": 200.0})
    r = settings.load("train", p)
    assert r.ok, "200.0 应被接受为整数，实际 %s：%s" % (r.code, r.message)
    v = r.value.values["epochs"]
    assert v == 200 and isinstance(v, int) and not isinstance(v, bool), repr(v)


def test_choices_enforced():
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"infer_mode": "快速模式"})
    r = settings.load("train", p)
    assert r.code == "invalid", r.code
    assert "infer_mode" in r.message, r.message
    assert r.value.values["infer_mode"] == "fine", r.value.values["infer_mode"]


def test_unknown_key_is_not_an_error():
    """新版本写的字段，老版本读到不该报警 —— 否则每次升级刷一屏"配置有问题"。"""
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": 10, "future_flag": {"a": 1}})
    r = settings.load("train", p)
    assert r.ok, "未知键不该让整份配置变成 invalid，实际 %s：%s" % (r.code, r.message)
    assert "future_flag" in r.value.unknown, r.value.unknown


def test_validate_names_every_bad_key():
    """直接测判据本身：三个坏键就要三条 Problem，且键名对得上。"""
    problems = settings.validate("train", {
        "epochs": "abc", "batch_size": 0, "resume": 1, "model": "ok"})
    keys = sorted(p.key for p in problems)
    assert keys == ["batch_size", "epochs", "resume"], keys
    assert all(p.message for p in problems), "每条问题都要有说明"


def test_save_roundtrip():
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    r = settings.save("train", {"epochs": 33, "batch_size": 2}, p)
    assert r.ok, "写配置失败：%s" % r.code_text
    got = settings.load("train", p)
    assert got.ok, got.code_text
    assert got.value.values["epochs"] == 33, got.value.values["epochs"]
    assert got.value.values["batch_size"] == 2, got.value.values["batch_size"]


def test_save_preserves_unknown_keys():
    """合并写：文件里 schema 没声明的键必须留着，不能被老版本一保存就抹掉。"""
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"future_flag": 7, "epochs": 11})
    r = settings.save("train", {"batch_size": 3}, p)
    assert r.ok, r.code_text
    raw = _read(p)
    assert raw.get("future_flag") == 7, "未知键被抹掉了：%r" % raw
    assert raw.get("epochs") == 11, "没传的键应保持原值：%r" % raw
    assert raw.get("batch_size") == 3, raw


def test_save_reports_ignored_keys():
    """★ 调用方打错字：不写进文件，但**必须在返回值里报出来**。

    ★ 审核 2026-11-12 指出的：原来这里静默丢弃 —— 调用方以为保存成功了，
      下次读出来还是老值，日志一个字都没有。那和 P0-1 要治的"失败没语义"是同一个病。
      不报错是对的（拼错的键确实不该写），但沉默不对。
    """
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    r = settings.save("train", {"epoches": 99, "epochs": 5}, p)     # 拼错的 epoches
    assert r.ok, "打错键不该让整次保存失败：%s" % r.code_text
    raw = _read(p)
    assert "epoches" not in raw, "拼错的键被写进配置了：%r" % raw
    assert raw.get("epochs") == 5, raw

    rep = r.value
    assert rep is not None, "save() 应返回 SaveReport"
    assert rep.ignored == ("epoches",), \
        "拼错的键必须出现在 ignored 里，实际 %r" % (rep.ignored,)
    assert "epochs" in rep.written, rep.written


def test_save_is_atomic_no_tmp_left():
    """原子写：必须走 `os.replace`，不能留临时文件，也不能出现"半个 JSON"。

    ★ 不能只断言"没留 .tmp 文件" —— 直接 `open(p, "w")` 覆盖也不留 .tmp，
      那种写法照样通过，等于没验。这里直接把 `os.replace` 换掉计数。
    """
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": 1})

    calls = []
    orig_replace = os.replace

    def spy(src, dst):
        calls.append((src, dst))
        return orig_replace(src, dst)

    os.replace = spy
    try:
        r = settings.save("train", {"epochs": 2}, p)
    finally:
        os.replace = orig_replace

    assert r.ok, r.code_text
    assert calls, "★ 保存没走 os.replace —— 不是原子写（写一半断电会留下半个 JSON）"
    assert os.path.normcase(calls[0][1]) == os.path.normcase(p), \
        "os.replace 的目标不是配置文件：%r" % (calls[0],)
    leftovers = [f for f in os.listdir(d) if f.endswith(".tmp")]
    assert not leftovers, "留下了临时文件：%r" % leftovers
    raw = _read(p)                       # 能解析 = 不是半个 JSON
    assert raw["epochs"] == 2, raw


# ---------------------------------------------------------------- 真实调用点

def test_panel_load_applies_valid_config():
    """真实调用点 ①：合法配置要真的灌进面板控件。"""
    w = window()
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": 45, "batch_size": 4, "k_folds": 3, "tile": True,
               "tile_size": 512, "model": "unet", "infer_mode": "fast"})
    w._train_settings_path = lambda: p
    _audit_log(w)
    w._load_panel_train_settings()
    assert w.epochs_spin.value() == 45, w.epochs_spin.value()
    assert w.batch_spin.value() == 4, w.batch_spin.value()
    assert w.kfold_spin.value() == 3, w.kfold_spin.value()
    assert w.tr_tile_check.isChecked() is True
    assert w.tr_tile_spin.value() == 512, w.tr_tile_spin.value()
    assert w.tr_model_combo.currentText() == "unet"


def test_panel_load_reports_bad_field():
    """★ 真实调用点 ②：坏字段必须**在日志里被点名**，而不是静默用默认值。

    这条是"校验接进执行路径"的证据。变异「调用方不看 res.ok、不打日志」会红。
    """
    w = window()
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"epochs": "200 次", "batch_size": 6})
    w._train_settings_path = lambda: p
    logs = _audit_log(w)
    w._load_panel_train_settings()

    assert w.batch_spin.value() == 6, "★ 一个坏字段拖垮了其它字段：batch=%r" % w.batch_spin.value()
    assert w.epochs_spin.value() == 200, "坏字段应退回默认值，实际 %r" % w.epochs_spin.value()
    assert any("epochs" in m for m in logs), \
        "★ 坏字段没有被报出来（日志里找不到 epochs）：%r" % logs
    assert any("训练参数" in m for m in logs), "日志应带上来源前缀：%r" % logs


def test_panel_load_untouched_when_file_missing():
    """真实调用点 ③：文件不存在时**一个控件都不动**（与改前行为一致）。"""
    w = window()
    d = _tmpdir()
    w._train_settings_path = lambda: os.path.join(d, "nope.json")
    w.epochs_spin.setValue(77)
    w.batch_spin.setValue(9)
    logs = _audit_log(w)
    w._load_panel_train_settings()
    assert w.epochs_spin.value() == 77, "文件不存在却动了面板：%r" % w.epochs_spin.value()
    assert w.batch_spin.value() == 9, w.batch_spin.value()
    assert not [m for m in logs if "训练参数" in m], \
        "第一次启动没有配置文件，不该刷日志：%r" % logs


def test_panel_load_reports_corrupt():
    """真实调用点 ④：配置坏了要说一句，但面板仍然不动（保住用户当前设置）。"""
    w = window()
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, "{ 坏掉的 json", raw=True)
    w._train_settings_path = lambda: p
    w.epochs_spin.setValue(66)
    logs = _audit_log(w)
    w._load_panel_train_settings()
    assert w.epochs_spin.value() == 66, "配置坏了却动了面板：%r" % w.epochs_spin.value()
    assert any("corrupt" in m for m in logs), "配置坏了要报出来：%r" % logs


def test_panel_save_roundtrip():
    """真实调用点 ⑤：面板 → 文件 → 再读回来，值要对得上。"""
    w = window()
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    w._train_settings_path = lambda: p
    w.epochs_spin.setValue(31)
    w.batch_spin.setValue(5)
    w.kfold_spin.setValue(2)
    logs = _audit_log(w)
    w._save_panel_train_settings()
    assert os.path.exists(p), "设置没写进文件"
    got = settings.load("train", p)
    assert got.ok, "写出来的文件自己都读不回来：%s" % got.code_text
    v = got.value.values
    assert v["epochs"] == 31, v["epochs"]
    assert v["batch_size"] == 5, v["batch_size"]
    assert v["k_folds"] == 2, v["k_folds"]
    assert not [m for m in logs if "训练参数" in m], "正常保存不该报错：%r" % logs


def test_panel_save_keeps_unknown_keys():
    """真实调用点 ⑥：面板保存不能抹掉文件里新版本写的字段。"""
    w = window()
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, {"future_flag": "别丢", "epochs": 1})
    w._train_settings_path = lambda: p
    _audit_log(w)
    w._save_panel_train_settings()
    raw = _read(p)
    assert raw.get("future_flag") == "别丢", "面板保存把未知键抹掉了：%r" % raw


def test_panel_save_logs_ignored_keys():
    """★ 真实调用点：拼错的键要被写进日志，不能沉默。

    怎么造：把 `_collect_panel_train_settings` 换成返回带 typo 的字典 ——
    面板正常收集不会拼错，不换成它这条分支就永远测不到（写了不验）。
    """
    w = window()
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    w._train_settings_path = lambda: p
    orig = w._collect_panel_train_settings
    w._collect_panel_train_settings = lambda: {"epochs": 12, "epohcs": 200}
    logs = _audit_log(w)
    try:
        w._save_panel_train_settings()
    finally:
        w._collect_panel_train_settings = orig

    raw = _read(p) if os.path.exists(p) else {}
    assert raw.get("epochs") == 12, "正常的键没写进去：%r" % raw
    assert "epohcs" not in raw, "拼错的键被写进文件了：%r" % raw
    assert any("epohcs" in m for m in logs), \
        "★ 拼错的键没有被报出来（日志里找不到 epohcs）：%r" % logs


# ---------------------------------------------------------------- 另外两个写入者

#: 训练参数对话框只负责这 9 个键（其余由面板 / 增强设置各自维护）
_DIALOG_KEYS = ("epochs", "batch_size", "model", "image_size", "k_folds",
                "loss", "resume", "unet_base", "unet_depth")

#: 对话框**不写**、但以前会被它抹掉的 6 个键
_DIALOG_MUST_KEEP = ("augment_flags", "early_stop_patience", "infer_mode",
                     "tile", "tile_size", "tile_stride")


def _full_config():
    """一份"满配"的配置：15 个键全给上非默认值，方便看谁被抹掉。"""
    return {
        "epochs": 200, "batch_size": 16, "model": "unet", "image_size": 256,
        "k_folds": 1, "loss": "ce+lovasz", "resume": True,
        "unet_base": 16, "unet_depth": 4,
        "augment_flags": {"enabled": True, "flip": False},
        "early_stop_patience": 30, "infer_mode": "fine",
        "tile": True, "tile_size": 1024, "tile_stride": 75,
    }


def test_dialog_save_preserves_other_keys():
    """★ `TrainSettingsDialog._save_and_accept()` 不许抹掉自己不负责的键。

    改前它是 `open(path,"w")` 全量覆盖、只写 9 个键 —— 文件有 15 个，
    点一次「确定」就删掉 6 个：augment_flags / early_stop_patience / infer_mode /
    tile / tile_size / tile_stride。用户只改一个 batch_size，
    分块训练设置、早停耐心、推理档位、增强参数全部回到默认，日志一个字没有。

    ⚠️ 这个对话框目前**没有被打开**（main_window 里构造它的两行被注释掉了，
       改成直接读面板），所以这是"埋着的雷"而不是正在丢数据 ——
       但它保留的意义就是"回退成弹窗模式"，一回退就会踩到。
    """
    from ui.widgets import TrainSettingsDialog

    w = window()
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, _full_config())

    orig_file = TrainSettingsDialog.SETTINGS_FILE
    TrainSettingsDialog.SETTINGS_FILE = p
    _audit_log(w)
    try:
        dlg = TrainSettingsDialog(w)
        # 只改一个键，模拟"用户进来只想改 batch_size"
        dlg.batch_spin.setValue(7)
        dlg._save_and_accept()
    finally:
        TrainSettingsDialog.SETTINGS_FILE = orig_file

    raw = _read(p)
    assert raw.get("batch_size") == 7, "对话框改的值没写进去：%r" % raw.get("batch_size")
    lost = [k for k in _DIALOG_MUST_KEEP if k not in raw]
    assert not lost, "★ 对话框把不属于它的键抹掉了：%s" % lost
    assert raw["tile_size"] == 1024, "分块尺寸被改了：%r" % raw.get("tile_size")
    assert raw["augment_flags"] == {"enabled": True, "flip": False}, \
        "增强参数被改了：%r" % (raw.get("augment_flags"),)


def test_dialog_save_is_atomic():
    """对话框也必须走 `os.replace` —— 不能又回到 `open(w)` 直接覆盖。"""
    from ui.widgets import TrainSettingsDialog

    w = window()
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, _full_config())

    calls = []
    orig_replace = os.replace

    def spy(src, dst):
        calls.append(dst)
        return orig_replace(src, dst)

    orig_file = TrainSettingsDialog.SETTINGS_FILE
    TrainSettingsDialog.SETTINGS_FILE = p
    os.replace = spy
    try:
        dlg = TrainSettingsDialog(w)
        dlg._save_and_accept()
    finally:
        os.replace = orig_replace
        TrainSettingsDialog.SETTINGS_FILE = orig_file

    assert calls, "★ 对话框没走 os.replace —— 不是原子写"
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")], "留了临时文件"


def test_dialog_save_reports_failure():
    """写不进去时要报出来（交给父窗口的日志），不能静默。

    造一个必然失败的目标：把 SETTINGS_FILE 的父目录指到一个**普通文件**上，
    `mkstemp` 会 NotADirectoryError → save 返回 io。
    """
    from ui.widgets import TrainSettingsDialog

    w = window()
    d = _tmpdir()
    blocker = os.path.join(d, "not_a_dir")
    _write(blocker, {"x": 1})                       # 这是个文件，不能当目录用
    bad_path = os.path.join(blocker, ".train_settings.json")

    orig_file = TrainSettingsDialog.SETTINGS_FILE
    TrainSettingsDialog.SETTINGS_FILE = bad_path
    logs = _audit_log(w)
    try:
        dlg = TrainSettingsDialog(w)
        dlg._save_and_accept()
    finally:
        TrainSettingsDialog.SETTINGS_FILE = orig_file

    assert any("训练参数" in m for m in logs), \
        "★ 保存失败却没有任何提示：%r" % logs


def test_augment_save_keeps_other_keys_and_is_atomic():
    """`training.augment_config.save_augment_flags()` 只改 augment_flags，且必须原子。

    它原来的"读-改-写"合并逻辑是对的（不丢键），但 `open(w)` 直接覆盖：
    写一半崩溃就留半个 JSON，下次启动 `settings.load` 判 corrupt、面板不动
    ⇒ 整份训练参数按默认走。
    """
    import training.augment_config as ac

    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    _write(p, _full_config())

    scope = settings.SCOPES["train"]
    orig_root = scope.root
    scope.root = d                      # 让默认路径指到临时目录
    calls = []
    orig_replace = os.replace

    def spy(src, dst):
        calls.append(dst)
        return orig_replace(src, dst)

    os.replace = spy
    try:
        ac.save_augment_flags({"flip": True, "noise": True})
    finally:
        os.replace = orig_replace
        scope.root = orig_root

    raw = _read(p)
    assert raw["augment_flags"].get("noise") is True, "增强开关没写进去：%r" % raw["augment_flags"]
    assert raw["epochs"] == 200 and raw["tile_size"] == 1024, \
        "★ 增强设置把别的键改了：%r" % raw
    assert len(raw) == 15, "键数从 15 变成 %d：%r" % (len(raw), sorted(raw))
    assert calls, "★ 增强设置没走 os.replace —— 不是原子写"


def main():
    print("=== 配置统一入口（core/settings.py）回归测试 ===")
    checks = [
        ("schema 覆盖真实文件的所有键", test_schema_covers_real_file),
        ("默认值来自 schema（且是拷贝）", test_defaults_come_from_schema),
        ("读合法文件", test_load_valid_file),
        ("文件不存在 → not_found / value=None", test_load_missing_file),
        ("JSON 坏了 → corrupt / value=None", test_load_corrupt_json),
        ("根不是对象 → corrupt", test_load_root_not_object),
        ("★ 类型错：报出来 + 退回默认 + 不拖垮别的键", test_bad_type_falls_back_and_reports),
        ("超范围 → 夹到边界并报告", test_out_of_range_is_clamped),
        ("bool 不能当 int", test_bool_is_not_int),
        ("200.0 按整数 200 接受", test_integral_float_accepted_as_int),
        ("枚举值受约束", test_choices_enforced),
        ("未知键不算错误", test_unknown_key_is_not_an_error),
        ("validate 为每个坏键各报一条", test_validate_names_every_bad_key),
        ("save → load 往返", test_save_roundtrip),
        ("save 保留未知键", test_save_preserves_unknown_keys),
        ("★ save 把打错的键报在 ignored 里", test_save_reports_ignored_keys),
        ("save 原子写、不留临时文件", test_save_is_atomic_no_tmp_left),
        ("★ 面板：合法配置真的灌进去", test_panel_load_applies_valid_config),
        ("★ 面板：坏字段在日志里被点名", test_panel_load_reports_bad_field),
        ("面板：文件不存在时不动控件", test_panel_load_untouched_when_file_missing),
        ("面板：配置坏了报一句但不动控件", test_panel_load_reports_corrupt),
        ("★ 面板：保存后能读回来", test_panel_save_roundtrip),
        ("★ 面板：保存不抹未知键", test_panel_save_keeps_unknown_keys),
        ("★ 面板：拼错的键写进日志", test_panel_save_logs_ignored_keys),
        ("★ 对话框：保存不抹别的键", test_dialog_save_preserves_other_keys),
        ("★ 对话框：走原子写", test_dialog_save_is_atomic),
        ("★ 对话框：写失败要报出来", test_dialog_save_reports_failure),
        ("★ 增强设置：只改自己的键 + 原子", test_augment_save_keeps_other_keys_and_is_atomic),
    ]
    for name, fn in checks:
        check(name, fn)

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
