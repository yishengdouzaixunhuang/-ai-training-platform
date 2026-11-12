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


def test_save_ignores_unknown_input_keys():
    """调用方打错字（传了个 schema 没有的键）不该被写进文件。"""
    d = _tmpdir()
    p = os.path.join(d, ".train_settings.json")
    settings.save("train", {"epoches": 99, "epochs": 5}, p)     # 拼错的 epoches
    raw = _read(p)
    assert "epoches" not in raw, "拼错的键被写进配置了：%r" % raw
    assert raw.get("epochs") == 5, raw


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
        ("save 忽略调用方打错的键", test_save_ignores_unknown_input_keys),
        ("save 原子写、不留临时文件", test_save_is_atomic_no_tmp_left),
        ("★ 面板：合法配置真的灌进去", test_panel_load_applies_valid_config),
        ("★ 面板：坏字段在日志里被点名", test_panel_load_reports_bad_field),
        ("面板：文件不存在时不动控件", test_panel_load_untouched_when_file_missing),
        ("面板：配置坏了报一句但不动控件", test_panel_load_reports_corrupt),
        ("★ 面板：保存后能读回来", test_panel_save_roundtrip),
        ("★ 面板：保存不抹未知键", test_panel_save_keeps_unknown_keys),
    ]
    for name, fn in checks:
        check(name, fn)

    print()
    if failures:
        print("❌ 有失败：%d / %d" % (len(failures), len(checks)))
        for name, exc in failures:
            print("   - %s: %s" % (name, exc))
        return 1
    print("✅ 全部通过  %d/%d" % (len(checks), len(checks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
