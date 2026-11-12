# -*- coding: utf-8 -*-
"""配置统一入口（重构清单 P0-2）。

## 为什么需要它

`.train_settings.json` 这类配置现在有 **20 份**，读写散在 `main_window` 的 7 个方法里。
最典型的坏味道是 `_load_panel_train_settings()`：

```python
self.epochs_spin.setValue(int(s.get("epochs", 200)))      # ← "abc" / null 直接抛
...
except Exception:
    pass                                                  # ← 整段静默放弃
```

一个字段写坏（比如手改成 `"epochs": "200 次"`），`int()` 抛异常 → 被最后那个
`except: pass` 吞掉 → **后面的字段一个都不再加载**，面板一半是旧值一半是文件值，
而且日志里一个字都没有。用户只会说"我的设置又没保存上"。

## 这个模块做的事

把配置声明成**一张表**，读取时逐字段校验：

- **缺失**（老文件没有新键）→ 用默认值，**不算错误**；
- **类型不对 / 超范围** → 记一条 `Problem`（带上是哪个键），该键退回默认值，
  其余键照常加载 —— 一个坏字段不再拖垮整份配置；
- **未知键**（新版本写的）→ 原样保留，不算错误（保存时不丢）；
- 文件不存在 / JSON 坏了 / 读不了 → 各自的错误码，`value` 为 `None`，
  调用方可以完全维持"不改任何界面"的老行为。

配合 `core/result.py`（P0-1）：返回值是 `Result`，`code` 能区分
`ok / invalid / not_found / corrupt / io`，`message` 里指名道姓是哪个键。

## 用法

    from core import settings

    res = settings.load("train", path)
    if res.value is None:            # 文件不存在或坏了
        if not res.benign:
            log(res.code_text)       # "corrupt / settings.load: ..."
        return                       # 不动界面（保持原行为）
    if not res.ok:
        log(res.code_text)           # ★ 有字段被替换成默认值，必须说出来
    s = res.value.values             # 已经过校验、该补的默认值都补齐了
    spin.setValue(s["epochs"])

    settings.save("train", {"epochs": 300}, path)   # 合并写 + 原子替换

## 三条设计约束（别"优化"掉）

1. **`validate()` 是唯一判据**：`merge()` 只认它返回的 `Problem` 列表来决定哪个键
   退回默认值。谁要是绕过 validate 自己做判断，两条路径迟早不一致。
2. **缺失 ≠ 错误**：老版本写的文件缺新键是正常情况，补默认值即可，不能报警 ——
   否则每次升级都刷一屏"配置有问题"，真问题反而被淹没。
3. **写入必须原子**：临时文件 + `os.replace()`。配置写一半断电，以前会留下
   半个 JSON，下次启动直接"配置损坏 → 全部默认"，用户所有设置一次丢光。
"""
import copy
import json
import os
import tempfile

from core.result import ErrCode, Result


# ---------------------------------------------------------------- 字段声明

class Problem(object):
    """一条校验问题：哪个键、什么毛病、拿到的是什么。"""

    __slots__ = ("key", "message", "got")

    def __init__(self, key, message, got=None):
        self.key = key
        self.message = message
        self.got = got

    def __str__(self):
        return "%s: %s" % (self.key, self.message)

    def __repr__(self):
        return "Problem(%r, %r)" % (self.key, self.message)


class Field(object):
    """一个配置字段的声明：默认值 + 类型 + 允许范围。"""

    kind = "any"

    def __init__(self, default, lo=None, hi=None, choices=None):
        self.default = default
        self.lo = lo
        self.hi = hi
        self.choices = tuple(choices) if choices else None

    # —— 子类实现 ——
    def check(self, value):
        """返回错误描述（str）表示不合法；返回 None 表示通过。"""
        raise NotImplementedError

    def coerce(self, value):
        """把**已通过校验**的值规范成目标类型。"""
        return value

    def clamp(self, value):
        """超范围时的取值。默认夹到边界。"""
        return value

    def __repr__(self):
        return "%s(default=%r)" % (type(self).__name__, self.default)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class Int(Field):
    kind = "int"

    def check(self, value):
        if isinstance(value, bool):
            return "要整数，给的是布尔 %r" % (value,)
        if isinstance(value, int):
            pass
        elif isinstance(value, float) and float(value).is_integer():
            pass                                   # 200.0 当 200，手改 JSON 常见
        else:
            return "要整数，给的是 %s %r" % (type(value).__name__, value)
        v = int(value)
        if self.lo is not None and v < self.lo:
            return "不得小于 %s，给的是 %s" % (self.lo, v)
        if self.hi is not None and v > self.hi:
            return "不得大于 %s，给的是 %s" % (self.hi, v)
        return None

    def coerce(self, value):
        return int(value)

    def clamp(self, value):
        v = int(value)
        if self.lo is not None:
            v = max(v, self.lo)
        if self.hi is not None:
            v = min(v, self.hi)
        return v


class Float(Field):
    kind = "float"

    def check(self, value):
        if not _is_number(value):
            return "要数字，给的是 %s %r" % (type(value).__name__, value)
        v = float(value)
        if self.lo is not None and v < self.lo:
            return "不得小于 %s，给的是 %s" % (self.lo, v)
        if self.hi is not None and v > self.hi:
            return "不得大于 %s，给的是 %s" % (self.hi, v)
        return None

    def coerce(self, value):
        return float(value)

    def clamp(self, value):
        v = float(value)
        if self.lo is not None:
            v = max(v, self.lo)
        if self.hi is not None:
            v = min(v, self.hi)
        return v


class Bool(Field):
    kind = "bool"

    def check(self, value):
        if not isinstance(value, bool):
            return "要 true/false，给的是 %s %r" % (type(value).__name__, value)
        return None

    def coerce(self, value):
        return bool(value)


class Str(Field):
    kind = "str"

    def check(self, value):
        if not isinstance(value, str):
            return "要字符串，给的是 %s %r" % (type(value).__name__, value)
        if self.choices and value not in self.choices:
            return "只能是 %s 之一，给的是 %r" % ("/".join(self.choices), value)
        return None

    def coerce(self, value):
        return str(value)


class Dict_(Field):
    """整体当字典用的字段（如 `augment_flags`），只校验最外层类型。

    内层结构由各自的业务方负责（`training.augment_config.merge_flags` 就做这件事）——
    在这里再抄一遍规则，两份迟早不一致。
    """

    kind = "dict"

    def check(self, value):
        if not isinstance(value, dict):
            return "要对象 {...}，给的是 %s %r" % (type(value).__name__, value)
        return None

    def coerce(self, value):
        return dict(value)


# ---------------------------------------------------------------- 作用域声明

class Scope(object):
    """一份配置文件 = 一个作用域。"""

    def __init__(self, filename, fields, root=None):
        self.filename = filename
        self.fields = fields
        self.root = root              # None = 相对于项目根（源码目录）

    def default_path(self):
        if self.root:
            return os.path.join(self.root, self.filename)
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.filename)

    def defaults(self):
        return {k: copy.deepcopy(f.default) for k, f in self.fields.items()}


DEFAULT_AUGMENT = {
    "enabled": True, "zoom_out": False, "zoom_in": False, "translate": False,
    "aspect": False, "shear": False, "rotate": False, "brightness": True,
    "contrast": True, "invert": False, "noise": False, "flip": True,
}

SCOPES = {
    # 训练参数面板（`_load_panel_train_settings` / `_save_panel_train_settings`）
    "train": Scope(".train_settings.json", {
        "epochs":              Int(200, 1, 100000),
        "early_stop_patience": Int(0, 0, 10000),
        "batch_size":          Int(8, 1, 4096),
        "k_folds":             Int(1, 1, 20),
        "resume":              Bool(False),
        "model":               Str("deeplabv3"),
        "image_size":          Int(256, 32, 8192),
        "loss":                Str("ce+lovasz"),
        "infer_mode":          Str("fine", choices=("fine", "fast")),
        "tile":                Bool(False),
        "tile_size":           Int(256, 32, 16384),
        "tile_stride":         Int(75, 1, 100),
        "unet_base":           Int(16, 4, 512),
        "unet_depth":          Int(4, 1, 12),
        "augment_flags":       Dict_(DEFAULT_AUGMENT),
    }),
}


def _scope(name):
    sc = SCOPES.get(name)
    if sc is None:
        raise KeyError("未知配置作用域：%r（已声明：%s）" % (name, ", ".join(sorted(SCOPES))))
    return sc


def schema(name):
    """返回 {键: Field} —— 给测试和文档用。"""
    return dict(_scope(name).fields)


def defaults(name):
    return _scope(name).defaults()


# ---------------------------------------------------------------- 校验与合并

def validate(name, payload):
    """逐字段校验，返回 [Problem]。**这是唯一的判据**（`merge` 只认它的结果）。

    缺失的键**不算问题**（老文件缺新键是正常的），只有"给了但给错"才记一笔。
    """
    sc = _scope(name)
    if not isinstance(payload, dict):
        return [Problem("<root>", "配置内容要是对象 {...}，实际是 %s"
                        % type(payload).__name__, payload)]
    problems = []
    for key, field in sc.fields.items():
        if key not in payload:
            continue                      # 缺失 → 用默认值，不是错误
        err = field.check(payload[key])
        if err:
            problems.append(Problem(key, err, payload[key]))
    return problems


class Loaded(object):
    """读取结果：已经补好默认值、规范化过类型的最终字典 + 问题清单。"""

    __slots__ = ("values", "problems", "unknown")

    def __init__(self, values, problems=(), unknown=()):
        self.values = values              # 只含 schema 里的键
        self.problems = tuple(problems)   # [Problem]
        self.unknown = tuple(unknown)     # 文件里有、schema 没声明的键

    def __repr__(self):
        return "Loaded(values=%d, problems=%d, unknown=%r)" % (
            len(self.values), len(self.problems), list(self.unknown))


def merge(name, payload):
    """把文件内容合并成最终字典。坏字段退回默认值（超范围的夹到边界）。"""
    sc = _scope(name)
    problems = validate(name, payload)            # ★ 唯一判据
    bad = {p.key for p in problems}
    values = {}
    for key, field in sc.fields.items():
        if key not in payload:
            values[key] = copy.deepcopy(field.default)     # 缺失 → 默认值（不是错误）
            continue
        if key in bad:
            # 分两种坏法：
            #   · 类型根本不对（"abc" 当整数）→ 默认值；硬 clamp 会抛异常
            #   · 只是超范围（0 当 epochs）→ 夹到边界，保住用户意图
            raw = payload[key]
            if field.kind in ("int", "float") and _is_number(raw):
                values[key] = field.clamp(raw)
            else:
                values[key] = copy.deepcopy(field.default)
            continue
        values[key] = field.coerce(payload[key])
    unknown = tuple(sorted(k for k in payload if k not in sc.fields))
    return Loaded(values, problems, unknown)


# ---------------------------------------------------------------- 读写

def load(name, path=None):
    """读一份配置。

    返回 `Result`，`value` 是 `Loaded`；文件不存在 / 坏了 / 读不了时 `value` 为 `None`
    （调用方据此维持"什么都不改"的老行为）。解析成功但有坏字段时 `code == invalid`，
    `value` 仍然可用（坏字段已退回默认值）。
    """
    sc = _scope(name)
    p = path or sc.default_path()
    where = "settings.load(%s)" % name

    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return Result.fail(ErrCode.NOT_FOUND, "配置文件不存在：%s" % p, where=where)
    except json.JSONDecodeError as e:
        return Result.fail(ErrCode.CORRUPT,
                           "配置 JSON 解析失败（第 %s 行）：%s" % (e.lineno, e.msg), where=where)
    except OSError as e:
        return Result.fail(ErrCode.IO, "读配置失败：%s" % e, where=where)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return Result.fail(ErrCode.CORRUPT,
                           "配置内容应是一个对象 {...}，实际是 %s" % type(raw).__name__,
                           where=where)

    data = merge(name, raw)
    if data.problems:
        return Result.fail(ErrCode.INVALID,
                           "；".join(str(x) for x in data.problems),
                           where=where, value=data)
    return Result.success(data)


def save(name, values, path=None):
    """合并写入（只覆盖 schema 里声明过的键），原子替换。

    文件里已有的**未知键会保留** —— 新版本写的字段不该被老版本一保存就抹掉。
    """
    sc = _scope(name)
    p = path or sc.default_path()
    where = "settings.save(%s)" % name

    if not isinstance(values, dict):
        return Result.fail(ErrCode.INVALID, "要写入的内容必须是 dict", where=where)

    existing = {}
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                got = json.load(f)
            if isinstance(got, dict):
                existing = got
            # 文件坏了就当空的：保存的意义就是把坏文件覆盖成好的
        except (json.JSONDecodeError, OSError):
            existing = {}

    payload = dict(existing)
    for key, val in values.items():
        if key in sc.fields:
            payload[key] = val
    # 只写 schema 键 + 原有未知键；传进来的未知键忽略（那是调用方打错字）

    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=".settings.", suffix=".tmp",
                                   dir=os.path.dirname(os.path.abspath(p)))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())      # 真落盘再替换，别只信页缓存
            except OSError:
                pass
        os.replace(tmp, p)
        return Result.success(True)
    except OSError as e:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return Result.fail(ErrCode.IO, "写配置失败：%s" % e, where=where)
