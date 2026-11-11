# -*- coding: utf-8 -*-
"""统一的返回与错误码（重构清单 P0-1）。

## 为什么需要它

全项目有 **610 处 `except Exception:`，其中 220 处紧跟 `pass`**（实测，见
`docs/重构清单_v1.md`）。后果不是"代码不好看"，而是**失败没有语义**：

- 调用方拿不到"为什么失败"，只能一律当"没有数据" —— 于是"缓存损坏"和"缓存不存在"
  走同一条路，用户永远不知道自己那份缓存被丢掉了（`core/mask_stats_cache` 就是这样）；
- 日志里没有任何线索，现场只能靠"重跑一次看看"；
- 想区分"可重试"和"没救了"时，无处可问。

这个模块只做一件事：**让失败带上一个可判定的错误码 + 一句人话**。
它**不**改变现有代码的异常行为（P0-1 是零侵入）：新代码用它，老代码照旧。

## 用法

    from core.result import Result, ErrCode, guard

    # 1) 手工构造
    def load(path):
        try:
            ...
        except FileNotFoundError:
            return Result.fail(ErrCode.NOT_FOUND, "缓存文件不存在", where="load")
        except ValueError as e:
            return Result.fail(ErrCode.CORRUPT, str(e), where="load")
        return Result.success(data)

    # 2) 判结果（`Result` 本身就带真假： ok 为真）
    r = load(p)
    if not r:
        log("读缓存失败：%s" % r.code_text)      # -> "corrupt / load: ..."
    data = r.unwrap_or({})

    # 3) 给"会抛异常的老函数"套一层，不用改函数体
    @guard(ErrCode.IO, where="save")
    def save(...): ...

## 三条设计约束（别"优化"掉）

1. **`ok` 是属性、不是字段**：字段只有 `code`，避免 `ok=True` 同时 `code=IO` 这种自相矛盾。
2. **`Result` 可当真假用**（`__bool__`）：老代码 `if save(...)` 这类写法不用改就能继续跑，
   迁移可以一个文件一个文件来，不必一次全换。
3. **不带异常对象**：`Result` 要能进日志、能序列化。
   原始异常信息用 `message` 带一句就够；真要看栈，调用方自己 `traceback`。
"""
from functools import wraps


class ErrCode(object):
    """错误码。刻意用**字符串常量**而不是 Enum。

    为什么要用字符串常量：
    - 日志/JSON 里直接就是 `"corrupt"`，不用 `.value`，也不会打出 `<ErrCode.X: 1>`；
    - 现场排查时能直接 grep；
    - 与 `task_queue.TaskState` 那种 Enum 不冲突（那个要参与比较和持久化，
      这里只要一个稳定的短字符串）。
    常量集中在这里，避免各处自己拼字符串拼错。
    """

    OK = "ok"                 # 成功
    NOT_FOUND = "not_found"   # 目标不存在（缓存没建过 / 文件被删）—— 通常**不是错误**
    CORRUPT = "corrupt"       # 文件在，但内容坏了（JSON 语法错、结构不对）
    VERSION = "version"       # 版本不符（老版本写的文件，需要重建）
    IO = "io"                 # 读写失败（权限、磁盘满、被独占）
    INVALID = "invalid"       # 调用方给的数据不合法
    CANCELED = "canceled"     # 用户取消
    UNKNOWN = "unknown"       # 没分类的异常（`guard` 兜底用）

    ALL = (OK, NOT_FOUND, CORRUPT, VERSION, IO, INVALID, CANCELED, UNKNOWN)

    #: 这些码表示"东西本来就没有/还没建"，调用方通常**不该**报错
    BENIGN = (OK, NOT_FOUND)


class Result(object):
    """一个"要么有值、要么有错误码"的返回。

    `Result.success(v)` / `Result.fail(code, msg)` 构造；`bool(r)` 等价于 `r.ok`。
    """

    __slots__ = ("code", "value", "message", "where")

    def __init__(self, code, value=None, message="", where=""):
        self.code = code
        self.value = value
        self.message = message
        self.where = where

    # ── 构造 ──
    @classmethod
    def success(cls, value=None):
        return cls(ErrCode.OK, value=value)

    @classmethod
    def fail(cls, code, message="", where="", value=None):
        if code == ErrCode.OK:
            # 用 fail 传 OK 一定是笔误（想复用变量？），当场纠成 UNKNOWN 更安全：
            # 否则会出现"ok 但带错误消息"的自相矛盾对象。
            code, message = ErrCode.UNKNOWN, message or "fail() 收到 OK"
        return cls(code, value=value, message=str(message), where=str(where))

    # ── 判定 ──
    @property
    def ok(self):
        return self.code == ErrCode.OK

    @property
    def benign(self):
        """失败是否属于"本来就没有"（缓存没建过这类，不该报警）。"""
        return self.code in ErrCode.BENIGN

    @property
    def code_text(self):
        """给日志用的一行：`code / where: message`。"""
        bits = [self.code or ErrCode.UNKNOWN]
        if self.where:
            bits.append(str(self.where))
        head = " / ".join(bits[:2])
        return ("%s: %s" % (head, self.message)) if self.message else head

    def __bool__(self):
        return self.ok

    def __repr__(self):
        if self.ok:
            return "Result.success(%r)" % (self.value,)
        return "Result.fail(%r, %r, where=%r)" % (self.code, self.message, self.where)

    # ── 取值 ──
    def unwrap_or(self, default):
        return self.value if self.ok else default

    def unwrap(self):
        """成功时取值，失败时抛 `ResultError`（带错误码，不是裸异常）。"""
        if not self.ok:
            raise ResultError(self)
        return self.value

    def raise_if_failed(self):
        if not self.ok:
            raise ResultError(self)
        return self

    def as_dict(self):
        """给日志 / JSON 用。"""
        return {"ok": self.ok, "code": self.code, "value": self.value,
                "message": self.message, "where": self.where}


class ResultError(RuntimeError):
    """`Result.unwrap()` 失败时抛这个 —— 带 code，调用方还能判。"""

    def __init__(self, result):
        super(ResultError, self).__init__(result.code_text)
        self.result = result
        self.code = result.code


def guard(code=ErrCode.UNKNOWN, where="", default=None):
    """给"会抛异常的函数"套一层：异常变成 `Result.fail(...)`。

    函数**本来就返回 Result** 时原样透传（不包第二层）—— 否则一个函数被
    `guard` 包装两次会得到 `Result(Result(...))`，取值处就得多问一次 `ok`。

        @guard(ErrCode.IO, where="mask_stats_cache.save")
        def save(...): ...
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                out = fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001  —— 这里就是要兜住一切
                return Result.fail(code, "%s: %s" % (type(e).__name__, e),
                                   where=where or fn.__name__)
            if isinstance(out, Result):
                return out
            return Result.success(out)
        return wrapper
    return deco
