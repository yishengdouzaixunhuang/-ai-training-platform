# -*- coding: utf-8 -*-
"""掩膜统计的工程级 sidecar 缓存。

为什么需要
----------
打开大图工程时，要为每个 `<base>_mask.png` 算「类别数 / 是否有内容」。
4096² 的掩膜单张 31.7ms（PIL getcolors），眉山中航那个工程 209 张就是 6.6 秒，
全部压在「打开工程」这一个动作上。而这个结果只跟文件内容有关，
完全可以在工程目录里存一份，下次打开直接读回来。

缓存文件：`<project_dir>/.mask_stats.json`

    {
      "version": 1,
      "entries": {
        "annotations/xxx_mask.png": [mtime_ns, size, classes, has],
        ...
      }
    }

为什么用 mtime_ns + size 双条件
--------------------------------
- 只比 mtime：FAT / 网络盘的时间精度常常只到秒，"同一秒内改两次"会被骗过
- 只比 size：改了像素但文件大小恰好不变（PNG 压缩后常见）会被骗过
两条都匹配才复用缓存。宁可多解码一张，也不能把旧数据当成「已标注」——
列表「标注」列显示错的类别数，用户会以为标注丢了。

写盘用「临时文件 + os.replace」
--------------------------------
断电时要么是旧文件、要么是新文件，不会留下半个 JSON。
这个习惯抄自 `ui/task_queue._save_history`。
缓存写不进去（权限、磁盘满）不算错误，只是这一次白算，不影响主流程；
但 `save()` 会带 `ErrCode.IO` 出来，调用方想提醒用户就能提醒（P0-1）。
"""
import json
import os
import tempfile

from core.result import ErrCode, Result

VERSION = 1
FILENAME = ".mask_stats.json"


def cache_path(project_dir):
    """缓存文件的完整路径（工程根目录下，`.` 开头，不干扰用户浏览文件）。"""
    return os.path.join(str(project_dir), FILENAME)


def stat_of(mask_path):
    """(mtime_ns, size)。取不到时返回 (0, -1)，调用方据此判为 miss。"""
    try:
        st = os.stat(mask_path)
        return int(st.st_mtime_ns), int(st.st_size)
    except OSError:
        return 0, -1


def load(project_dir):
    """读缓存 → `Result`，`value` 是 {相对路径: (mtime_ns, size, classes, has)}。

    ★ 2026-11-11（P0-1 第一个真实迁移点）：原来"文件不存在 / JSON 坏 / 版本不符 /
      结构不对"**四种情况一律返回空 dict**，调用方无从区分 —— 缓存坏了被静默丢掉，
      用户只感觉"这次打开慢了 6 秒"，永远查不到原因。现在 `code` 分得清：
      `NOT_FOUND`（正常，还没建过）/ `CORRUPT`（内容坏了）/ `VERSION`（版本旧了）/ `IO`。

    仍然**绝不抛异常** —— 缓存是加速手段，坏了就该退化成"全量重算"，不能把工程打开卡死。
    `Result` 可当真假用（`if load(...)`），老的调用写法不改也能跑；
    要默认值的用 `.unwrap_or({})` —— **失败时 `value` 是 `None`**（`Result` 的统一语义）。
    """
    try:
        with open(cache_path(project_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return Result.fail(ErrCode.NOT_FOUND, "缓存还没建过", where="mask_stats_cache.load")
    except ValueError as e:                 # JSON 语法错
        return Result.fail(ErrCode.CORRUPT, str(e), where="mask_stats_cache.load")
    except OSError as e:
        return Result.fail(ErrCode.IO, str(e), where="mask_stats_cache.load")

    if not isinstance(data, dict) or data.get("version") != VERSION:
        return Result.fail(ErrCode.VERSION,
                           "版本 %r != %r" % ((data.get("version") if isinstance(data, dict) else None), VERSION),
                           where="mask_stats_cache.load")

    entries = data.get("entries")
    if not isinstance(entries, dict):
        return Result.fail(ErrCode.CORRUPT, "entries 不是 dict",
                           where="mask_stats_cache.load")

    out = {}
    dropped = 0
    for rel, v in entries.items():
        try:
            if isinstance(v, (list, tuple)) and len(v) == 4:
                out[str(rel)] = (int(v[0]), int(v[1]), int(v[2]), bool(v[3]))
            else:
                dropped += 1    # 长度不对也算坏，别静默
        except (TypeError, ValueError):
            dropped += 1        # 单条坏了就丢这一条，其余照用
            continue
    res = Result.success(out)
    if dropped:
        # 部分条目坏了：值照用，但把原因带出去（调用方可以决定要不要提醒）
        res.message = "丢弃 %d 条坏记录" % dropped
    return res


def save(project_dir, entries):
    """原子写入 → `Result`。

    写不进去（权限、磁盘满）**不算错误**（只是这一次白算），
    但现在会带 `ErrCode.IO` + 原因，调用方想提醒用户就能提醒。
    """
    project_dir = str(project_dir)
    tmp_path = None
    try:
        payload = {
            "version": VERSION,
            "entries": {
                str(k): [int(v[0]), int(v[1]), int(v[2]), bool(v[3])]
                for k, v in entries.items()
            },
        }
        fd, tmp_path = tempfile.mkstemp(
            prefix=".mask_stats.", suffix=".tmp", dir=project_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            try:
                os.fsync(f.fileno())      # 真落盘再替换，别只信页缓存
            except OSError:
                pass
        os.replace(tmp_path, cache_path(project_dir))
        return Result.success(True)
    except Exception as e:  # noqa: BLE001
        if tmp_path:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
        return Result.fail(ErrCode.IO, "%s: %s" % (type(e).__name__, e),
                           where="mask_stats_cache.save")
