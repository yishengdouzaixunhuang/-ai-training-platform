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
缓存写不进去（权限、磁盘满）不算错误，只是这一次白算，不影响主流程。
"""
import json
import os
import tempfile

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
    """读缓存 → {相对路径: (mtime_ns, size, classes, has)}。

    文件不存在、JSON 坏了、版本不符，一律返回空 dict（绝不抛异常）。
    缓存是加速手段，坏了就应该退化成「全量重算」，不能把工程打开卡死。
    """
    try:
        with open(cache_path(project_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict) or data.get("version") != VERSION:
        return {}

    entries = data.get("entries")
    if not isinstance(entries, dict):
        return {}

    out = {}
    for rel, v in entries.items():
        try:
            if isinstance(v, (list, tuple)) and len(v) == 4:
                out[str(rel)] = (int(v[0]), int(v[1]), int(v[2]), bool(v[3]))
        except (TypeError, ValueError):
            continue            # 单条坏了就丢这一条，其余照用
    return out


def save(project_dir, entries):
    """原子写入。成功返回 True，失败返回 False（失败不抛异常）。"""
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
        return True
    except Exception:
        if tmp_path:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
        return False
