# -*- coding: utf-8 -*-
"""数据删除 / 恢复（回收站语义 + 可撤销）。

设计要点（为什么不是简单 os.remove）：

1. **可撤销优先**：系统回收站（Recycle Bin）里的文件**无法可靠地用程序还原**
   —— Windows Shell API 只能"送进去"，没有稳定的"取回来"接口。所以本模块把
   "删除"实现为 **移动到工程外的备份目录**：

       <project_dir>/../_deleted_data/<时间戳>/

   备份目录保持**相对路径**（images/xxx.bmp、annotations/xxx_mask.png），
   因此 `undo_last()` 可以精确还原。备份目录在**工程目录之外**，
   不会被平台的扫描（discover_labels / 数据集构建）看到。

2. **索引必须同步**：光删文件会导致训练/统计出错 ——
   - `train_test_split.json` 里残留的键 → 训练时报"文件不存在"（最严重）
   - 分类工程 `annotations/class_labels.json` 的 mapping 残留 → Class 表计数虚高
   - `tile_index.json`（多视图分块）残留 → 分块训练找不到图
   所以删除时**连带清理这些索引，并把旧值记进 journal**，撤销时一并还原。

3. **按"组"删除**：多视图工程一台料有 te_0/te_1/te_2 三张图，图片列表里只占一行。
   只删视图会造成"残缺组"（三视图不齐 → 该料直接废掉且污染训练），
   因此 `collect_targets()` 会把选中行**展开成整组**。

4. **失败不静默**：文件被占用（推理/训练正在读）时记录到 report.failed，
   并且**只为真正移走的 base 清理索引**，保证"文件 ↔ 索引"始终一致。

用法：
    report = delete_data(project_dir, row_bases)     # row_bases: 选中行的 base 名
    undo_last(project_dir)                           # 撤销最近一次
    purge_backup(project_dir)                        # 清空备份（送系统回收站）
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

# 多视图命名约定：<组名>_te_<序号>
GROUP_RE = re.compile(r"^(?P<g>.+)_te_(?P<i>\d+)$")
IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
BACKUP_ROOTNAME = "_deleted_data"          # 备份根目录名（在工程目录的父目录下）
JOURNAL_NAME = "journal.json"              # 每次删除一份，撤销靠它


# ---------------------------------------------------------------- 工具函数

def group_of(base: str) -> str:
    """`xxx_te_1` → `xxx`；单视图 `yyy` → `yyy`。"""
    m = GROUP_RE.match(base)
    return m.group("g") if m else base


def _img_dir(project_dir: str) -> str:
    return os.path.join(project_dir, "images")


def _ann_dir(project_dir: str) -> str:
    return os.path.join(project_dir, "annotations")


def group_views(project_dir: str, base: str) -> list[str]:
    """返回该 base 所属**组**在磁盘上的全部视图 base 名（按序号排序）。

    单视图工程：返回 [base]。多视图：返回 [xxx_te_0, xxx_te_1, ...]。
    **以磁盘实际存在的文件为准** —— 不凭空生成视图名，避免删到不存在的文件。
    """
    img_dir = _img_dir(project_dir)
    if not os.path.isdir(img_dir):
        return [base]
    g = group_of(base)
    if g == base:                                   # 单视图
        return [base]
    found: list[tuple[int, str]] = []
    pat = re.compile(re.escape(g) + r"_te_(\d+)$")
    for fn in os.listdir(img_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in IMG_EXTS:
            continue
        m = pat.match(stem)
        if m:
            found.append((int(m.group(1)), stem))
    if not found:
        return [base]
    found.sort()
    return [s for _, s in found]


def _image_path(project_dir: str, base: str) -> str | None:
    """找到 base 对应的图片文件（扩展名不限）。"""
    img_dir = _img_dir(project_dir)
    if not os.path.isdir(img_dir):
        return None
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def _derived_files(project_dir: str, base: str) -> list[str]:
    """annotations/ 下属于该 base 的派生文件（掩膜、叠加图等）。

    匹配规则带边界（`base.` 或 `base_`），避免 `a` 误匹配 `ab_mask.png`。
    """
    ann = _ann_dir(project_dir)
    out: list[str] = []
    if not os.path.isdir(ann):
        return out
    pat = re.compile(re.escape(base) + r"[._].*$")
    for fn in os.listdir(ann):
        if pat.match(fn):
            out.append(os.path.join(ann, fn))
    return out


# ---------------------------------------------------------------- 收集目标

def collect_targets(project_dir: str, row_bases: Iterable[str]) -> dict:
    """把"选中行"展开成要处理的文件与 base 清单。

    返回：{files, bases, groups}
      files  : 图片 / JSON / annotations 派生文件（去重、已存在的）
      bases  : 涉及的**视图 base**（用于清理 split / class_labels / tile_index）
      groups : 涉及的组名（打印用）
    """
    files: list[str] = []
    bases: list[str] = []
    groups: list[str] = []
    by_base: dict[str, list[str]] = {}
    seen_f: set[str] = set()
    seen_b: set[str] = set()
    for rb in row_bases:
        if not rb:
            continue
        g = group_of(rb)
        if g not in groups:
            groups.append(g)
        for base in group_views(project_dir, rb):
            if base in seen_b:
                continue
            seen_b.add(base)
            bases.append(base)
            cand: list[str] = []
            img = _image_path(project_dir, base)
            if img:
                cand.append(img)
            js = os.path.join(_img_dir(project_dir), base + ".json")
            if os.path.exists(js):
                cand.append(js)
            cand += _derived_files(project_dir, base)
            per: list[str] = []
            for p in cand:
                k = os.path.normcase(os.path.abspath(p))
                if k not in seen_f and os.path.isfile(p):
                    seen_f.add(k)
                    files.append(p)
                    per.append(p)
            by_base[base] = per
    # by_base 让调用方知道"每个 base 各自有哪些文件"，删除后才能精确判断
    # "哪些 base 真的被移走了"（只按文件名反推会踩 `images/` 前缀的坑）
    return {"files": files, "bases": bases, "groups": groups, "by_base": by_base}


# ---------------------------------------------------------------- 索引清理

def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)                     # 原子替换，避免写坏原文件


def _purge_split(project_dir: str, bases: Sequence[str]) -> dict:
    """从 train_test_split.json 移除这些 base，返回被移除的 {base: split}。"""
    path = os.path.join(project_dir, "train_test_split.json")
    if not os.path.exists(path):
        return {}
    data = _load_json(path, {})
    if not isinstance(data, dict):
        return {}
    removed = {}
    for base in bases:
        for key in (base, "images/" + base, os.path.basename(base)):
            if key in data:
                removed[base] = data.pop(key)
                break
    if removed:
        _save_json(path, data)
    return removed


def _purge_class_mapping(project_dir: str, bases: Sequence[str]) -> dict:
    """从 annotations/class_labels.json 的 mapping 移除这些 base。"""
    path = os.path.join(_ann_dir(project_dir), "class_labels.json")
    if not os.path.exists(path):
        return {}
    data = _load_json(path, {})
    mapping = data.get("mapping") if isinstance(data, dict) else None
    if not isinstance(mapping, dict):
        return {}
    removed = {}
    wants = set()
    for b in bases:
        wants |= {b, "images/" + b, os.path.basename(b)}
    for rel in list(mapping.keys()):
        stem = os.path.splitext(rel.replace("\\", "/"))[0]
        if stem in wants or os.path.basename(stem) in wants:
            removed[rel] = mapping.pop(rel)
    if removed:
        _save_json(path, data)
    return removed


def _purge_tile_index(project_dir: str, bases: Sequence[str]) -> dict:
    """从 tile_index.json（多视图分块索引）移除这些 base。"""
    path = os.path.join(project_dir, "tile_index.json")
    if not os.path.exists(path):
        return {}
    data = _load_json(path, {})
    if not isinstance(data, dict):
        return {}
    removed = {}
    for base in bases:
        if base in data:
            removed[base] = data.pop(base)
    if removed:
        _save_json(path, data)
    return removed


# ---------------------------------------------------------------- 备份/还原

def _backup_root(project_dir: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(project_dir)), BACKUP_ROOTNAME)


def _rel(project_dir: str, path: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(project_dir)).replace("\\", "/")


@dataclass
class DeleteReport:
    token: str = ""                     # 备份目录名（时间戳）＝撤销令牌
    backup_dir: str = ""
    moved: list[str] = field(default_factory=list)        # 已移走的文件（工程内相对路径）
    failed: list[tuple[str, str]] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    split_removed: dict = field(default_factory=dict)
    class_removed: dict = field(default_factory=dict)
    tile_removed: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.moved) and not self.failed

    def as_dict(self) -> dict:
        return asdict(self)


def delete_data(project_dir: str, row_bases: Iterable[str], log=print) -> DeleteReport:
    """删除选中行（按组展开）的图片 + JSON + 派生文件，并同步清理索引。

    顺序刻意设计为"先移文件、成功后才清索引"，保证任何中断下
    "文件存在 ↔ 索引存在"始终一致。
    """
    project_dir = os.path.abspath(project_dir)
    tgt = collect_targets(project_dir, row_bases)
    rep = DeleteReport(groups=tgt["groups"], bases=tgt["bases"])
    if not tgt["files"]:
        log("  没有可删除的文件（可能已被删除）")
        return rep

    token = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(_backup_root(project_dir), token)
    n = 2
    while os.path.exists(backup_dir):            # 同一秒内连续删除不能撞目录
        token = "%s_%d" % (time.strftime("%Y%m%d_%H%M%S"), n)
        backup_dir = os.path.join(_backup_root(project_dir), token)
        n += 1
    rep.token, rep.backup_dir = token, backup_dir

    moved_rels: list[str] = []
    for src in tgt["files"]:
        rel = _rel(project_dir, src)
        dst = os.path.join(backup_dir, rel)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                base, ext = os.path.splitext(dst)
                dst = f"{base}_{int(time.time()*1000)%100000}{ext}"
            shutil.move(src, dst)
            moved_rels.append(rel)
        except Exception as e:                       # 被占用 / 权限 / 路径过长
            rep.failed.append((rel, f"{type(e).__name__}: {e}"))
            log(f"  ✗ 移动失败 {rel}: {type(e).__name__}: {e}")
    rep.moved = moved_rels

    # 只为"图片真的被移走"的 base 清索引 ——
    # 依据是 collect_targets 的 by_base（不能拿相对路径反推 base 名，会带上 images/ 前缀）
    moved_set = set(moved_rels)

    def _img_moved(base: str) -> bool:
        for f in tgt.get("by_base", {}).get(base, []):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                return _rel(project_dir, f) in moved_set
        return False

    ok_bases = [b for b in tgt["bases"] if _img_moved(b)]
    rep.split_removed = _purge_split(project_dir, ok_bases)
    rep.class_removed = _purge_class_mapping(project_dir, ok_bases)
    rep.tile_removed = _purge_tile_index(project_dir, ok_bases)

    try:
        os.makedirs(backup_dir, exist_ok=True)
        with open(os.path.join(backup_dir, JOURNAL_NAME), "w", encoding="utf-8") as f:
            json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "project_dir": project_dir,
                       "moved": moved_rels,
                       "split_removed": rep.split_removed,
                       "class_removed": rep.class_removed,
                       "tile_removed": rep.tile_removed},
                      f, indent=2, ensure_ascii=False)
    except Exception as e:
        log(f"  ⚠ journal 写入失败（撤销将不可用）: {e}")

    log(f"  已删除 {len(moved_rels)} 个文件 → 备份 {token}"
        + (f"；失败 {len(rep.failed)} 个" if rep.failed else "")
        + (f"；清理划分键 {len(rep.split_removed)} 个" if rep.split_removed else ""))
    return rep


def list_backups(project_dir: str) -> list[str]:
    """返回可撤销的令牌（时间戳），新的在前。"""
    root = _backup_root(project_dir)
    if not os.path.isdir(root):
        return []
    out = [d for d in os.listdir(root)
           if os.path.isfile(os.path.join(root, d, JOURNAL_NAME))]
    return sorted(out, reverse=True)


def undo_last(project_dir: str, token: str | None = None, log=print) -> dict:
    """撤销一次删除：文件搬回原位 + 还原 split / class_labels / tile_index。"""
    project_dir = os.path.abspath(project_dir)
    token = token or (list_backups(project_dir) or [None])[0]
    if not token:
        log("  没有可撤销的删除记录")
        return {"restored": 0}
    backup_dir = os.path.join(_backup_root(project_dir), token)
    jpath = os.path.join(backup_dir, JOURNAL_NAME)
    if not os.path.exists(jpath):
        log(f"  找不到撤销记录: {jpath}")
        return {"restored": 0}
    j = _load_json(jpath, {})
    restored, failed = 0, []
    for rel in j.get("moved", []):
        src = os.path.join(backup_dir, rel)
        dst = os.path.join(project_dir, rel)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            restored += 1
        except Exception as e:
            failed.append((rel, str(e)))
            log(f"  ✗ 还原失败 {rel}: {e}")

    # 还原索引
    if j.get("split_removed"):
        path = os.path.join(project_dir, "train_test_split.json")
        data = _load_json(path, {})
        if isinstance(data, dict):
            data.update(j["split_removed"])
            _save_json(path, data)
    if j.get("class_removed"):
        path = os.path.join(_ann_dir(project_dir), "class_labels.json")
        data = _load_json(path, {})
        if isinstance(data, dict) and isinstance(data.get("mapping"), dict):
            data["mapping"].update(j["class_removed"])
            _save_json(path, data)
    if j.get("tile_removed"):
        path = os.path.join(project_dir, "tile_index.json")
        data = _load_json(path, {})
        if isinstance(data, dict):
            data.update(j["tile_removed"])
            _save_json(path, data)

    # 还原完就删掉这份 journal（避免重复撤销）；备份目录空了就移除
    try:
        os.remove(jpath)
        for root, dirs, files in os.walk(backup_dir, topdown=False):
            if not os.listdir(root):
                os.rmdir(root)
    except Exception:
        pass
    log(f"  已撤销：还原 {restored} 个文件"
        + (f"；还原划分键 {len(j.get('split_removed', {}))} 个" if j.get("split_removed") else "")
        + (f"；失败 {len(failed)} 个" if failed else ""))
    return {"restored": restored, "failed": failed}


def _send_to_trash(paths: Sequence[str]) -> str:
    """把文件/目录送进系统回收站。返回实际使用的方式。"""
    try:                                          # ① 首选 send2trash
        from send2trash import send2trash          # type: ignore
        for p in paths:
            send2trash(p)
        return "send2trash"
    except Exception:
        pass
    import ctypes                                  # ② Shell API
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    FO_DELETE, FOF_ALLOWUNDO, FOF_NOCONFIRMATION, FOF_SILENT = 3, 0x0040, 0x0010, 0x0004
    buf = "\0".join(os.path.abspath(p) for p in paths) + "\0\0"
    op = SHFILEOPSTRUCTW(None, FO_DELETE, buf, None,
                         FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT, False, None, None)
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if rc != 0:
        raise OSError(f"SHFileOperationW 返回 {rc}")
    return "shell"


def purge_backup(project_dir: str, token: str | None = None, log=print) -> int:
    """清空备份（默认全部）：优先送系统回收站，失败则彻底删除。

    这是"二次保险"步骤 —— 备份目录本身已经是可撤销的回收站，
    只有确认不再需要时才清空。
    """
    root = _backup_root(project_dir)
    if not os.path.isdir(root):
        return 0
    tokens = [token] if token else [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    n = 0
    for t in tokens:
        p = os.path.join(root, t)
        try:
            _send_to_trash([p])
            n += 1
        except Exception as e1:
            try:
                shutil.rmtree(p)
                n += 1
                log(f"  （回收站不可用，已彻底删除 {t}: {e1}）")
            except Exception as e2:
                log(f"  ✗ 清空 {t} 失败: {e2}")
    log(f"  已清空 {n} 份删除备份")
    return n
