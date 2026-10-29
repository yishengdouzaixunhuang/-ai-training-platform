# -*- coding: utf-8 -*-
"""类别表发现与合并：**图片的标注 JSON 才是类别的唯一真源**。

背景（真实工程踩过的坑）：
    集成商_233_客户 工程建好后 project.json 里是 "classes": []，
    用户把 JSON 标注批量导入 images/ 后，类别表始终是空的：
      · Class 表一行都没有；
      · 画布 LabelManager 只有 ["background"]，手工标注**选不到缺陷类别**；
      · 更糟的是 shapes_to_mask 当时会给未知标签"临时编号"，
        于是同一标签在不同图片里拿到不同掩膜值 —— 掩膜 PNG 直接污染。

本模块的职责：
    1. discover_labels()  扫描工程里所有标注 JSON，汇总出现过的类别名；
    2. merge_classes()    把发现的类别并进 project.json（保序、去重、写盘）；
    3. 供 GUI 在"打开工程 / 导入 JSON / 刷新"时调用，保证
       图片列表、预览窗口、Class 表三者用的是同一份类别表。
"""
import json
import os
import re

from core.label_scan import scan_labels

# 与 tools/import_farasis_labelme.py 的 DEFAULT_CLASSES 保持一致：
# 掩膜值 = 该列表下标，255 = __ignore__。顺序一变，已有掩膜的语义就会整体错位。
BASE_CLASSES = ["background", "Zw", "Ys", "Zz", "Hh", "Atd"]

# 这些标签不当类别：background 是 0 号类；ignore 类写入 255，不占类别号
_IGNORE_RE = re.compile(r"ignore|gnore", re.IGNORECASE)
_AUTO_NAME_RE = re.compile(r"^class[_\s]?\d+$", re.IGNORECASE)


def is_real_class(label):
    """该标签是否应作为一档类别进类别表。"""
    if not label or not isinstance(label, str):
        return False
    lab = label.strip()
    if not lab or lab == "background":
        return False
    if _IGNORE_RE.search(lab):            # __ignore__ / ignore / 不关注
        return False
    if _AUTO_NAME_RE.match(lab):          # class_1 / Class 2 这类自动名
        return False
    return True


def _json_dirs(project_dir):
    """工程里可能放标注 JSON 的目录（多视图工程的 JSON 就放在 images/ 里）。"""
    return [os.path.join(project_dir, "images"),
            os.path.join(project_dir, "annotations"),
            os.path.join(project_dir, "labels")]


def discover_labels(project_dir, extra_dirs=(), limit=None, on_progress=None):
    """扫描工程内所有标注 JSON，返回出现过的类别名（首次出现顺序，已去重）。

    limit: 最多扫描多少个 JSON（调试/提速用），None = 全部。
    """
    seen = []
    n_read = 0
    dirs = list(extra_dirs) + _json_dirs(project_dir)
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _sub, files in os.walk(d):
            for fn in sorted(files):
                if not fn.lower().endswith(".json"):
                    continue
                if fn in ("project.json", "class_labels.json", "train_test_split.json"):
                    continue
                if limit is not None and n_read >= limit:
                    return seen
                n_read += 1
                path = os.path.join(root, fn)
                # ★★ 2026-10-28：这里原来 `json.load` 整个文件。某些工程的 labelme JSON
                #   单个就有 14~18MB（掩膜轮廓上万个顶点），489 张合计 1GB —— 而本函数
                #   只想知道"有哪些类别"。改用 label_scan（bytes 正则 + 共享缓存）：
                #   全工程 4024ms → 820ms（冷）/ 19ms（热），且与 class_table 共用同一份缓存，
                #   不再"打开工程时同一批 JSON 被扫两遍"。
                info = scan_labels(path)
                if info is None:
                    continue
                for lab in info[0]:
                    if is_real_class(lab):
                        lab = lab.strip()
                        if lab not in seen:
                            seen.append(lab)
                if on_progress is not None and n_read % 200 == 0:
                    try:
                        on_progress(n_read, list(seen))
                    except Exception:
                        pass
    return seen


def newest_json_signature(project_dir):
    """标注 JSON 的"指纹"：文件名+个数+最新修改时间。**只 stat，不解析**，所以很快。

    用途：打开工程时判断"要不要重扫 402 个 JSON"。实测全量扫一遍要 ~1.2s，
    但内容没变时纯属白等 —— 指纹一样就直接用 project.json 里已有的类别表。
    """
    sig = []
    for d in _json_dirs(project_dir):
        if not d or not os.path.isdir(d):
            continue
        try:
            names = []
            mx = 0.0
            for fn in os.scandir(d):
                if fn.is_file() and fn.name.lower().endswith(".json"):
                    names.append(fn.name)
                    try:
                        mx = max(mx, fn.stat().st_mtime)
                    except OSError:
                        pass
            sig.append("%s:%d:%.0f" % (os.path.basename(d), len(names), mx))
        except OSError:
            continue
    return "|".join(sig)


def set_classes_sig(project_dir, sig):
    """把指纹写进 project.json（和类别表放一起，下次打开就能直接比对）。"""
    path = os.path.join(project_dir, "project.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return
    meta["classes_json_sig"] = sig
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def read_project_meta(project_dir):
    try:
        with open(os.path.join(project_dir, "project.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_project_classes(project_dir):
    """读工程 project.json 里的类别表（按平台约定补齐 background）。"""
    classes = []
    path = os.path.join(project_dir, "project.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            classes = list(meta.get("classes", []) or [])
        except Exception:
            classes = []
    if "background" not in classes:
        classes.insert(0, "background")
    return classes


def merge_classes(project_dir, discovered, write=True):
    """把 discovered 并进 project.json 的类别表。

    返回 (classes, added)：
      classes = 合并后的完整类别表（下标即掩膜值）
      added   = 本次新增的类别名（空表示没变）

    排序规则（**顺序就是掩膜语义，稳定压倒一切**）：
      · background 恒在 0 号位，255 恒为 ignore；
      · 工程原本一个类别都没有（classes 为空 / 只有 background）时，按 BASE_CLASSES
        的标准顺序排 —— 新建工程与训练数据集、已有工具的口径就一致了；
      · 工程**已经有**类别时，**原顺序一律不动**，新类别追加到末尾。
        绝不能因为"又发现了一个更靠前的标准类别"就重排：那会让所有已有掩膜 PNG
        的像素值整体错位（Ys 从 1 变成 2，旧掩膜立刻全错）。
    """
    cur = read_project_classes(project_dir)
    discovered = [c.strip() for c in (discovered or []) if is_real_class(c)]
    old_known = [c for c in cur if c != "background"]
    new_known = [c for c in discovered if c not in old_known]
    if old_known:
        ordered_known = old_known + [c for c in BASE_CLASSES
                                     if c != "background" and c in new_known]
        ordered_known += [c for c in new_known if c not in ordered_known]
    else:
        ordered_known = [c for c in BASE_CLASSES if c != "background" and c in new_known]
        ordered_known += [c for c in new_known if c not in ordered_known]
    classes = ["background"] + ordered_known
    added = [c for c in classes if c not in cur]
    removed = [c for c in cur if c not in classes]      # 正常为 0，仅用于日志
    if write and (added or removed or classes != cur):
        path = os.path.join(project_dir, "project.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        meta["classes"] = classes
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    return classes, added
