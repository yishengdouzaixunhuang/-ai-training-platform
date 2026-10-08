# -*- coding: utf-8 -*-
"""core/file_ops.py 的单元测试（不依赖 GUI / torch）。

造一个假工程，覆盖三件最容易出错的事：
  ① 多视图"整组删"（选中 te_1 也要把 te_0/te_2 一起删）
  ② 索引同步（train_test_split.json / class_labels.json / tile_index.json）
  ③ 撤销能完整还原（文件 + 索引）
跑法：python tools/test_file_ops.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import file_ops as fo   # noqa: E402

FAIL = []


def check(cond, msg):
    print(("   ✓ " if cond else "   ✗ ") + msg)
    if not cond:
        FAIL.append(msg)


def build_project(root: str) -> str:
    proj = os.path.join(root, "proj_多视图")
    img = os.path.join(proj, "images")
    ann = os.path.join(proj, "annotations")
    os.makedirs(img, exist_ok=True)
    os.makedirs(ann, exist_ok=True)
    views = ["xxx_te_0", "xxx_te_1", "xxx_te_2"]
    singles = ["yyy"]
    for b in views + singles:
        with open(os.path.join(img, b + ".bmp"), "wb") as f:
            f.write(b"\x00" * 16)                                   # 假图
        with open(os.path.join(img, b + ".json"), "w", encoding="utf-8") as f:
            json.dump({"version": "5.3.1", "shapes": [], "imagePath": b + ".bmp"}, f)
        with open(os.path.join(ann, b + "_mask.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")                            # 假掩膜
    with open(os.path.join(proj, "train_test_split.json"), "w", encoding="utf-8") as f:
        json.dump({b: ("val" if b.endswith("_te_1") else "train") for b in views + singles}, f)
    with open(os.path.join(ann, "class_labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": ["OK", "Ys"], "mapping": {b + ".bmp": 1 for b in views + singles}}, f)
    with open(os.path.join(proj, "tile_index.json"), "w", encoding="utf-8") as f:
        json.dump({b: [[0, 0, 64, 64]] for b in views + singles}, f)
    return proj


def exists(proj, rel):
    return os.path.exists(os.path.join(proj, rel))


def ld(proj, rel):
    with open(os.path.join(proj, rel), "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    root = tempfile.mkdtemp(prefix="aitp_delfs_")
    print("测试工程根目录:", root)
    proj = build_project(root)
    print("\n【1】多视图整组删：选中行 base = xxx_te_1（应把整组 te_0/1/2 都删掉）")
    rep = fo.delete_data(proj, ["xxx_te_1"])
    check(rep.groups == ["xxx"], "识别出组名 xxx（得到 %s）" % rep.groups)
    check(sorted(rep.bases) == ["xxx_te_0", "xxx_te_1", "xxx_te_2"], "展开为 3 个视图：%s" % rep.bases)
    check(len(rep.moved) == 9, "移走 9 个文件（3 图 + 3 JSON + 3 掩膜），实际 %d" % len(rep.moved))
    check(not rep.failed, "无失败项")
    for b in ("xxx_te_0", "xxx_te_1", "xxx_te_2"):
        check(not exists(proj, "images/%s.bmp" % b), "图片已移走 %s.bmp" % b)
        check(not exists(proj, "images/%s.json" % b), "JSON 已移走 %s.json" % b)
        check(not exists(proj, "annotations/%s_mask.png" % b), "掩膜已移走 %s_mask.png" % b)
    check(exists(proj, "images/yyy.bmp"), "未选中的 yyy.bmp 未被误删")

    print("\n【2】索引同步")
    sp = ld(proj, "train_test_split.json")
    check(all(("xxx_te_%d" % i) not in sp for i in range(3)), "train_test_split 里的 3 个视图键已清理")
    check("yyy" in sp, "yyy 的划分键保留")
    cl = ld(proj, "annotations/class_labels.json")
    check(all(("xxx_te_%d.bmp" % i) not in cl["mapping"] for i in range(3)), "class_labels.mapping 已清理")
    check("yyy.bmp" in cl["mapping"], "yyy 的分类映射保留")
    ti = ld(proj, "tile_index.json")
    check(all(("xxx_te_%d" % i) not in ti for i in range(3)), "tile_index 已清理")
    check("yyy" in ti, "yyy 的分块索引保留")
    check(os.path.isfile(os.path.join(rep.backup_dir, fo.JOURNAL_NAME)), "journal.json 已写入")

    print("\n【3】撤销：文件与索引都要还原")
    r = fo.undo_last(proj)
    check(r["restored"] == 9, "还原 9 个文件（实际 %d）" % r["restored"])
    for b in ("xxx_te_0", "xxx_te_1", "xxx_te_2"):
        check(exists(proj, "images/%s.bmp" % b) and exists(proj, "images/%s.json" % b)
              and exists(proj, "annotations/%s_mask.png" % b), "%s 的图/JSON/掩膜都已还原" % b)
    sp2 = ld(proj, "train_test_split.json")
    check(sp2.get("xxx_te_1") == "val" and sp2.get("xxx_te_0") == "train", "划分键还原且值正确（te_1=val）")
    cl2 = ld(proj, "annotations/class_labels.json")
    check(cl2["mapping"].get("xxx_te_2.bmp") == 1, "分类映射还原且值正确（=1）")
    check(ld(proj, "tile_index.json").get("xxx_te_0") == [[0, 0, 64, 64]], "分块索引还原且内容正确")
    check(fo.list_backups(proj) == [], "撤销后不再有可撤销记录（不会重复撤销）")

    print("\n【4】单视图删（yyy）→ 撤销")
    rep2 = fo.delete_data(proj, ["yyy"])
    check(len(rep2.moved) == 3, "单视图只移走 3 个文件（实际 %d）" % len(rep2.moved))
    check(not exists(proj, "images/yyy.bmp"), "yyy.bmp 已移走")
    check("yyy" not in ld(proj, "train_test_split.json"), "yyy 划分键已清理")
    fo.undo_last(proj)
    check(exists(proj, "images/yyy.bmp") and "yyy" in ld(proj, "train_test_split.json"), "yyy 完整还原")

    print("\n【5】删除不存在的行 / 重复删除（幂等性）")
    rep3 = fo.delete_data(proj, ["not_exist_base"])
    check(rep3.moved == [] and rep3.failed == [], "删不存在的 base：不报错、不动任何文件")

    print("\n【6】清空备份（送系统回收站 / 兜底彻底删）")
    fo.delete_data(proj, ["yyy"])
    n = fo.purge_backup(proj)
    check(n >= 1, "清空了 %d 份备份" % n)
    check(fo.list_backups(proj) == [], "备份列表已空")

    shutil.rmtree(root, ignore_errors=True)
    print("\n" + ("全部通过 ✓" if not FAIL else "失败 %d 项：\n  - %s" % (len(FAIL), "\n  - ".join(FAIL))))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
