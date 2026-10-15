# -*- coding: utf-8 -*-
"""客户三视图 LabelMe 导出 → 可训练工程（原分辨率切图）

背景（233 训练集实测）：
  * 缺陷中位仅 11x12 px（原图 ~1450x3740），整图缩放会把缺陷缩没 → 必须原分辨率切 tile
  * 图像尺寸有 30+ 种 → 以 te_0(漫反射) 尺寸为基准，其余视图 resize 对齐
  * 三视图 JSON 完全相同 → 只光栅化一份掩膜，三视图共用
  * __ignore__ 是"不确定/不关注"区域 → 掩膜写入 255（训练时 ignore_index）

用法：
  python -m tools.import_farasis_labelme --src <导出根> --dst <工程目录> [--limit 150]
输出工程布局（与 training/multiview.MultiViewSegDataset 完全兼容）：
  images/{gid}_t{k}_te_{0,1,2}.jpg
  annotations/{gid}_t{k}_mask.png      像素值 = 类别索引，255 = 忽略
  project.json / train_test_split.json / tile_index.json / import_report.json
  preview/*.jpg                        人工抽检用（原图+掩膜叠加）
"""
from __future__ import annotations
import argparse, json, re, sys, time, collections, random, hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

VIEWS = ("te_0", "te_1", "te_2")
VIEW_CN = {"te_0": "漫反射", "te_1": "形状图1", "te_2": "形状图2"}
LEGACY_VIEW = {"漫反射": 0, "形状图1": 1, "形状图2": 2}
IGNORE = 255
IGNORE_LABEL = "__ignore__"

# 类别顺序固定（background 必须为 0）。中文名已经现场确认。
DEFAULT_CLASSES = ["background", "Zw", "Ys", "Zz", "Hh", "Atd"]
CLASS_CN_GUESS = {"background": "背景", "Zw": "脏污", "Ys": "压伤", "Zz": "褶皱",
                  "Hh": "划痕", "Atd": "凹凸点"}


# ─────────────────────── 命名解析 ───────────────────────
def parse_stem(stem: str):
    """返回 (group_key, view_idx)。view_idx: 0/1/2，无法判定返回 -1。"""
    m = re.match(r"^(.*)_te_(\d+)$", stem)
    if m:
        v = int(m.group(2))
        return m.group(1), (v if v in (0, 1, 2) else -1)
    m = re.search(r"(漫反射|形状图1|形状图2)", stem)
    if m:
        # 旧命名：03NCB...260漫反射-20260814-110044-431 → 去掉视图词与尾部毫秒
        g = stem[: m.start()] + "#" + stem[m.end():]
        g = re.sub(r"-\d+$", "", g)
        return g, LEGACY_VIEW[m.group(1)]
    return stem, -1


# ─────────────────────── 扫描与分组 ───────────────────────
def scan_source(src: Path):
    img_roots, lab_roots = [], []
    for d in src.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("原图"):
            tr = d / "Train"
            if tr.exists():
                img_roots.append(tr)
        elif d.name.lower().startswith("label"):
            tr = d / "Train"
            if tr.exists():
                lab_roots.append(tr)
    if not img_roots or not lab_roots:
        # 退化：直接在 src 下找
        img_roots = img_roots or [src]
        lab_roots = lab_roots or [src]
    return img_roots, lab_roots


def build_index(img_roots, lab_roots, log=print):
    imgs = {}     # stem -> (view_idx, path)
    conflict = []
    for root in img_roots:
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            g, v = parse_stem(p.stem)
            if v < 0:
                # 用所在目录名兜底（导出结构 labeled/<0|1|2>/）
                try:
                    di = int(p.parent.name)
                    v = di if di in (0, 1, 2) else -1
                except ValueError:
                    v = -1
            if p.stem in imgs and imgs[p.stem][0] != v:
                conflict.append(p.stem)
            imgs[p.stem] = (v, p)

    jsons = {}
    for root in lab_roots:
        for p in root.rglob("*.json"):
            jsons[p.stem] = p

    log(f"[index] 图片 {len(imgs)}  标注 {len(jsons)}")
    only_i = set(imgs) - set(jsons)
    only_j = set(jsons) - set(imgs)
    if only_i:
        log(f"[index] 仅图片无标注 {len(only_i)}：{list(only_i)[:3]}")
    if only_j:
        log(f"[index] 仅标注无图片 {len(only_j)}：{list(only_j)[:3]}")
    if conflict:
        log(f"[index] 视图号冲突 {len(conflict)}：{conflict[:3]}")

    groups = collections.defaultdict(dict)     # gkey -> {view: stem}
    stems = collections.defaultdict(list)
    for stem, (v, _) in imgs.items():
        if stem not in jsons:
            continue
        g, _ = parse_stem(stem)
        stems[g].append((v, stem))
    for g, lst in stems.items():
        for v, stem in sorted(lst):
            if v < 0:
                continue
            if v in groups[g]:
                continue
            groups[g][v] = stem
    complete = {g: d for g, d in groups.items() if set(d) >= {0, 1, 2}}
    log(f"[index] 总计 {len(imgs)} 图 / {len(groups)} 组 / 完整三视图 {len(complete)} 组")
    return imgs, jsons, complete


# ─────────────────────── 掩膜光栅化 ───────────────────────
def rasterize(shapes, classes, hw, ignore_id=IGNORE):
    """shapes → uint8 掩膜。classes: [background, ...]；未知标签按背景处理。"""
    h, w = hw
    cls_idx = {c: i for i, c in enumerate(classes)}
    # 先铺 ignore，再画缺陷，最后 ignore 形状覆盖（ignore 优先，语义=不确定）
    m = np.zeros((h, w), np.uint8)
    ign = []
    for s in shapes:
        lab = s.get("label", "")
        if lab == IGNORE_LABEL:
            ign.append(s)
            continue
        if lab not in cls_idx:
            continue
        idx = cls_idx[lab]
        if idx == 0:
            continue                       # background 形状不画
        pts = np.asarray(s["points"], np.float64)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], int(idx))
    for s in ign:
        pts = np.asarray(s["points"], np.float64)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], int(ignore_id))
    return m


def shape_boxes(shapes, classes):
    """非 ignore 形状的 (label, x0,y0,x1,y1)。"""
    out = []
    for s in shapes:
        lab = s.get("label", "")
        if lab == IGNORE_LABEL or lab not in classes or lab == "background":
            continue
        pts = np.asarray(s["points"], np.float64)
        if pts.shape[0] < 3:
            continue
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        out.append((lab, float(x0), float(y0), float(x1), float(y1)))
    return out


# ─────────────────────── 切图窗口规划 ───────────────────────
def plan_tiles(boxes, hw, tile, margin, neg_ratio, rng):
    """正样本: 以缺陷为中心；同窗缺陷合并。负样本: 无缺陷区域随机采样。"""
    H, W = hw
    half = tile // 2.0
    pos, covered = [], []
    used = [False] * len(boxes)
    order = sorted(range(len(boxes)),
                   key=lambda i: -((boxes[i][3] - boxes[i][1]) * (boxes[i][4] - boxes[i][2])))
    for i in order:
        if used[i]:
            continue
        lab, x0, y0, x1, y1 = boxes[i]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # 窗口以缺陷为中心（缺陷完整落在窗内 + margin 上下文）
        wx0 = int(round(min(max(cx - half, 0), max(W - tile, 0))))
        wy0 = int(round(min(max(cy - half, 0), max(H - tile, 0))))
        labs = {lab}
        used[i] = True
        for j in range(len(boxes)):
            if used[j]:
                continue
            l2, a0, b0, a1, b1 = boxes[j]
            if a0 >= wx0 and a1 <= wx0 + tile and b0 >= wy0 and b1 <= wy0 + tile:
                used[j] = True
                labs.add(l2)
        pos.append((wx0, wy0, sorted(labs)))
        covered.append((wx0, wy0))

    n_neg = int(round(len(pos) * neg_ratio))
    if not pos:                       # 全空白组：也要采负样本（纯背景很有价值）
        n_neg = max(n_neg, 2)
    neg = []
    if n_neg > 0 and H >= tile and W >= tile:
        mask_def = np.zeros((H, W), np.uint8)
        for _, x0, y0, x1, y1 in boxes:
            mask_def[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = 1
        # 积分图快速判断窗口内是否含缺陷
        ii = cv2.integral(mask_def)
        tries = 0
        while len(neg) < n_neg and tries < n_neg * 40:
            tries += 1
            wx0 = int(rng.integers(0, W - tile + 1))
            wy0 = int(rng.integers(0, H - tile + 1))
            s = ii[wy0 + tile, wx0 + tile] - ii[wy0, wx0 + tile] - ii[wy0 + tile, wx0] + ii[wy0, wx0]
            if s > 0:
                continue
            # 与已有正样本窗口重叠过多则跳过
            if any(abs(wx0 - a) < tile * 0.5 and abs(wy0 - b) < tile * 0.5 for a, b in covered):
                continue
            neg.append((wx0, wy0, []))
    return pos + neg


# ─────────────────────── 划分 ───────────────────────
def plan_split(gkeys, jsons, view_stems, val_ratio, seed, log=print):
    """两遍法：按缺陷数分层 + 确定性划分。

    为什么不能用顺序 rng 抽签（原来的做法，已废弃）：
      1) 抽签发生在切图之后，而切图消耗的随机数个数随缺陷数变化 →
         不同切图尺寸的工程会得到**不同的** train/val 划分，模型之间无法比较；
      2) 缺陷数分布极度长尾（实测单组最多 1300+ 个微小连通域），
         纯随机划分会让 val 被个别"病态图"整体主导，指标失去代表性。

    做法：先统计每组缺陷数并分层（0 / 1-2 / 3-5 / 6-20 / 21-100 / >100），
    每层内按 (seed, 组名) 的哈希排序后取前 val_ratio 比例进 val。
    → 同一组在任何工程里都进同一个 split；各层比例一致，val 有代表性。
    """
    counts = {}
    for g in gkeys:
        jp = jsons.get(view_stems[g][0])
        n = 0
        if jp is not None:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                n = sum(1 for s in meta.get("shapes", []) if s.get("label") != IGNORE_LABEL)
            except Exception:
                n = 0
        counts[g] = n

    bounds = (0, 2, 5, 20, 100, 10 ** 9)

    def bucket(n):
        for i, hi in enumerate(bounds):
            if n <= hi:
                return i
        return len(bounds) - 1

    buckets = collections.defaultdict(list)
    for g in gkeys:
        buckets[bucket(counts[g])].append(g)
    split = {}
    for b, gs in buckets.items():
        gs = sorted(gs, key=lambda g: hashlib.md5(f"{seed}:{g}".encode("utf-8")).hexdigest())
        nv = int(round(len(gs) * val_ratio))
        for i, g in enumerate(gs):
            split[g] = "val" if i < nv else "train"

    for sp in ("train", "val"):
        gs = [g for g in gkeys if split[g] == sp]
        log(f"[split] {sp:<5} 组={len(gs):<4} 缺陷实例={sum(counts[g] for g in gs):<6} "
            f"(共 {sum(counts.values())} 个)")
    return split, counts


# ─────────────────────── 人工抽检图 ───────────────────────
_PREVIEW_COLORS = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255),
                   (255, 255, 0), (0, 255, 255)]


def make_previews(project_dir, n=24, seed=42, log=print):
    """从已生成的工程里随机抽 n 个切片，输出"原图 + 掩膜叠加"图，供人工核对标注是否解析正确。

    可以独立调用（导入已完成也能补生成）。早期版本的掩膜文件名是 {base}_mask.png，
    后来统一成 {base}_te_0_mask.png —— 两种都兼容，避免再出现找不到文件却静默跳过。
    """
    project_dir = Path(project_dir)
    img_dir, ann_dir, prev_dir = project_dir / "images", project_dir / "annotations", project_dir / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)
    ti = project_dir / "tile_index.json"
    bases = []
    if ti.exists():
        with open(ti, "r", encoding="utf-8") as f:
            bases = [t["base"] for t in json.load(f) if not t.get("labels") or True]
    if not bases:
        bases = sorted({p.name.split("_te_0")[0] for p in img_dir.glob("*_te_0.jpg")})
    try:
        with open(project_dir / "project.json", "r", encoding="utf-8") as f:
            classes = json.load(f).get("classes", DEFAULT_CLASSES)
    except Exception:
        classes = DEFAULT_CLASSES
    nc = len(classes)
    rng = random.Random(seed)
    rng.shuffle(bases)
    made, failed = 0, []
    for i, base in enumerate(bases[:n]):
        try:
            im = np.asarray(Image.open(img_dir / f"{base}_te_0.jpg").convert("RGB"))
            mp = ann_dir / f"{base}_te_0_mask.png"
            if not mp.exists():
                mp = ann_dir / f"{base}_mask.png"
            mm = np.asarray(Image.open(mp))
            ov = im.copy()
            for c in range(1, nc):
                sel = mm == c
                if sel.any():
                    ov[sel] = np.array(_PREVIEW_COLORS[(c - 1) % len(_PREVIEW_COLORS)], np.uint8)
            ign = mm == IGNORE
            if ign.any():
                ov[ign] = (ov[ign] * 0.4 + np.array([80, 80, 80]) * 0.6).astype(np.uint8)
            blend = (im * 0.45 + ov * 0.55).astype(np.uint8)
            cv2.putText(blend, base, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            Image.fromarray(blend).save(prev_dir / f"qc_{i:02d}_{base}.jpg", quality=90)
            made += 1
        except Exception as e:
            failed.append(f"{base}: {type(e).__name__}: {e}")
    if failed:
        log(f"  [警告] {len(failed)} 张抽检图生成失败，前 3 条：{failed[:3]}")
    return made


# ─────────────────────── 主流程 ───────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="导出根目录（含 原图_* 与 Label_*）")
    ap.add_argument("--dst", required=True, help="输出工程目录")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--margin", type=int, default=96, help="缺陷框外的上下文像素")
    ap.add_argument("--neg-ratio", type=float, default=1.5, help="负样本 tile 数 / 正样本 tile 数")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 组（0=全部）")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview", type=int, default=12, help="输出多少张人工抽检图")
    ap.add_argument("--preview-only", action="store_true",
                    help="只补生成抽检图（用于导入已完成的工程）")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    t0 = time.time()
    src, dst = Path(args.src), Path(args.dst)
    if args.preview_only:
        n = make_previews(dst, n=args.preview, seed=args.seed)
        print(f"已补生成 {n} 张抽检图 → {dst / 'preview'}")
        return
    img_dir = dst / "images"; ann_dir = dst / "annotations"; prev_dir = dst / "preview"
    for d in (img_dir, ann_dir, prev_dir):
        d.mkdir(parents=True, exist_ok=True)

    imgs, jsons, complete = build_index(*scan_source(src))
    gkeys = sorted(complete.keys())
    rng.shuffle(gkeys)
    if args.limit:
        gkeys = gkeys[: args.limit]

    classes = list(DEFAULT_CLASSES)
    nc = len(classes)
    group_split, defect_counts = plan_split(gkeys, jsons, complete, args.val_ratio, args.seed)

    split_map, tile_index = {}, []
    stat = collections.Counter()
    cls_px = collections.Counter()
    cls_inst = collections.Counter()
    gid_used = 0
    preview_pool = []

    for gi, g in enumerate(gkeys):
        view_stems = complete[g]
        jp = jsons.get(view_stems[0])
        try:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            stat["bad_json"] += 1
            continue
        shapes = meta.get("shapes", [])
        H, W = int(meta.get("imageHeight", 0)), int(meta.get("imageWidth", 0))
        if H < args.tile or W < args.tile:
            stat["too_small"] += 1
            continue

        # 读三视图（te_0 为基准尺寸）
        arrs = []
        ok = True
        for v in range(3):
            stem = view_stems[v]
            p = imgs[stem][1]
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                ok = False
                break
            if (im.height, im.width) != (H, W):
                im = im.resize((W, H), Image.BILINEAR)
            arrs.append(np.asarray(im, dtype=np.uint8))
        if not ok or len(arrs) != 3:
            stat["bad_image"] += 1
            continue

        mask = rasterize(shapes, classes, (H, W))
        boxes = shape_boxes(shapes, classes)
        for lab, *_ in boxes:
            cls_inst[lab] += 1
        for c in range(1, nc):
            cls_px[classes[c]] += int((mask == c).sum())
        has_def = len(boxes) > 0
        stat["groups_with_defect" if has_def else "groups_empty"] += 1

        tiles = plan_tiles(boxes, (H, W), args.tile, args.margin, args.neg_ratio, rng)
        if not tiles:
            stat["no_tile"] += 1
            continue

        gid = f"g{gid_used:04d}"
        gid_used += 1
        split = group_split.get(g, "train")
        for k, (wx0, wy0, labs) in enumerate(tiles):
            x1, y1 = wx0 + args.tile, wy0 + args.tile
            px0, py0 = max(0, -wx0), max(0, -wy0)
            ex1, ey1 = min(W, x1), min(H, y1)
            tw, th = ex1 - wx0, ey1 - wy0
            if tw <= 0 or th <= 0:
                continue
            base = f"{gid}_t{k}"
            for vi in range(3):
                crop = arrs[vi][wy0:ey1, wx0:ex1]
                if crop.shape[0] != args.tile or crop.shape[1] != args.tile:
                    crop = cv2.copyMakeBorder(crop, py0, args.tile - py0 - crop.shape[0],
                                              px0, args.tile - px0 - crop.shape[1],
                                              cv2.BORDER_REPLICATE)
                Image.fromarray(crop).save(img_dir / f"{base}_te_{vi}.jpg",
                                           quality=args.jpeg_quality, subsampling=0)
            mc = mask[wy0:ey1, wx0:ex1]
            if mc.shape[0] != args.tile or mc.shape[1] != args.tile:
                mc = cv2.copyMakeBorder(mc, py0, args.tile - py0 - mc.shape[0],
                                        px0, args.tile - px0 - mc.shape[1],
                                        cv2.BORDER_CONSTANT, value=IGNORE)
            Image.fromarray(mc, mode="L").save(ann_dir / f"{base}_te_0_mask.png")
            split_map[f"{base}_te_0"] = split
            tile_index.append({"base": base, "gid": gid, "group": g, "tile": k,
                               "rect": [int(wx0), int(wy0), int(x1), int(y1)],
                               "labels": labs, "split": split,
                               "src": {VIEWS[vi2]: str(imgs[view_stems[vi2]][1]) for vi2 in range(3)}})
            stat["tiles_pos" if labs else "tiles_neg"] += 1
            if len(preview_pool) < 400:
                preview_pool.append(base)

        if (gi + 1) % 25 == 0 or gi + 1 == len(gkeys):
            el = time.time() - t0
            print(f"  [{gi+1}/{len(gkeys)}] tiles={stat['tiles_pos']+stat['tiles_neg']} "
                  f"用时 {el:.0f}s", flush=True)

    # ── 抽检图（原图 + 掩膜半透明叠加）──
    n = make_previews(dst, n=args.preview, seed=args.seed)
    print(f"\n人工抽检图 {n} 张 → {prev_dir}")

    # ── 工程元数据 ──
    proj = {"name": dst.name, "classes": classes,
            "class_cn": {c: CLASS_CN_GUESS.get(c, c) for c in classes},
            "ignore_index": IGNORE, "views": list(VIEWS), "view_names": VIEW_CN,
            "in_channels": len(VIEWS) * 3, "tile_size": args.tile,
            "source": str(src), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "由 tools/import_farasis_labelme.py 从 LabelMe 三视图导出生成；"
                    "掩膜 255 = __ignore__（不确定区）"}
    with open(dst / "project.json", "w", encoding="utf-8") as f:
        json.dump(proj, f, ensure_ascii=False, indent=2)
    with open(dst / "train_test_split.json", "w", encoding="utf-8") as f:
        json.dump(split_map, f, ensure_ascii=False, indent=1)
    with open(dst / "tile_index.json", "w", encoding="utf-8") as f:
        json.dump(tile_index, f, ensure_ascii=False, indent=1)

    report = {
        "source": str(src), "dest": str(dst), "elapsed_sec": round(time.time() - t0, 1),
        "groups_scanned": len(complete), "groups_used": gid_used,
        "tiles_total": stat["tiles_pos"] + stat["tiles_neg"],
        "tiles_pos": stat["tiles_pos"], "tiles_neg": stat["tiles_neg"],
        "train": sum(1 for v in split_map.values() if v == "train"),
        "val": sum(1 for v in split_map.values() if v == "val"),
        "instances_per_class": dict(cls_inst),
        "pixels_per_class": dict(cls_px),
        "skipped": {k: v for k, v in stat.items() if k.startswith(("bad_", "too_", "no_"))},
        "groups_with_defect": stat["groups_with_defect"],
        "groups_empty": stat["groups_empty"],
        "tile": args.tile, "margin": args.margin, "neg_ratio": args.neg_ratio,
        "split_defects": {sp: {"groups": sum(1 for g in gkeys if group_split.get(g) == sp),
                               "instances": sum(defect_counts[g] for g in gkeys
                                                if group_split.get(g) == sp)}
                          for sp in ("train", "val")},
        "max_instances_per_group": max(defect_counts.values()) if defect_counts else 0,
    }
    with open(dst / "import_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== 导入完成 ===")
    for k in ("groups_used", "tiles_pos", "tiles_neg", "train", "val", "elapsed_sec"):
        print(f"  {k}: {report[k]}")
    print(f"  类别实例: {report['instances_per_class']}")
    print(f"  工程目录: {dst}")


if __name__ == "__main__":
    main()
