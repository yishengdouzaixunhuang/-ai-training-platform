# -*- coding: utf-8 -*-
"""整图滑窗推理 → 真正的产线指标（按电芯统计漏检率 / 过杀率）

与 eval_farasis.py 的区别（重要）：
  eval_farasis 在"已切好的 tile"上评估，负样本是人为采样的 →
      它给出的"过杀率"只代表 tile 级误报，不代表整颗电芯的过杀
  本工具在**整张原图**上滑窗拼接，逐个缺陷实例与电芯判定做匹配 →
      漏检率 / 过杀率可直接对接产线口径（漏检 ≤0.2%；过杀＝无特征却报了，**目标 0**）

匹配口径：
  * 实例级：预测连通域与标注连通域有像素交集即算检出（同类别）
  * 电芯级：该电芯只要有一个未检出的标注缺陷 → 漏检；只要报了任何缺陷 → 判 NG
            电芯级过杀 = 判 NG 但实际无缺陷的电芯占比

用法：
  python -m tools.infer_cell_farasis --project <工程> --ckpt <模型.pth>
        --src <导出根> [--split val] [--limit 0] [--window 512] [--stride 384]
        [--min-areas 0,16,64,256] [--probs 0.5] [--out <报告前缀>]
"""
from __future__ import annotations
import argparse, json, sys, time, collections
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[1]))

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

from tools.import_farasis_labelme import rasterize, shape_boxes, IGNORE, VIEWS
from tools.eval_farasis import _cc, _match_counts, _pct, forward_probs


def load_model(proj: Path, ckpt: str):
    from training.multiview import create_multiview_model
    ck = torch.load(proj / "models" / ckpt, map_location="cpu", weights_only=False)
    m = create_multiview_model(ck.get("model_name", "deeplabv3"), ck["num_classes"],
                               tuple(ck["views"]), pretrained=False)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev), ck, dev


def build_json_index(src: Path):
    idx = {}
    for d in src.iterdir():
        if d.is_dir() and d.name.lower().startswith("label"):
            for tr in (d / "Train", d):
                if tr.exists():
                    for p in tr.rglob("*.json"):
                        idx[p.stem] = p
    return idx


@torch.inference_mode()
def infer_one(model, arrs, ck, dev, window, stride, batch=8, tta=False, median=0):
    """三视图整图 → 全分辨率概率图 [C,H,W] float32"""
    nc = ck["num_classes"]
    H, W = arrs[0].shape[:2]
    prob = np.zeros((nc, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    xs = list(range(0, max(W - window, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - window, 0) + 1, stride)) or [0]
    if xs[-1] != W - window:
        xs.append(max(W - window, 0))
    if ys[-1] != H - window:
        ys.append(max(H - window, 0))
    wins = [(x, y) for y in ys for x in xs]
    mean = torch.tensor([0.485, 0.456, 0.406] * 3).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225] * 3).view(-1, 1, 1)
    for i in range(0, len(wins), batch):
        blk = wins[i:i + batch]
        xs_ = []
        for (x, y) in blk:
            ch = []
            for vi in range(3):
                c = arrs[vi][y:y + window, x:x + window]
                if c.shape[0] != window or c.shape[1] != window:
                    c = cv2.copyMakeBorder(c, 0, window - c.shape[0], 0, window - c.shape[1],
                                           cv2.BORDER_REPLICATE)
                ch.append(c.astype(np.float32) / 255.0)
            xs_.append(np.concatenate(ch, axis=2))
        xb = torch.from_numpy(np.stack(xs_)).permute(0, 3, 1, 2).to(dev)
        xb = (xb - mean.to(dev)) / std.to(dev)
        p = forward_probs(model, xb, dev, tta=tta, size=(window, window), median=median)
        for bi, (x, y) in enumerate(blk):
            prob[:, y:y + window, x:x + window] += p[bi]
            cnt[y:y + window, x:x + window] += 1.0
    return prob / np.maximum(cnt, 1e-6), wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True, help="原始导出根（用于取 LabelMe 标注做 GT）")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", type=int, default=0, help="0 = 自动取模型训练尺寸")
    ap.add_argument("--stride", type=int, default=0, help="0 = 窗口的 75%%")
    ap.add_argument("--min-areas", default="0,16,64,256,1024")
    ap.add_argument("--probs", default="0.5")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tta", action="store_true", help="测试时增强（水平翻转平均，价翻倍但更稳）")
    ap.add_argument("--median", type=int, default=0, help="概率图中值滤波核（0=关，建议 3）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    proj = Path(args.project)
    src = Path(args.src)
    model, ck, dev = load_model(proj, args.ckpt)
    classes = ck["classes"]; nc = ck["num_classes"]
    if not args.window:
        args.window = int((ck.get("image_size") or [512, 512])[0])
    if not args.stride:
        args.stride = max(32, int(args.window * 0.75))
    print(f"模型 {ck.get('model_name')}  类数 {nc}  设备 {dev}  窗口 {args.window} 步长 {args.stride}")

    with open(proj / "tile_index.json", "r", encoding="utf-8") as f:
        tiles = json.load(f)
    with open(proj / "train_test_split.json", "r", encoding="utf-8") as f:
        split_map = json.load(f)

    # 按 gid 归并 → 每个原始组一份（三视图原图路径）
    groups = {}
    for t in tiles:
        gid = t["gid"]
        if gid in groups:
            continue
        if args.split != "all":
            # ★ 2026-10-22：未划分不再默认算 train（与训练侧同一口径），
            #   没登记的组不属于任何集 —— 用它做评估等于拿"没核对过的图"算指标。
            s = split_map.get(f"{t['base']}_te_0", split_map.get(t["base"]))
            if s != args.split:
                continue
        groups[gid] = t["src"]
    gids = sorted(groups)
    if args.limit:
        gids = gids[:args.limit]
    jidx = build_json_index(src)
    print(f"评估 {len(gids)} 组（整图滑窗）")

    min_areas = [int(x) for x in args.min_areas.split(",")]
    probs_thr = [float(x) for x in args.probs.split(",")]
    inst = {(p, ma): {c: dict(gt=0, hit=0, pred=0, phit=0) for c in range(1, nc)}
            for p in probs_thr for ma in min_areas}
    cell = {(p, ma): dict(ng_gt=0, ng_hit=0, ok_gt=0, ok_flag=0, miss_cell=0, over_cell=0)
            for p in probs_thr for ma in min_areas}
    t0 = time.time()
    used = 0

    for gi, gid in enumerate(gids):
        srcs = groups[gid]
        stems = [Path(srcs[v]).stem for v in VIEWS]
        jp = None
        for st in stems:
            if st in jidx:
                jp = jidx[st]
                break
        if jp is None:
            continue
        try:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        H, W = int(meta["imageHeight"]), int(meta["imageWidth"])
        arrs = []
        ok = True
        for v in VIEWS:
            try:
                im = Image.open(srcs[v]).convert("RGB")
            except Exception:
                ok = False; break
            if (im.height, im.width) != (H, W):
                im = im.resize((W, H), Image.BILINEAR)
            arrs.append(np.asarray(im, dtype=np.uint8))
        if not ok:
            continue
        used += 1

        gt_mask = rasterize(meta.get("shapes", []), classes, (H, W))
        per_class_gt = {}
        for c in range(1, nc):
            lab, ids, _ = _cc(gt_mask == c)
            per_class_gt[c] = (lab, ids)
        n_gt_defect = sum(len(v[1]) for v in per_class_gt.values())
        n_gt_ign = int((gt_mask == IGNORE).sum())

        prob, wins = infer_one(model, arrs, ck, dev, args.window, args.stride, args.batch,
                               args.tta, args.median)
        am = prob.argmax(0)

        for p in probs_thr:
            for ma in min_areas:
                any_pred = False
                for c in range(1, nc):
                    pb = (am == c) & (prob[c] >= p) & (gt_mask != IGNORE)
                    if pb.any():
                        p_lab, p_ids, p_areas = _cc(pb)
                        keep = [i for i, a in zip(p_ids, p_areas) if a >= ma]
                    else:
                        p_lab, p_ids, keep = None, [], []
                    if keep:
                        any_pred = True
                    g_lab, g_ids = per_class_gt[c]
                    st = inst[(p, ma)][c]
                    st["gt"] += len(g_ids); st["pred"] += len(keep)
                    if keep and g_ids:
                        mg, mp = _match_counts(g_lab, g_ids, p_lab, keep)
                        st["hit"] += mg; st["phit"] += mp
                cs = cell[(p, ma)]
                if n_gt_defect > 0:
                    cs["ng_gt"] += 1
                    if any_pred:
                        cs["ng_hit"] += 1
                    else:
                        cs["miss_cell"] += 1
                else:
                    cs["ok_gt"] += 1
                    if any_pred:
                        cs["ok_flag"] += 1      # 过杀（OK 电芯被判 NG）
                        cs["over_cell"] += 1
        if (gi + 1) % 5 == 0 or gi + 1 == len(gids):
            el = time.time() - t0
            print(f"  [{gi+1}/{len(gids)}] {el:.0f}s  {el/max(gi+1,1):.1f}s/组", flush=True)

    sweep = []
    for p in probs_thr:
        for ma in min_areas:
            tg = th = tp = tph = 0
            per = {}
            for c in range(1, nc):
                st = inst[(p, ma)][c]
                per[classes[c]] = {
                    "gt_inst": st["gt"], "gt_hit": st["hit"],
                    "miss_rate": round(1 - st["hit"] / st["gt"], 5) if st["gt"] else None,
                    "pred_inst": st["pred"], "pred_hit": st["phit"],
                    "overkill_rate": round(1 - st["phit"] / st["pred"], 5) if st["pred"] else None}
                tg += st["gt"]; th += st["hit"]; tp += st["pred"]; tph += st["phit"]
            cs = cell[(p, ma)]
            sweep.append({
                "prob_thr": p, "min_area": ma,
                "instance": {"gt": tg, "hit": th, "miss_rate": round(1 - th / tg, 5) if tg else None,
                             "pred": tp, "phit": tph,
                             "overkill_rate": round(1 - tph / tp, 5) if tp else None},
                "cell": {"ng_gt": cs["ng_gt"], "ng_hit": cs["ng_hit"], "ok_gt": cs["ok_gt"],
                         "ok_flag": cs["ok_flag"],
                         "cell_miss_rate": round(cs["miss_cell"] / cs["ng_gt"], 5) if cs["ng_gt"] else None,
                         "cell_overkill_rate": round(cs["ok_flag"] / cs["ok_gt"], 5) if cs["ok_gt"] else None},
                "classes": per})

    name = args.out or f"{Path(args.ckpt).stem}_cell"
    rep = {"ckpt": args.ckpt, "project": str(proj), "groups_evaluated": used,
           "split": args.split, "window": args.window, "stride": args.stride,
           "elapsed_sec": round(time.time() - t0, 1), "sweep": sweep}
    with open(proj / "models" / f"{name}_cell.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    L = [f"# 整图评估报告（产线口径）— {Path(args.ckpt).stem}", "",
         f"- 工程：`{proj}`  切分：{args.split}  评估电芯组数：**{used}**",
         f"- 滑窗：{args.window}  步长：{args.stride}  用时 {rep['elapsed_sec']}s", "",
         "## 电芯级（漏检 ≤0.2%；过杀=无特征却报了，目标 0）", "",
         "| 概率阈值 | 最小面积 | NG电芯 | 报出NG | **电芯漏检率** | OK电芯 | 误报NG | **电芯过杀率** |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sweep:
        c = r["cell"]
        L.append(f"| {r['prob_thr']} | {r['min_area']} | {c['ng_gt']} | {c['ng_hit']} | "
                 f"**{_pct(c['cell_miss_rate'])}** | {c['ok_gt']} | {c['ok_flag']} | "
                 f"**{_pct(c['cell_overkill_rate'])}** |")
    L += ["", "## 缺陷实例级", "",
          "| 概率阈值 | 最小面积 | GT实例 | 漏检率 | 预测实例 | 过杀率 |", "|---|---|---|---|---|---|"]
    for r in sweep:
        i = r["instance"]
        L.append(f"| {r['prob_thr']} | {r['min_area']} | {i['gt']} | {_pct(i['miss_rate'])} | "
                 f"{i['pred']} | {_pct(i['overkill_rate'])} |")
    L += ["", "## 分类别（实例级，最小面积=0）", "",
          "| 类别 | GT实例 | 漏检率 | 预测实例 | 过杀率 |", "|---|---|---|---|---|"]
    for r in sweep:
        if r["min_area"] != 0:
            continue
        for cn, v in r["classes"].items():
            L.append(f"| {cn} | {v['gt_inst']} | {_pct(v['miss_rate'])} | "
                     f"{v['pred_inst']} | {_pct(v['overkill_rate'])} |")
    rp = proj / "models" / f"{name}_cell_report.md"
    rp.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n报告: {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
