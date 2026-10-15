# -*- coding: utf-8 -*-
"""客户三图分割评估：把像素 IoU 翻译成业务语言（漏检率 / 过杀率）

为什么需要它：像素 IoU 无法回答"这条线能不能用"。业务上真正关心的是
    漏检率 = 应为 NG 却没报出来的缺陷个数占比（目标 ≤0.2%）
    过杀率 = 报出来但实际不是缺陷的个数占比（**目标 0**）

⚠️ 这里曾经写过"过杀率 目标 ≤20~30%"，是**错的**（用户明确纠正）：
    判定对错的唯一依据是"图上有没有这个特征" —— 有特征必须报（漏检 ≤0.2%），
    没特征绝不能报（过杀=报错，不存在"允许 20~30% 余量"）。
    只有"机器初筛 + 人工复判"的降级模式才可能容忍过杀，那种模式必须单独写明数值。
    所以选阈值时**不能**只盯漏检：同一工作点下必须同时看
    ① 缺陷级/电芯级漏检率 ② **无缺陷样本上的误报数**。

做法：
  1. 模型输出类别图 + 类别概率
  2. 按类别做连通域，丢弃面积 < min_area 的碎点（等价于厂商的"面积公差"）
  3. 与标注连通域匹配：有像素交集即算检出
  4. 扫描 min_area × 概率阈值，输出"漏检-过杀"工作曲线供选阈值

用法：
  python -m tools.eval_farasis --project <工程> --ckpt mv_xxx.pth [--split val]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[1]))

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def _cc(binary):
    """连通域：返回 (标签图, 标签id列表, 面积列表)"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    ids = list(range(1, n))
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in ids]
    return lab, ids, areas


def _match_counts(gt_lab, gt_ids, pred_lab, pred_ids):
    """有像素交集的配对 → (被匹配到的GT数, 被匹配到的预测数)"""
    if not gt_ids or not pred_ids:
        return 0, 0
    gsub = np.isin(gt_lab, gt_ids)
    psub = np.isin(pred_lab, pred_ids)
    inter = gsub & psub
    if not inter.any():
        return 0, 0
    pairs = set(zip(gt_lab[inter].tolist(), pred_lab[inter].tolist()))
    return len({a for a, _ in pairs}), len({b for _, b in pairs})


def _match_sets(gt_lab, gt_ids, pred_lab, pred_ids):
    """同 _match_counts，但返回被匹配到的 GT 标签集合（用于按尺寸分层统计）"""
    if not gt_ids or not pred_ids:
        return set(), set()
    inter = np.isin(gt_lab, gt_ids) & np.isin(pred_lab, pred_ids)
    if not inter.any():
        return set(), set()
    pairs = set(zip(gt_lab[inter].tolist(), pred_lab[inter].tolist()))
    return {a for a, _ in pairs}, {b for _, b in pairs}


# 尺寸分层（按外接框最大边，单位=像素，原分辨率）
BUCKETS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 10 ** 9)]


def _bucket_of(side):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= side < hi:
            return i
    return len(BUCKETS) - 1


def _bucket_label(i):
    lo, hi = BUCKETS[i]
    return f"<{hi}px" if lo == 0 else (f">={lo}px" if hi > 10 ** 8 else f"{lo}-{hi}px")


def _bbox_sides(lab, ids):
    """每个连通域外接框的最大边"""
    out = []
    for i in ids:
        ys, xs = np.where(lab == i)
        if len(xs) == 0:
            out.append(0)
            continue
        out.append(int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    return out


def _pct(v):
    return "—" if v is None else f"{v * 100:.2f}%"


def forward_probs(model, xb, dev, tta=False, size=None, median=0):
    """前向 → 概率图 (N,C,H,W) numpy。

    tta=True  水平翻转平均，通常同时小幅改善召回与查准（三视图一起翻，物理上等价）
    median>1  对每类概率图做中值滤波，专治全分辨率模型输出的孤立碎点。
              比形态学开运算安全：7~12px 的小脏污不会被抹掉，只去掉 1~2px 的噪点。
              （cv2.medianBlur 对 float32 只支持 ksize=3/5）
    """
    x = xb.to(dev)

    def _run(inp):
        o = model(inp)["out"]
        if size is not None and o.shape[-2:] != tuple(size):
            o = F.interpolate(o, size=tuple(size), mode="bilinear", align_corners=False)
        return torch.softmax(o.float(), 1)

    p = _run(x)
    if tta:
        p2 = _run(torch.flip(x, dims=[3]))
        p = (p + torch.flip(p2, dims=[3])) * 0.5
    arr = p.cpu().numpy()
    if median and median > 1:
        arr = np.stack([np.stack([cv2.medianBlur(arr[n, c], median)
                                  for c in range(arr.shape[1])])
                        for n in range(arr.shape[0])])
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--size", type=int, default=0, help="0 = 自动取模型 checkpoint 里的尺寸")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-areas", default="0,4,16,64,256,1024")
    ap.add_argument("--probs", default="0.5")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tta", action="store_true", help="测试时增强（水平翻转平均）")
    ap.add_argument("--median", type=int, default=0, help="概率图中值滤波核（0=关，建议 3）")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    proj = Path(args.project)
    ck = torch.load(proj / "models" / args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]; nc = ck["num_classes"]; views = tuple(ck["views"])
    ignore_index = ck.get("ignore_index", 255)
    model_name = ck.get("model_name", "deeplabv3")
    if not args.size:
        isz = ck.get("image_size") or [512, 512]
        args.size = int(isz[0])

    from training.multiview import MultiViewSegDataset, create_multiview_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_multiview_model(model_name, nc, views, pretrained=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(dev)

    ds = MultiViewSegDataset(proj, args.split, args.size, views, augment=False)
    if len(ds) == 0:
        print("[错误] 评估集为空")
        return 2
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False,
                                         num_workers=args.workers)
    print(f"评估 {len(ds)} 个 tile  模型 {model_name}  设备 {dev}")

    min_areas = [int(x) for x in args.min_areas.split(",")]
    probs_thr = [float(x) for x in args.probs.split(",")]
    keys = [(p, ma) for p in probs_thr for ma in min_areas]
    # 统计: key -> per-class dict / tile 级误报
    S = {k: {c: dict(gt=0, hit=0, pred=0, phit=0,
                     gtb=[0] * len(BUCKETS), hitb=[0] * len(BUCKETS))
             for c in range(1, nc)} for k in keys}
    tile_fp = {k: 0 for k in keys}
    tiles = 0; empty_tiles = 0; ignore_px = 0
    t0 = time.time()

    with torch.inference_mode():
        for xb, yb in loader:
            pr = forward_probs(model, xb, dev, tta=args.tta, median=args.median)
            if pr.shape[-2:] != tuple(yb.shape[-2:]):
                # 正常不会走到（数据集已把图与掩膜统一到 image_size，模型也按输入尺寸输出）
                raise RuntimeError(f"输出尺寸 {pr.shape[-2:]} 与掩膜 {tuple(yb.shape[-2:])} 不一致")
            gt = yb.numpy()
            for si in range(pr.shape[0]):
                tiles += 1
                g = gt[si]
                v = g != ignore_index
                ignore_px += int((~v).sum())
                am = pr[si].argmax(0)
                gt_empty = not (g > 0).any()
                if gt_empty:
                    empty_tiles += 1
                for p in probs_thr:
                    flag = {}                       # (p,ma) -> 该 tile 是否误报
                    for c in range(1, nc):
                        gt_bin = (g == c)
                        if gt_bin.any():
                            g_lab, g_ids, _ = _cc(gt_bin)
                            g_sides = _bbox_sides(g_lab, g_ids)
                        else:
                            g_lab, g_ids, g_sides = None, [], []
                        pb = (am == c) & (pr[si][c] >= p) & v
                        if pb.any():
                            p_lab, p_ids, p_areas = _cc(pb)
                        else:
                            p_lab, p_ids, p_areas = None, [], []
                        for ma in min_areas:
                            keep = [i for i, a in zip(p_ids, p_areas) if a >= ma]
                            st = S[(p, ma)][c]
                            st["gt"] += len(g_ids)
                            st["pred"] += len(keep)
                            if keep and g_ids:
                                mg, mp = _match_sets(g_lab, g_ids, p_lab, keep)
                                st["hit"] += len(mg)
                                st["phit"] += len(mp)
                                for gid_, side in zip(g_ids, g_sides):
                                    bi = _bucket_of(side)
                                    st["gtb"][bi] += 1
                                    if gid_ in mg:
                                        st["hitb"][bi] += 1
                            elif g_ids:
                                for side in g_sides:
                                    st["gtb"][_bucket_of(side)] += 1
                            if keep and gt_empty:
                                flag[(p, ma)] = True
                    for k, hit in flag.items():
                        if hit:
                            tile_fp[k] += 1

    # ── 汇总 ──
    sweep = []
    for p, ma in keys:
        row = {"prob_thr": p, "min_area": ma, "classes": {}}
        tg = th = tp = tph = 0
        gtb_all = [0] * len(BUCKETS); hitb_all = [0] * len(BUCKETS)
        for c in range(1, nc):
            st = S[(p, ma)][c]
            miss = (1 - st["hit"] / st["gt"]) if st["gt"] else None
            over = (1 - st["phit"] / st["pred"]) if st["pred"] else None
            row["classes"][classes[c]] = {
                "gt_inst": st["gt"], "gt_hit": st["hit"], "miss_rate": None if miss is None else round(miss, 5),
                "pred_inst": st["pred"], "pred_hit": st["phit"],
                "overkill_rate": None if over is None else round(over, 5)}
            tg += st["gt"]; th += st["hit"]; tp += st["pred"]; tph += st["phit"]
            for i in range(len(BUCKETS)):
                gtb_all[i] += st["gtb"][i]; hitb_all[i] += st["hitb"][i]
        row["by_size"] = {_bucket_label(i): {
            "gt": gtb_all[i], "hit": hitb_all[i],
            "miss_rate": round(1 - hitb_all[i] / gtb_all[i], 5) if gtb_all[i] else None}
            for i in range(len(BUCKETS))}
        row["all"] = {"gt_inst": tg, "gt_hit": th,
                      "miss_rate": round(1 - th / tg, 5) if tg else None,
                      "pred_inst": tp, "pred_hit": tph,
                      "overkill_rate": round(1 - tph / tp, 5) if tp else None,
                      "tile_fp": tile_fp[(p, ma)]}
        sweep.append(row)

    name = args.name or Path(args.ckpt).stem
    out = {"ckpt": args.ckpt, "project": str(proj), "split": args.split, "tiles": tiles,
           "empty_tiles": empty_tiles, "ignore_px": ignore_px, "tta": bool(args.tta),
           "elapsed_sec": round(time.time() - t0, 1), "sweep": sweep}
    with open(proj / "models" / f"{name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    L = [f"# 评估报告 — {name}", "",
         f"- 工程：`{proj}`  切分：{args.split}  tile 数：{tiles}（其中无缺陷 tile {empty_tiles}）",
         f"- TTA：{'开' if args.tta else '关'}",
         f"- 忽略像素占比：{ignore_px / max(tiles * args.size * args.size, 1) * 100:.2f}%", "",
         "## 漏检-过杀工作曲线（全类别合计）", "",
         "| 概率阈值 | 最小面积 | GT实例 | 检出 | **漏检率** | 预测实例 | 命中 | **过杀率** | 空tile误报 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sweep:
        a = r["all"]
        L.append(f"| {r['prob_thr']} | {r['min_area']} | {a['gt_inst']} | {a['gt_hit']} | "
                 f"**{_pct(a['miss_rate'])}** | {a['pred_inst']} | {a['pred_hit']} | "
                 f"**{_pct(a['overkill_rate'])}** | {a['tile_fp']} |")
    ref = 0.5 if 0.5 in probs_thr else probs_thr[0]
    L += ["", f"## 分类别（概率阈值={ref}）", "",
          "| 类别 | 最小面积 | GT实例 | 漏检率 | 预测实例 | 过杀率 |", "|---|---|---|---|---|---|"]
    for r in sweep:
        if r["prob_thr"] != ref:
            continue
        for cname, v in r["classes"].items():
            L.append(f"| {cname} | {r['min_area']} | {v['gt_inst']} | {_pct(v['miss_rate'])} | "
                     f"{v['pred_inst']} | {_pct(v['overkill_rate'])} |")
    rp = proj / "models" / f"{name}_eval_report.md"
    L += ["", "## 漏检率 × 缺陷尺寸（关键：小缺陷是不是漏检主因）", "",
          "尺寸 = 标注缺陷外接框最大边（原图像素）", "",
          "| 概率阈值 | 最小面积 | " + " | ".join(_bucket_label(i) for i in range(len(BUCKETS))) + " |",
          "|---|---|" + "---|" * len(BUCKETS)]
    for r in sweep:
        cells = []
        for i in range(len(BUCKETS)):
            b = r["by_size"][_bucket_label(i)]
            cells.append(f"{_pct(b['miss_rate'])}<br><sub>n={b['gt']}</sub>")
        L.append(f"| {r['prob_thr']} | {r['min_area']} | " + " | ".join(cells) + " |")
    rp.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n完整报告: {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
