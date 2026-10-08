# -*- coding: utf-8 -*-
"""并排比较多个模型的评估结果（读取 tools/eval_farasis.py 产出的 *_eval.json）

用法：
  python -m tools.compare_models --project <工程目录> [--names a,b] [--prob 0.5] [--ma 0]
                                  [--out 对比报告.md]
不指定 --names 时自动读取 models/ 下所有 *_eval.json。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[1]))


def _pct(v):
    return "—" if v is None else f"{v * 100:.2f}%"


def load_all(proj: Path):
    out = {}
    for p in sorted((proj / "models").glob("*_eval.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = p.name[: -len("_eval.json")]
        out[name] = d
    return out


def pick(sweep, prob, ma):
    best = None
    for r in sweep:
        if abs(r["prob_thr"] - prob) < 1e-6 and r["min_area"] == ma:
            return r
        if best is None or (abs(r["prob_thr"] - prob) + abs(r["min_area"] - ma) <
                            abs(best["prob_thr"] - prob) + abs(best["min_area"] - ma)):
            best = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--names", default="")
    ap.add_argument("--prob", type=float, default=0.5)
    ap.add_argument("--ma", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    proj = Path(args.project)
    runs = load_all(proj)
    if not runs:
        print(f"[错误] {proj / 'models'} 下没有 *_eval.json（先跑 tools.eval_farasis）")
        return 2
    if args.names:
        want = [x.strip() for x in args.names.split(",") if x.strip()]
        runs = {k: v for k, v in runs.items() if k in want}

    L = [f"# 模型对比 — {proj.name}", "",
         f"取 概率阈值={args.prob}，最小面积={args.ma}", "",
         "| 模型 | 切图 | TTA | GT实例 | **漏检率** | 预测实例 | **过杀率** | 空tile误报 | 无缺陷tile |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for name, d in runs.items():
        r = pick(d.get("sweep", []), args.prob, args.ma)
        if not r:
            continue
        a = r["all"]
        rows.append((name, d, r, a))
        L.append(f"| {name} | {d.get('split', '')} | {'开' if d.get('tta') else '关'} | "
                 f"{a['gt_inst']} | **{_pct(a['miss_rate'])}** | {a['pred_inst']} | "
                 f"**{_pct(a['overkill_rate'])}** | {a['tile_fp']} | {d.get('empty_tiles', '')} |")

    # 分类别
    L += ["", "## 分类别漏检率", "",
          "| 模型 | " + " | ".join(list(rows[0][2]["classes"].keys()) if rows else []) + " |",
          "|---|" + "---|" * (len(rows[0][2]["classes"]) if rows else 0)]
    for name, d, r, a in rows:
        L.append(f"| {name} | " + " | ".join(
            _pct(v["miss_rate"]) for v in r["classes"].values()) + " |")

    # 尺寸分层
    if rows and "by_size" in rows[0][2]:
        keys = list(rows[0][2]["by_size"].keys())
        L += ["", "## 漏检率 × 缺陷尺寸", "",
              "| 模型 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for name, d, r, a in rows:
            bs = r.get("by_size", {})
            L.append(f"| {name} | " + " | ".join(
                _pct(bs.get(k, {}).get("miss_rate")) for k in keys) + " |")

    txt = "\n".join(L)
    print(txt)
    out = Path(args.out) if args.out else (proj / "models" / "compare_report.md")
    out.write_text(txt, encoding="utf-8")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
