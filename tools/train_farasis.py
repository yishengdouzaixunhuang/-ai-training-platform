# -*- coding: utf-8 -*-
"""客户三图语义分割训练入口

面向 233 训练集这类"极小缺陷"数据集的默认策略：
  * loss = ce_dice（Dice 只对前景类平均，避免被背景淹没）
  * class_weighted = sqrt（逆平方根，比逆频率稳）
  * ignore_index = 255（__ignore__ 不确定区不参与梯度与指标）
  * 指标按类报 IoU / 召回(1-漏检) / 查准(1-过杀)

用法：
  python -m tools.train_farasis --project <工程目录> [--model deeplabv3|fcn|lraspp]
                               [--epochs 30] [--batch 8] [--size 512] [--lr 3e-4]
                               [--name mv233_deeplabv3]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[1]))

import torch  # noqa: E402  必须早于其它重型库


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--model", default="deeplabv3", choices=["deeplabv3", "fcn", "lraspp", "unet"])
    ap.add_argument("--base", type=int, default=16, help="U-Net 首层通道数")
    ap.add_argument("--depth", type=int, default=4, help="U-Net 下采样层数")
    ap.add_argument("--jitter", type=int, default=-1, help="增强抖动像素；-1=自动(窗口/8)，0=关闭")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=0, help="0 = 自动取 project.json 的 tile_size")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", default="ce_dice", choices=["ce", "dice", "ce_dice"])
    ap.add_argument("--class-weighted", default="sqrt", choices=["none", "inv", "sqrt"])
    ap.add_argument("--ignore-index", type=int, default=255)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="CPU 线程数；0=自动(实测 6 最快)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--resume", default="", help="从已有 .pth 继续训练（可写文件名或全路径）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    proj = Path(args.project)
    if not (proj / "project.json").exists():
        print(f"[错误] {proj} 不是有效工程（缺 project.json）")
        return 2

    with open(proj / "project.json", "r", encoding="utf-8") as f:
        pj = json.load(f)
    if not args.size:
        args.size = int(pj.get("tile_size", 512))
    name = args.name or f"mv_{proj.name}_{args.model}"
    save_name = f"{name}.pth"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"工程 {proj.name}  类别 {pj.get('classes')}  tile {args.size}")
    print(f"设备 {dev}" + (f"  {torch.cuda.get_device_name(0)}" if dev == "cuda" else "  (无 GPU，极慢)"))
    print(f"保存为 models/{save_name}")

    from training.multiview import train_multiview
    resume = args.resume
    if resume and not Path(resume).exists():
        cand = proj / "models" / resume
        resume = str(cand) if cand.exists() else ""
    t0 = time.time()
    metrics = train_multiview(
        proj, model_name=args.model, epochs=args.epochs, batch_size=args.batch,
        image_size=args.size, lr=args.lr, weight_decay=args.weight_decay,
        num_workers=args.workers, device=dev, augment=not args.no_augment,
        class_weighted=(False if args.class_weighted == "none" else args.class_weighted),
        save_name=save_name, pretrained=not args.no_pretrained,
        ignore_index=args.ignore_index, loss=args.loss,
        jitter=(None if args.jitter < 0 else args.jitter),
        unet_base=args.base, unet_depth=args.depth, resume=(resume or None))
    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    metrics["model"] = args.model
    metrics["name"] = name
    metrics["batch"] = args.batch
    metrics["lr"] = args.lr
    metrics["epochs"] = args.epochs
    metrics["device"] = dev
    metrics["project"] = str(proj)

    out = proj / "models" / f"{name}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cls = metrics["ious"].keys()
    lines = [f"# 三图分割训练报告 — {name}", "",
             f"- 工程：`{proj}`", f"- 模型：{args.model}  设备：{dev}  batch={args.batch} lr={args.lr}",
             f"- 训练轮数：{args.epochs}  最优轮：{metrics['best_epoch']}  用时：{metrics['elapsed_sec']}s",
             f"- loss：{args.loss}  类别权重：{args.class_weighted}  忽略值：{args.ignore_index}", "",
             f"**mIoU={metrics['best_mIoU']:.4f}  前景mIoU={metrics['best_mIoU_fg']:.4f}**", "",
             "| 类别 | IoU | 召回(1-漏检) | 查准(1-过杀) |", "|---|---|---|---|"]
    for c in cls:
        lines.append(f"| {c} | {metrics['ious'][c]:.4f} | {metrics['recall'][c]:.4f} | {metrics['precision'][c]:.4f} |")
    lines += ["", "## 逐轮", "", "| epoch | loss | mIoU | 前景mIoU |", "|---|---|---|---|"]
    for h in metrics["history"]:
        lines.append(f"| {h['epoch']} | {h['loss']} | {h['mIoU']} | {h['mIoU_fg']} |")
    rp = proj / "models" / f"{name}_report.md"
    rp.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {rp}")
    print(f"指标: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
