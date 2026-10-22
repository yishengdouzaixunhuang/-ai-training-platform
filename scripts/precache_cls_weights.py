# -*- coding: utf-8 -*-
"""把分类可选模型的 torchvision 预训练权重预先下到本地缓存。

背景（踩过的坑）：
  DL Studio 的 `training/models.py` 里有一行
      os.environ.setdefault("TORCH_HOME", ~/Documents/Codex/.torch_cache)
  ⇒ **GUI 里训练**用的权重缓存目录是 `~/Documents/Codex/.torch_cache/hub/checkpoints`，
    而命令行脚本默认用 `~/.cache/torch/hub/checkpoints`，两者不是同一个目录。
  现场网络受限时，训练一开始现下 45MB+ 权重会卡住甚至直接失败。

用法：
    C:\\aitp_venv\\Scripts\\python.exe scripts/precache_cls_weights.py
    ... --models resnet18,resnet34,efficientnet_b0
    ... --all                     # 补齐 CLS_MODELS 里全部（体积很大）
    ... --list                    # 只看状态，不下载
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402  必须早于 PyQt5，且必须先于 training.models 的 TORCH_HOME 设置生效

# 触发与 GUI 相同的 TORCH_HOME（training/models.py 用 setdefault，不会覆盖已有值）
try:
    import training.models  # noqa: F401
except Exception:
    pass

from classification.trainer import (  # noqa: E402
    CLS_MODELS, _create_classification_model, pretrained_weight_status,
)

DEFAULT_SET = ["resnet18", "resnet34", "resnet50",
               "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_SET),
                    help="逗号分隔的模型名（默认一组常用小模型）")
    ap.add_argument("--all", action="store_true", help="CLS_MODELS 全部（体积很大）")
    ap.add_argument("--list", action="store_true", help="只报告缓存状态，不下载")
    args = ap.parse_args()

    names = list(CLS_MODELS.keys()) if args.all else [m.strip() for m in args.models.split(",") if m.strip()]
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints")
    print(f"权重缓存目录（GUI 同款）: {ckpt}")
    print(f"待处理模型: {len(names)} 个\n")

    miss = []
    for n in names:
        cached, f = pretrained_weight_status(n)
        if cached:
            print(f"  [已缓存] {n:22s} {f}")
        else:
            print(f"  [缺失  ] {n:22s}")
            miss.append(n)

    if args.list or not miss:
        print("\n无缺失项就绪。" if not miss else "\n仅列出状态，未下载。")
        return 0

    print(f"\n开始下载 {len(miss)} 个模型（联网慢时请耐心，已缓存的会跳过）...")
    ok, bad = [], []
    for n in miss:
        try:
            # 走产品自己的建模函数，保证与训练时完全一致的权重口径
            _create_classification_model(n, 2, pretrained=True)
            cached, f = pretrained_weight_status(n)
            print(f"  [OK] {n:22s} {f}")
            ok.append(n)
        except Exception as e:
            print(f"  [FAIL] {n:22s} {type(e).__name__}: {e}")
            bad.append(n)

    print(f"\n完成：成功 {len(ok)}，失败 {len(bad)}")
    if bad:
        print("失败项：" + ", ".join(bad))
        print("现场网络受限时可以：① 在能上网的机器上跑本脚本，把 checkpoints 目录整体拷过来；")
        print("                ② 或把该模型的「预训练」勾选框取消（从零训练，精度会明显下降）。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
