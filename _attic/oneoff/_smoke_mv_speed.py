"""三图工程训练链路冒烟 + 速度实测（只读工程数据，结果写临时目录）。

用途：回答"这个三图工程现在能不能训、训一轮要多久"。
- 用**产品自己的** MultiViewSegDataset / create_multiview_model，不另写一套
- 开分块（滑窗 256 / 步长 75%）与不开分块（整图缩放）两种模式都测，直接看块数差
- 不写任何文件到工程目录
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.multiview import MultiViewSegDataset, create_multiview_model  # noqa: E402

PROJ = sys.argv[1] if len(sys.argv) > 1 else None
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N_BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 20
VIEWS = ("te_0", "te_1", "te_2")


def build(tile):
    logs = []
    t0 = time.time()
    ds = MultiViewSegDataset(
        PROJ, "train", image_size=256,
        tile_size=tile, tile_mode="slide", tile_stride_pct=75,
        log=logs.append,
    )
    return ds, len(ds.samples), len(ds), time.time() - t0, logs


def main():
    print("=" * 72)
    print("三图工程训练链路冒烟（只读）")
    print("工程:", PROJ)
    print("torch", torch.__version__, "| cuda =", torch.cuda.is_available())
    print("=" * 72)

    for tag, tile in (("整图缩放（未开分块）", 0), ("原分辨率滑窗（开分块 256px/75%）", 256)):
        ds, ngrp, nblk, t_build, logs = build(tile)
        print("\n[%s]" % tag)
        print("  组数=%d  训练块数=%d  构建耗时=%.1fs" % (ngrp, nblk, t_build))
        for m in [x for x in logs if "分块" in str(x) or "多视图样本" in str(x)][-3:]:
            print("   ·", m)

    # ── 用开分块的数据集实测吞吐（CPU / GPU 由环境决定）──
    ds, ngrp, nblk, _, _ = build(256)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_multiview_model("unet", num_classes=6, views=VIEWS,
                                   unet_base=16, unet_depth=4).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss(ignore_index=255)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    model.train()
    done = 0
    t0 = None
    for x, y in dl:
        if t0 is None:
            t0 = time.time()          # 跳过第一批（含首次 cudnn/DLL 预热）
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model(x)["out"]
        loss = crit(out, y)
        loss.backward()
        opt.step()
        done += 1
        if done >= N_BATCH:
            break
    dt = time.time() - t0
    per_bs = dt / max(1, done - 1)

    print("\n" + "-" * 72)
    print("模型: U-Net base=16 depth=4 | 参数 %.2fM | 设备 %s" % (n_par, dev))
    print("%d 个 batch（bs=%d，已跳过首批预热）：%.1fs  →  %.2f s/batch" % (done - 1, BATCH, dt, per_bs))
    n_iter = nblk / float(BATCH)
    per_ep = n_iter * per_bs
    print("1 epoch = %.0f 个 batch（%d 块 ÷ bs=%d）→ 约 **%.1f 分钟**" % (n_iter, nblk, BATCH, per_ep / 60))
    for e in (20, 50, 200):
        print("   %3d epoch ≈ %6.2f 小时" % (e, per_ep * e / 3600))
    print("-" * 72)


if __name__ == "__main__":
    if not PROJ:
        print("用法: _smoke_mv_speed.py <工程目录> [batch] [n_batch]")
        sys.exit(2)
    main()
